# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C2A full decode attention with mHC."""


import pypto.language as pl
from config import (
    FLASH as M, LOCAL_T_DYN, T_DYN, ORI_BLOCKS_DYN, CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, B_DYN, TABLE_DYN,
    Q_START_DYN, STATE_BLOCKS_DYN, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, STATE_CAPACITY,
    WINDOW_CACHE_GROUP,
)
from compressor import (
    c2a_compressor_publish, STATE_WIDTH, c2a_compressor_project, c2a_compressor_pool_state,
)
from hc import hc_head_norm, hc_project, hc_sinkhorn, hc_post
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj_rope
from sparse_attention import (
    gather_packed, attend_selected, publish_window_row, GATHER_WORKERS as GATHER_WORKER_TILE, attend_packed_cohorts,
)
from decode_indexer import decode_c2a_indexer
from o_proj import o_proj

# model config
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk
MX_GROUP = M.weight_block_size[0]

IDX_GROUP = INDEX_CACHE_GROUP
CMP_GROUP = COMPRESSED_CACHE_GROUP
RATIO = 2


@pl.jit.inline(auto_scope=False)
def decode_c2a_full(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    cache_ready = hc_head_norm(x_hc, pre_mix, attn_norm_weight, x_norm)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    proj_payload, proj_scales = quantize_projection_input(x_norm, num_tokens)
    qr, query, qb_payload, qb_scales = q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, query,
        num_tokens,
    )
    kv_proj_rope(
        proj_payload, proj_scales,
        wkv, wkv_scale, kv_norm_weight,
        rope_cos, rope_sin,
        kv_projection, window_kv,
        num_tokens,
    )
    publish_window_row(window_kv, window_slots, window_cache, window_cache_scale, num_tokens)

    compressor_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    compressor_score = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    c2a_compressor_project(x_norm, compressor_wkv, compressor_wgate, compressor_kv, compressor_score, num_tokens)
    latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    pool_tid, state_tid = c2a_compressor_pool_state(
        compressor_kv, compressor_score, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, compressor_norm_weight, latent, num_tokens, cache_ready,
    )

    select_tid = decode_c2a_indexer(
        x_norm, qb_payload, qb_scales, latent, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
        index_weights_proj, rope_cos, rope_sin, compressed_rope_cos, compressed_rope_sin,
        compressed_slots, index_cache, index_cache_scale, index_block_table, token_to_req_indices,
        host_token_to_req_indices,
        compressed_lens, topk_indices, num_tokens, cache_ready, pool_tid,
    )
    compressed_publish_tid = c2a_compressor_publish(
        latent, compressed_rope_cos, compressed_rope_sin, compressed_slots,
        compressed_cache, compressed_cache_scale, num_tokens, cache_ready, pool_tid, max_workers=4,
    )

    chunk_done = pl.system.task_dummy(deps=[select_tid, state_tid, compressed_publish_tid])

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    if num_tokens <= 12:
        attend_packed_cohorts(
            query, window_indices, window_cache, window_cache_scale, topk_indices,
            compressed_cache, compressed_cache_scale, attn_sink, attended, num_tokens, chunk_done,
            gather_name="c2a_gather", attn_name="c2a_sparse_attn",
        )
    else:
        selected = pl.create_tensor([tokens, 640, HEAD_DIM], dtype=pl.BF16)
        indices = pl.create_tensor([tokens, 640], dtype=pl.INT32)
        selected_ret, indices_ret, gather_tid = gather_packed(
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            window_indices, topk_indices, selected, indices, num_tokens, chunk_done, max_workers=GATHER_WORKER_TILE,
            name_hint="c2a_gather",
        )
        attend_selected(query, selected, indices, attn_sink, attended, num_tokens, gather_tid)

    inv_rms, partials = hc_project(x_hc, hc_attn_fn)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        attn_out, num_tokens,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(attn_out, x_hc, post_mix, residual_mix, output, deps=hc_post_deps)
    return output, next_pre_mix


@pl.jit
def l2_decode_c2a_full(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    query_start_loc.bind_dynamic(0, Q_START_DYN)
    state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
    return decode_c2a_full(
        x_hc, pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale,
        token_to_req_indices, host_token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table,
        position_ids,
        compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_wgate,
        query_start_loc, state_block_table, state_cache,
        compressor_norm_weight,
        compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices,
        output, next_pre_mix,
        num_tokens,
    )


@pl.jit.host
def l3_decode_c2a_full(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    query_start_loc.bind_dynamic(0, Q_START_DYN)
    state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
    wq_a_scale_mx: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale
    wq_b_scale_mx: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale
    wkv_scale_mx: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale
    wo_b_scale_mx: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale
    index_wq_b_scale_mx: pl.Tensor[[Q_LORA // MX_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale
    l2_decode_c2a_full(
        x_hc, pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale_mx, q_norm_weight, wq_b, wq_b_scale_mx,
        wkv, wkv_scale_mx, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale_mx,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale,
        token_to_req_indices, host_token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table,
        position_ids,
        compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_wgate,
        query_start_loc, state_block_table, state_cache,
        compressor_norm_weight,
        compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_wq_b_scale_mx, index_weights_proj,
        topk_indices,
        output, next_pre_mix,
        num_tokens, device=0,
    )


def golden_c2a(tensors, mode="full"):
    """Compute mHC pre, C2A attention, cache updates, and mHC post."""
    import torch
    from compressor import golden_c2a_compressor
    from hc import golden_mhc_mixes, golden_hc_collapse, golden_hc_post
    from rmsnorm import golden_rms_norm
    from qkv_proj_rope import golden_qkv_proj_rope, golden_rope_tail
    from decode_indexer import golden_c2a_index
    from sparse_attention import golden_selected_attention, KEY_TILE, PACKED_KEY_TILE
    from o_proj import golden_o_proj
    from functools import partial
    from quant import quantize_mxfp8_cache, quantize_mxfp4_cache, dequantize_mxfp4_cache, unpack_bf16_nz
    from utils import _publish_quantized_cache

    t = dict(tensors)
    if "x_hc" in t:
        x_hc = t["x_hc"]
        next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
            x_hc, t["hc_attn_fn"], t["hc_attn_scale"], t["hc_attn_base"],
        )
        attention_input = golden_hc_collapse(x_hc, t["pre_mix"]).to(torch.bfloat16)
        x = golden_rms_norm(attention_input, t["attn_norm_weight"])
    else:
        x = t["x"]
    t["x"] = x
    head_dim = t["window_cache"].shape[-1]

    # Query/KV projection and rotary embedding.
    q, kv, qr = golden_qkv_proj_rope(
        x, t["wq_a"], t["wq_a_scale"], t["q_norm_weight"], t["wq_b"], t["wq_b_scale"],
        t["wkv"], t["wkv_scale"], t["kv_norm_weight"], t["rope_cos"], t["rope_sin"],
        sequential_query=True,
    )

    # Sliding-window cache publication.
    window_cache, window_cache_scale = _publish_quantized_cache(
        t["window_cache"], t["window_cache_scale"], kv, t["window_slots"], quantize_mxfp8_cache,
    )

    if mode == "full":
        # Ratio-2 compressor and ring-state update.
        state_cache = t["state_cache"].clone()
        latent, publish = golden_c2a_compressor(
            x, t["query_start_loc"], t["position_ids"], t["token_to_req_indices"],
            t["state_block_table"], state_cache, t["compressor_wkv"], t["compressor_wgate"],
            t["compressor_norm_weight"],
        )
        # Compressed KV publication.
        rotated = golden_rope_tail(latent, t["compressed_rope_cos"], t["compressed_rope_sin"])
        compressed_slots = t["compressed_slots"].masked_fill(~publish, -1)
        compressed_quantize = partial(quantize_mxfp4_cache, group_size=CMP_GROUP, scale_format="bf16")
        compressed_cache, compressed_cache_scale = _publish_quantized_cache(
            t["compressed_cache"], t["compressed_cache_scale"], rotated, compressed_slots, compressed_quantize,
        )

        # Publish RNE E2M1 index keys with group-32 E8M0 scales.
        index_projection = latent.float() @ unpack_bf16_nz(t["index_wk"]).float()
        index_projection = index_projection.to(torch.bfloat16)
        index_key = golden_rms_norm(index_projection, t["index_norm_weight"])
        index_key = golden_rope_tail(index_key, t["compressed_rope_cos"], t["compressed_rope_sin"])
        index_quantize = partial(quantize_mxfp4_cache, group_size=IDX_GROUP, scale_format="e8m0", round_to_even=True)
        index_cache, index_cache_scale = _publish_quantized_cache(
            t["index_cache"], t["index_cache_scale"], index_key, compressed_slots, index_quantize,
        )

        # Score FP4 queries in original coordinate order with the shared BF16 epilogue.
        topk_indices = golden_c2a_index(t, qr, index_cache, index_cache_scale)

    else:
        compressed_cache = t["compressed_cache"]
        compressed_cache_scale = t["compressed_cache_scale"]
        index_cache = index_cache_scale = state_cache = None
        topk_indices = t["compressed_indices"]

    # Sparse attention over window and compressed rows.
    window_values = window_cache.float()
    window_scales = window_cache_scale.float().repeat_interleave(MX_GROUP, -1)
    window_values = (window_values * window_scales).reshape(-1, head_dim)
    compressed_values = dequantize_mxfp4_cache(compressed_cache, compressed_cache_scale, CMP_GROUP, "bf16")
    compressed_values = compressed_values.flatten(0, -2)
    attended = golden_selected_attention(
        q, window_values.reshape(-1, 1, 1, head_dim), t["window_indices"],
        compressed_values.reshape(-1, 1, 1, head_dim), topk_indices, t["attn_sink"],
        key_tile=KEY_TILE if q.shape[0] > 12 else PACKED_KEY_TILE,
    )

    # Inverse RoPE and output projection.
    output = golden_o_proj(
        attended, t["rope_cos"], t["rope_sin"], t["wo_a"], t["wo_b"], t["wo_b_scale"],
        wo_a_scale=t["wo_a_scale"], sequential_projection=True,
    )

    for name, value in (
        ("window_cache", window_cache), ("window_cache_scale", window_cache_scale),
        ("compressed_cache", compressed_cache), ("compressed_cache_scale", compressed_cache_scale),
        ("index_cache", index_cache), ("index_cache_scale", index_cache_scale),
        ("state_cache", state_cache), ("topk_indices", topk_indices), ("candidate_mask", None),
        ("pool_block_ids", None), ("pool_block_counts", None),
    ):
        if value is None:
            tensors[name] = None
        elif name in tensors and isinstance(tensors[name], torch.Tensor):
            tensors[name].view(torch.uint8).copy_(value.contiguous().view(torch.uint8))
        else:
            tensors[name] = value
    if "x_hc" in t:
        hidden = golden_hc_post(output.bfloat16(), x_hc, post_mix, residual_mix)
        tensors["output"].copy_(hidden)
        tensors["next_pre_mix"].copy_(next_pre_mix)
        tensors["attention_input"] = attention_input
    else:
        tensors["output"] = output
    tensors["attention_output"] = output


def build_c2a_specs(
    mode, tokens=24, requests=6, seed=17, case="mixed", start_pos=None, query_lens=None, *, param_names=None,
):
    """Build C2A mHC, attention, cache, and runtime specifications."""
    import torch
    import math
    from config import MAX_BATCH_PER_DP, BLOCK_SIZE
    from utils import packed_query_lengths, paged_slots, window_metadata
    from quant import (
        pack_bf16_nz, pack_fp8_nz,
        quantize_grouped_mxfp8_weight, quantize_mxfp4_cache, quantize_mxfp8_cache,
        quantize_mxfp8_upward_reference, pack_mx_b_scale,
    )
    from rope_tables import precompute_rope_tables

    from functools import partial
    from golden import ScalarSpec, TensorSpec

    if case not in ("mixed", "long", "masked", "zero"):
        raise ValueError(f"unsupported case {case!r}")
    if not 1 <= requests <= min(MAX_BATCH_PER_DP, tokens):
        raise ValueError(f"requests must be in [1, min({MAX_BATCH_PER_DP}, tokens)]")
    gen = torch.Generator().manual_seed(seed)

    lengths = [tokens // requests + (index < tokens % requests) for index in range(requests)]
    prefix_pattern = (1, 63, 128, 255, 0, 511, 129, 256)
    if case == "long":
        prefix_pattern = (1024, 1027, 1290, 63, 1535, 128, 2048, 255)
    prefixes = [prefix_pattern[index % len(prefix_pattern)] for index in range(requests)]
    if query_lens is not None:
        lengths = packed_query_lengths(tokens, requests, query_lens)
    if start_pos is not None:
        if len(start_pos) != requests:
            raise ValueError("start_pos must contain one prefix per request")
        prefixes = list(start_pos)
    token_to_req_indices = torch.repeat_interleave(torch.arange(requests), torch.tensor(lengths))
    positions = torch.cat([torch.arange(prefix, prefix + length) for prefix, length in zip(prefixes, lengths)])
    maximum = int(positions.max())
    window_columns = maximum // BLOCK_SIZE + 1
    compressed_columns = max(4, (maximum // RATIO) // BLOCK_SIZE + 1)

    # One unmapped control page per cache.
    window_pages = requests * window_columns + 1
    compressed_pages = requests * compressed_columns + 1
    window_rows = torch.randperm(window_pages, generator=gen)[: requests * window_columns]
    window_table = window_rows.reshape(requests, window_columns).to(torch.int32)
    index_rows = torch.randperm(compressed_pages, generator=gen)[: requests * compressed_columns]
    index_block_table = index_rows.reshape(requests, compressed_columns).to(torch.int32)

    window_slots, window_indices, _ = window_metadata(positions, token_to_req_indices, window_table)
    compressed_slots = paged_slots(positions, token_to_req_indices, index_block_table, BLOCK_SIZE, RATIO, True)
    compressed_lens = torch.div(positions + 1, RATIO, rounding_mode="floor").to(torch.int32)

    # Token-local rotary tables.
    table_rows = int(positions.max()) + 1
    freqs_cos, freqs_sin = precompute_rope_tables(table_rows, compressed_attention=True)
    compressed_freqs_cos, compressed_freqs_sin = freqs_cos, freqs_sin
    rope_cos, rope_sin = freqs_cos[positions], freqs_sin[positions]
    complete = (positions + 1).remainder(RATIO) == 0
    pair_start = torch.where(complete, positions + 1 - RATIO, torch.full_like(positions, -1))
    compressed_rope_cos = compressed_freqs_cos[pair_start.clamp_min(0)].masked_fill(~complete[:, None], 1.0)
    compressed_rope_sin = compressed_freqs_sin[pair_start.clamp_min(0)].masked_fill(~complete[:, None], 0.0)

    weights = {}
    shapes = {
        "wq_a": (D, Q_LORA),
        "wq_b": (Q_LORA, LOCAL_H * HEAD_DIM),
        "wkv": (D, HEAD_DIM),
        "wo_b": (LOCAL_O_WIDTH, D),
        "index_wq_b": (Q_LORA, INDEX_H * INDEX_DIM),
    }
    for name, (k, n) in shapes.items():
        weight = torch.randn(n, k, generator=gen) / math.sqrt(k)
        payload, codes = quantize_mxfp8_upward_reference(weight)
        weights[name] = payload.T.contiguous()
        if name in ("wq_a", "wq_b", "wkv", "wo_b", "index_wq_b"):
            weights[name] = pack_fp8_nz(weights[name])
        weights[name + "_scale"] = pack_mx_b_scale(codes.T.contiguous()).view(torch.float8_e8m0fnu)

    compressor_wkv = (torch.randn(D, HEAD_DIM, generator=gen) / math.sqrt(D)).bfloat16()
    compressor_wgate = (torch.randn(D, HEAD_DIM, generator=gen) / math.sqrt(D)).bfloat16()
    wo_a_values = torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, generator=gen)
    # Predecessor projections in each request's ring block.
    pending = torch.randn(MAX_BATCH_PER_DP, D, generator=gen).bfloat16().float()
    state_cache = torch.zeros(MAX_BATCH_PER_DP + 1, STATE_CAPACITY, STATE_WIDTH)
    state_block_table = torch.arange(requests - 1, -1, -1, dtype=torch.int32).unsqueeze(1) + 1
    for request, prefix in enumerate(prefixes):
        if prefix:
            block = int(state_block_table[request, 0])
            slot = (prefix - 1) % STATE_CAPACITY
            state_cache[block, slot, :HEAD_DIM] = pending[request] @ compressor_wkv.float()
            state_cache[block, slot, HEAD_DIM:] = pending[request] @ compressor_wgate.float()
    query_start_loc = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)

    window_values = torch.randn(window_pages, BLOCK_SIZE, 1, HEAD_DIM, generator=gen).bfloat16()
    window_cache, window_cache_scales = quantize_mxfp8_cache(window_values)
    compressed_values = torch.randn(compressed_pages, BLOCK_SIZE, 1, HEAD_DIM, generator=gen).bfloat16()
    compressed_cache, compressed_cache_scale = quantize_mxfp4_cache(compressed_values, CMP_GROUP, "bf16")
    index_values = torch.randn(compressed_pages, BLOCK_SIZE, 1, INDEX_DIM, generator=gen).bfloat16()
    index_cache, index_cache_codes = quantize_mxfp4_cache(index_values, IDX_GROUP, "e8m0")

    values = {
        "x": torch.randn(tokens, D, generator=gen).bfloat16(),
        "wq_a": weights["wq_a"],
        "wq_a_scale": weights["wq_a_scale"],
        "q_norm_weight": (torch.randn(Q_LORA, generator=gen) * 0.1 + 1).bfloat16(),
        "wq_b": weights["wq_b"],
        "wq_b_scale": weights["wq_b_scale"],
        "wkv": weights["wkv"],
        "wkv_scale": weights["wkv_scale"],
        "kv_norm_weight": (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "attn_sink": torch.randn(LOCAL_H, generator=gen) * 2,
        "wo_a": (wo_a_values / math.sqrt(O_GROUP_IN)).bfloat16(),
        "wo_b": weights["wo_b"],
        "wo_b_scale": weights["wo_b_scale"],
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "window_slots": window_slots,
        "window_indices": window_indices,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scales,
        "compressed_cache": compressed_cache,
        "compressed_cache_scale": compressed_cache_scale,
        "token_to_req_indices": token_to_req_indices.to(torch.int32),
        "compressed_lens": compressed_lens,
        "index_cache": index_cache,
        "index_cache_scale": index_cache_codes.view(torch.float8_e8m0fnu),
        "index_block_table": index_block_table,
        "position_ids": positions.to(torch.int32),
        "compressed_rope_cos": compressed_rope_cos,
        "compressed_rope_sin": compressed_rope_sin,
        "compressor_wkv": pack_bf16_nz(compressor_wkv),
        "compressor_wgate": pack_bf16_nz(compressor_wgate),
        "query_start_loc": query_start_loc,
        "state_block_table": state_block_table,
        "state_cache": state_cache,
        "compressor_norm_weight": (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "compressed_slots": compressed_slots,
        "index_wk": (torch.randn(HEAD_DIM, INDEX_DIM, generator=gen) / math.sqrt(HEAD_DIM)).bfloat16(),
        "index_norm_weight": (torch.randn(INDEX_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "index_wq_b": weights["index_wq_b"],
        "index_wq_b_scale": weights["index_wq_b_scale"],
        "index_weights_proj": (torch.randn(D, INDEX_H, generator=gen) / math.sqrt(D)).bfloat16(),
    }

    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(values["wo_a"])

    for name in ("index_wk", "index_weights_proj"):
        values[name] = pack_bf16_nz(values[name])

    if case == "masked":
        values["window_slots"] = torch.full_like(window_slots, -1)
        values["window_indices"] = torch.full_like(window_indices, -1)
        values["compressed_slots"] = torch.full_like(compressed_slots, -1)
        values["compressed_lens"] = torch.zeros_like(compressed_lens)
        values["state_block_table"] = torch.full_like(state_block_table, -1)
    if case == "zero":
        values["x"] = torch.zeros_like(values["x"])
        values["window_cache"] = torch.zeros_like(window_cache.view(torch.uint8)).view(window_cache.dtype)
        values["compressed_cache"] = torch.zeros_like(compressed_cache)
        values["index_cache"] = torch.zeros_like(index_cache)
        values["state_cache"] = torch.zeros_like(state_cache)

    print(
        f"[FIXTURE] c2a tokens={tokens} requests={requests} case={case} "
        f"lengths={lengths} prefixes={prefixes} "
        f"max_compressed_len={int(values['compressed_lens'].max())} "
        f"window_pages={window_pages} compressed_pages={compressed_pages} "
        f"index_block_table={tuple(index_block_table.shape)}"
    )
    if mode == "reuse":
        source = values
        published = {name: value.clone() for name, value in source.items()}
        golden_c2a(published)
        values["compressed_cache"] = published["compressed_cache"]
        values["compressed_cache_scale"] = published["compressed_cache_scale"]
        values["compressed_indices"] = published["topk_indices"]
        reachable = int((published["topk_indices"] >= 0).sum())
        print(
            f"[FIXTURE] c2a reuse tokens={tokens} requests={requests} case={case} "
            f"selected_rows={reachable} source_published_rows="
            f"{int((source['compressed_slots'] >= 0).sum())}"
        )
    del values["x"]
    generator = torch.Generator().manual_seed(seed + 1000)
    shapes = {
        "x_hc": [tokens, HC_MULT, D],
        "pre_mix": [tokens, HC_MULT],
        "hc_attn_fn": [MIX_HC, HC_DIM],
        "hc_attn_scale": [3],
        "hc_attn_base": [MIX_HC],
        "attn_norm_weight": [D],
    }
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator)
        if name == "hc_attn_fn":
            value /= HC_DIM**0.5
        elif name == "attn_norm_weight":
            value = torch.ones(shape, dtype=torch.bfloat16)
        if name in ("x_hc", "pre_mix"):
            value = torch.randn(shape, generator=generator)
            if name == "pre_mix":
                value = torch.softmax(value, dim=-1)
        values[name] = value
    if mode == "full":
        values["topk_indices"] = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    values["output"] = torch.zeros(tokens, HC_MULT, D)
    values["next_pre_mix"] = torch.zeros(tokens, HC_MULT)
    names = l3_decode_c2a_full.param_names if param_names is None else param_names
    specs = [
        TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=partial(torch.clone, values[name]))
        for name in names if name != "num_tokens"
    ]
    specs.append(ScalarSpec("num_tokens", torch.int32, tokens))
    return specs


if __name__ == "__main__":
    from config import BLOCK_SIZE

    from golden import ratio_allclose, mapped_pool_ratio_allclose, run

    from decode_indexer import mapped_fp4_cache_compare
    from utils import configure_decode_workload, parse_decode_query_lens, parse_decode_startpos

    import argparse
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1 A5 mHC C2A full validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one starting position per request; omitted: default validation workload"
    parser.add_argument("--start-pos", "--startpos", dest="startpos", type=parse_decode_startpos, help=startpos_help)
    query_help = "one query length per start-pos; omitted: config.DECODE_ROWS_PER_REQUEST each"
    parser.add_argument("--query-lens", type=parse_decode_query_lens, help=query_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["mixed", "long", "masked", "zero"], default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=72, requests=12, startpos=[8192] * 12)
    args = parser.parse_args()
    configure_decode_workload(args, parser)
    if args.tokens != 72:
        parser.error("this decode entry requires 72 query tokens")
    host = l3_decode_c2a_full
    specs = build_c2a_specs(
        "full", args.tokens, args.requests, args.seed, args.case,
        start_pos=args.startpos, query_lens=args.query_lens, param_names=host.param_names,
    )
    result = run(
        fn=host,
        specs=specs,
        golden_fn=golden_c2a,
        compare_fn={
            "output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "window_cache": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window_cache",
                mapping_shape=(args.tokens,),
                block_size=BLOCK_SIZE, leading_rank_axis=False,
                atol=2 ** -9, rtol=2 ** -3, max_error_ratio=0.0,
            ),
            "window_cache_scale": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window_cache_scale",
                mapping_shape=(args.tokens,),
                block_size=BLOCK_SIZE, leading_rank_axis=False,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "compressed_cache": mapped_fp4_cache_compare(
                "compressed_slots", pool_name="compressed_cache",
                mapping_shape=(args.tokens,),
                block_size=BLOCK_SIZE, leading_rank_axis=False,
                max_error_ratio=0.005,
            ),
            "compressed_cache_scale": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="compressed_cache_scale",
                mapping_shape=(args.tokens,),
                block_size=BLOCK_SIZE, leading_rank_axis=False,
                atol=0.0, rtol=2 ** -7, max_error_ratio=0.0,
            ),
            "index_cache": mapped_fp4_cache_compare(
                "compressed_slots", pool_name="index_cache",
                mapping_shape=(args.tokens,),
                block_size=BLOCK_SIZE, leading_rank_axis=False,
                max_error_ratio=0.005,
            ),
            "index_cache_scale": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="index_cache_scale",
                mapping_shape=(args.tokens,),
                block_size=BLOCK_SIZE, leading_rank_axis=False,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "state_cache": ratio_allclose(atol=1e-3, rtol=1e-3),
            "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
        },
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            enable_chip_swimlane=args.enable_chip_swimlane,
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
        ),
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
