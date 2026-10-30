# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C1A full decode attention with mHC."""


import pypto.language as pl
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj_rope
from o_proj import o_proj
from decode_indexer import (
    select_candidate_block_pool, c1a_index, use_pretransposed_index_query, score_select_c1a_groups,
    publish_index_rows, c1a_project_index_key, c1a_rotate_index_key,
)
from sparse_attention import (
    attend_selected, gather_packed, publish_window_row, GATHER_WORKERS as GATHER_WORKER_TILE,
)
from hc import hc_project, hc_sinkhorn, hc_head_norm, hc_post
from config import (
    LOCAL_T_DYN, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, WINDOW_CACHE_GROUP, FLASH as M, B_DYN,
    CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, INDEX_POOL_BLOCKS_DYN, ORI_BLOCKS_DYN, TABLE_DYN, T_DYN,
)
from rmsnorm import rms_norm
from compressor import c1a_compressor_project, compressor_rope, c1a_compressor_publish

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


@pl.jit.inline(auto_scope=False)
def decode_c1a_full(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    device_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run C1A with separate graph-control and device request mappings.

    host_token_to_req_indices is a CPU INT32 tensor under HBG. device_token_to_req_indices is
    the identical token-to-batch-row mapping consumed by device tasks. Both
    come from one scheduling snapshot and remain valid until completion.
    Ordinary HBG calls rebuild routing for each snapshot. External graph
    capture must recapture when request-run boundaries or shapes change.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    inv_rms, partials = hc_project(x_hc, hc_attn_fn)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    x_ready = hc_head_norm(x_hc, pre_mix, attn_norm_weight, x_norm)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
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
        kv_projection, kv,
        num_tokens,
    )
    publish_window_row(kv, window_slots, window_cache, window_cache_scale, num_tokens)
    projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    projected, pc_tid = c1a_compressor_project(x_norm, compressor_wkv, projected, num_tokens, x_ready)
    latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    with pl.spmd((num_tokens + 7) // 8, name_hint="norm_cmp", deps=[pc_tid]) as nc_tid:
        norm_worker = pl.tile.get_block_idx()
        rms_norm(projected, compressor_norm_weight, latent, norm_worker, num_tokens, width=HEAD_DIM)
    projected_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    pk_tid = c1a_project_index_key(latent, index_wk, projected_key, num_tokens, nc_tid)
    normalized_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    with pl.spmd((num_tokens + 7) // 8, name_hint="norm_idx_k", deps=[pk_tid]) as nk_tid:
        norm_worker = pl.tile.get_block_idx()
        rms_norm(projected_key, index_norm_weight, normalized_key, norm_worker, num_tokens, width=INDEX_DIM)
    index_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    ik_tid = c1a_rotate_index_key(
        normalized_key, compressed_rope_cos, compressed_rope_sin, index_key, num_tokens, nk_tid,
    )
    index_ready = publish_index_rows(
        index_key, compressed_slots, index_cache, index_cache_scale, num_tokens, x_ready, ik_tid,
    )
    compressed_rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    compressed_rotated = compressor_rope(
        latent, compressed_rope_cos, compressed_rope_sin, compressed_rotated, num_tokens,
    )
    compressed_ready = c1a_compressor_publish(
        compressed_rotated, compressed_slots, compressed_cache, compressed_cache_scale, num_tokens, x_ready,
    )
    # Logical index score capacity.
    score_width = pl.tensor.dim(index_block_table, 1) * 128
    private_route = use_pretransposed_index_query(
        host_token_to_req_indices, compressed_lens, num_tokens, score_width, skip_short=False,
    )
    private_query = pl.create_tensor([16, INDEX_H * INDEX_DIM], dtype=pl.FP8E4M3FN)
    (index_query, index_query_scale, index_weights, scores, query_tid, weights_tid) = c1a_index(
        x_norm, index_wq_b, index_wq_b_scale, index_weights_proj, rope_cos, rope_sin,
        num_tokens, score_width, x_ready, private_query, private_route,
        qb_payload, qb_scales,
        pretransposed=True,
    )
    topk_tid, scores_tid = score_select_c1a_groups(
        index_query, index_query_scale, index_weights,
        index_cache, index_cache_scale, index_block_table, device_token_to_req_indices, host_token_to_req_indices, compressed_lens,
        scores, topk_indices,
        num_tokens, index_ready, query_tid, weights_tid, private_query, private_route,
        pl.tensor.dim(pool_block_ids, 1),
    )
    pool_tid = select_candidate_block_pool(
        scores, compressed_lens, pool_block_ids, pool_block_counts, num_tokens, scores_tid,
    )
    selection_ready = pl.system.task_dummy(deps=[topk_tid, pool_tid, compressed_ready])
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected = pl.create_tensor([tokens, 640, HEAD_DIM], dtype=pl.BF16)
    indices = pl.create_tensor([tokens, 640], dtype=pl.INT32)
    selected_ret, indices_ret, gather_tid = gather_packed(
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        window_indices, topk_indices, selected, indices, num_tokens, selection_ready,
        max_workers=GATHER_WORKER_TILE, name_hint="c1a_gather",
    )
    attend_selected(query, selected, indices, attn_sink, attended, num_tokens, gather_tid)
    attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        attn_out, num_tokens,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(attn_out, x_hc, post_mix, residual_mix, output, deps=hc_post_deps)
    return output


@pl.jit
def l2_decode_c1a_full(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    device_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.Out[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
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
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    return decode_c1a_full(
        x_hc, pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        host_token_to_req_indices, device_token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table,
        compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_norm_weight, compressed_slots,
        index_wk, index_norm_weight,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices, pool_block_ids, pool_block_counts,
        output, next_pre_mix,
        num_tokens,
    )


@pl.jit.host
def l3_decode_c1a_full(
    x_hc: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    host_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    device_token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.Out[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
    output: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, LOCAL_T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    # MX_B_NN scale views.
    wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale
    wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale
    wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale
    wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale
    index_wq_b_scale_r: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale
    l2_decode_c1a_full(
        x_hc, pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale_r, q_norm_weight,
        wq_b, wq_b_scale_r,
        wkv, wkv_scale_r, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale_r,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        host_token_to_req_indices, device_token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table,
        compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_norm_weight, compressed_slots,
        index_wk, index_norm_weight,
        index_wq_b, index_wq_b_scale_r, index_weights_proj,
        topk_indices, pool_block_ids, pool_block_counts,
        output, next_pre_mix,
        num_tokens,
        device=0,
    )


def golden_c1a(tensors, mode="full"):
    """Compute mHC pre, C1A attention, cache updates, and mHC post."""
    import torch

    from compressor import golden_c1a_compressor
    from qkv_proj_rope import golden_qkv_proj_rope, golden_rope_tail
    from o_proj import golden_o_proj
    from decode_indexer import (
        golden_index_key, golden_select_candidate_block_pool, golden_c1a_index_scores,
        golden_pool_index_scores, golden_c1a_index_topk,
    )
    from sparse_attention import golden_selected_attention
    from quant import (
        quantize_mxfp4_cache, quantize_mxfp8_cache, dequantize_mxfp4_cache, dequantize_mxfp8_cache,
    )
    from utils import _publish_quantized_cache

    from hc import golden_mhc_mixes, golden_hc_collapse, golden_hc_post
    from rmsnorm import golden_rms_norm

    a = dict(tensors)
    if mode == "full":
        from utils import validate_request_id_tensors

        active_tokens = int(a.get("num_tokens", a["device_token_to_req_indices"].shape[0]))
        validate_request_id_tensors(a["host_token_to_req_indices"], a["device_token_to_req_indices"], active_tokens)
        a["token_to_req_indices"] = a["device_token_to_req_indices"]
    x_hc = a["x_hc"]
    next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        x_hc, a["hc_attn_fn"], a["hc_attn_scale"], a["hc_attn_base"],
    )
    attention_input = golden_hc_collapse(x_hc, a["pre_mix"]).to(torch.bfloat16)
    x = golden_rms_norm(attention_input, a["attn_norm_weight"])
    a["x"] = x
    q, kv, qr = golden_qkv_proj_rope(
        x, a["wq_a"], a["wq_a_scale"], a["q_norm_weight"], a["wq_b"], a["wq_b_scale"],
        a["wkv"], a["wkv_scale"], a["kv_norm_weight"], a["rope_cos"], a["rope_sin"], sequential_query=True,
    )
    window, window_scale = _publish_quantized_cache(
        a["window_cache"], a["window_cache_scale"], kv, a["window_slots"], quantize_mxfp8_cache,
    )
    compressed = a["compressed_cache"].view(torch.uint8).clone()
    compressed_scale = a["compressed_cache_scale"].clone()
    index = a.get("index_cache") if mode != "reuse" else None
    index_scale = a.get("index_cache_scale") if mode != "reuse" else None
    if index is not None:
        index = index.view(torch.uint8).clone()
        index_scale = index_scale.view(torch.uint8).clone()
    if mode == "full":
        latent = golden_c1a_compressor(x, a["compressor_wkv"], a["compressor_norm_weight"])
        key = golden_index_key(
            latent, a["index_wk"], a["index_norm_weight"], a["compressed_rope_cos"], a["compressed_rope_sin"],
        )
        index, index_scale = _publish_quantized_cache(
            index, index_scale, key, a["compressed_slots"],
            lambda value: quantize_mxfp4_cache(value, 32, "e8m0", round_to_even=True),
        )
        rotated = golden_rope_tail(latent, a["compressed_rope_cos"], a["compressed_rope_sin"])
        compressed, compressed_scale = _publish_quantized_cache(
            compressed, compressed_scale, rotated, a["compressed_slots"],
            lambda value: quantize_mxfp4_cache(value, 16, "bf16", round_to_even=True),
        )
    pool_block_ids = a.get("pool_block_ids") if mode == "reindex" else None
    pool_block_counts = a.get("pool_block_counts") if mode == "reindex" else None
    topk = a.get("compressed_indices")
    if mode != "reuse":
        lengths = a["compressed_lens"]
        scores, physical = golden_c1a_index_scores(a, qr, index, index_scale)
        if mode == "reindex":
            if pool_block_ids is None or pool_block_counts is None:
                raise ValueError("C1A Reindex requires compact block-pool metadata")
            scores, physical = golden_pool_index_scores(scores, physical, pool_block_ids, pool_block_counts)
        topk = golden_c1a_index_topk(scores, physical, lengths, pool_block_counts)
        if mode == "full":
            pool_block_ids, pool_block_counts = golden_select_candidate_block_pool(scores, lengths)
    window_value = dequantize_mxfp8_cache(window, window_scale).to(x.dtype)
    compressed_value = dequantize_mxfp4_cache(compressed, compressed_scale, 16, "bf16").to(x.dtype)
    attended = golden_selected_attention(q, window_value, a["window_indices"], compressed_value, topk, a["attn_sink"])
    output = golden_o_proj(
        attended, a["rope_cos"], a["rope_sin"], a["wo_a"], a["wo_b"], a["wo_b_scale"],
        wo_a_scale=a["wo_a_scale"], sequential_projection=True,
    )
    hidden = golden_hc_post(output.to(torch.bfloat16), x_hc, post_mix, residual_mix)
    tensors["next_pre_mix"].copy_(next_pre_mix)
    for name, value in (
        ("window_cache", window), ("window_cache_scale", window_scale),
        ("compressed_cache", compressed), ("compressed_cache_scale", compressed_scale),
        ("index_cache", index), ("index_cache_scale", index_scale),
        ("state_cache", None), ("topk_indices", topk), ("candidate_mask", None),
        ("pool_block_ids", pool_block_ids), ("pool_block_counts", pool_block_counts),
    ):
        if value is None:
            tensors[name] = None
        elif name in tensors and isinstance(tensors[name], torch.Tensor):
            tensors[name].view(torch.uint8).copy_(value.contiguous().view(torch.uint8))
        else:
            tensors[name] = value
    tensors["output"].copy_(hidden)
    tensors["attention_output"] = output
    tensors["attention_input"] = attention_input


def build_c1a_specs(
    mode, tokens, pages, requests=1, seed=17, case="random", start_pos=None, query_lens=None, *,
    param_names=None, request_order=None,
):
    """Build C1A mHC, attention, cache, and runtime tensor specifications."""
    import torch

    from functools import partial
    from golden import ScalarSpec, TensorSpec
    from decode_indexer import index_pool_block_capacity
    from quant import quantize_grouped_mxfp8_weight, quantize_mxfp4_cache, quantize_mxfp8_cache
    from utils import packed_query_lengths
    from rope_tables import precompute_rope_tables
    import math

    TOKENS, PAGES, REQUESTS = tokens, pages, requests
    from quant import pack_bf16_nz, pack_fp8_nz, pack_mx_b_scale

    cache_pages = PAGES * REQUESTS if start_pos is not None else PAGES
    torch.manual_seed(seed)
    # Packed token-to-request split.
    lengths = packed_query_lengths(TOKENS, REQUESTS, query_lens)
    order = list(range(REQUESTS)) if request_order is None else list(request_order)
    if sorted(order) != list(range(REQUESTS)):
        raise ValueError("request_order must be a permutation of the batch request rows")
    ordered_lengths = torch.tensor([lengths[request] for request in order])
    token_to_req = torch.repeat_interleave(torch.tensor(order), ordered_lengths).to(torch.int32)
    values = {}
    shapes = {
        "wq_a": (D, Q_LORA),
        "wq_b": (Q_LORA, LOCAL_H * HEAD_DIM),
        "wkv": (D, HEAD_DIM),
        "wo_b": (LOCAL_O_WIDTH, D),
    }
    for name, (k, n) in shapes.items():
        values[name] = torch.randn(k, n).to(torch.float8_e4m3fn)
        if name in ("wq_a", "wq_b", "wkv", "wo_b"):
            values[name] = pack_fp8_nz(values[name])
        exponent = round(127 - math.log2(k) / 2)
        weight_scale_codes = torch.full((k // 32, n), exponent, dtype=torch.uint8)
        values[name + "_scale"] = pack_mx_b_scale(weight_scale_codes).view(torch.float8_e8m0fnu)
    values["x"] = torch.randn(TOKENS, D, dtype=torch.bfloat16)
    values["q_norm_weight"] = torch.ones(Q_LORA, dtype=torch.bfloat16)
    values["kv_norm_weight"] = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
    values["attn_sink"] = torch.randn(LOCAL_H)
    values["wo_a"] = (torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(torch.bfloat16)
    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(values["wo_a"])
    positions = torch.arange(PAGES * 128 - 1, PAGES * 128 - TOKENS - 1, -1)
    if start_pos is not None:
        positions = torch.cat([torch.arange(start_pos[request], start_pos[request] + lengths[request]) for request in order])
    freqs_cos, freqs_sin = precompute_rope_tables(PAGES * 128, compressed_attention=True)
    values["rope_cos"] = freqs_cos[positions]
    values["rope_sin"] = freqs_sin[positions]
    values["window_slots"] = torch.arange(TOKENS, dtype=torch.int64)
    values["window_indices"] = torch.arange(128, dtype=torch.int32).view(1, 128).expand(TOKENS, -1).contiguous()
    values["compressed_indices"] = torch.full((TOKENS, INDEX_TOPK), -1, dtype=torch.int32)
    values["compressed_indices"][:, :128] = torch.arange(128, dtype=torch.int32)
    wc, ws = quantize_mxfp8_cache(torch.randn(cache_pages, 128, 1, HEAD_DIM))
    cc, cs = quantize_mxfp4_cache(torch.randn(cache_pages, 128, 1, HEAD_DIM), 16, "bf16")
    values["window_cache"], values["window_cache_scale"] = wc, ws
    values["compressed_cache"], values["compressed_cache_scale"] = cc, cs
    values["output"] = torch.zeros(TOKENS, D, dtype=torch.bfloat16)
    values["index_wq_b"] = pack_fp8_nz(torch.randn(Q_LORA, INDEX_H * INDEX_DIM).to(torch.float8_e4m3fn))
    index_weight_scale_codes = torch.full((Q_LORA // 32, INDEX_H * INDEX_DIM), 122, dtype=torch.uint8)
    values["index_wq_b_scale"] = pack_mx_b_scale(index_weight_scale_codes).view(torch.float8_e8m0fnu)
    # Non-negative index head weights.
    values["index_weights_proj"] = pack_bf16_nz((torch.rand(D, INDEX_H) / math.sqrt(D)).to(torch.bfloat16))
    values["token_to_req_indices"] = token_to_req
    values["host_token_to_req_indices"] = token_to_req.clone()
    values["device_token_to_req_indices"] = token_to_req.clone()
    values["compressed_lens"] = torch.tensor([PAGES * 128 - t for t in range(TOKENS)], dtype=torch.int32)
    values["index_block_table"] = torch.randperm(PAGES).int().view(1, PAGES).expand(REQUESTS, -1).contiguous()
    if start_pos is not None:
        from utils import window_metadata

        table = torch.arange(cache_pages, dtype=torch.int32).reshape(REQUESTS, PAGES)
        slots, indices, _ = window_metadata(positions, token_to_req, table)
        values["window_slots"] = slots
        values["window_indices"] = indices
        values["compressed_lens"] = (positions + 1).int()
        values["index_block_table"] = table
        selected = torch.arange(INDEX_TOPK).expand(TOKENS, -1)
        valid = selected < values["compressed_lens"][:, None]
        physical = token_to_req[:, None] * PAGES * 128 + selected
        values["compressed_indices"] = physical.masked_fill(~valid, -1).int()
        values["compressed_indices"] = values["compressed_indices"]

    # Page-level index-key magnitude ladder.
    quiet_page = torch.arange(cache_pages).remainder(PAGES) >= PAGES // 2
    page_gain = torch.where(quiet_page, torch.full((cache_pages,), 2.0**-10), torch.ones(cache_pages))
    keys = torch.randn(cache_pages, 128, 1, INDEX_DIM) * page_gain.view(cache_pages, 1, 1, 1)
    ic, ics = quantize_mxfp4_cache(keys.to(torch.bfloat16), 32, "e8m0", round_to_even=True)
    values["index_cache"], values["index_cache_scale"] = (ic, ics.view(torch.float8_e8m0fnu))
    values["topk_indices"] = torch.zeros(TOKENS, INDEX_TOPK, dtype=torch.int32)
    if mode == "full":
        values["compressed_rope_cos"] = values["rope_cos"].clone()
        values["compressed_rope_sin"] = values["rope_sin"].clone()
        values["compressor_wkv"] = pack_bf16_nz((torch.randn(D, HEAD_DIM) / math.sqrt(D)).to(torch.bfloat16))
        values["compressor_norm_weight"] = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
        values["index_wk"] = pack_bf16_nz((torch.randn(HEAD_DIM, INDEX_DIM) / math.sqrt(HEAD_DIM)).to(torch.bfloat16))
        # Quiet published index keys.
        values["index_norm_weight"] = torch.full((INDEX_DIM,), 2.0**-10, dtype=torch.bfloat16)
        positions = values["compressed_lens"].long() - 1
        request_rows = values["token_to_req_indices"][:, None].long().expand(-1, PAGES)
        token_tables = values["index_block_table"].gather(0, request_rows)
        physical_pages = token_tables.gather(1, (positions // 128)[:, None]).squeeze(-1).long()
        values["compressed_slots"] = physical_pages * 128 + positions % 128
    if case == "ragged":
        for t in range(TOKENS):
            length = (PAGES * 128, 0, 7, 129)[t % 4]
            values["compressed_lens"][t] = min(length, PAGES * 128)
            values["window_indices"][t, (0 if t % 4 == 1 else 17) :] = -1
            values["compressed_indices"][t, (0 if t % 4 == 1 else 31) :] = -1
            if t % 4 == 1:
                values["window_slots"][t] = -1
        if mode == "full":
            positions = values["compressed_lens"].long() - 1
            safe = positions.clamp_min(0)
            request_rows = values["token_to_req_indices"][:, None].long().expand(-1, PAGES)
            token_tables = values["index_block_table"].gather(0, request_rows)
            physical_pages = token_tables.gather(1, (safe // 128)[:, None]).squeeze(-1).long()
            values["compressed_slots"] = physical_pages * 128 + safe % 128
            values["compressed_slots"][positions < 0] = -1
    elif case == "masked":
        values["window_indices"].fill_(-1)
        values["compressed_indices"].fill_(-1)
        values["compressed_lens"].zero_()
        if mode == "full":
            values["compressed_slots"].fill_(-1)
    elif case == "sink":
        values["attn_sink"].fill_(10000)

    # Build sorted complete-block fixtures with the newest causal block.
    pool_capacity = index_pool_block_capacity(PAGES * 128)
    pool_ids = torch.full((TOKENS, pool_capacity), -1, dtype=torch.int32)
    pool_counts = torch.zeros((TOKENS), dtype=torch.int32)
    for token in range(TOKENS):
        visible = max(0, min(int(values["compressed_lens"][token]), PAGES * 128))
        block_count = (visible + M.candidate_block_size - 1) // M.candidate_block_size
        selected = min(block_count, pool_capacity)
        if selected == 0:
            continue
        if selected == block_count:
            chosen = torch.arange(selected, dtype=torch.int32)
        else:
            previous_blocks = torch.arange(selected - 1, dtype=torch.int32)
            newest_block = torch.tensor([block_count - 1], dtype=torch.int32)
            chosen = torch.cat((previous_blocks, newest_block))
        pool_ids[token, :selected] = chosen
        pool_counts[token] = selected
    values["pool_block_ids"] = pool_ids
    values["pool_block_counts"] = pool_counts
    del values["x"]
    generator = torch.Generator().manual_seed(seed)
    local_tokens = tokens
    values["x_hc"] = torch.randn(local_tokens, HC_MULT, D, generator=generator)
    # Incoming staggered mHC mix.
    values["pre_mix"] = torch.sigmoid(torch.randn(local_tokens, HC_MULT, generator=generator))
    values["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)
    values["hc_attn_scale"] = torch.randn(3, generator=generator)
    values["hc_attn_base"] = torch.randn(MIX_HC, generator=generator)
    # Attention RMSNorm weights.
    values["attn_norm_weight"] = (torch.randn(D, generator=generator) * 0.1 + 1.0).to(torch.bfloat16)
    values["output"] = torch.zeros(local_tokens, HC_MULT, D, dtype=torch.float32)
    values["next_pre_mix"] = torch.zeros(local_tokens, HC_MULT, dtype=torch.float32)
    names = l3_decode_c1a_full.param_names if param_names is None else param_names
    specs = [
        TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=partial(torch.clone, values[name]))
        for name in names if name != "num_tokens"
    ]
    specs.append(ScalarSpec("num_tokens", torch.int32, tokens, compile_runtime=True))
    return specs


def validate_request_metadata():
    """Validate reordered, ragged and padded Host/Device mapping snapshots."""
    import torch

    from utils import validate_request_id_tensors

    host = torch.tensor([2, 0, 0, 1, 1, 1, -1], dtype=torch.int32)
    device = host.clone()
    validate_request_id_tensors(host, device, 6)
    device[-1] = 9
    validate_request_id_tensors(host, device, 6)
    invalid = device.clone()
    invalid[2] = 2
    cases = ((host, invalid, 6), (host.long(), device, 6), (host, device[:-1], 6), (host, device, 8))
    for control, snapshot, active in cases:
        try:
            validate_request_id_tensors(control, snapshot, active)
        except ValueError:
            continue
        raise RuntimeError("invalid request metadata was accepted")
    print("[REQUEST_METADATA] PASS: ragged/reordered mapping, inactive padding and four rejection cases")


if __name__ == "__main__":
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run
    from config import BLOCK_SIZE

    from decode_indexer import mapped_fp4_cache_compare
    from utils import configure_decode_workload, parse_decode_query_lens, parse_decode_startpos

    import argparse

    from pypto import passes
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1 A5 mHC C1A full validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one starting position per request; omitted: default validation workload"
    parser.add_argument("--start-pos", "--startpos", dest="startpos", type=parse_decode_startpos, help=startpos_help)
    query_help = "one query length per start-pos; omitted: config.DECODE_ROWS_PER_REQUEST each"
    parser.add_argument("--query-lens", type=parse_decode_query_lens, help=query_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["random", "ragged", "masked", "sink"], default="random")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--check-request-metadata", action="store_true", help="run CPU mapping contract checks")
    parser.add_argument("--runtime", choices=["host_build_graph", "tensormap_and_ringbuffer"], default="tensormap_and_ringbuffer")
    parser.add_argument("--request-order", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=72, requests=12, pages=8, startpos=[8192] * 12)
    args = parser.parse_args()
    if args.check_request_metadata:
        validate_request_metadata()
        raise SystemExit(0)
    configure_decode_workload(args, parser)
    if args.tokens != 72:
        parser.error("this decode entry requires 72 query tokens")
    if args.startpos is not None:
        max_end = max(pos + length for pos, length in zip(args.startpos, args.query_lens))
        args.pages = (max_end + 127) // 128
    host = l3_decode_c1a_full
    specs = build_c1a_specs(
        "full", args.tokens, args.pages, args.requests, args.seed, args.case,
        start_pos=args.startpos, query_lens=args.query_lens, param_names=host.param_names,
        request_order=args.request_order,
    )

    runtime = getattr(passes.RuntimeKind, args.runtime.upper())
    with passes.PassContext([], runtime=runtime):
        result = run(
            fn=host,
            specs=specs,
            golden_fn=golden_c1a,
            compile_only=args.compile_only,
            save_data=args.save_data,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                dump_passes=args.dump_passes,
                enable_chip_swimlane=args.enable_chip_swimlane,
                distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
                "pool_block_ids": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
                "pool_block_counts": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
                "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "output": ratio_allclose(atol=1e-2, rtol=1e-2, max_error_ratio=0.01),
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
                    atol=0.0, rtol=2 ** -7, max_error_ratio=0.0,
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
            },
        )
    if not result.passed:
        raise SystemExit(result.error or 1)
