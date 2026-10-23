# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous-batch decode C1A full attention wired through mHC."""


import inspect

import pypto.language as pl
import torch

from golden import ScalarSpec, TensorSpec, mapped_pool_ratio_allclose, ratio_allclose, run
from compressor import golden_compressor_ratio1
from qkv_proj_rope import (
    golden_qkv_proj_rope,
    golden_rope_tail,
    qkv_proj_rope,
)
from o_proj import o_proj, o_proj_hc_post, golden_o_proj
from decode_indexer import (
    golden_index_key, golden_select_candidate_block_pool, select_candidate_block_pool,
    c1a_index, use_pretransposed_index_query, score_full_pretransposed,
    golden_c1a_index_scores,
    golden_pool_index_scores,
    golden_c1a_index_topk,
    make_fp4_publish,
    select_index_topk,
)
from sparse_attention import golden_selected_attention, attend_selected, gather_packed, publish_window_row
from sparse_attention import GATHER_WORKERS, attend_packed_cohorts
from quant import quantize_grouped_mxfp8_weight, quantize_mxfp4_cache, quantize_mxfp8_cache, dequantize_mxfp4_cache, dequantize_mxfp8_cache
from utils import _publish_quantized_cache
from rope_tables import precompute_rope_tables
from utils import AttentionGoldenResult, packed_query_lengths
from decode_indexer import index_pool_block_capacity

from hc import golden_hc_pre, hc_pre
from config import (
    AttentionMode, BLOCK_SIZE,
    LOCAL_T_DYN,
    COMPRESSED_CACHE_GROUP,
    INDEX_CACHE_GROUP,
    WINDOW_CACHE_GROUP,
    FLASH as M,
    B_DYN,
    CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN,
    INDEX_POOL_BLOCKS_DYN,
    ORI_BLOCKS_DYN,
    TABLE_DYN,
    T_DYN,
    TP_SIZE,
)


from hc import golden_mhc_mixes, hc_mixes
from hc import golden_hc_post, hc_post, _hc_post_from_partial
from rmsnorm import golden_rms_norm, rms_norm, rms_norm_tiled


from compressor import decode_c1a_compressor_project, decode_c1a_compressor_rope
from decode_indexer import c1a_project_index_key, c1a_rotate_index_key

# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
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


# tiling
Q_B_WORKERS = 24 if TP_SIZE == 1 else 2147483647
INDEX_QUERY_WORKERS = 8 if TP_SIZE == 1 else 2147483647


# Validation.
OUTPUT_ATOL = 1e-2
OUTPUT_RTOL = 1e-2
OUTPUT_MAX_ERROR_RATIO = 0.01


publish_compressed = make_fp4_publish(HEAD_DIM, 16, pl.BF16, CMP_BLOCKS_DYN, name_hint="pub_cmp")
publish_index = make_fp4_publish(INDEX_DIM, 32, pl.FP8E8M0, INDEX_BLOCKS_DYN, name_hint="pub_idx")


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
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix, sinkhorn_blocks=64)
    x_attn = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(x_hc, pre_mix, x_attn)
    x_norm = pl.create_tensor([tokens, D], dtype=pl.BF16)
    with pl.spmd((tokens + 7) // 8, name_hint="norm_in") as x_ready:
        norm_worker = pl.tile.get_block_idx()
        rms_norm_tiled(x_attn, attn_norm_weight, x_norm, norm_worker, tokens, output_fp32=False)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qkv_proj_rope(
        x_norm, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin,
        qr, query, kv_projection, kv, num_tokens, max_workers=32,
        fuse_qnorm=False, q_b_n_tile=128, q_b_max_workers=Q_B_WORKERS,
    )
    publish_window_row(kv, window_slots, window_cache, window_cache_scale, num_tokens)
    projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    pc_tid = decode_c1a_compressor_project(x_norm, compressor_wkv, projected, num_tokens, x_ready)
    latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    norm_blocks = (num_tokens + 7) // 8
    with pl.spmd(norm_blocks, name_hint="norm_cmp", deps=[pc_tid]) as nc_tid:
        norm_worker = pl.tile.get_block_idx()
        rms_norm(projected, compressor_norm_weight, latent, norm_worker, num_tokens, width=HEAD_DIM)
    projected_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    pk_tid = c1a_project_index_key(latent, index_wk, projected_key, num_tokens, nc_tid)
    normalized_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    norm_blocks = (num_tokens + 7) // 8
    with pl.spmd(norm_blocks, name_hint="norm_idx_k", deps=[pk_tid]) as nk_tid:
        norm_worker = pl.tile.get_block_idx()
        rms_norm(projected_key, index_norm_weight, normalized_key, norm_worker, num_tokens, width=INDEX_DIM)
    index_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    ik_tid = c1a_rotate_index_key(
        normalized_key, compressed_rope_cos, compressed_rope_sin, index_key, num_tokens, nk_tid,
    )
    index_ready = publish_index(
        index_key, compressed_slots, index_cache, index_cache_scale, num_tokens, x_ready, ik_tid,
    )
    rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    rc_tid = decode_c1a_compressor_rope(latent, compressed_rope_cos, compressed_rope_sin, rotated, num_tokens, nc_tid)
    compressed_ready = publish_compressed(
        rotated, compressed_slots, compressed_cache, compressed_cache_scale, num_tokens, x_ready, rc_tid,
    )
    # Logical index score capacity.
    score_width = pl.tensor.dim(index_block_table, 1) * 128
    private_route = use_pretransposed_index_query(
        request_ids, compressed_lens, num_tokens, score_width, skip_short=False,
    )
    private_query = pl.create_tensor([16, INDEX_H * INDEX_DIM], dtype=pl.FP8E4M3FN)
    (index_query, index_query_scale, index_weights, scores, query_tid, weights_tid) = c1a_index(
        x_norm, qr, index_wq_b, index_wq_b_scale, index_weights_proj, rope_cos, rope_sin,
        num_tokens, score_width, x_ready, private_query, private_route,
        pretransposed=True, query_max_workers=INDEX_QUERY_WORKERS,
    )
    scores_tid = score_full_pretransposed(
        index_query, index_query_scale, index_weights,
        index_cache, index_cache_scale, index_block_table, request_ids, compressed_lens,
        scores,
        num_tokens, index_ready, query_tid, weights_tid, private_query, private_route,
    )
    topk_tid = select_index_topk(scores, index_block_table, request_ids, topk_indices, num_tokens, scores_tid)
    pool_tid = select_candidate_block_pool(
        scores, compressed_lens, pool_block_ids, pool_block_counts, num_tokens, scores_tid,
    )
    selection_ready = pl.system.task_dummy(deps=[topk_tid, pool_tid, compressed_ready])
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    if TP_SIZE == 1 and num_tokens <= 12:
        attend_packed_cohorts(
            query, window_indices, window_cache, window_cache_scale, topk_indices,
            compressed_cache, compressed_cache_scale, attn_sink, attended, num_tokens, selection_ready,
            gather_name="c1a_gather", attn_name="c1a_sparse_attn",
        )
    else:
        selected = pl.create_tensor([tokens, 640, HEAD_DIM], dtype=pl.BF16)
        indices = pl.create_tensor([tokens, 640], dtype=pl.INT32)
        selected_ret, indices_ret, gather_tid = gather_packed(
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            window_indices, topk_indices, selected, indices, num_tokens, selection_ready,
            max_workers=GATHER_WORKERS, name_hint="c1a_gather",
        )
        attend_selected(query, selected, indices, attn_sink, attended, num_tokens, gather_tid)
    if TP_SIZE == 1 and num_tokens > 0 and num_tokens <= tokens and tokens <= 16:
        attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
        o_proj_hc_post(
            attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            attn_out, x_hc, post_mix, residual_mix, output, num_tokens,
        )
    else:
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        o_proj(
            attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, partial, num_tokens, grouped_max_workers=64,
        )
        attn_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
        if TP_SIZE == 1 and num_tokens > 0 and num_tokens <= 16:
            _hc_post_from_partial(partial, attn_out, x_hc, post_mix, residual_mix, output, num_tokens)
        else:
            with pl.spmd(tokens, name_hint="cast_out"):
                row = pl.tile.get_block_idx()
                for col in pl.range(0, D, 512):
                    value = pl.tile.full([1, 512], dtype=pl.BF16, value=0.0)
                    if row < num_tokens:
                        partial_tile = pl.load(partial, [row, col], [1, 512])
                        value = pl.cast(partial_tile, pl.BF16, mode="rint")
                    pl.store(value, [row, col], attn_out)
            hc_post(attn_out, x_hc, post_mix, residual_mix, output)
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
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
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
        request_ids, compressed_lens,
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
    x_hc: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[TP_SIZE, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[TP_SIZE, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]
    ],
    request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]
    ],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[TP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[TP_SIZE, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.Out[pl.Tensor[[TP_SIZE, T_DYN], pl.INT32]],
    output: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(1, LOCAL_T_DYN)
    rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
    # MX_B_NN scale views.
    wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[0]
    wq_b_scale_r: pl.Tensor[
        [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = wq_b_scale[0]
    wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[0]
    wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[0]
    index_wq_b_scale_r: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ] = index_wq_b_scale[0]
    l2_decode_c1a_full(
        x_hc[0], pre_mix[0],
        hc_attn_fn[0], hc_attn_scale[0], hc_attn_base[0],
        attn_norm_weight[0],
        wq_a[0], wq_a_scale_r, q_norm_weight[0],
        wq_b[0], wq_b_scale_r,
        wkv[0], wkv_scale_r, kv_norm_weight[0],
        attn_sink[0],
        wo_a[0], wo_a_scale[0], wo_b[0], wo_b_scale_r,
        rope_cos[0], rope_sin[0],
        window_slots[0], window_indices[0],
        window_cache[0], window_cache_scale[0], compressed_cache[0], compressed_cache_scale[0],
        request_ids[0], compressed_lens[0],
        index_cache[0], index_cache_scale[0], index_block_table[0],
        compressed_rope_cos[0], compressed_rope_sin[0],
        compressor_wkv[0], compressor_norm_weight[0], compressed_slots[0],
        index_wk[0], index_norm_weight[0],
        index_wq_b[0], index_wq_b_scale_r, index_weights_proj[0],
        topk_indices[0], pool_block_ids[0], pool_block_counts[0],
        output[0], next_pre_mix[0],
        num_tokens,
        device=0,
    )


def golden_c1a_attention(mode: AttentionMode, **a) -> AttentionGoldenResult:
    """Compose C1A operator references and publish mapped cache rows."""
    x = a["x"]
    q, kv, qr = golden_qkv_proj_rope(
        x, a["wq_a"], a["wq_a_scale"], a["q_norm_weight"], a["wq_b"], a["wq_b_scale"],
        a["wkv"], a["wkv_scale"], a["kv_norm_weight"], a["rope_cos"], a["rope_sin"], sequential_query=True,
    )
    window, window_scale = _publish_quantized_cache(
        a["window_cache"], a["window_cache_scale"], kv, a["window_slots"], quantize_mxfp8_cache,
    )
    compressed = a["compressed_cache"].view(torch.uint8).clone()
    compressed_scale = a["compressed_cache_scale"].clone()
    index = a.get("index_cache")
    index_scale = a.get("index_cache_scale")
    if index is not None:
        index = index.view(torch.uint8).clone()
        index_scale = index_scale.view(torch.uint8).clone()
    if mode == AttentionMode.FULL:
        latent = golden_compressor_ratio1(x, a["compressor_wkv"], a["compressor_norm_weight"])
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
    pool_block_ids = a.get("pool_block_ids")
    pool_block_counts = a.get("pool_block_counts")
    topk = a.get("compressed_indices")
    if mode != AttentionMode.REUSE:
        lengths = a["compressed_lens"]
        scores, physical = golden_c1a_index_scores(a, qr, index, index_scale)
        if mode == AttentionMode.REINDEX:
            if pool_block_ids is None or pool_block_counts is None:
                raise ValueError("C1A Reindex requires compact block-pool metadata")
            scores, physical = golden_pool_index_scores(scores, physical, pool_block_ids, pool_block_counts)
        topk = golden_c1a_index_topk(scores, physical, lengths, pool_block_counts)
        if mode == AttentionMode.FULL:
            pool_block_ids, pool_block_counts = golden_select_candidate_block_pool(scores, lengths)
    window_value = dequantize_mxfp8_cache(window, window_scale).to(x.dtype)
    compressed_value = dequantize_mxfp4_cache(compressed, compressed_scale, 16, "bf16").to(x.dtype)
    attended = golden_selected_attention(
        q, window_value, a["window_indices"], compressed_value, topk, a["attn_sink"],
    )
    output = golden_o_proj(
        attended, a["rope_cos"], a["rope_sin"], a["wo_a"], a["wo_b"], a["wo_b_scale"],
        wo_a_scale=a["wo_a_scale"], sequential_projection=True,
    )
    return AttentionGoldenResult(
        output, window, window_scale, compressed, compressed_scale, index, index_scale, None, topk,
        pool_block_ids=pool_block_ids, pool_block_counts=pool_block_counts,
    )


def golden_decode_attn_c1a_full(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    request_ids: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    compressed_rope_cos: torch.Tensor,
    compressed_rope_sin: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    compressed_slots: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate C1A full attention and cache state."""
    return golden_c1a_attention(
        mode=AttentionMode.FULL,
        x=x, wq_a=wq_a, wq_a_scale=wq_a_scale, q_norm_weight=q_norm_weight, wq_b=wq_b, wq_b_scale=wq_b_scale,
        wkv=wkv, wkv_scale=wkv_scale, kv_norm_weight=kv_norm_weight, attn_sink=attn_sink, wo_a=wo_a, wo_a_scale=wo_a_scale, wo_b=wo_b,
        wo_b_scale=wo_b_scale, rope_cos=rope_cos, rope_sin=rope_sin, window_slots=window_slots,
        window_indices=window_indices, window_cache=window_cache, window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache, compressed_cache_scale=compressed_cache_scale,
        request_ids=request_ids, compressed_lens=compressed_lens, index_cache=index_cache,
        index_cache_scale=index_cache_scale, index_block_table=index_block_table,
        compressed_rope_cos=compressed_rope_cos, compressed_rope_sin=compressed_rope_sin,
        compressor_wkv=compressor_wkv, compressor_norm_weight=compressor_norm_weight,
        compressed_slots=compressed_slots, index_wk=index_wk, index_norm_weight=index_norm_weight,
        index_wq_b=index_wq_b, index_wq_b_scale=index_wq_b_scale, index_weights_proj=index_weights_proj,
    )


def build_validation_values(
    mode, tokens, pages, requests=1, seed=17, case="random", start_pos=None, query_lens=None,
):
    """Deterministic synthetic tensors at released model dimensions."""
    import math

    TOKENS, PAGES, RANKS, REQUESTS = tokens, pages, TP_SIZE, requests
    from quant import (
        pack_mx_b_scale,
        quantize_mxfp8_cache,
        quantize_mxfp4_cache,
    )

    cache_pages = PAGES * REQUESTS if start_pos is not None else PAGES
    torch.manual_seed(seed)
    # Packed token-to-request split.
    lengths = packed_query_lengths(TOKENS, REQUESTS, query_lens)
    token_to_req = torch.repeat_interleave(
        torch.arange(REQUESTS), torch.tensor(lengths)
    ).to(torch.int32)
    values = {}
    shapes = {
        "wq_a": (D, Q_LORA),
        "wq_b": (Q_LORA, LOCAL_H * HEAD_DIM),
        "wkv": (D, HEAD_DIM),
        "wo_b": (LOCAL_O_WIDTH, D),
    }
    for name, (k, n) in shapes.items():
        values[name] = torch.randn(RANKS, k, n).to(torch.float8_e4m3fn)
        exponent = round(127 - math.log2(k) / 2)
        values[name + "_scale"] = pack_mx_b_scale(
            torch.full((RANKS, k // 32, n), exponent, dtype=torch.uint8)
        ).view(torch.float8_e8m0fnu)
    values["x"] = torch.randn(1, TOKENS, D, dtype=torch.bfloat16).expand(RANKS, -1, -1).contiguous()
    values["q_norm_weight"] = torch.ones(RANKS, Q_LORA, dtype=torch.bfloat16)
    values["kv_norm_weight"] = torch.ones(RANKS, HEAD_DIM, dtype=torch.bfloat16)
    values["attn_sink"] = torch.randn(RANKS, LOCAL_H)
    values["wo_a"] = (torch.randn(RANKS, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(
        torch.bfloat16
    )
    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(values["wo_a"])
    positions = torch.arange(PAGES * 128 - 1, PAGES * 128 - TOKENS - 1, -1)
    if start_pos is not None:
        positions = torch.cat([torch.arange(pos, pos + length) for pos, length in zip(start_pos, lengths)])
    freqs_cos, freqs_sin = precompute_rope_tables(PAGES * 128, compressed_attention=True)
    values["rope_cos"] = freqs_cos[positions].unsqueeze(0).expand(RANKS, -1, -1).contiguous()
    values["rope_sin"] = freqs_sin[positions].unsqueeze(0).expand(RANKS, -1, -1).contiguous()
    values["window_slots"] = (
        torch.arange(TOKENS, dtype=torch.int64).unsqueeze(0).expand(RANKS, -1).contiguous()
    )
    values["window_indices"] = (
        torch.arange(128, dtype=torch.int32).view(1, 1, 128).expand(RANKS, TOKENS, -1).contiguous()
    )
    values["compressed_indices"] = torch.full((RANKS, TOKENS, INDEX_TOPK), -1, dtype=torch.int32)
    values["compressed_indices"][:, :, :128] = torch.arange(128, dtype=torch.int32)
    wc, ws = quantize_mxfp8_cache(torch.randn(1, cache_pages, 128, 1, HEAD_DIM).expand(RANKS, -1, -1, -1, -1).contiguous())
    cc, cs = quantize_mxfp4_cache(
        torch.randn(1, cache_pages, 128, 1, HEAD_DIM).expand(RANKS, -1, -1, -1, -1).contiguous(), 16, "bf16"
    )
    values["window_cache"], values["window_cache_scale"] = wc, ws
    values["compressed_cache"], values["compressed_cache_scale"] = cc, cs
    values["output"] = torch.zeros(RANKS, TOKENS, D, dtype=torch.bfloat16)
    values["index_wq_b"] = torch.randn(RANKS, Q_LORA, INDEX_H * INDEX_DIM).to(torch.float8_e4m3fn)
    values["index_wq_b_scale"] = pack_mx_b_scale(
        torch.full((RANKS, Q_LORA // 32, INDEX_H * INDEX_DIM), 122, dtype=torch.uint8)
    ).view(torch.float8_e8m0fnu)
    # Non-negative head weights keep the score monotone in the key magnitude, so the page ladder
    # above turns the cutoff into a magnitude gap instead of a dense band. Signed weights would
    # let a quiet row outrank a loud one and put the cut back on a knife edge.
    values["index_weights_proj"] = (torch.rand(RANKS, D, INDEX_H) / math.sqrt(D)).to(torch.bfloat16)
    values["request_ids"] = token_to_req.unsqueeze(0).expand(RANKS, -1).contiguous()
    values["compressed_lens"] = torch.tensor(
        [PAGES * 128 - t for t in range(TOKENS)], dtype=torch.int32
    ).repeat(RANKS, 1)
    values["index_block_table"] = (
        torch.randperm(PAGES).int().view(1, 1, PAGES).expand(RANKS, REQUESTS, -1).contiguous()
    )
    if start_pos is not None:
        from utils import window_metadata

        table = torch.arange(cache_pages, dtype=torch.int32).reshape(REQUESTS, PAGES)
        slots, indices, _ = window_metadata(positions, token_to_req, table)
        values["window_slots"] = slots.unsqueeze(0).expand(RANKS, -1).contiguous()
        values["window_indices"] = indices.unsqueeze(0).expand(RANKS, -1, -1).contiguous()
        values["compressed_lens"] = (positions + 1).int().unsqueeze(0).expand(RANKS, -1).contiguous()
        values["index_block_table"] = table.unsqueeze(0).expand(RANKS, -1, -1).contiguous()
        selected = torch.arange(INDEX_TOPK).expand(TOKENS, -1)
        valid = selected < values["compressed_lens"][0, :, None]
        physical = token_to_req[:, None] * PAGES * 128 + selected
        values["compressed_indices"] = physical.masked_fill(~valid, -1).int().unsqueeze(0)
        values["compressed_indices"] = values["compressed_indices"].expand(RANKS, -1, -1).contiguous()

    # Index keys carry a page-level magnitude ladder: the first half of the pages dominates the
    # second half by 2^10. The candidate window then ends in a magnitude gap instead of a dense
    # score band, so a ulp-level stack difference cannot reorder the cut. The index query is
    # quantized to E2M1 inside the operator, so a cutoff decided by a dense band would move with
    # any build. A few quiet positions stay visible so the mask still excludes something.
    quiet_page = torch.arange(cache_pages).remainder(PAGES) >= PAGES // 2
    page_gain = torch.where(quiet_page, torch.full((cache_pages,), 2.0**-10), torch.ones(cache_pages))
    keys = torch.randn(RANKS, cache_pages, 128, 1, INDEX_DIM) * page_gain.view(1, cache_pages, 1, 1, 1)
    ic, ics = quantize_mxfp4_cache(keys.to(torch.bfloat16), 32, "e8m0", round_to_even=True)
    values["index_cache"], values["index_cache_scale"] = (
        ic,
        ics.view(torch.float8_e8m0fnu),
    )
    values["topk_indices"] = torch.zeros(RANKS, TOKENS, INDEX_TOPK, dtype=torch.int32)
    if mode == "full":
        values["compressed_rope_cos"] = values["rope_cos"].clone()
        values["compressed_rope_sin"] = values["rope_sin"].clone()
        values["compressor_wkv"] = (torch.randn(RANKS, D, HEAD_DIM) / math.sqrt(D)).to(torch.bfloat16)
        values["compressor_norm_weight"] = torch.ones(RANKS, HEAD_DIM, dtype=torch.bfloat16)
        values["index_wk"] = (torch.randn(RANKS, HEAD_DIM, INDEX_DIM) / math.sqrt(HEAD_DIM)).to(torch.bfloat16)
        # The published index key lands in the quiet half of the ladder: the loud half already
        # fills INDEX_TOPK slots exactly, so a key competing at the cut would make the last slot
        # depend on a ulp. RMSNorm cancels any scale on the projection, so the norm weight sets
        # the published key magnitude.
        values["index_norm_weight"] = torch.full((RANKS, INDEX_DIM), 2.0**-10, dtype=torch.bfloat16)
        positions = values["compressed_lens"].long() - 1
        request_rows = values["request_ids"][:, :, None].long().expand(-1, -1, PAGES)
        token_tables = values["index_block_table"].gather(1, request_rows)
        physical_pages = token_tables.gather(2, (positions // 128)[:, :, None]).squeeze(-1).long()
        values["compressed_slots"] = physical_pages * 128 + positions % 128
    for name in (
        "wq_a",
        "wq_a_scale",
        "wkv",
        "wkv_scale",
        "index_wq_b",
        "index_wq_b_scale",
        "index_weights_proj",
        "index_cache",
        "index_cache_scale",
        "compressor_wkv",
        "index_wk",
    ):
        if name in values:
            values[name] = values[name][0:1].expand_as(values[name]).contiguous()
    if case == "ragged":
        for t in range(TOKENS):
            length = (PAGES * 128, 0, 7, 129)[t % 4]
            values["compressed_lens"][:, t] = min(length, PAGES * 128)
            values["window_indices"][:, t, (0 if t % 4 == 1 else 17) :] = -1
            values["compressed_indices"][:, t, (0 if t % 4 == 1 else 31) :] = -1
            if t % 4 == 1:
                values["window_slots"][:, t] = -1
        if mode == "full":
            positions = values["compressed_lens"].long() - 1
            safe = positions.clamp_min(0)
            request_rows = values["request_ids"][:, :, None].long().expand(-1, -1, PAGES)
            token_tables = values["index_block_table"].gather(1, request_rows)
            physical_pages = token_tables.gather(2, (safe // 128)[:, :, None]).squeeze(-1).long()
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
    pool_ids = torch.full((RANKS, TOKENS, pool_capacity), -1, dtype=torch.int32)
    pool_counts = torch.zeros((RANKS, TOKENS), dtype=torch.int32)
    for rank in range(RANKS):
        for token in range(TOKENS):
            visible = max(0, min(int(values["compressed_lens"][rank, token]), PAGES * 128))
            block_count = (visible + M.candidate_block_size - 1) // M.candidate_block_size
            selected = min(block_count, pool_capacity)
            if selected == 0:
                continue
            if selected == block_count:
                chosen = torch.arange(selected, dtype=torch.int32)
            else:
                chosen = torch.cat(
                    (
                        torch.arange(selected - 1, dtype=torch.int32),
                        torch.tensor([block_count - 1], dtype=torch.int32),
                    )
                )
            pool_ids[rank, token, :selected] = chosen
            pool_counts[rank, token] = selected
    values["pool_block_ids"] = pool_ids
    values["pool_block_counts"] = pool_counts
    return values


CACHE_SLOTS = {
    "window_cache": "window_slots",
    "window_cache_scale": "window_slots",
    "compressed_cache": "compressed_slots",
    "compressed_cache_scale": "compressed_slots",
    "index_cache": "compressed_slots",
    "index_cache_scale": "compressed_slots",
}


def cache_compare(world_size, tokens, mode):
    """Mapped rows must match byte for byte and every other cache row must stay unchanged.

    Only FULL publishes the compressor and index caches: REINDEX and REUSE read them, so the
    reference hands those back unmodified and they stay an exact byte contract.
    """
    options = dict(
        mapping_shape=(world_size, tokens),
        block_size=BLOCK_SIZE,
        leading_rank_axis=True,
        atol=0.0,
        rtol=0.0,
        max_error_ratio=0.0,
    )
    compare = {
        name: mapped_pool_ratio_allclose(slots, pool_name=name, **options)
        for name, slots in CACHE_SLOTS.items()
        if mode == "full" or name.startswith("window")
    }
    if mode != "full":
        for name in ("compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale"):
            compare[name] = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    return compare


def build_hc_validation_values(
    mode, tokens, pages, requests=1, seed=17, case="random", start_pos=None, query_lens=None,
):
    """Create local HC token slabs and full-range attention operands."""
    import math

    values = build_validation_values(
        mode, tokens, pages, requests, seed, case, start_pos=start_pos, query_lens=query_lens,
    )
    del values["x"]
    generator = torch.Generator().manual_seed(seed)
    local_tokens = (tokens + TP_SIZE - 1) // TP_SIZE
    values["x_hc"] = torch.randn(TP_SIZE, local_tokens, HC_MULT, D, generator=generator)
    # The coefficients are staggered: this site collapses with the pre-mix the previous
    # sub-layer produced (identity one-hot at the very first site). The fixture carries a
    # general sigmoid-distributed mix instead of the seed so every lane participates.
    values["pre_mix"] = torch.sigmoid(torch.randn(TP_SIZE, local_tokens, HC_MULT, generator=generator))
    values["hc_attn_fn"] = torch.randn(TP_SIZE, MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)
    values["hc_attn_scale"] = torch.randn(TP_SIZE, 3, generator=generator)
    values["hc_attn_base"] = torch.randn(TP_SIZE, MIX_HC, generator=generator)
    # The block normalizes the collapsed stream before attention; the checkpoint carries one
    # weight per layer, so the fixture does too. BF16: the harness takes each spec's dtype from
    # the fixture tensor, and the kernel declares the weight in the activation dtype.
    values["attn_norm_weight"] = (
        torch.randn(TP_SIZE, D, generator=generator) * 0.1 + 1.0
    ).to(torch.bfloat16)
    values["output"] = torch.zeros(TP_SIZE, local_tokens, HC_MULT, D, dtype=torch.float32)
    values["next_pre_mix"] = torch.zeros(TP_SIZE, local_tokens, HC_MULT, dtype=torch.float32)
    return values


CACHE_STATE_NAMES = (
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "index_cache",
    "index_cache_scale",
    "topk_indices",
    "pool_block_ids",
    "pool_block_counts",
)


def golden_c1a_hc_case(tensors, attn_golden):
    """Compute TP1 mHC, C1A attention, cache updates, and mHC post."""
    x_hc = tensors["x_hc"][0]
    next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        x_hc, tensors["hc_attn_fn"][0], tensors["hc_attn_scale"][0], tensors["hc_attn_base"][0],
    )
    tensors["next_pre_mix"][0].copy_(next_pre_mix)
    x_attn = golden_hc_pre(x_hc, tensors["pre_mix"][0]).to(torch.bfloat16)
    x_norm = golden_rms_norm(x_attn, tensors["attn_norm_weight"][0])
    parameters = inspect.signature(attn_golden).parameters
    inputs = {name: tensors[name][0] for name in parameters if name != "x"}
    result = attn_golden(x=x_norm, **inputs)
    for name in CACHE_STATE_NAMES:
        value = getattr(result, name, None)
        if value is not None and name in tensors:
            tensors[name][0].view(torch.uint8).copy_(value.contiguous().view(torch.uint8))
    sublayer = result.output.to(torch.bfloat16)
    tensors["output"][0].copy_(golden_hc_post(sublayer, x_hc, post_mix, residual_mix).to(torch.float32))


def golden_decode_c1a_full_case(tensors):
    """Evaluate the TP1 C1A full mHC composition."""
    golden_c1a_hc_case(tensors, golden_decode_attn_c1a_full)


if __name__ == "__main__":
    from utils import configure_decode_workload, parse_decode_query_lens, parse_decode_startpos

    import argparse

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
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=1, requests=1, pages=8)
    args = parser.parse_args()
    configure_decode_workload(args, parser)
    if args.startpos is not None:
        max_end = max(pos + length for pos, length in zip(args.startpos, args.query_lens))
        args.pages = (max_end + 127) // 128
    host = l3_decode_c1a_full
    values = build_hc_validation_values(
        "full", args.tokens, args.pages, args.requests, args.seed, args.case,
        start_pos=args.startpos, query_lens=args.query_lens,
    )
    # Independent initial storage for each harness allocation.
    specs = [
        TensorSpec(n, list(values[n].shape), values[n].dtype, init_value=lambda n=n: values[n].clone())
        for n in host.param_names if n != "num_tokens"
    ]

    specs.append(ScalarSpec("num_tokens", torch.int32, args.tokens))

    output_budget = ratio_allclose(atol=OUTPUT_ATOL, rtol=OUTPUT_RTOL, max_error_ratio=OUTPUT_MAX_ERROR_RATIO)
    comparisons = {
        "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
        "pool_block_ids": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
        "pool_block_counts": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
        "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
        "output": output_budget,
        **cache_compare(1, args.tokens, "full"),
    }
    result = run(
        fn=host,
        specs=specs,
        golden_fn=golden_decode_c1a_full_case,
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
        compare_fn=comparisons,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
