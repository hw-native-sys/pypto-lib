# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C1A reindex decode attention with mHC."""

import pypto.language as pl
from config import (
    COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, WINDOW_CACHE_GROUP, FLASH as M, B_DYN, CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN, INDEX_POOL_BLOCKS_DYN, ORI_BLOCKS_DYN, TABLE_DYN, T_DYN, LOCAL_T_DYN,
)
from hc import hc_head_norm, hc_project, hc_sinkhorn, hc_post
from decode_indexer import c1a_index, score_reindex, select_pool_index_topk
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj_rope
from sparse_attention import (
    gather_packed, attend_selected, publish_window_row, GATHER_WORKERS as GATHER_WORKER_TILE, attend_packed_cohorts,
)
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


@pl.jit.inline(auto_scope=False)
def decode_c1a_reindex(
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
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
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
    score_width = pl.tensor.dim(pool_block_ids, 1) * 8
    index_query, index_query_scale, index_weights, scores, query_tid, weights_tid = c1a_index(
        x_norm, index_wq_b, index_wq_b_scale, index_weights_proj, rope_cos, rope_sin,
        num_tokens, score_width, x_ready, index_wq_b, 0,
        qb_payload, qb_scales,
        pretransposed=False,
    )
    scores_tid = score_reindex(
        index_query, index_query_scale, index_weights, index_cache, index_cache_scale,
        index_block_table, token_to_req_indices, compressed_lens, pool_block_ids, pool_block_counts,
        scores, num_tokens, x_ready, query_tid, weights_tid, x_ready,
    )
    selection_ready = select_pool_index_topk(
        scores, pool_block_ids, pool_block_counts, index_block_table, token_to_req_indices, compressed_lens,
        topk_indices, num_tokens, scores_tid,
    )
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    if num_tokens <= 12:
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
            max_workers=GATHER_WORKER_TILE, name_hint="c1a_gather",
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
    return output


@pl.jit
def l2_decode_c1a_reindex(
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
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
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
    return decode_c1a_reindex(
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
        token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table,
        pool_block_ids, pool_block_counts,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices,
        output, next_pre_mix,
        num_tokens,
    )


@pl.jit.host
def l3_decode_c1a_reindex(
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
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
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
    l2_decode_c1a_reindex(
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
        token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table,
        pool_block_ids, pool_block_counts,
        index_wq_b, index_wq_b_scale_r, index_weights_proj,
        topk_indices,
        output, next_pre_mix,
        num_tokens,
        device=0,
    )


if __name__ == "__main__":
    from functools import partial
    from config import BLOCK_SIZE
    from decode_c1a_full import build_c1a_specs, golden_c1a
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run

    from utils import configure_decode_workload, parse_decode_query_lens, parse_decode_startpos

    import argparse

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1 A5 mHC C1A reindex validation")
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
    host = l3_decode_c1a_reindex
    specs = build_c1a_specs(
        "reindex", args.tokens, args.pages, args.requests, args.seed, args.case,
        start_pos=args.startpos, query_lens=args.query_lens, param_names=host.param_names,
    )

    result = run(
        fn=host,
        specs=specs,
        golden_fn=partial(golden_c1a, mode="reindex"),
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
            "compressed_cache": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "compressed_cache_scale": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "index_cache": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "index_cache_scale": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
