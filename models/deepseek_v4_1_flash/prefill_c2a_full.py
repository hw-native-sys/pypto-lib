# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C2A Full wired through mHC, and the validation shared with Reuse.

One attention sublayer of the text backbone, in the order the official ``Block.forward`` runs it::

    attn_pre, attn_post, attn_comb = hc_mixes(x_hc)   # coefficients from this sublayer's input
    x = rms_norm(hc_pre(x_hc, pre_mix))              # pre_mix: the previous sublayer's delayed mix
    x = allgather(x)                                # normalized input in full token order
    x = reduce_scatter(attention_partial(x))         # FP32 head-TP sum, local token rows
    x_hc = hc_post(x, x_hc, attn_post, attn_comb)

``attn_pre`` comes back as ``next_pre_mix``; the FFN sublayer of the same layer collapses its input
with it.
"""

import argparse
import math
import os
import pypto.language as pl
import pypto.language.distributed as pld
import torch

from hc import hc_pre
from hc import hc_mixes
from golden import ScalarSpec, TensorSpec, mapped_pool_ratio_allclose, ratio_allclose, ratio_reldiff, run
from config import (
    FLASH as M, B_DYN, CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN, TABLE_DYN, Q_START_DYN,
    STATE_BLOCKS_DYN, TP_SIZE, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, MAX_BATCH_PER_DP,
    PREFILL_MAX_TOKENS, STATE_CAPACITY, WINDOW_CACHE_GROUP, SP_T_DYN, BLOCK_SIZE,
)
from compressor import STATE_WIDTH, project_ratio2, compressor_ratio2, publish_c2a_compressed
from prefill_cp_allgather import prefill_sp_input_allgather
from o_proj import (
    golden_o_proj, prefill_sp_output_reduce_scatter, o_proj,
)
from hc import prefill_sp_post, golden_hc_post
from hc import golden_mhc_mixes
from hc import golden_hc_pre
from rmsnorm import golden_rms_norm, rms_norm_tiled
from rope_tables import ROPE_ROWS_DYN, materialize_rope_rows, precompute_rope_tables
from qkv_proj_rope import golden_rope_tail, q_proj_qr, q_proj_rope, kv_proj_rope
from quant import (
    dequantize_mxfp4_cache, golden_bf16_gemm, golden_mxfp8_linear, golden_mxfp8_quantize, pack_mx_b_scale,
    quantize_grouped_mxfp8_weight, quantize_mxfp4_cache, quantize_mxfp8_cache,
)
from utils import paged_slots, window_metadata
from sparse_attention import (
    PACKED_QUERY_TILE as QUERY_TILE, SELECTED_KEYS as SPARSE_WIDTH, gather_packed_chunk,
    prefill_attend_sparse, prefill_publish_window,
)
from prefill_indexer import (
    LEAF, prefill_c2a_index_key_project, prefill_c2a_rotate_index_key, prefill_c2a_publish_index_key,
    prefill_c2a_project_index_q, prefill_c2a_index_query, prefill_c2a_index_weights,
    prefill_c2a_index_select,
)


from compressor import prefill_c2a_compressed_rope

# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4

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
def prefill_c2a_full_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
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
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Return replicated output for standalone C2A attention callers."""
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="c2a_previous_epoch", allow_early_resolve=False
    ) as cache_ready:
        for peer in pl.range(TP_SIZE):
            pld.system.wait(
                output_arrived, offsets=[peer, 0], expected=(attention_epoch - 1) * 2,
                cmp=pld.WaitCmp.Ge,
            )
    tokens = pl.tensor.dim(x, 0)

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, qr, num_tokens)
    q_proj_rope(qr, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens, max_workers=2147483647)
    kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, kv_projection, window_kv, num_tokens)
    window_publish_tid = prefill_publish_window(window_kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)

    compressor_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    compressor_score = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.FP32)
    project_ratio2(x, compressor_wkv, compressor_wgate, compressor_kv, compressor_score, num_tokens)
    latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    pool_tid, state_tid = compressor_ratio2(
        compressor_kv, compressor_score, query_start_loc, position_ids, token_to_req_indices,
        state_block_table, state_cache, compressor_norm_weight, latent, num_tokens, cache_ready,
    )

    # Index keys come from the unrotated latent; the compressed cache stores the rotated one.
    index_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    prefill_c2a_index_key_project(latent, index_wk, index_norm_weight, index_key, num_tokens, pool_tid)
    rotated_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    prefill_c2a_rotate_index_key(index_key, compressed_rope_cos, compressed_rope_sin, rotated_key, num_tokens)
    index_publish_tid = prefill_c2a_publish_index_key(
        rotated_key, compressed_slots, index_cache, index_cache_scale, num_tokens, cache_ready
    )
    rotated_latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    prefill_c2a_compressed_rope(latent, compressed_rope_cos, compressed_rope_sin, rotated_latent, num_tokens)
    compressed_publish_tid = publish_c2a_compressed(
        rotated_latent, compressed_slots, compressed_cache, compressed_cache_scale,
        num_tokens, cache_ready, pool_tid,
    )

    index_query_raw = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    prefill_c2a_project_index_q(qr, index_wq_b, index_wq_b_scale, index_query_raw, num_tokens)
    index_query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM], dtype=pl.BF16)
    prefill_c2a_index_query(index_query_raw, rope_cos, rope_sin, index_query, num_tokens)
    index_weight = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    prefill_c2a_index_weights(x, index_weights_proj, index_weight, num_tokens)
    leaf_scores = pl.create_tensor([tokens, LEAF], dtype=pl.FP32)
    leaf_rows = pl.create_tensor([tokens, LEAF], dtype=pl.INT32)
    select_tid = prefill_c2a_index_select(
        index_query, index_weight, index_cache, index_cache_scale, index_block_table,
        token_to_req_indices, compressed_lens, leaf_scores, leaf_rows, topk_indices,
        num_tokens, index_publish_tid,
    )

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH, HEAD_DIM], dtype=pl.BF16)
    combined = pl.create_tensor([QUERY_TILE, SPARSE_WIDTH], dtype=pl.INT32)
    chunk_done = pl.system.task_dummy(deps=[select_tid, state_tid, window_publish_tid, compressed_publish_tid])
    for start in pl.range(0, num_tokens, QUERY_TILE):
        active = pl.min(QUERY_TILE, num_tokens - start)
        gather_tid = gather_packed_chunk(
            window_cache, window_cache_scale, window_indices, compressed_cache,
            compressed_cache_scale, topk_indices, selected, combined, start, active, chunk_done,
        )
        chunk_done = prefill_attend_sparse(
            query, selected, combined, attn_sink, attended, start, active, gather_tid
        )

    o_proj(
        attended, wo_a, wo_a_scale, wo_b,
        wo_b_scale, rope_cos, rope_sin,
        partial, num_tokens, grouped_max_workers=4096,
    )
    return partial


@pl.jit.inline
def prefill_c2a_full_sp(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
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
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    x_hc_out: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one C2A Full attention sublayer between mHC pre and post.

    ``x_hc`` and ``pre_mix`` contain contiguous local token rows. ``num_tokens`` is
    the valid global count of this DP group; metadata follows the gathered group order.
    ``attn_input`` is gathered, while ``attn_output``, ``next_pre_mix`` and ``x_hc_out``
    stay local. Inactive local outputs are zero. Outputs must not alias inputs.
    """
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    local_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_mixes(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix,
    )
    hc_collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(x_hc, pre_mix, hc_collapsed)
    hc_norm_tokens = pl.tensor.dim(hc_collapsed, 0)
    for hc_norm_block in pl.spmd((hc_norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(hc_collapsed, attn_norm_weight, local_hidden, hc_norm_block, hc_norm_tokens)
    prefill_sp_input_allgather(
        local_hidden, input_window, input_arrived, attn_input,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    group_tokens = pl.tensor.dim(attn_input, 0)
    partial = pl.create_tensor([group_tokens, D], dtype=pl.FP32)
    for row in pl.spmd(group_tokens, name_hint="sp_topk_padding"):
        if row >= num_tokens:
            topk_indices[row:row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
    if num_tokens > 0:
        prefill_c2a_full_attention(
            attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b,
            wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale, compressed_cache,
            compressed_cache_scale, token_to_req_indices, compressed_lens, index_cache, index_cache_scale,
            index_block_table, position_ids, compressed_rope_cos, compressed_rope_sin, compressor_wkv,
            compressor_wgate, query_start_loc, state_block_table, state_cache, compressor_norm_weight,
            compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
            index_weights_proj, topk_indices, output_arrived, partial, num_tokens,
            attention_epoch,
        )
    prefill_sp_output_reduce_scatter(
        partial, output_window, output_arrived, attn_output,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    prefill_sp_post(
        attn_output, x_hc, post_mix, residual_mix, x_hc_out, next_pre_mix, tp_rank, num_tokens,
    )
    return x_hc_out


def golden_attention_input(
    x_hc: torch.Tensor,
    pre_mix: torch.Tensor,
    attn_norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Collapse the streams with the delayed pre-mix, round to BF16, then apply RMSNorm."""
    return golden_rms_norm(golden_hc_pre(x_hc, pre_mix).to(torch.bfloat16), attn_norm_weight)


HC_INPUT_NAMES = ("x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight")
MX_GROUP = M.weight_block_size[0]

CMP_SCALES = HEAD_DIM // 16

IDX_SCALES = INDEX_DIM // 32

CMP_PACKED = HEAD_DIM // 2

IDX_PACKED = INDEX_DIM // 2

WINDOW = M.sliding_window

ATTEND_TILE = 64

E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

RATIO = 2

TOPK = M.index_topk

CMP_GROUP = COMPRESSED_CACHE_GROUP

IDX_GROUP = INDEX_CACHE_GROUP

def _write_rows(cache, rows, payload, width):
    """Rewrite only ``rows`` of a paged cache, leaving every other byte alone."""
    cache.view(torch.uint8).reshape(-1, width)[rows] = payload.contiguous().view(torch.uint8).reshape(-1, width)

def _flat_cache_rows(payload, scale, group_size, scale_format):
    """Packed E2M1 paged cache -> flattened FP32 physical rows."""
    return dequantize_mxfp4_cache(payload, scale, group_size, scale_format).flatten(0, -2)

def golden_reference_c2a(inputs: dict, *, fp32_output=False) -> dict:
    """Ratio-2 compressed sparse attention; publishes only the addressed cache rows.

    Returns the per-rank partial output (before TP reduction) plus every mutated state: the
    three paged caches with scales, the compressor state, and the physical Top-K rows.
    """
    t = inputs
    x = t["x"]
    tokens = x.shape[0]
    head_dim = t["window_cache"].shape[-1]
    local_heads = t["attn_sink"].numel()
    index_dim = t["index_norm_weight"].numel()
    index_heads = t["index_weights_proj"].shape[-1]

    # a. Query/KV projection and rotary embedding.
    qr = golden_rms_norm(golden_mxfp8_linear(x, t["wq_a"], t["wq_a_scale"]), t["q_norm_weight"])
    q = golden_mxfp8_linear(qr, t["wq_b"], t["wq_b_scale"]).unflatten(-1, (local_heads, head_dim))
    q = golden_rope_tail(q, t["rope_cos"], t["rope_sin"])
    kv = golden_rms_norm(golden_mxfp8_linear(x, t["wkv"], t["wkv_scale"]), t["kv_norm_weight"])
    kv = golden_rope_tail(kv, t["rope_cos"], t["rope_sin"])

    # b. Sliding-window cache: MXFP8, group 32, BF16 scales; addressed rows only.
    window_cache = t["window_cache"].clone()
    window_cache_scale = t["window_cache_scale"].clone()
    window_payload, window_scales = quantize_mxfp8_cache(kv)
    window_slots = t["window_slots"].long()
    window_valid = window_slots >= 0
    window_rows = window_slots[window_valid]
    # Boolean indexing of Float8 on CPU requires an exact FP32 roundtrip.
    visible_payload = window_payload.float()[window_valid].to(window_payload.dtype)
    _write_rows(window_cache, window_rows, visible_payload, head_dim)
    window_cache_scale.reshape(-1, head_dim // MX_GROUP)[window_rows] = window_scales[window_valid]

    # c. Ratio-2 compressor: per-channel two-way softmax over the pair, pooled row rounded to
    #    BF16 before RMSNorm, published exactly when position_ids % 2 == 1. Ring rows
    #    contain the KV projection followed by the gate score.
    state_cache = t["state_cache"].clone()
    pending_kv = x.float() @ t["compressor_wkv"].float()
    pending_score = x.float() @ t["compressor_wgate"].float()
    position_ids = t["position_ids"].long()
    latent = torch.zeros(tokens, head_dim, dtype=torch.bfloat16)
    publish = torch.zeros(tokens, dtype=torch.bool)
    capacity = state_cache.shape[1]
    valid_tokens = min(tokens, int(t["query_start_loc"][-1]))
    for token in range(valid_tokens):
        request = int(t["token_to_req_indices"][token])
        if not 0 <= request < t["state_block_table"].shape[0]:
            continue
        block = int(t["state_block_table"][request, 0])
        if not 0 <= block < state_cache.shape[0]:
            continue
        position = int(position_ids[token])
        if position < 0:
            continue
        if position % RATIO:
            previous = state_cache[block, (position - 1) % capacity]
            pair_kv = torch.stack((previous[:head_dim], pending_kv[token]))
            pair_score = torch.stack((previous[head_dim:], pending_score[token]))
            pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0)
            latent[token] = golden_rms_norm(pooled.to(torch.bfloat16), t["compressor_norm_weight"])
            publish[token] = True
        state_cache[block, position % capacity, :head_dim] = pending_kv[token]
        state_cache[block, position % capacity, head_dim:] = pending_score[token]

    # d. Compressed KV: rotate the latent tail, then packed FP4 with group-16 BF16 scales.
    compressed_cache = t["compressed_cache"].clone()
    compressed_cache_scale = t["compressed_cache_scale"].clone()
    rotated = golden_rope_tail(latent, t["compressed_rope_cos"], t["compressed_rope_sin"])
    compressed_slots = t["compressed_slots"].long()
    compressed_valid = (compressed_slots >= 0) & publish
    compressed_rows = compressed_slots[compressed_valid]
    if compressed_rows.numel():
        payload, scale = quantize_mxfp4_cache(rotated[compressed_valid], CMP_GROUP, "bf16")
        _write_rows(compressed_cache, compressed_rows, payload, head_dim // 2)
        compressed_cache_scale.reshape(-1, head_dim // CMP_GROUP)[compressed_rows] = scale

    # e. Index keys come from the UNROTATED latent; packed FP4, group 32, E8M0, same slots.
    index_cache = t["index_cache"].clone()
    index_cache_scale = t["index_cache_scale"].clone()
    index_key = golden_rms_norm(golden_bf16_gemm(latent, t["index_wk"]), t["index_norm_weight"])
    index_key = golden_rope_tail(index_key, t["compressed_rope_cos"], t["compressed_rope_sin"])
    if compressed_rows.numel():
        payload, scale = quantize_mxfp4_cache(index_key[compressed_valid], IDX_GROUP, "e8m0")
        _write_rows(index_cache, compressed_rows, payload, index_dim // 2)
        _write_rows(index_cache_scale, compressed_rows, scale, index_dim // IDX_GROUP)

    index_values = _flat_cache_rows(index_cache, index_cache_scale, IDX_GROUP, "e8m0")

    # f. Indexer: score every causal compressed position against the freshly written keys.
    index_q = golden_mxfp8_linear(qr, t["index_wq_b"], t["index_wq_b_scale"]).unflatten(
        -1, (index_heads, index_dim)
    )
    index_q = golden_rope_tail(index_q, t["rope_cos"], t["rope_sin"])
    head_weights = golden_bf16_gemm(x, t["index_weights_proj"])
    head_weights = head_weights * index_dim**-0.5 * index_heads**-0.5
    lens = t["compressed_lens"].long()
    max_len = int(lens.max()) if lens.numel() else 0
    topk_indices = torch.full((tokens, TOPK), -1, dtype=torch.int32)
    if max_len:
        positions = torch.arange(max_len)
        blocks = torch.div(positions, BLOCK_SIZE, rounding_mode="floor")
        offsets = positions.remainder(BLOCK_SIZE)
        pages = t["index_block_table"][t["token_to_req_indices"].long().unsqueeze(-1), blocks]
        physical_rows = pages.to(torch.int64) * BLOCK_SIZE + offsets
        keys = index_values[physical_rows.reshape(-1)].reshape(tokens, max_len, index_dim)
        scores = torch.einsum("thd,tkd->thk", index_q.float(), keys.float()).relu()
        scores = (scores * head_weights.float().unsqueeze(-1)).sum(dim=-2)
        scores = scores.masked_fill(positions.unsqueeze(0) >= lens.unsqueeze(-1), -torch.inf)
        count = min(TOPK, max_len)
        logical = scores.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
        selected_scores = scores.gather(-1, logical)
        selected_rows = physical_rows.gather(-1, logical)
        rows = torch.where(torch.isfinite(selected_scores), selected_rows, -1)
        for token in range(tokens):
            if int(lens[token]) > TOPK:
                # The scored device path orders winners by physical cache row.
                rows[token] = rows[token].sort().values
        topk_indices[:, :count] = rows.to(torch.int32)

    # g. Sparse attention: one online softmax over window rows plus selected compressed rows.
    window_values = (
        window_cache.float() * window_cache_scale.float().repeat_interleave(MX_GROUP, -1)
    ).reshape(-1, head_dim)
    compressed_values = _flat_cache_rows(compressed_cache, compressed_cache_scale, CMP_GROUP, "bf16")
    window_indices = t["window_indices"].long()
    gathered = torch.cat(
        (
            window_values[window_indices.clamp_min(0).reshape(-1)].reshape(tokens, -1, head_dim),
            compressed_values[topk_indices.long().clamp_min(0).reshape(-1)].reshape(tokens, -1, head_dim),
        ),
        dim=1,
    ).to(torch.bfloat16)
    visible = torch.cat((window_indices >= 0, topk_indices >= 0), dim=1)
    gathered = gathered.masked_fill(~visible[..., None], 0)

    maximum = torch.full((tokens, local_heads), -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros(tokens, local_heads, head_dim, dtype=torch.float32)
    for start in range(0, gathered.shape[1], ATTEND_TILE):
        values = gathered[:, start : start + ATTEND_TILE].float()
        logits = torch.einsum("thd,tkd->thk", q.float(), values) * head_dim**-0.5
        logits = logits.masked_fill(~visible[:, None, start : start + ATTEND_TILE], -torch.inf)
        new_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = (maximum - new_maximum).exp()
        probabilities = (logits - new_maximum[..., None]).exp()
        denominator = denominator * correction + probabilities.sum(-1)
        numerator = numerator * correction[..., None] + torch.einsum(
            "thk,tkd->thd", probabilities.to(torch.bfloat16).float(), values
        )
        maximum = new_maximum
    final_maximum = torch.maximum(maximum, t["attn_sink"].float()[None])
    correction = (maximum - final_maximum).exp()
    denominator = denominator * correction + (t["attn_sink"].float()[None] - final_maximum).exp()
    attended = (numerator * (correction / denominator)[..., None]).to(torch.bfloat16)

    # h. Inverse RoPE, grouped low-rank output projection, per-rank partial.
    output = golden_o_proj(
        attended, t["rope_cos"], t["rope_sin"], t["wo_a"], t["wo_b"], t["wo_b_scale"],
        wo_a_scale=t["wo_a_scale"], output_dtype=torch.float32 if fp32_output else None,
        sequential_projection=True,
    )

    return {
        "output": output,
        "window_cache": window_cache,
        "window_cache_scale": window_cache_scale,
        "compressed_cache": compressed_cache,
        "compressed_cache_scale": compressed_cache_scale,
        "index_cache": index_cache,
        "index_cache_scale": index_cache_scale,
        "state_cache": state_cache,
        "topk_indices": topk_indices,
    }

INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale",
    "compressed_cache", "compressed_cache_scale", "token_to_req_indices", "compressed_lens",
    "index_cache", "index_cache_scale", "index_block_table", "position_ids",
    "compressed_rope_cos", "compressed_rope_sin", "compressor_wkv", "compressor_wgate",
    "query_start_loc", "state_block_table", "state_cache", "compressor_norm_weight", "compressed_slots",
    "index_wk", "index_norm_weight", "index_wq_b", "index_wq_b_scale", "index_weights_proj",
)

FULL_ROPE_NAMES = {
    "rope_cos": "freqs_cos", "rope_sin": "freqs_sin",
    "compressed_rope_cos": "compressed_freqs_cos",
    "compressed_rope_sin": "compressed_freqs_sin",
}

PROGRAM_INPUT_NAMES = tuple(
    entry
    for name in INPUT_NAMES
    for entry in (("compressed_rope_positions", name) if name == "compressor_wkv"
                  else (FULL_ROPE_NAMES.get(name, name),))
)

_PREFIXES = {
    ("mixed", "decode"): (1, 63, 128, 255, 0, 511, 129, 256),
    ("mixed", "prefill"): (0, 1, 64, 128, 0, 255, 2, 127),
    ("long", "decode"): (1024, 1027, 1290, 63, 1535, 128, 2048, 255),
    ("long", "prefill"): (1024, 1027, 1291, 62, 1534, 129, 2049, 254),
}

def _weight(gen, k, n):
    """MXFP8 input-major weight with Cube-ordered MX_B_NN scales."""
    payload, codes = golden_mxfp8_quantize(torch.randn(n, k, generator=gen) / math.sqrt(k))
    return payload.T.contiguous(), pack_mx_b_scale(codes.T.contiguous()).view(E8M0)

def _plan(tokens, requests, case, mode, start_pos=None):
    """Ragged query lengths and per-request KV prefixes."""
    lengths = [tokens // requests + (index < tokens % requests) for index in range(requests)]
    if mode == "prefill" and requests > 1 and lengths[-1] > 4:
        lengths[0] += 3
        lengths[-1] -= 3
    pattern = _PREFIXES[(case if case == "long" else "mixed", mode)]
    prefixes = [pattern[index % len(pattern)] for index in range(requests)]
    if start_pos is not None:
        prefixes = list(start_pos)
    return lengths, prefixes

def make_c2a_inputs(tokens=24, requests=6, seed=17, case="mixed", mode="decode", full_rope_tables=False, start_pos=None):
    """Deterministic inputs for ``golden_prefill_attn_c2a_full``.

    case: "mixed" keeps contexts short so every causal position is selected, "long" pushes
    part of the batch past index_topk so the scored Top-K path runs, "masked" clears every
    slot and index, "zero" zeroes activations and all three cache payloads.  mode: "decode"
    (few rows per request, long prefixes) or "prefill" (ragged multi-token requests).
    """
    if mode not in ("decode", "prefill"):
        raise ValueError(f"mode must be 'decode' or 'prefill', got {mode!r}")
    if case not in ("mixed", "long", "masked", "zero"):
        raise ValueError(f"unsupported case {case!r}")
    if not 1 <= requests <= min(MAX_BATCH_PER_DP, tokens):
        raise ValueError(f"requests must be in [1, min({MAX_BATCH_PER_DP}, tokens)]")
    gen = torch.Generator().manual_seed(seed)

    lengths, prefixes = _plan(tokens, requests, case, mode, start_pos)
    token_to_req_indices = torch.repeat_interleave(torch.arange(requests), torch.tensor(lengths))
    positions = torch.cat(
        [torch.arange(prefix, prefix + length) for prefix, length in zip(prefixes, lengths)]
    )
    maximum = int(positions.max())
    window_columns = maximum // BLOCK_SIZE + 1
    compressed_columns = max(4, (maximum // RATIO) // BLOCK_SIZE + 1)

    # One spare page per cache is never mapped: a control row for untouched-byte checks.
    window_pages = requests * window_columns + 1
    compressed_pages = requests * compressed_columns + 1
    def table(pages, columns):
        rows = torch.randperm(pages, generator=gen)[: requests * columns]
        return rows.reshape(requests, columns).to(torch.int32)

    window_table = table(window_pages, window_columns)
    index_block_table = table(compressed_pages, compressed_columns)

    window_slots, window_indices, _ = window_metadata(positions, token_to_req_indices, window_table)
    compressed_slots = paged_slots(positions, token_to_req_indices, index_block_table, BLOCK_SIZE, RATIO, True)
    compressed_lens = torch.div(positions + 1, RATIO, rounding_mode="floor").to(torch.int32)

    # Re-derive every piece of addressing independently of the metadata helpers.
    assert window_slots.unique().numel() == tokens
    for row, (request, position) in enumerate(zip(token_to_req_indices.tolist(), positions.tolist())):
        expected_slot = int(window_table[request, position // BLOCK_SIZE]) * BLOCK_SIZE
        assert int(window_slots[row]) == expected_slot + position % BLOCK_SIZE
        history = list(range(max(0, position - WINDOW + 1), position + 1))
        expected = [
            int(window_table[request, p // BLOCK_SIZE]) * BLOCK_SIZE + p % BLOCK_SIZE
            for p in history
        ]
        assert window_indices[row, : len(expected)].tolist() == expected
        assert bool((window_indices[row, len(expected) :] == -1).all())
        logical = position // RATIO
        published = (position + 1) % RATIO == 0
        expected_compressed = (
            int(index_block_table[request, logical // BLOCK_SIZE]) * BLOCK_SIZE
            + logical % BLOCK_SIZE
            if published
            else -1
        )
        assert int(compressed_slots[row]) == expected_compressed
        assert int(compressed_lens[row]) == (position + 1) // RATIO
    published_slots = compressed_slots[compressed_slots >= 0]
    assert published_slots.unique().numel() == published_slots.numel()

    # Fixture initialization only: production callers retain these full tables.
    table_rows = int(positions.max()) + 1
    freqs_cos, freqs_sin = precompute_rope_tables(table_rows, compressed_attention=True)
    compressed_freqs_cos, compressed_freqs_sin = freqs_cos, freqs_sin
    rope_cos, rope_sin = freqs_cos[positions], freqs_sin[positions]
    complete = (positions + 1).remainder(RATIO) == 0
    pair_start = torch.where(complete, positions + 1 - RATIO, torch.full_like(positions, -1))
    compressed_rope_cos = compressed_freqs_cos[pair_start.clamp_min(0)].masked_fill(~complete[:, None], 1.0)
    compressed_rope_sin = compressed_freqs_sin[pair_start.clamp_min(0)].masked_fill(~complete[:, None], 0.0)

    wq_a, wq_a_scale = _weight(gen, D, Q_LORA)
    wq_b, wq_b_scale = _weight(gen, Q_LORA, LOCAL_H * HEAD_DIM)
    wkv, wkv_scale = _weight(gen, D, HEAD_DIM)
    wo_b, wo_b_scale = _weight(gen, LOCAL_O_WIDTH, D)
    index_wq_b, index_wq_b_scale = _weight(gen, Q_LORA, INDEX_H * INDEX_DIM)

    compressor_wkv = torch.randn(D, HEAD_DIM, generator=gen) / math.sqrt(D)
    compressor_wgate = torch.randn(D, HEAD_DIM, generator=gen) / math.sqrt(D)
    wo_a_values = torch.randn(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, generator=gen)
    wo_a, wo_a_scale = quantize_grouped_mxfp8_weight((wo_a_values / math.sqrt(O_GROUP_IN)).bfloat16())
    # Seed each request's predecessor projections in its physical ring block.
    pending = torch.randn(MAX_BATCH_PER_DP, D, generator=gen).bfloat16().float()
    state_cache = torch.zeros(MAX_BATCH_PER_DP + 1, STATE_CAPACITY, STATE_WIDTH)
    state_block_table = torch.arange(requests - 1, -1, -1, dtype=torch.int32).unsqueeze(1) + 1
    for request, prefix in enumerate(prefixes):
        if prefix:
            block = int(state_block_table[request, 0])
            slot = (prefix - 1) % STATE_CAPACITY
            state_cache[block, slot, :HEAD_DIM] = pending[request] @ compressor_wkv
            state_cache[block, slot, HEAD_DIM:] = pending[request] @ compressor_wgate
    query_start_loc = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)

    window_cache, window_cache_scales = quantize_mxfp8_cache(
        torch.randn(window_pages, BLOCK_SIZE, 1, HEAD_DIM, generator=gen).bfloat16()
    )
    compressed_cache, compressed_cache_scale = quantize_mxfp4_cache(
        torch.randn(compressed_pages, BLOCK_SIZE, 1, HEAD_DIM, generator=gen).bfloat16(),
        CMP_GROUP,
        "bf16",
    )
    index_cache, index_cache_codes = quantize_mxfp4_cache(
        torch.randn(compressed_pages, BLOCK_SIZE, 1, INDEX_DIM, generator=gen).bfloat16(),
        IDX_GROUP,
        "e8m0",
    )

    values = {
        "x": torch.randn(tokens, D, generator=gen).bfloat16(),
        "wq_a": wq_a,
        "wq_a_scale": wq_a_scale,
        "q_norm_weight": (torch.randn(Q_LORA, generator=gen) * 0.1 + 1).bfloat16(),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": wkv,
        "wkv_scale": wkv_scale,
        "kv_norm_weight": (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "attn_sink": torch.randn(LOCAL_H, generator=gen) * 2,
        "wo_a": wo_a,
        "wo_a_scale": wo_a_scale,
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
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
        "index_cache_scale": index_cache_codes.view(E8M0),
        "index_block_table": index_block_table,
        "position_ids": positions.to(torch.int32),
        "compressed_rope_cos": compressed_rope_cos,
        "compressed_rope_sin": compressed_rope_sin,
        "compressor_wkv": compressor_wkv,
        "compressor_wgate": compressor_wgate,
        "query_start_loc": query_start_loc,
        "state_block_table": state_block_table,
        "state_cache": state_cache,
        "compressor_norm_weight": (torch.randn(HEAD_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "compressed_slots": compressed_slots,
        "index_wk": (torch.randn(HEAD_DIM, INDEX_DIM, generator=gen) / math.sqrt(HEAD_DIM)).bfloat16(),
        "index_norm_weight": (torch.randn(INDEX_DIM, generator=gen) * 0.1 + 1).bfloat16(),
        "index_wq_b": index_wq_b,
        "index_wq_b_scale": index_wq_b_scale,
        "index_weights_proj": (torch.randn(D, INDEX_H, generator=gen) / math.sqrt(D)).bfloat16(),
    }

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
        f"[FIXTURE] c2a tokens={tokens} requests={requests} mode={mode} case={case} "
        f"lengths={lengths} prefixes={prefixes} "
        f"max_compressed_len={int(values['compressed_lens'].max())} "
        f"window_pages={window_pages} compressed_pages={compressed_pages} "
        f"index_block_table={tuple(index_block_table.shape)}"
    )
    if full_rope_tables:
        values.update(
            freqs_cos=freqs_cos, freqs_sin=freqs_sin,
            compressed_freqs_cos=compressed_freqs_cos, compressed_freqs_sin=compressed_freqs_sin,
            compressed_rope_positions=pair_start.to(torch.int32),
        )
        return {name: values[name] for name in PROGRAM_INPUT_NAMES}
    return {name: values[name] for name in INPUT_NAMES}

MUTABLE_NAMES = (
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "index_cache",
    "index_cache_scale",
    "state_cache",
)

SHARDED_NAMES = ("wq_b", "wq_b_scale", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale")

CACHE_SLOTS = {
    "window_cache": "window_slots",
    "window_cache_scale": "window_slots",
    "compressed_cache": "compressed_slots",
    "compressed_cache_scale": "compressed_slots",
    "index_cache": "compressed_slots",
    "index_cache_scale": "compressed_slots",
}


def cache_compare(world_size, tokens, names=None, payload_error_ratio=0.01):
    """Scales stored as E8M0 match exactly; everything else allows a one-code mismatch.

    An E8M0 scale is a bare exponent, coarse enough that device and CPU agree exactly. A
    BF16 scale (``compressed_cache_scale``) is a derived quantity (amax over a group), so
    the device and the CPU golden can differ by a ULP; it shares the payload's boundary-flip
    tolerance instead of the exact scale budget. Mapped rows elsewhere stay unchanged.
    """
    options = dict(
        mapping_shape=(world_size, tokens),
        block_size=BLOCK_SIZE,
        leading_rank_axis=True,
        atol=0.0,
        rtol=0.0,
    )
    exact_scales = {
        scale_name for scale_name, _, scale_format in CACHE_PAYLOADS.values()
        if scale_format in (None, "e8m0")
    }
    return {
        name: mapped_pool_ratio_allclose(
            CACHE_SLOTS[name],
            pool_name=name,
            max_error_ratio=0.0 if name in exact_scales else payload_error_ratio,
            **options,
        )
        for name in (CACHE_SLOTS if names is None else names)
    }

CACHE_PAYLOADS = {
    "window_cache": ("window_cache_scale", None, None),
    "compressed_cache": ("compressed_cache_scale", 16, "bf16"),
    "index_cache": ("index_cache_scale", 32, "e8m0"),
}

REUSE_INPUT_NAMES = (
    "x", "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale", "compressed_cache",
    "compressed_cache_scale", "compressed_indices",
)

REUSE_MUTABLE_NAMES = ("window_cache", "window_cache_scale")

REUSE_SHARDED_NAMES = ("wq_b", "wq_b_scale", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale")


MODES = {
    "full": (INPUT_NAMES, MUTABLE_NAMES, SHARDED_NAMES),
    "reuse": (REUSE_INPUT_NAMES, REUSE_MUTABLE_NAMES, REUSE_SHARDED_NAMES),
}
# Per-rank attention results besides the TP-reduced output.
ATTENTION_STATE = {"full": MUTABLE_NAMES + ("topk_indices",), "reuse": REUSE_MUTABLE_NAMES}


def make_hc_inputs(tokens, seed, case):
    """mHC inputs with distinct per-token magnitudes so both RMS statistics stay observable."""
    gen = torch.Generator().manual_seed(seed)
    magnitudes = (0.5 + torch.arange(tokens) % 4).reshape(-1, 1, 1)
    # The residual stream holds BF16 values; FP32 storage only keeps hc_post's single rounding.
    x_hc = (torch.randn(tokens, HC_MULT, D, generator=gen) * magnitudes).bfloat16().float()
    values = {
        "x_hc": x_hc,
        "pre_mix": torch.sigmoid(torch.randn(tokens, HC_MULT, generator=gen)) + M.hc_eps,
        "hc_attn_fn": torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM),
        "hc_attn_scale": torch.randn(3, generator=gen),
        "hc_attn_base": torch.randn(MIX_HC, generator=gen),
        "attn_norm_weight": (torch.randn(D, generator=gen) * 0.1 + 1).bfloat16(),
    }
    if case == "zero":
        values["x_hc"] = torch.zeros_like(x_hc)
    return values


def make_attention_inputs(mode, tokens, requests, seed, case, *, make_inputs=None):
    """Attention fixture of the chained mode, without ``x``: the chain derives it from x_hc."""
    make = make_c2a_inputs if make_inputs is None else make_inputs
    options = {"full_rope_tables": True} if mode == "full" else {}
    values = make(tokens=tokens, requests=requests, seed=seed, case=case, mode="prefill", **options)
    return {name: value for name, value in values.items() if name != "x"}


def token_shards(value, local_tokens):
    """Pad scheduled rows and split them in contiguous TP rank order."""
    padded = value.new_zeros(TP_SIZE * local_tokens, *value.shape[1:])
    padded[:value.shape[0]].copy_(value)
    return padded.reshape(TP_SIZE, local_tokens, *value.shape[1:])


def reference_attention(mode, tensors, x, ranks, *, reference=golden_reference_c2a):
    """Evaluate valid global rows and retain each head rank's cache side effects."""
    names, mutable_names, _ = MODES[mode]
    partials, results = [], []
    counts = tensors["num_tokens"]
    active = int(counts[ranks[0], 0]) if isinstance(counts, torch.Tensor) and counts.ndim == 2 else int(counts)
    token_names = {
        "rope_cos", "rope_sin", "window_slots", "window_indices", "token_to_req_indices",
        "compressed_lens", "position_ids", "compressed_rope_cos", "compressed_rope_sin",
        "compressed_slots", "compressed_indices",
    }
    for rank in ranks:
        inputs = {
            name: x[:active] if name == "x" else tensors[name][rank]
            for name in names if mode != "full" or name not in FULL_ROPE_NAMES
        }
        if mode == "full":
            for row_name, table_name in FULL_ROPE_NAMES.items():
                positions_name = "compressed_rope_positions" if row_name.startswith("compressed_") else "position_ids"
                positions = tensors[positions_name][rank, :active].to(torch.long)
                identity = 1.0 if row_name.endswith("cos") else 0.0
                inputs[row_name] = tensors[table_name][rank][positions.clamp_min(0)].masked_fill(
                    positions[:, None] < 0, identity
                )
        for name in token_names.intersection(inputs):
            inputs[name] = inputs[name][:active]
        if "query_start_loc" in inputs:
            inputs["query_start_loc"] = inputs["query_start_loc"].clamp_max(active)
        output = torch.zeros_like(x, dtype=torch.float32)
        if active:
            # Head partials remain FP32 until the TP sum is complete.
            result = reference(inputs, fp32_output=True)
            output[:active].copy_(result["output"])
        else:
            result = {name: inputs[name].clone() for name in mutable_names}
        if mode == "full":
            topk = torch.full((x.shape[0], INDEX_TOPK), -1, dtype=torch.int32)
            if active:
                topk[:active].copy_(result["topk_indices"])
            result["topk_indices"] = topk
        result["output"] = output
        partials.append(output)
        results.append(result)
    return sum(partials).bfloat16(), results


def make_golden(mode, *, attention_reference=None, state_names=None):
    """Reconstruct scheduled rows, evaluate full references, then shard local state."""

    def golden(tensors):
        local = tensors["x_hc"].shape[1]
        tokens = tensors["attn_input"].shape[1]
        for base in range(0, tensors["x_hc"].shape[0], TP_SIZE):
            ranks = range(base, base + TP_SIZE)
            group = slice(base, base + TP_SIZE)
            active = int(tensors["num_tokens"][base, 0])
            x_hc = tensors["x_hc"][group].flatten(0, 1)[:tokens].clone()
            x_hc[active:] = 0
            incoming = tensors["pre_mix"][group].flatten(0, 1)[:tokens]
            pre, post, comb = golden_mhc_mixes(
                x_hc, tensors["hc_attn_fn"][base], tensors["hc_attn_scale"][base], tensors["hc_attn_base"][base]
            )
            x = golden_attention_input(x_hc, incoming, tensors["attn_norm_weight"][base])
            x[active:] = 0
            reference = attention_reference or reference_attention
            attention, results = reference(mode, tensors, x, ranks)
            for rank, result in zip(ranks, results):
                for name in ATTENTION_STATE[mode] if state_names is None else state_names:
                    tensors[name][rank].copy_(result[name])
            output = golden_hc_post(attention, x_hc, post, comb).float()
            output[active:] = 0
            pre[active:] = 0
            tensors["attn_input"][group].copy_(x.expand_as(tensors["attn_input"][group]))
            for name, value in (("attn_output", attention), ("next_pre_mix", pre), ("x_hc_out", output)):
                tensors[name][group].copy_(token_shards(value, local))

    return golden


def _report(name, actual, expected):
    """Print and return (rel_l2, worst token-row rel_l2); rows are the leading axis."""
    diff = actual.double() - expected.double()
    rel_l2 = (diff.norm() / expected.double().norm().clamp_min(1e-12)).item()
    rows = diff.flatten(1).norm(dim=-1) / expected.double().flatten(1).norm(dim=-1).clamp_min(1e-12)
    print(
        f"[PRECISION] {name} rel_l2={rel_l2:.6g} max_abs={diff.abs().max().item():.6g} "
        f"max_row_rel_l2={rows.max().item():.6g}"
    )
    return rel_l2, rows.max().item()


class StagedAttentionReference:
    """The attention reference re-run on the device's own ``attn_input`` (teacher forcing).

    One BF16 ULP of the attention input can move an MXFP8 group scale or an E2M1 code, and with
    it the attention output by close to 1%. The device's mHC pre and RMSNorm reduce in another
    order than torch and flip a handful of such ULPs, so the attention stage keeps its standalone
    budget against this reference while ``attn_input`` itself is checked against the golden input.
    """

    def __init__(self, mode, initial_state, *, attention_reference=None, state_names=None):
        """Keep the mode and pre-run cache and state of every rank."""
        self.mode = mode
        self.initial_state = initial_state
        self.outputs = None
        self.reference = attention_reference or reference_attention
        self.state_names = ATTENTION_STATE[mode] if state_names is None else state_names

    def __call__(self, inputs, actual_outputs):
        """Reference every rank's attention on the device's ``attn_input``, once per run."""
        if self.outputs is None:
            tensors = {**inputs, **self.initial_state}
            x = actual_outputs["attn_input"]
            names = ("attn_output",) + self.state_names
            outputs = {name: torch.empty_like(actual_outputs[name]) for name in names}
            for base in range(0, x.shape[0], TP_SIZE):
                ranks = range(base, base + TP_SIZE)
                attention, results = self.reference(self.mode, tensors, x[base], ranks)
                if isinstance(inputs["num_tokens"], torch.Tensor) and inputs["num_tokens"].ndim == 2:
                    outputs["attn_output"][base : base + TP_SIZE] = token_shards(
                        attention, actual_outputs["attn_output"].shape[1]
                    )
                else:
                    outputs["attn_output"][base : base + TP_SIZE] = attention
                for rank, result in zip(ranks, results):
                    for name in self.state_names:
                        outputs[name][rank] = result[name]
            self.outputs = outputs
        return self.outputs

    def compare(self, name, check):
        """Hold one attention output to ``check`` against the teacher-forced reference."""

        def staged(actual, expected, *, inputs, actual_outputs, expected_outputs, **kwargs):
            """Report the end-to-end difference, then compare against the forced reference."""
            if name == "attn_output":
                for base in range(0, actual.shape[0], TP_SIZE):
                    counts = inputs["num_tokens"]
                    if isinstance(counts, torch.Tensor) and counts.ndim == 2:
                        active = int(counts[base, 0])
                        if active:
                            _report(
                                "attn_output(end-to-end, not gated)",
                                actual[base : base + TP_SIZE].flatten(0, 1)[:active],
                                expected[base : base + TP_SIZE].flatten(0, 1)[:active],
                            )
                    else:
                        _report("attn_output(end-to-end, not gated)", actual[base], expected[base])
            forced = self(inputs, actual_outputs)
            return check(
                actual, forced[name], inputs=inputs, actual_outputs=actual_outputs, expected_outputs=forced, **kwargs
            )

        return staged


def make_compare(mode, world_size, tokens):
    """Compare the final outputs and published state with the default comparators."""
    compare = {
        "attn_input": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
        "attn_output": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        "x_hc_out": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
    }
    if mode == "full":
        compare["topk_indices"] = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02)
        compare["state_cache"] = ratio_allclose(atol=1e-3, rtol=1e-3)
        compare.update(cache_compare(world_size, tokens))
    else:
        compare.update(cache_compare(world_size, tokens, REUSE_MUTABLE_NAMES))
    return compare


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_prefill_c2a_full(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
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
    freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_freqs_cos: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_freqs_sin: pl.Tensor[[ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_positions: pl.Tensor[[T_DYN], pl.INT32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32]],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Bind the runtime shapes and run the sublayer once on one rank."""
    freqs_cos.bind_dynamic(0, ROPE_ROWS_DYN)
    x_hc.bind_dynamic(0, SP_T_DYN)
    attn_input.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    query_start_loc.bind_dynamic(0, Q_START_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
    active = pl.read(num_tokens, [0])
    tokens = pl.tensor.dim(attn_input, 0)
    rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    compressed_rope_cos = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    compressed_rope_sin = pl.create_tensor([tokens, ROPE_DIM // 2], dtype=pl.FP32)
    materialize_rope_rows(freqs_cos, freqs_sin, position_ids, active, rope_cos, rope_sin)
    materialize_rope_rows(
        compressed_freqs_cos, compressed_freqs_sin, compressed_rope_positions, active,
        compressed_rope_cos, compressed_rope_sin,
    )
    prefill_c2a_full_sp(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table, position_ids,
        compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_wgate,
        query_start_loc, state_block_table, state_cache, compressor_norm_weight,
        compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale,
        index_weights_proj, topk_indices, input_window, input_arrived, output_window, output_arrived, attn_input,
        attn_output, next_pre_mix, x_hc_out, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE,
        active, attention_epoch,
    )
    return (
        x_hc_out, next_pre_mix, attn_output, attn_input, topk_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, index_cache, index_cache_scale, state_cache,
    )


@pl.jit.host
def l3_prefill_c2a_full(
    x_hc: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    freqs_cos: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    freqs_sin: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, CMP_PACKED], pl.UINT8]],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, CMP_SCALES], pl.BF16]
    ],
    token_to_req_indices: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, IDX_PACKED], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, IDX_SCALES], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[RANKS_DYN, B_DYN, TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressed_freqs_cos: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_freqs_sin: pl.Tensor[[RANKS_DYN, ROPE_ROWS_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_positions: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressor_wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP32],
    compressor_wgate: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[RANKS_DYN, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[RANKS_DYN, B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[
        pl.Tensor[[RANKS_DYN, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]
    ],
    compressor_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[RANKS_DYN, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS_DYN, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[RANKS_DYN, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32]],
    attn_input: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Allocate the TP communication windows and launch one rank entry per device."""
    freqs_cos.bind_dynamic(1, ROPE_ROWS_DYN)
    x_hc.bind_dynamic(0, RANKS_DYN)
    x_hc.bind_dynamic(1, SP_T_DYN)
    attn_input.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    query_start_loc.bind_dynamic(1, Q_START_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    input_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    input_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    data_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
    signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        input_data = pld.window(input_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        input_signal = pld.window(input_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        data = pld.window(data_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        signal = pld.window(signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        # Each rank consumes packed MX_B_NN scale rows.
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        index_wq_b_scale_r: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale[rank]
        l2_prefill_c2a_full(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank],
            wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank],
            attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale_r, freqs_cos[rank],
            freqs_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank],
            window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
            token_to_req_indices[rank], compressed_lens[rank], index_cache[rank],
            index_cache_scale[rank], index_block_table[rank], position_ids[rank],
            compressed_freqs_cos[rank], compressed_freqs_sin[rank], compressed_rope_positions[rank], compressor_wkv[rank],
            compressor_wgate[rank], query_start_loc[rank], state_block_table[rank], state_cache[rank],
            compressor_norm_weight[rank], compressed_slots[rank], index_wk[rank],
            index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r,
            index_weights_proj[rank], topk_indices[rank], attn_input[rank], attn_output[rank],
            next_pre_mix[rank], x_hc_out[rank], input_data, input_signal, data, signal, rank, num_tokens[rank], attention_epoch,
            device=rank,
        )


def build_specs(args, mode, initial_state, *, make_inputs=None):
    """Stacked specs whose per-rank fixtures are drawn once, on first use.

    The comparators see pure inputs only, so the initial cache and state are kept here for the
    staged attention reference.
    """
    world_size = TP_SIZE * args.dp
    names, mutable_names, sharded_names = MODES[mode]
    program_names = PROGRAM_INPUT_NAMES if mode == "full" else names
    attention_names = tuple(name for name in program_names if name != "x")
    ranks = {}
    local = (args.tokens + TP_SIZE - 1) // TP_SIZE
    counts = getattr(args, "dp_tokens", None) or [args.tokens] * args.dp

    def initialize(name):
        """Draw every rank's fixture on the first spec that needs it, then stack one tensor."""
        if not ranks:
            for rank in range(world_size):
                ranks[rank] = make_attention_inputs(
                    mode, args.tokens, args.requests, args.seed + rank, args.case,
                    make_inputs=make_inputs,
                )
            for rank in range(world_size):
                leader = rank // TP_SIZE * TP_SIZE
                # Every rank of a TP group sees the same tokens, residual stream, metadata and caches.
                ranks[rank].update(
                    make_hc_inputs(args.tokens, args.seed + 7919 * (leader + 1), args.case)
                    if rank == leader else {key: ranks[leader][key] for key in HC_INPUT_NAMES}
                )
                for key in attention_names:
                    if key not in sharded_names:
                        ranks[rank][key] = ranks[leader][key]
            for rank in range(world_size):
                active = counts[rank // TP_SIZE]
                start = rank % TP_SIZE * local
                valid = max(0, min(local, active - start))
                for key in ("x_hc", "pre_mix"):
                    original = ranks[rank][key]
                    shard = original.new_full((local, *original.shape[1:]), 17.0)
                    shard[:valid].copy_(original[start:start + valid])
                    ranks[rank][key] = shard
                for key in ("window_slots", "compressed_slots", "window_indices", "compressed_indices"):
                    if key in ranks[rank]:
                        ranks[rank][key] = ranks[rank][key].clone()
                        ranks[rank][key][active:] = -1
                if "query_start_loc" in ranks[rank]:
                    ranks[rank]["query_start_loc"] = ranks[rank]["query_start_loc"].clamp_max(active)
        column = [ranks[rank][name] for rank in range(world_size)]
        if column[0].dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
            stacked = torch.stack([value.view(torch.uint8) for value in column]).view(column[0].dtype)
        else:
            stacked = torch.stack(column)
        if name in mutable_names:
            initial_state[name] = stacked.clone()
        return stacked

    shapes = make_attention_inputs(mode, args.tokens, args.requests, args.seed, args.case, make_inputs=make_inputs)
    shapes.update(make_hc_inputs(args.tokens, args.seed, args.case))
    specs = [
        TensorSpec(
            name,
            [world_size, local, *shapes[name].shape[1:]] if name in ("x_hc", "pre_mix")
            else [world_size, *shapes[name].shape],
            shapes[name].dtype,
            init_value=(lambda name=name: initialize(name)),
            resident="stacked",
        )
        for name in HC_INPUT_NAMES + attention_names
    ]
    if mode == "full":
        specs.append(
            TensorSpec("topk_indices", [world_size, args.tokens, INDEX_TOPK], torch.int32, resident="stacked")
        )
    specs += [
        TensorSpec("attn_input", [world_size, args.tokens, D], torch.bfloat16, resident="stacked"),
        TensorSpec("attn_output", [world_size, local, D], torch.bfloat16, resident="stacked"),
        TensorSpec("next_pre_mix", [world_size, local, HC_MULT], torch.float32, resident="stacked"),
        TensorSpec("x_hc_out", [world_size, local, HC_MULT, D], torch.float32, resident="stacked"),
        TensorSpec("num_tokens", [world_size, 1], torch.int32, resident="stacked",
                   init_value=torch.tensor(counts, dtype=torch.int32).repeat_interleave(TP_SIZE).reshape(-1, 1)),
        ScalarSpec(
            "attention_epoch",
            torch.int32,
            1,
            compile_runtime=True,
            benchmark_step=1 if args.bench else None,
        ),
    ]
    return specs


PREFILL_C2A_FULL_RING_HEAP = (2 * 1024 * 1024 * 1024,) * 4


if __name__ == "__main__":
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C2A full with mHC: A5 precision")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--dp-tokens", type=str, help="comma-separated active counts, one per DP group")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--case", default="mixed", choices=["mixed", "long", "masked", "zero"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    devices = list(range(TP_SIZE * args.dp))
    if args.device:
        devices = [int(value) for value in args.device.split(",")]
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error(f"--device must name {TP_SIZE * args.dp} distinct non-negative device IDs")
    if not 1 <= args.tokens <= PREFILL_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {PREFILL_MAX_TOKENS}]")
    if not 1 <= args.requests <= min(MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"--requests must be in [1, {min(MAX_BATCH_PER_DP, args.tokens)}]")
    args.dp_tokens = [int(value) for value in args.dp_tokens.split(",")] if args.dp_tokens else [args.tokens] * args.dp
    if len(args.dp_tokens) != args.dp or any(count < 0 or count > args.tokens for count in args.dp_tokens):
        parser.error("--dp-tokens must provide one count in [0, tokens] per DP group")
    torch.set_num_threads(8)

    print(
        f"[C2A-HC] mode=full tokens={args.tokens} requests={args.requests} TP={TP_SIZE} "
        f"DP={args.dp} case={args.case} seed={args.seed} devices={devices}"
    )
    initial_state = {}
    result = run(
        fn=l3_prefill_c2a_full,
        specs=build_specs(args, "full", initial_state),
        golden_fn=make_golden("full"),
        compile_only=args.compile_only,
        config=dict(
            platform=args.platform,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            # Full retains the K-chunk products for both compressor projections.
            ring_heap=PREFILL_C2A_FULL_RING_HEAP,
        ),
        compare_fn=make_compare("full", len(devices), args.tokens),
    )
    if args.compile_only and result.passed:
        print("[C2A-HC] Compilation passed; device accuracy was NOT validated.")
    if not result.passed:
        raise SystemExit(result.error or 1)
