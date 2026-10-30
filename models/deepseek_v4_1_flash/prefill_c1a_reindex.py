# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C1A reindex attention wired through delayed mHC mixing."""

import math
from functools import partial

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from golden import ScalarSpec, TensorSpec, mapped_pool_ratio_allclose, ratio_allclose, run
from config import (
    FLASH as M, BLOCK_SIZE, TP_SIZE, T_DYN, ORI_BLOCKS_DYN, CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, B_DYN, TABLE_DYN,
    INDEX_POOL_BLOCKS_DYN, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, PREFILL_MAX_TOKENS, WINDOW_CACHE_GROUP,
    SP_T_DYN,
)
from hc import golden_mhc_mixes
from hc import golden_hc_post, hc_post, prefill_sp_post
from hc import golden_hc_collapse, hc_pre
from rmsnorm import golden_rms_norm, rms_norm_tiled
from sparse_attention import (
    attend_packed, golden_compressed_attention, golden_prefill_c1a_attention,
    prefill_publish_window,
)
from prefill_c1a_full import (
    CASE_DEFAULT, CASE_MAX_TOKENS, CASE_NAMES, apply_distributed_golden, fixture_token_count, make_fixture_values, topk_indices_compare, wrap_attention_compare, COMMON_INPUT_NAMES,
)
from prefill_cp_allgather import prefill_sp_input_allgather
from o_proj import retire_prefill_signals, prefill_sp_output_reduce_scatter, o_proj, prefill_tp_output_all_reduce
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj_rope
from prefill_indexer import (
    prefill_indexer, golden_prefill_indexer,
    golden_prefill_index_scores as golden_index_scores,
)


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


PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4


REINDEX_INPUT_NAMES = COMMON_INPUT_NAMES + (
    "token_to_req_indices",
    "compressed_lens",
    "index_cache",
    "index_cache_scale",
    "index_block_table",
    "pool_block_ids", "pool_block_counts",
    "index_wq_b",
    "index_wq_b_scale",
    "index_weights_proj",
)


@pl.jit.inline(auto_scope=False)
def prefill_c1a_reindex_attention(
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
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    scatter_output: pl.constexpr = False,
    use_candidates: pl.constexpr = True,
):
    """Refresh ratio-1 Top-K rows inside a supplied candidate set."""
    tokens = pl.tensor.dim(x, 0)
    partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
    if num_tokens > 0:
        query_latent = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        proj_payload, proj_scales = quantize_projection_input(x, num_tokens)
        q_proj_rope(
            proj_payload, proj_scales,
            wq_a, wq_a_scale, q_norm_weight,
            wq_b, wq_b_scale,
            rope_cos, rope_sin,
            query_latent, query,
            num_tokens,
        )
        cache_ready = pl.system.task_dummy(deps=[])
        indexer_completion = prefill_indexer(
            x, query_latent, token_to_req_indices, compressed_lens, index_cache, index_cache_scale, index_block_table,
            rope_cos, rope_sin, index_wq_b, index_wq_b_scale, index_weights_proj,
            pool_block_ids, pool_block_counts, topk_indices, num_tokens, cache_ready, pool=use_candidates, c2a=False,
        )
        window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        kv_proj_rope(
            proj_payload, proj_scales,
            wkv, wkv_scale, kv_norm_weight,
            rope_cos, rope_sin,
            kv_projection, window_kv,
            num_tokens,
        )
        window_ready = prefill_publish_window(
            window_kv, window_slots, window_cache, window_cache_scale, num_tokens, indexer_completion,
        )

        attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        attend_packed(
            query,
            window_indices, window_cache, window_cache_scale,
            topk_indices, compressed_cache, compressed_cache_scale,
            attn_sink, attended,
            num_tokens, window_ready,
        )
        o_proj(
            attended, wo_a, wo_a_scale, wo_b,
            wo_b_scale, rope_cos, rope_sin,
            partial, num_tokens,
        )
    if scatter_output:
        prefill_sp_output_reduce_scatter(
            partial, output_window, output_arrived, output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    else:
        prefill_tp_output_all_reduce(
            partial, output_window, output_arrived, output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    return output


def golden_prefill_attn_c1a_reindex(
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
    token_to_req_indices: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    pool_block_ids: torch.Tensor,
    pool_block_counts: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> dict:
    """Return one rank's FP32 partial output and reference cache state."""
    return golden_compressed_attention(
        mode='reindex', ratio=1,
        x=x,
        wq_a=wq_a, wq_a_scale=wq_a_scale, q_norm_weight=q_norm_weight, wq_b=wq_b, wq_b_scale=wq_b_scale,
        wkv=wkv, wkv_scale=wkv_scale, kv_norm_weight=kv_norm_weight, attn_sink=attn_sink,
        wo_a=wo_a, wo_a_scale=wo_a_scale, wo_b=wo_b, wo_b_scale=wo_b_scale,
        rope_cos=rope_cos, rope_sin=rope_sin,
        window_slots=window_slots, window_indices=window_indices, window_cache=window_cache, window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache, compressed_cache_scale=compressed_cache_scale, compressed_indices=None,
        compressor_wkv=None, compressor_wgate=None, compressor_norm_weight=None,
        state_block_table=None, state_cache=None,
        compressed_slots=None, position_ids=None, compressed_lens=compressed_lens, compressed_rope_cos=None, compressed_rope_sin=None,
        index_wk=None, index_norm_weight=None, index_wq_b=index_wq_b, index_wq_b_scale=index_wq_b_scale, index_weights_proj=index_weights_proj,
        index_cache=index_cache, index_cache_scale=index_cache_scale, index_block_table=index_block_table, token_to_req_indices=token_to_req_indices, candidate_mask=None,
        indexer_fn=partial(golden_prefill_indexer, pool_block_ids=pool_block_ids,
                           pool_block_counts=pool_block_counts),
        attention_fn=golden_prefill_c1a_attention, sequential_query=True, round_cache_to_even=True, output_dtype=torch.float32,
    )


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_prefill_c1a_reindex_sharded(
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
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
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
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32]],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, SP_T_DYN)
    pre_mix.bind_dynamic(0, SP_T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    compressed_cache_scale.bind_dynamic(0, CMP_BLOCKS_DYN)
    token_to_req_indices.bind_dynamic(0, T_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_cache_scale.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    compressed_rope_cos.bind_dynamic(0, T_DYN)
    compressed_rope_sin.bind_dynamic(0, T_DYN)
    compressed_slots.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    pool_block_ids.bind_dynamic(0, T_DYN)
    pool_block_counts.bind_dynamic(0, T_DYN)
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    compressed_indices.bind_dynamic(0, T_DYN)
    attn_input.bind_dynamic(0, T_DYN)
    attn_output.bind_dynamic(0, SP_T_DYN)
    next_pre_mix.bind_dynamic(0, SP_T_DYN)
    x_hc_out.bind_dynamic(0, SP_T_DYN)
    active = pl.read(num_tokens, [0])
    tokens = pl.tensor.dim(x_hc, 0)
    group_tokens = pl.tensor.dim(attn_input, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    local_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        hc_collapsed, next_pre_mix, post_mix, residual_mix,
    )
    hc_norm_tokens = pl.tensor.dim(hc_collapsed, 0)
    for hc_norm_block in pl.spmd((hc_norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(hc_collapsed, attn_norm_weight, local_hidden, hc_norm_block, hc_norm_tokens, output_fp32=False)
    prefill_sp_input_allgather(
        local_hidden, input_window, input_arrived, attn_input, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE,
        active, attention_epoch,
    )
    for row in pl.spmd(group_tokens, name_hint='c1a_sp_inactive_topk'):
        if row >= active:
            topk_indices[row:row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
    prefill_c1a_reindex_attention(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
        attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table, pool_block_ids, pool_block_counts, index_wq_b, index_wq_b_scale,
        index_weights_proj, topk_indices, output_window, output_arrived, attn_output,
        rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active, attention_epoch,
        scatter_output=True,
        use_candidates=True,
    )
    prefill_sp_post(attn_output, x_hc, post_mix, residual_mix, x_hc_out, next_pre_mix, rank % TP_SIZE, active)
    retire_prefill_signals(x_hc_out, input_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    retire_prefill_signals(x_hc_out, output_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    return x_hc_out


@pl.jit.host
def l3_prefill_c1a_reindex_sharded(
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
    rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    token_to_req_indices: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[RANKS_DYN, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[RANKS_DYN, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS_DYN, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[RANKS_DYN, D, INDEX_H], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.InOut[pl.Tensor[[RANKS_DYN, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32]],
    compressed_indices: pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32],
    attn_input: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, RANKS_DYN)
    x_hc.bind_dynamic(1, SP_T_DYN)
    pre_mix.bind_dynamic(1, SP_T_DYN)
    rope_cos.bind_dynamic(1, T_DYN)
    rope_sin.bind_dynamic(1, T_DYN)
    window_slots.bind_dynamic(1, T_DYN)
    window_indices.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    compressed_cache_scale.bind_dynamic(1, CMP_BLOCKS_DYN)
    token_to_req_indices.bind_dynamic(1, T_DYN)
    compressed_lens.bind_dynamic(1, T_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_cache_scale.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    compressed_rope_cos.bind_dynamic(1, T_DYN)
    compressed_rope_sin.bind_dynamic(1, T_DYN)
    compressed_slots.bind_dynamic(1, T_DYN)
    topk_indices.bind_dynamic(1, T_DYN)
    pool_block_ids.bind_dynamic(1, T_DYN)
    pool_block_counts.bind_dynamic(1, T_DYN)
    pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
    compressed_indices.bind_dynamic(1, T_DYN)
    attn_input.bind_dynamic(1, T_DYN)
    attn_output.bind_dynamic(1, SP_T_DYN)
    next_pre_mix.bind_dynamic(1, SP_T_DYN)
    x_hc_out.bind_dynamic(1, SP_T_DYN)
    input_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    input_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    output_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    output_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        input_window = pld.window(input_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        input_arrived = pld.window(input_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        output_window = pld.window(output_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        output_arrived = pld.window(output_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        index_wq_b_scale_r: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale[rank]
        l2_prefill_c1a_reindex_sharded(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
            wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank],
            wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
            window_cache[rank], window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
            token_to_req_indices[rank], compressed_lens[rank], index_cache[rank], index_cache_scale[rank],
            index_block_table[rank], compressed_rope_cos[rank], compressed_rope_sin[rank],
            compressor_wkv[rank], compressor_norm_weight[rank], compressed_slots[rank], index_wk[rank],
            index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r, index_weights_proj[rank],
            topk_indices[rank], pool_block_ids[rank], pool_block_counts[rank], compressed_indices[rank], attn_input[rank],
            attn_output[rank], next_pre_mix[rank], x_hc_out[rank], input_window, input_arrived, output_window,
            output_arrived, num_tokens[rank], rank, attention_epoch, device=rank,
        )


@pl.jit.inline(auto_scope=False)
def prefill_c1a_reindex(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
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
    window_cache_scale: pl.Tensor[
        [ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16
    ],
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
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run official mHC mixes, collapse, attention RMSNorm, C1A, then mHC post."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        hidden, next_pre_mix, post_mix, residual_mix,
    )
    norm_tokens = pl.tensor.dim(hidden, 0)
    for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(hidden, attn_norm_weight, attn_input, norm_block, norm_tokens, output_fp32=False)
    prefill_c1a_reindex_attention(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table, pool_block_ids, pool_block_counts,
        index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices,
        output_window, output_arrived, attn_out,
        group_base, tp_rank, num_tokens, attention_epoch,
        scatter_output=False, use_candidates=True,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(attn_out, x_hc, post_mix, residual_mix, output, deps=hc_post_deps)
    return output


@pl.jit
def l2_prefill_c1a_reindex(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
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
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run one TP rank of the delayed-mix C1A reindex path."""
    x_hc.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    prefill_c1a_reindex(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table, pool_block_ids, pool_block_counts,
        index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices,
        output_window, output_arrived, hidden, attn_input, attn_out, output, next_pre_mix,
        0, tp_rank, num_tokens, 1,
    )
    retire_prefill_signals(output, output_arrived, 0, tp_rank)
    return output


@pl.jit.host
def l3_prefill_c1a_reindex(
    x_hc: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT], pl.FP32],
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
    window_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]
    ],
    window_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.Tensor[
        [TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8
    ],
    compressed_cache_scale: pl.Tensor[
        [TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    token_to_req_indices: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.Tensor[
        [TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8
    ],
    index_cache_scale: pl.Tensor[
        [TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[TP_SIZE, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0
    ],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT], pl.FP32]],
    hidden: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    output_window_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    output_arrived_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        output_window = pld.window(output_window_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        output_arrived = pld.window(output_arrived_buf, [TP_SIZE, 1], dtype=pl.INT32)
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        index_wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = index_wq_b_scale[rank]
        l2_prefill_c1a_reindex(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
            wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank],
            wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank],
            window_indices[rank], window_cache[rank], window_cache_scale[rank],
            compressed_cache[rank], compressed_cache_scale[rank], token_to_req_indices[rank],
            compressed_lens[rank], index_cache[rank], index_cache_scale[rank],
            index_block_table[rank], pool_block_ids[rank], pool_block_counts[rank], index_wq_b[rank],
            index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank],
            output_window, output_arrived, next_pre_mix[rank], hidden[rank], attn_input[rank], attn_out[rank],
            output[rank], rank, num_tokens, device=rank,
        )


def build_hc_tensor_specs(token_count=32, case_name=CASE_DEFAULT, active_tokens=None):
    """Build C1A inputs plus a pre-mix produced by the preceding mHC block."""
    values = make_fixture_values(token_count, case_name)
    tokens = values["x"].shape[1]
    active_tokens = tokens if active_tokens is None else active_tokens
    if not 0 <= active_tokens <= tokens:
        raise ValueError(f"active_tokens must be in [0, {tokens}], got {active_tokens}")
    generator = torch.Generator().manual_seed(2026)
    x_hc = torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16().float()
    previous_x_hc = torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16().float()
    hc_attn_fn = torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)
    hc_attn_scale = torch.randn(3, generator=generator)
    hc_attn_base = torch.randn(MIX_HC, generator=generator)
    pre_mix, _, _ = golden_mhc_mixes(previous_x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    hc_values = {
        "x_hc": x_hc.unsqueeze(0).repeat(TP_SIZE, 1, 1, 1),
        "pre_mix": pre_mix.unsqueeze(0).repeat(TP_SIZE, 1, 1),
        "hc_attn_fn": hc_attn_fn.unsqueeze(0).repeat(TP_SIZE, 1, 1),
        "hc_attn_scale": hc_attn_scale.unsqueeze(0).repeat(TP_SIZE, 1),
        "hc_attn_base": hc_attn_base.unsqueeze(0).repeat(TP_SIZE, 1),
        "attn_norm_weight": values["attn_norm_weight"],
    }
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in hc_values.items()
    ]
    specs += [
        TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name])
        for name in REINDEX_INPUT_NAMES if name != "x"
    ]
    specs += [
        TensorSpec("topk_indices", [TP_SIZE, tokens, INDEX_TOPK], torch.int32),
        TensorSpec("next_pre_mix", [TP_SIZE, tokens, HC_MULT], torch.float32),
        TensorSpec("hidden", [TP_SIZE, tokens, D], torch.bfloat16),
        TensorSpec("attn_input", [TP_SIZE, tokens, D], torch.bfloat16),
        TensorSpec(
            "attn_out", [TP_SIZE, tokens, D], torch.bfloat16,
            init_value=torch.zeros(TP_SIZE, tokens, D, dtype=torch.bfloat16),
        ),
        TensorSpec("output", [TP_SIZE, tokens, HC_MULT, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]
    return specs


def golden_prefill_c1a_reindex_case(tensors):
    """Reference delayed pre-mix, current coefficient generation, attention, and post-mix."""
    next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"][0], tensors["hc_attn_fn"][0],
        tensors["hc_attn_scale"][0], tensors["hc_attn_base"][0],
    )
    hidden = golden_hc_collapse(tensors["x_hc"][0], tensors["pre_mix"][0]).to(torch.bfloat16)
    attn_input = golden_rms_norm(hidden.to(torch.bfloat16), tensors["attn_norm_weight"][0])
    active = int(tensors["num_tokens"])
    output = tensors["output"]
    tensors["x"] = attn_input.unsqueeze(0).expand(TP_SIZE, -1, -1)
    tensors["output"] = tensors["attn_out"]
    apply_distributed_golden("reindex", golden_prefill_attn_c1a_reindex, tensors, active)
    tensors["attn_out"][:, active:].zero_()
    tensors["attn_input"][:] = attn_input.unsqueeze(0)
    hc_output = golden_hc_post(tensors["attn_out"][0], tensors["x_hc"][0], post_mix, residual_mix)
    tensors["output"] = output
    tensors["next_pre_mix"][:] = next_pre_mix.unsqueeze(0)
    tensors["hidden"][:] = hidden.unsqueeze(0)
    tensors["output"][:] = hc_output.unsqueeze(0)


if __name__ == "__main__":
    import argparse

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C1A reindex with mHC validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=(1, 2, 4))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dp", type=int, default=1, choices=(1, 2))
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--active-tokens", type=int)
    parser.add_argument("--case", default=CASE_DEFAULT, choices=CASE_NAMES)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--wiring", choices=("replicated", "sharded"), default="replicated")
    parser.add_argument("--dp-tokens")
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        parser.error(f"--tp was parsed as TP{TP_SIZE}, got --tp {args.tp}")
    if args.device is None:
        args.device = ",".join(str(rank) for rank in range(TP_SIZE * args.dp))
    if args.wiring == "sharded":
        from prefill_c1a_full import validate_c1a_sp
        result = validate_c1a_sp("reindex", args, l3_prefill_c1a_reindex_sharded, golden_prefill_attn_c1a_reindex)
        if not result.passed:
            raise SystemExit(result.error or 1)
        raise SystemExit(0)
    if args.dp != 1:
        parser.error("replicated wiring requires --dp 1")
    if args.case == "causal" and not 1 <= args.tokens <= CASE_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {CASE_MAX_TOKENS}]")
    if args.active_tokens is not None and not 0 <= args.active_tokens <= args.tokens:
        parser.error(f"--active-tokens must be in [0, {args.tokens}]")
    devices = [int(device) for device in args.device.split(",")]
    if len(devices) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {devices}")
    result = run(
        fn=l3_prefill_c1a_reindex,
        specs=build_hc_tensor_specs(args.tokens, args.case, args.active_tokens),
        golden_fn=golden_prefill_c1a_reindex_case,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config={
            "backend_type": args.backend_type, "execution_mode": args.execution_mode,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            "dump_passes": args.dump_passes,
            "enable_chip_swimlane": args.enable_chip_swimlane,
            "ring_heap": PREFILL_ATTN_RING_HEAP,
        },
        compare_fn={
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "hidden": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "attn_input": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "attn_out": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "window_cache": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window_cache",
                mapping_shape=(len(devices), fixture_token_count(args.tokens, args.case)),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "window_cache_scale": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window_cache_scale",
                mapping_shape=(len(devices), fixture_token_count(args.tokens, args.case)),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "topk_indices": wrap_attention_compare(topk_indices_compare("reindex", golden_index_scores)),
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
