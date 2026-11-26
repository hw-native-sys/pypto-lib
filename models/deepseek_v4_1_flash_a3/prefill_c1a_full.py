# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C1A full attention wired through delayed mHC mixing."""

from . import config as C
import pypto.language as pl
from .activation_quant import quantize_projection_activation
import pypto.language.distributed as pld
from .config import (
    FLASH as M, TP_SIZE, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, PREFILL_MAX_TOKENS, WINDOW_CACHE_GROUP,
)
from .dsl_types import (
    T_DYN, ORI_BLOCKS_DYN, CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, B_DYN, TABLE_DYN, INDEX_POOL_BLOCKS_DYN,
    SP_T_DYN,
)
from .hc import hc_post, prefill_sp_post
from .hc import hc_pre
from .rmsnorm import rms_norm_tiled, rms_norm
from .sparse_attention import (
    KEY_TILE, TOKEN_TILE, attend_packed, prefill_publish_window,
)
from .prefill_cp_allgather import prefill_sp_input_allgather
from .o_proj import retire_prefill_signals, prefill_sp_output_reduce_scatter, prefill_o_proj_nz as o_proj, prefill_tp_output_all_reduce
from .qkv_proj_rope import q_proj_rope, kv_proj_rope
from .compressor import c1a_compressor_project, compressor_rope, c1a_compressor_publish
from .prefill_indexer import (
    clear_inactive_index_pool, prefill_indexer, publish_c1a_index_key,
)


CASE_DEFAULT = 'multi_1k'
CASE_MAX_TOKENS = C.PREFILL_MAX_TOKENS
CASE_PAGE = 128
CASE_NAMES = (CASE_DEFAULT, 'causal', 'mixed', 'long')
COMMON_INPUT_NAMES = ('x', 'wq_a', 'wq_a_scale', 'q_norm_weight', 'wq_b', 'wq_b_scale', 'wkv', 'wkv_scale', 'kv_norm_weight', 'attn_sink', 'wo_a', 'wo_a_scale', 'wo_b', 'wo_b_scale', 'rope_cos', 'rope_sin', 'window_slots', 'window_indices', 'window_cache', 'window_cache_scale', 'compressed_cache', 'compressed_cache_scale')


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


LEAF_NAMES = ('wq_a', 'wq_a_scale', 'q_norm_weight', 'wq_b', 'wq_b_scale', 'wkv', 'wkv_scale', 'kv_norm_weight', 'attn_sink', 'wo_a', 'wo_a_scale', 'wo_b', 'wo_b_scale', 'rope_cos', 'rope_sin', 'window_slots', 'window_indices', 'window_cache', 'window_cache_scale', 'compressed_cache', 'compressed_cache_scale', 'request_ids', 'compressed_lens', 'index_cache', 'index_cache_scale', 'index_block_table', 'compressed_rope_cos', 'compressed_rope_sin', 'compressor_wkv', 'compressor_norm_weight', 'compressed_slots', 'index_wk', 'index_norm_weight', 'index_wq_b', 'index_wq_b_scale', 'index_weights_proj', 'topk_indices', "pool_block_ids", "pool_block_counts", 'compressed_indices')

FULL_INPUT_NAMES = COMMON_INPUT_NAMES + (
    "request_ids",
    "compressed_lens",
    "index_cache",
    "index_cache_scale",
    "index_block_table",
    "compressed_rope_cos",
    "compressed_rope_sin",
    "compressor_wkv",
    "compressor_norm_weight",
    "compressed_slots",
    "index_wk",
    "index_norm_weight",
    "index_wq_b",
    "index_wq_b_scale",
    "index_weights_proj",
)


@pl.jit.inline(auto_scope=False)
def prefill_c1a_full_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND],
    index_wq_b_scale: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    scatter_output: pl.constexpr = False,
):
    """Publish ratio-1 caches, select sparse rows, and compute packed C1A."""
    tokens = pl.tensor.dim(x, 0)
    partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
    if num_tokens > 0:
        proj_payload, proj_scales = quantize_projection_activation(x, num_tokens, D, True, True)
        published = prefill_publish_decoder(
            x, compressor_wkv, compressor_norm_weight, index_wk, index_norm_weight,
            compressed_rope_cos, compressed_rope_sin, compressed_slots,
            compressed_cache, compressed_cache_scale, index_cache, index_cache_scale, num_tokens,
        )
        query_latent = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        _, query, qb_payload, qb_scales = q_proj_rope(
            proj_payload, proj_scales,
            wq_a, wq_a_scale, q_norm_weight,
            wq_b, wq_b_scale,
            rope_cos, rope_sin,
            query_latent, query,
            num_tokens,
        )
        indexer_completion = prefill_indexer(
            x, qb_payload, qb_scales, request_ids, compressed_lens,
            index_cache, index_cache_scale, index_block_table,
            rope_cos, rope_sin, index_wq_b, index_wq_b_scale, index_weights_proj,
            pool_block_ids, pool_block_counts, topk_indices, num_tokens, published, pool=False, c2a=False,
        )
        source_ready = pl.system.task_dummy(deps=[published, indexer_completion])
        window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        kv_proj_rope(
            proj_payload, proj_scales,
            wkv, wkv_scale, kv_norm_weight,
            rope_cos, rope_sin,
            kv_projection, window_kv,
            num_tokens,
        )
        window_ready = prefill_publish_window(window_kv, window_slots, window_cache, window_cache_scale, num_tokens, source_ready)

        attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        attend_packed(
            query,
            window_indices, window_cache, window_cache_scale,
            topk_indices, compressed_cache, compressed_cache_scale,
            attn_sink, attended,
            num_tokens, window_ready, KEY_TILE, TOKEN_TILE, shared_prefix=-1,
        )
        o_proj(
            attended, wo_a, wo_b, rope_cos, rope_sin, partial, num_tokens,
        )
    elif tokens > 0:
        clear_inactive_index_pool(pool_block_ids, pool_block_counts, num_tokens, pl.system.task_dummy(deps=[]))
    if scatter_output:
        prefill_sp_output_reduce_scatter(
            partial, output_window, output_arrived, output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    else:
        prefill_tp_output_all_reduce(
            partial, output_window, output_arrived, output,
            group_base, tp_rank, num_tokens, attention_epoch, owner_rows=False,
        )
    return output


@pl.jit.inline(auto_scope=False)
def prefill_publish_decoder(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8
    ],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x, 0)
    compressed_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    projection_ready = pl.system.task_dummy(deps=[])
    compressed_projection, projection_done = c1a_compressor_project(
        x, compressor_wkv, compressed_projection, num_tokens, projection_ready,
    )
    compressed_latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    norm_blocks = (num_tokens + 7) // 8
    norm_workers = pl.max(1, pl.min(norm_blocks, pl.system.available_aiv_count()))
    for norm_worker in pl.spmd(norm_workers, name_hint="attention_rmsnorm", deps=[projection_done]):
        for norm_block in pl.range(norm_worker, norm_blocks, norm_workers):
            rms_norm(compressed_projection, compressor_norm_weight, compressed_latent, norm_block, num_tokens, width=HEAD_DIM)
    compressed_rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    compressed_rotated = compressor_rope(
        compressed_latent, compressed_rope_cos, compressed_rope_sin, compressed_rotated, num_tokens,
    )
    compressed_ready = c1a_compressor_publish(
        compressed_rotated, compressed_slots, compressed_cache, compressed_cache_scale,
        num_tokens, projection_ready,
    )
    index_ready = publish_c1a_index_key(
        compressed_latent, index_wk, index_norm_weight, compressed_rope_cos, compressed_rope_sin,
        compressed_slots, index_cache, index_cache_scale, num_tokens,
    )
    ready = pl.system.task_dummy(deps=[compressed_ready, index_ready])
    return ready


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_prefill_c1a_full_sharded(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND],
    index_wq_b_scale: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
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
    request_ids.bind_dynamic(0, T_DYN)
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
        prefill=True,
    )
    hc_norm_tokens = pl.tensor.dim(hc_collapsed, 0)
    norm_tiles = (hc_norm_tokens + 7) // 8
    norm_workers = pl.max(1, pl.min(norm_tiles, pl.system.available_aiv_count()))
    for norm_worker in pl.spmd(norm_workers, name_hint="rms_norm"):
        for hc_norm_block in pl.range(norm_worker, norm_tiles, norm_workers):
            rms_norm_tiled(hc_collapsed, attn_norm_weight, local_hidden, hc_norm_block, hc_norm_tokens, output_fp32=False, compensated=False)
    prefill_sp_input_allgather(
        local_hidden, input_window, input_arrived, attn_input, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE,
        active, attention_epoch,
    )
    topk_tiles = group_tokens
    topk_workers = pl.max(1, pl.min(topk_tiles, pl.system.available_aiv_count()))
    for topk_worker in pl.spmd(topk_workers, name_hint='c1a_sp_inactive_topk'):
        for row in pl.range(topk_worker, topk_tiles, topk_workers):
            if row >= active:
                topk_indices[row:row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
    prefill_c1a_full_attention(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens, index_cache, index_cache_scale, index_block_table, compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, pool_block_ids, pool_block_counts, output_window, output_arrived, attn_output, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active, attention_epoch, scatter_output=True,
    )
    prefill_sp_post(attn_output, x_hc, post_mix, residual_mix, x_hc_out, next_pre_mix, rank % TP_SIZE, active)
    retire_prefill_signals(x_hc_out, input_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    retire_prefill_signals(x_hc_out, output_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    return x_hc_out


@pl.jit.host
def l3_prefill_c1a_full_sharded(
    x_hc: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[RANKS_DYN, 1, Q_LORA], pl.FP32],
    q_norm_weight: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[RANKS_DYN, 1, LOCAL_H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[RANKS_DYN, 1, HEAD_DIM], pl.FP32],
    kv_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8]],
    window_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    request_ids: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8]],
    index_block_table: pl.Tensor[[RANKS_DYN, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[RANKS_DYN, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS_DYN, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, INDEX_H * INDEX_DIM], pl.INT8],
    index_wq_b_scale: pl.Tensor[[RANKS_DYN, 1, INDEX_H * INDEX_DIM], pl.FP32],
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
    request_ids.bind_dynamic(1, T_DYN)
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
    # These signal windows are fresh on every standalone launch.
    for rank in pl.range(pld.world_size()):
        input_window = pld.window(input_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        input_arrived = pld.window(input_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        output_window = pld.window(output_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        output_arrived = pld.window(output_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        wq_a_scale_r: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND] = wkv_scale[rank]
        index_wq_b_scale_r: pl.Tensor[[1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND] = index_wq_b_scale[rank]
        l2_prefill_c1a_full_sharded(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank], rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank], index_cache[rank], index_cache_scale[rank], index_block_table[rank], compressed_rope_cos[rank], compressed_rope_sin[rank], compressor_wkv[rank], compressor_norm_weight[rank], compressed_slots[rank], index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], pool_block_ids[rank], pool_block_counts[rank], compressed_indices[rank], attn_input[rank], attn_output[rank], next_pre_mix[rank], x_hc_out[rank], input_window, input_arrived, output_window, output_arrived, num_tokens[rank], rank, 1, device=rank,
        )


STATE_NAMES = {'full': ('window_cache', 'window_cache_scale', 'compressed_cache', 'compressed_cache_scale', 'index_cache', 'index_cache_scale', 'topk_indices', "pool_block_ids", "pool_block_counts"), 'reindex': ('window_cache', 'window_cache_scale', 'topk_indices'), 'reuse': ('window_cache', 'window_cache_scale')}

TOKEN_NAMES = {'rope_cos', 'rope_sin', 'window_slots', 'window_indices', 'compressed_indices', 'request_ids', 'compressed_lens', "pool_block_ids", "pool_block_counts", 'compressed_rope_cos', 'compressed_rope_sin', 'compressed_slots'}


@pl.jit.inline(auto_scope=False)
def prefill_c1a_full(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
    window_cache_scale: pl.Tensor[
        [ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16
    ],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND],
    index_wq_b_scale: pl.Tensor[
        [1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
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
        prefill=True,
    )
    norm_tokens = pl.tensor.dim(hidden, 0)
    norm_tiles = (norm_tokens + 7) // 8
    norm_workers = pl.max(1, pl.min(norm_tiles, pl.system.available_aiv_count()))
    for norm_worker in pl.spmd(norm_workers, name_hint="rms_norm"):
        for norm_block in pl.range(norm_worker, norm_tiles, norm_workers):
            rms_norm_tiled(hidden, attn_norm_weight, attn_input, norm_block, norm_tokens, output_fp32=False, compensated=False)
    prefill_c1a_full_attention(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens, index_cache, index_cache_scale, index_block_table, compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, pool_block_ids, pool_block_counts, output_window, output_arrived, attn_out, group_base, tp_rank, num_tokens, attention_epoch, scatter_output=False,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(
        attn_out, x_hc, post_mix, residual_mix, output,
        deps=hc_post_deps, prefill=True,
    )
    return output


@pl.jit
def l2_prefill_c1a_full(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]
    ],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ]
    ],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale: pl.InOut[
        pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8
        ]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.ND],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.ND],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.INT8, pl.ND],
    index_wq_b_scale: pl.Tensor[
        [1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.ND],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.Out[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
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
    """Run one TP rank of the delayed-mix full C1A path."""
    x_hc.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    prefill_c1a_full(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, compressed_cache, compressed_cache_scale, request_ids, compressed_lens, index_cache, index_cache_scale, index_block_table, compressed_rope_cos, compressed_rope_sin, compressor_wkv, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices, pool_block_ids, pool_block_counts, output_window, output_arrived, hidden, attn_input, attn_out, output, next_pre_mix, 0, tp_rank, num_tokens, 1,
    )
    retire_prefill_signals(output, output_arrived, 0, tp_rank)
    return output


@pl.jit.host
def l3_prefill_c1a_full(
    x_hc: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[TP_SIZE, 1, Q_LORA], pl.FP32],
    q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[TP_SIZE, 1, LOCAL_H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[TP_SIZE, 1, HEAD_DIM], pl.FP32],
    kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8]
    ],
    window_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]
    ],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[
            [TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ]
    ],
    request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale: pl.InOut[
        pl.Tensor[
            [TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.UINT8
        ]
    ],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[TP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[TP_SIZE, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.INT8],
    index_wq_b_scale: pl.Tensor[
        [TP_SIZE, 1, INDEX_H * INDEX_DIM], pl.FP32
    ],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.Out[
        pl.Tensor[[TP_SIZE, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]
    ],
    pool_block_counts: pl.Out[
        pl.Tensor[[TP_SIZE, T_DYN], pl.INT32]
    ],
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
        wq_a_scale_r: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[
            [1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND
        ] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND] = wkv_scale[rank]
        index_wq_b_scale_r: pl.Tensor[
            [1, INDEX_H * INDEX_DIM], pl.FP32, pl.ND
        ] = index_wq_b_scale[rank]
        l2_prefill_c1a_full(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank], rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank], index_cache[rank], index_cache_scale[rank], index_block_table[rank], compressed_rope_cos[rank], compressed_rope_sin[rank], compressor_wkv[rank], compressor_norm_weight[rank], compressed_slots[rank], index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r, index_weights_proj[rank], topk_indices[rank], pool_block_ids[rank], pool_block_counts[rank], output_window, output_arrived, next_pre_mix[rank], hidden[rank], attn_input[rank], attn_out[rank], output[rank], rank, num_tokens, device=rank,
        )
