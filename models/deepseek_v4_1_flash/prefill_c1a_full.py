# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""C1A full prefill."""


import functools

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from config import (
    FLASH as M, RANKS_DYN, LOCAL_T_DYN, T_DYN, ORI_BLOCKS_DYN, B_DYN, TABLE_DYN, INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP, INDEX_POOL_BLOCKS_DYN, CP_SIZE, PREFILL_MAX_TOKENS,
    WINDOW_CACHE_BYTES, CMP_BLOCKS_DYN, COMPRESSED_CACHE_BYTES,
)
from compressor import c1a_compressor_project, c1a_compressor_publish
from hc import (
    make_hc_inputs, golden_hc_head_norm, golden_hc_mixes, golden_hc_post, hc_head_norm, hc_project,
    hc_sinkhorn, zero_bf16_padding,
)
from o_proj import o_proj_hc_post, retire_prefill_signals
from prefill_cp_allgather import prefill_cp_input_allgather
from prefill_indexer import (
    index_pool_block_capacity, golden_prefill_indexer, golden_index_key, golden_select_candidate_block_pool,
    compare_compressed_topk,
    prefill_indexer, publish_c1a_index_key,
)
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj, window_publish
from quant import (
    pack_bf16_nz, pack_fp8_nz, pack_mx_b_scale, pack_quantized_cache, unpack_quantized_cache,
    quantize_mxfp4_cache, dequantize_mxfp4_cache, packed_cache_compare,
)
from rope_tables import precompute_rope_tables
from sparse_attention import gather_sparse_attention
from utils import prefill_batch_layout, prefill_fixture_lengths, prefill_window_pages
from utils import prefill_head_tail_rows, paged_slots


# model config
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
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


# tiling
Q_TILE = 1024
WORKER_TILE = 64
POOL_TILE = M.candidate_topk_blocks


@pl.jit.inline(auto_scope=False)
def prefill_publish_decoder(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_compressor_kv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    w_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    w_index_k: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    w_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x, 0)
    compressed_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    projection_ready = pl.system.task_dummy(deps=[])
    compressed_projection, projection_done = c1a_compressor_project(
        x, w_compressor_kv, compressed_projection, num_tokens, projection_ready, m_tile=80, n_tile=128,
    )
    compressed_latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    compressed_ready = c1a_compressor_publish(
        compressed_projection, w_compressor_norm, compressed_latent, compressed_rope_cos, compressed_rope_sin,
        compressed_write_slots, compressed_cache,
        num_tokens, projection_ready, projection_done,
    )
    index_ready = publish_c1a_index_key(
        compressed_latent, w_index_k, w_index_norm, compressed_rope_cos, compressed_rope_sin,
        compressed_write_slots, index_cache, index_cache_scale, num_tokens,
    )
    ready = pl.system.task_dummy(deps=[compressed_ready, index_ready])
    return ready


@pl.jit.inline(auto_scope=False)
def prefill_c1a_full(
    # HC input and weights.
    hc_input: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[D], pl.BF16],
    # Q projection weights.
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wq_norm: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    # KV projection weights and attention sink.
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    # Output projection weights.
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    # Compressor weights.
    w_compressor_kv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    w_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Index weights.
    w_index_k: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    w_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16],
    # Index query and score weights.
    w_index_q: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    w_index_q_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    w_index_score: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    q_window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    q_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    q_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Scalar[pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    # Compressed cache publication.
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    # Outputs.
    kv_hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    hc_output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    q_topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    q_pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    q_pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    publish_compressed: pl.constexpr = True,
):
    q_capacity = pl.tensor.dim(hc_input, 0)
    post_mix = pl.create_tensor([q_capacity, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([q_capacity, HC_MULT, HC_MULT], dtype=pl.FP32)
    normalized_ready = hc_head_norm(hc_input, hc_pre_mix, w_attn_norm, kv_hidden, worker_limit=64)
    inv_rms, partials = hc_project(hc_input, hc_attn_fn, rms_worker_limit=64)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, hc_next_pre_mix, post_mix, residual_mix)
    zero_bf16_padding(kv_hidden, kv_token_count)
    if kv_token_count > 0:
        kv_capacity = pl.tensor.dim(kv_hidden, 0)
        kv_projection = pl.create_tensor([kv_capacity, HEAD_DIM], dtype=pl.BF16)
        kv_input_payload, kv_input_scales, kv_input_payload_padded_rows = quantize_projection_input(
            kv_hidden, kv_token_count, row_tile=256,
        )
        kv_projection = kv_proj(
            kv_input_payload, kv_input_scales,
            wkv, wkv_scale,
            kv_projection, kv_token_count, worker_limit=8, row_tile=256,
        )
        kv_ready = window_publish(
            kv_projection, wkv_norm, kv_rope_cos, kv_rope_sin,
            kv_write_slots, window_cache,
            kv_token_count, normalized_ready,
        )
        compressed_ready = normalized_ready
        if publish_compressed:
            compressed_ready = prefill_publish_decoder(
                kv_hidden, w_compressor_kv, w_compressor_norm, w_index_k, w_index_norm,
                compressed_rope_cos, compressed_rope_sin, compressed_write_slots,
                compressed_cache, index_cache, index_cache_scale, kv_token_count,
            )
        chunk_hidden = pl.create_tensor([Q_TILE, D], dtype=pl.BF16)
        chunk_rope_cos = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_rope_sin = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_window_indices = pl.create_tensor([Q_TILE, 128], dtype=pl.INT32)
        pool_capacity = pl.tensor.dim(q_pool_block_ids, 1)
        chunk_request_ids = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        chunk_compressed_lens = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        chunk_topk_indices = pl.create_tensor([Q_TILE, INDEX_TOPK], dtype=pl.INT32)
        chunk_pool_block_ids = pl.create_tensor([Q_TILE, pool_capacity], dtype=pl.INT32)
        chunk_pool_block_counts = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        q_latent = pl.create_tensor([Q_TILE, Q_LORA], dtype=pl.BF16)
        q = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)

        chunk_done = normalized_ready
        for start in pl.range(0, kv_token_count, Q_TILE):
            active = pl.min(Q_TILE, kv_token_count - start)
            # The prior chunk must finish reading every reusable scratch buffer.
            with pl.spmd(WORKER_TILE, name_hint="prefill_c1a_full_stage", deps=[chunk_done]) as stage_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    source_row = start + row
                    chunk_hidden[row:row + 1, :] = kv_hidden[source_row:source_row + 1, :]
                    chunk_rope_cos[row:row + 1, :] = kv_rope_cos[source_row:source_row + 1, :]
                    chunk_rope_sin[row:row + 1, :] = kv_rope_sin[source_row:source_row + 1, :]
                    chunk_window_indices[row:row + 1, :] = q_window_indices[source_row:source_row + 1, :]
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage_query_metadata", deps=[chunk_done]):
                for row in pl.range(active):
                    request_ids_value = pl.read(q_request_ids, [start + row])
                    pl.write(chunk_request_ids, [row], request_ids_value)
                    compressed_lens_value = pl.read(q_compressed_lens, [start + row])
                    pl.write(chunk_compressed_lens, [row], compressed_lens_value)
            q_input_payload, q_input_scales, q_input_payload_padded_rows = quantize_projection_input(
                chunk_hidden, active, row_tile=256,
            )
            q_proj_rope(
                q_input_payload, q_input_scales,
                wq_a, wq_a_scale, wq_norm,
                wq_b, wq_b_scale,
                chunk_rope_cos, chunk_rope_sin,
                q_latent, q,
                active, qa_worker_limit=24, qb_worker_limit=24,
                padded_rows=q_input_payload_padded_rows,
                qa_row_tile=256, qb_row_tile=128,
            )
            attended = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
            index_inputs_ready = pl.system.task_dummy(deps=[stage_tid, compressed_ready])
            index_ready = prefill_indexer(
                chunk_hidden, q_latent, chunk_request_ids, chunk_compressed_lens,
                index_cache, index_cache_scale, index_block_table,
                chunk_rope_cos, chunk_rope_sin, w_index_q, w_index_q_scale, w_index_score,
                chunk_pool_block_ids, chunk_pool_block_counts, chunk_topk_indices, active, index_inputs_ready,
                pool=False, c2a=False,
            )
            attention_ready = pl.system.task_dummy(deps=[stage_tid, kv_ready, index_ready])
            gather_sparse_attention(
                q, chunk_window_indices, window_cache, chunk_topk_indices, compressed_cache,
                attn_sink, attended, active, attention_ready, compressed_keys=INDEX_TOPK,
            )
            _chunk_output, projection_done = o_proj_hc_post(
                attended, wo_a, wo_a_scale, wo_b, wo_b_scale, chunk_rope_cos, chunk_rope_sin,
                hc_input[start:start + active, :, :], post_mix[start:start + active, :],
                residual_mix[start:start + active, :, :], hc_output[start:start + active, :, :], active,
                oa_row_tile=128, ob_row_tile=128,
            )

            with pl.spmd(WORKER_TILE, name_hint="prefill_c1a_full_collect", deps=[projection_done]) as collect_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    q_topk_indices[start + row:start + row + 1, :] = chunk_topk_indices[row:row + 1, :]
                    pool_row = pl.slice(chunk_pool_block_ids, [1, POOL_TILE], [row, 0], valid_shape=[1, pool_capacity])
                    q_pool_block_ids[start + row:start + row + 1, :] = pool_row
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="collect_pool_counts") as counts_tid:
                for row in pl.range(active):
                    count = pl.read(chunk_pool_block_counts, [row])
                    pl.write(q_pool_block_counts, [start + row], count)
            chunk_done = pl.system.task_dummy(deps=[collect_tid, counts_tid])
    if kv_token_count < q_capacity:
        with pl.spmd(32, name_hint="prefill_c1a_full_zero_padding"):
            worker = pl.tile.get_block_idx()
            for row in pl.range(kv_token_count + worker, q_capacity, 32):
                zero_output = pl.full([1, HC_MULT * D], dtype=pl.FP32, value=0.0)
                hc_output[row:row + 1, :, :] = pl.reshape(zero_output, [1, HC_MULT, D])
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                hc_next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
                q_topk_indices[row:row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
                pool_capacity = pl.tensor.dim(q_pool_block_ids, 1)
                empty_pool = pl.full([1, POOL_TILE], dtype=pl.INT32, value=-1)
                q_pool_block_ids[row:row + 1, :] = pl.set_validshape(empty_pool, 1, pool_capacity)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="clear_pool_counts"):
            for row in pl.range(kv_token_count, q_capacity):
                pl.write(q_pool_block_counts, [row], pl.cast(0, pl.INT32))
    return hc_output, hc_next_pre_mix


@pl.jit
def l2_prefill_c1a_full(
    # HC input and weights.
    hc_input: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[D], pl.BF16],
    # Q projection weights.
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wq_norm: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    # KV projection weights and attention sink.
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    # Output projection weights.
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    # Compressor weights.
    w_compressor_kv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    w_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Index weights.
    w_index_k: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    w_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16],
    # Index query and score weights.
    w_index_q: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    w_index_q_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    w_index_score: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    q_window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    q_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    q_compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed cache publication.
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    # Compressed and index caches.
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    q_topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    q_pool_block_ids: pl.Out[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    q_pool_block_counts: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
):
    hc_input.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    q_pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    kv_tokenactive = pl.read(kv_token_count, [0])
    prefill_c1a_full(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, w_attn_norm,
        wq_a, wq_a_scale, wq_norm, wq_b, wq_b_scale,
        wkv, wkv_scale, wkv_norm, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        w_compressor_kv, w_compressor_norm,
        w_index_k, w_index_norm,
        w_index_q, w_index_q_scale, w_index_score,
        q_window_indices,
        q_request_ids, q_compressed_lens,
        kv_rope_cos, kv_rope_sin, kv_write_slots, kv_tokenactive,
        window_cache,
        compressed_rope_cos, compressed_rope_sin, compressed_write_slots,
        compressed_cache,
        index_cache, index_cache_scale,
        index_block_table,
        kv_hidden,
        hc_output, hc_next_pre_mix,
        q_topk_indices,
        q_pool_block_ids, q_pool_block_counts,
        publish_compressed=True,
    )
    return window_cache, compressed_cache, index_cache, index_cache_scale, kv_hidden, hc_output, hc_next_pre_mix, q_topk_indices, q_pool_block_ids, q_pool_block_counts


@pl.jit.host
def l3_prefill_c1a_full(
    # HC input and weights.
    hc_input: pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    # Q projection weights.
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    wq_norm: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    # KV projection weights and attention sink.
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    wkv_norm: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    # Output projection weights.
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    # Compressor weights.
    w_compressor_kv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.BF16],
    w_compressor_norm: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    # Index weights.
    w_index_k: pl.Tensor[[RANKS_DYN, HEAD_DIM, INDEX_DIM], pl.BF16],
    w_index_norm: pl.Tensor[[RANKS_DYN, INDEX_DIM], pl.BF16],
    # Index query and score weights.
    w_index_q: pl.Tensor[[RANKS_DYN, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    w_index_q_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    w_index_score: pl.Tensor[[RANKS_DYN, D, INDEX_H], pl.BF16],
    q_window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    q_request_ids: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    q_compressed_lens: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed cache publication.
    compressed_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    # Compressed and index caches.
    compressed_cache: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[RANKS_DYN, B_DYN, TABLE_DYN], pl.INT32],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT], pl.FP32]],
    q_topk_indices: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32]],
    q_pool_block_ids: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    q_pool_block_counts: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32]],
):
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    q_pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
    for rank in pl.range(pld.world_size()):
        rank_wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        rank_wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        rank_wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        rank_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        rank_w_compressor_kv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = w_compressor_kv[rank]
        rank_w_index_k: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = w_index_k[rank]
        rank_w_index_q: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ] = w_index_q[rank]
        rank_w_index_q_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = w_index_q_scale[rank]
        rank_w_index_score: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = w_index_score[rank]
        l2_prefill_c1a_full(
            hc_input[rank], hc_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], w_attn_norm[rank],
            wq_a[rank], rank_wq_a_scale, wq_norm[rank], wq_b[rank], rank_wq_b_scale,
            wkv[rank], rank_wkv_scale, wkv_norm[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], rank_wo_b_scale,
            rank_w_compressor_kv, w_compressor_norm[rank],
            rank_w_index_k, w_index_norm[rank],
            rank_w_index_q, rank_w_index_q_scale, rank_w_index_score,
            q_window_indices[rank],
            q_request_ids[rank], q_compressed_lens[rank],
            kv_rope_cos[rank], kv_rope_sin[rank], kv_write_slots[rank], kv_token_count[rank],
            window_cache[rank],
            compressed_rope_cos[rank], compressed_rope_sin[rank], compressed_write_slots[rank],
            compressed_cache[rank],
            index_cache[rank], index_cache_scale[rank],
            index_block_table[rank],
            kv_hidden[rank],
            hc_output[rank], hc_next_pre_mix[rank],
            q_topk_indices[rank],
            q_pool_block_ids[rank], q_pool_block_counts[rank],
            device=rank,
        )

@pl.jit.inline(auto_scope=False)
def prefill_c1a_full_cp(
    # HC input and weights.
    hc_input: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[D], pl.BF16],
    # Q projection weights.
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wq_norm: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    # KV projection weights and attention sink.
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    # Output projection weights.
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    # Compressor weights.
    w_compressor_kv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    w_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Index weights.
    w_index_k: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    w_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16],
    # Index query and score weights.
    w_index_q: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    w_index_q_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    w_index_score: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    # Local Q metadata.
    q_rope_cos: pl.Tensor[[LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_window_indices: pl.Tensor[[LOCAL_T_DYN, 128], pl.INT32],
    q_request_ids: pl.Tensor[[LOCAL_T_DYN], pl.INT32],
    q_compressed_lens: pl.Tensor[[LOCAL_T_DYN], pl.INT32],
    q_token_count: pl.Scalar[pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Scalar[pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    # Compressed cache publication.
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    # CP communication and row restoration.
    cp_hidden_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    cp_hidden_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    kv_restore_indices: pl.Tensor[[T_DYN], pl.INT32],
    cp_group_base: pl.Scalar[pl.INT32],
    cp_rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    # Outputs.
    kv_hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    hc_output: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    hc_next_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    q_topk_indices: pl.Tensor[[LOCAL_T_DYN, INDEX_TOPK], pl.INT32],
    q_pool_block_ids: pl.Tensor[[LOCAL_T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    q_pool_block_counts: pl.Tensor[[LOCAL_T_DYN], pl.INT32],
    publish_compressed: pl.constexpr = True,
):
    q_capacity = pl.tensor.dim(hc_input, 0)
    q_hidden = pl.create_tensor([q_capacity, D], dtype=pl.BF16)
    post_mix = pl.create_tensor([q_capacity, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([q_capacity, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_head_norm(hc_input, hc_pre_mix, w_attn_norm, q_hidden, worker_limit=64)
    inv_rms, partials = hc_project(hc_input, hc_attn_fn, rms_worker_limit=64)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, hc_next_pre_mix, post_mix, residual_mix)
    zero_bf16_padding(q_hidden, q_token_count)
    kv_hidden, gathered = prefill_cp_input_allgather(
        q_hidden, cp_hidden_window, cp_hidden_arrived, kv_restore_indices, kv_hidden,
        cp_group_base, cp_rank, attention_epoch, group_size=CP_SIZE,
    )
    publication = pl.array.create(1, pl.TASK_ID)
    publication[0] = gathered
    if kv_token_count > 0:
        kv_capacity = pl.tensor.dim(kv_hidden, 0)
        kv_projection = pl.create_tensor([kv_capacity, HEAD_DIM], dtype=pl.BF16)
        # Gate every projection tile before scratch storage is reused by a new epoch.
        with pl.spmd(WORKER_TILE, name_hint="prefill_c1a_full_cp_begin", deps=[gathered]):
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, kv_token_count, WORKER_TILE):
                kv_projection[row : row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        kv_input_payload, kv_input_scales, kv_input_payload_padded_rows = quantize_projection_input(
            kv_hidden, kv_token_count, row_tile=256,
        )
        kv_projection = kv_proj(
            kv_input_payload, kv_input_scales,
            wkv, wkv_scale,
            kv_projection, kv_token_count, worker_limit=8, row_tile=256,
        )
        kv_ready = window_publish(
            kv_projection, wkv_norm, kv_rope_cos, kv_rope_sin,
            kv_write_slots, window_cache,
            kv_token_count, gathered,
        )
        compressed_ready = gathered
        if publish_compressed:
            compressed_ready = prefill_publish_decoder(
                kv_hidden, w_compressor_kv, w_compressor_norm, w_index_k, w_index_norm,
                compressed_rope_cos, compressed_rope_sin, compressed_write_slots,
                compressed_cache, index_cache, index_cache_scale, kv_token_count,
            )
        publication[0] = pl.system.task_dummy(deps=[kv_ready, compressed_ready])
        chunk_hidden = pl.create_tensor([Q_TILE, D], dtype=pl.BF16)
        chunk_rope_cos = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_rope_sin = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_window_indices = pl.create_tensor([Q_TILE, 128], dtype=pl.INT32)
        pool_capacity = pl.tensor.dim(q_pool_block_ids, 1)
        chunk_request_ids = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        chunk_compressed_lens = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        chunk_topk_indices = pl.create_tensor([Q_TILE, INDEX_TOPK], dtype=pl.INT32)
        chunk_pool_block_ids = pl.create_tensor([Q_TILE, pool_capacity], dtype=pl.INT32)
        chunk_pool_block_counts = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        q_latent = pl.create_tensor([Q_TILE, Q_LORA], dtype=pl.BF16)
        q = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)

        chunk_done = gathered
        for start in pl.range(0, q_token_count, Q_TILE):
            active = pl.min(Q_TILE, q_token_count - start)
            # The prior chunk must finish reading every reusable scratch buffer.
            with pl.spmd(WORKER_TILE, name_hint="prefill_c1a_full_cp_stage", deps=[chunk_done]) as stage_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    source_row = start + row
                    chunk_hidden[row:row + 1, :] = q_hidden[source_row:source_row + 1, :]
                    chunk_rope_cos[row:row + 1, :] = q_rope_cos[source_row:source_row + 1, :]
                    chunk_rope_sin[row:row + 1, :] = q_rope_sin[source_row:source_row + 1, :]
                    chunk_window_indices[row:row + 1, :] = q_window_indices[source_row:source_row + 1, :]
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage_query_metadata", deps=[chunk_done]):
                for row in pl.range(active):
                    request_ids_value = pl.read(q_request_ids, [start + row])
                    pl.write(chunk_request_ids, [row], request_ids_value)
                    compressed_lens_value = pl.read(q_compressed_lens, [start + row])
                    pl.write(chunk_compressed_lens, [row], compressed_lens_value)
            q_input_payload, q_input_scales, q_input_payload_padded_rows = quantize_projection_input(
                chunk_hidden, active, row_tile=256,
            )
            q_proj_rope(
                q_input_payload, q_input_scales,
                wq_a, wq_a_scale, wq_norm,
                wq_b, wq_b_scale,
                chunk_rope_cos, chunk_rope_sin,
                q_latent, q,
                active, qa_worker_limit=24, qb_worker_limit=24,
                padded_rows=q_input_payload_padded_rows,
                qa_row_tile=256, qb_row_tile=128,
            )
            attended = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
            index_inputs_ready = pl.system.task_dummy(deps=[stage_tid, compressed_ready])
            index_ready = prefill_indexer(
                chunk_hidden, q_latent, chunk_request_ids, chunk_compressed_lens,
                index_cache, index_cache_scale, index_block_table,
                chunk_rope_cos, chunk_rope_sin, w_index_q, w_index_q_scale, w_index_score,
                chunk_pool_block_ids, chunk_pool_block_counts, chunk_topk_indices, active, index_inputs_ready,
                pool=False, c2a=False,
            )
            attention_ready = pl.system.task_dummy(deps=[stage_tid, kv_ready, index_ready])
            gather_sparse_attention(
                q, chunk_window_indices, window_cache, chunk_topk_indices, compressed_cache,
                attn_sink, attended, active, attention_ready, compressed_keys=INDEX_TOPK,
            )
            _chunk_output, projection_done = o_proj_hc_post(
                attended, wo_a, wo_a_scale, wo_b, wo_b_scale, chunk_rope_cos, chunk_rope_sin,
                hc_input[start:start + active, :, :], post_mix[start:start + active, :],
                residual_mix[start:start + active, :, :], hc_output[start:start + active, :, :], active,
                oa_row_tile=128, ob_row_tile=128,
            )

            with pl.spmd(WORKER_TILE, name_hint="prefill_c1a_full_cp_collect", deps=[projection_done]) as collect_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    q_topk_indices[start + row:start + row + 1, :] = chunk_topk_indices[row:row + 1, :]
                    pool_row = pl.slice(chunk_pool_block_ids, [1, POOL_TILE], [row, 0], valid_shape=[1, pool_capacity])
                    q_pool_block_ids[start + row:start + row + 1, :] = pool_row
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="collect_pool_counts") as counts_tid:
                for row in pl.range(active):
                    count = pl.read(chunk_pool_block_counts, [row])
                    pl.write(q_pool_block_counts, [start + row], count)
            chunk_done = pl.system.task_dummy(deps=[collect_tid, counts_tid])
    if q_token_count < q_capacity:
        with pl.spmd(32, name_hint="prefill_c1a_full_cp_zero_padding", deps=[publication[0]]):
            worker = pl.tile.get_block_idx()
            for row in pl.range(q_token_count + worker, q_capacity, 32):
                zero_output = pl.full([1, HC_MULT * D], dtype=pl.FP32, value=0.0)
                hc_output[row:row + 1, :, :] = pl.reshape(zero_output, [1, HC_MULT, D])
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                hc_next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
                q_topk_indices[row:row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
                pool_capacity = pl.tensor.dim(q_pool_block_ids, 1)
                empty_pool = pl.full([1, POOL_TILE], dtype=pl.INT32, value=-1)
                q_pool_block_ids[row:row + 1, :] = pl.set_validshape(empty_pool, 1, pool_capacity)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="clear_pool_counts"):
            for row in pl.range(q_token_count, q_capacity):
                pl.write(q_pool_block_counts, [row], pl.cast(0, pl.INT32))
    return hc_output, hc_next_pre_mix


@pl.jit
def l2_prefill_c1a_full_cp(
    # HC input and weights.
    hc_input: pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[D], pl.BF16],
    # Q projection weights.
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wq_norm: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    # KV projection weights and attention sink.
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    # Output projection weights.
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    # Compressor weights.
    w_compressor_kv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    w_compressor_norm: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Index weights.
    w_index_k: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    w_index_norm: pl.Tensor[[INDEX_DIM], pl.BF16],
    # Index query and score weights.
    w_index_q: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    w_index_q_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    w_index_score: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    # Local Q metadata.
    q_rope_cos: pl.Tensor[[LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_window_indices: pl.Tensor[[LOCAL_T_DYN, 128], pl.INT32],
    q_request_ids: pl.Tensor[[LOCAL_T_DYN], pl.INT32],
    q_compressed_lens: pl.Tensor[[LOCAL_T_DYN], pl.INT32],
    q_token_count: pl.Tensor[[1], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed cache publication.
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    # Compressed and index caches.
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    # CP communication and row restoration.
    cp_hidden_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    cp_hidden_arrived: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    kv_restore_indices: pl.Tensor[[T_DYN], pl.INT32],
    global_rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[LOCAL_T_DYN, HC_MULT], pl.FP32]],
    q_topk_indices: pl.Out[pl.Tensor[[LOCAL_T_DYN, INDEX_TOPK], pl.INT32]],
    q_pool_block_ids: pl.Out[pl.Tensor[[LOCAL_T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    q_pool_block_counts: pl.Out[pl.Tensor[[LOCAL_T_DYN], pl.INT32]],
):
    hc_input.bind_dynamic(0, LOCAL_T_DYN)
    kv_rope_cos.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    q_pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    q_tokenactive = pl.read(q_token_count, [0])
    kv_tokenactive = pl.read(kv_token_count, [0])
    prefill_c1a_full_cp(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, w_attn_norm,
        wq_a, wq_a_scale, wq_norm, wq_b, wq_b_scale,
        wkv, wkv_scale, wkv_norm, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        w_compressor_kv, w_compressor_norm,
        w_index_k, w_index_norm,
        w_index_q, w_index_q_scale, w_index_score,
        q_rope_cos, q_rope_sin,
        q_window_indices,
        q_request_ids, q_compressed_lens, q_tokenactive,
        kv_rope_cos, kv_rope_sin, kv_write_slots, kv_tokenactive,
        window_cache,
        compressed_rope_cos, compressed_rope_sin, compressed_write_slots,
        compressed_cache,
        index_cache, index_cache_scale,
        index_block_table,
        cp_hidden_window, cp_hidden_arrived, kv_restore_indices, global_rank // CP_SIZE * CP_SIZE, global_rank % CP_SIZE, attention_epoch,
        kv_hidden,
        hc_output, hc_next_pre_mix,
        q_topk_indices,
        q_pool_block_ids, q_pool_block_counts,
        publish_compressed=True,
    )
    retire_prefill_signals(hc_output, cp_hidden_arrived, global_rank // CP_SIZE * CP_SIZE, global_rank % CP_SIZE, group_size=CP_SIZE)
    return window_cache, compressed_cache, index_cache, index_cache_scale, kv_hidden, hc_output, hc_next_pre_mix, q_topk_indices, q_pool_block_ids, q_pool_block_counts


@pl.jit.host
def l3_prefill_c1a_full_cp(
    # HC input and weights.
    hc_input: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT, D], pl.FP32],
    hc_pre_mix: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    w_attn_norm: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    # Q projection weights.
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    wq_norm: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    # KV projection weights and attention sink.
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    wkv_norm: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    # Output projection weights.
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    # Compressor weights.
    w_compressor_kv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.BF16],
    w_compressor_norm: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    # Index weights.
    w_index_k: pl.Tensor[[RANKS_DYN, HEAD_DIM, INDEX_DIM], pl.BF16],
    w_index_norm: pl.Tensor[[RANKS_DYN, INDEX_DIM], pl.BF16],
    # Index query and score weights.
    w_index_q: pl.Tensor[[RANKS_DYN, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    w_index_q_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    w_index_score: pl.Tensor[[RANKS_DYN, D, INDEX_H], pl.BF16],
    # Local Q metadata.
    q_rope_cos: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_window_indices: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, 128], pl.INT32],
    q_request_ids: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN], pl.INT32],
    q_compressed_lens: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN], pl.INT32],
    q_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed cache publication.
    compressed_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    # Compressed and index caches.
    compressed_cache: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[RANKS_DYN, B_DYN, TABLE_DYN], pl.INT32],
    kv_restore_indices: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT], pl.FP32]],
    q_topk_indices: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, INDEX_TOPK], pl.INT32]],
    q_pool_block_ids: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    q_pool_block_counts: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN], pl.INT32]],
):
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, LOCAL_T_DYN)
    kv_rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    q_pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
    cp_hidden_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    cp_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        cp_hidden_window = pld.window(cp_hidden_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        cp_hidden_arrived = pld.window(cp_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        rank_wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        rank_wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        rank_wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        rank_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        rank_w_compressor_kv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = w_compressor_kv[rank]
        rank_w_index_k: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = w_index_k[rank]
        rank_w_index_q: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ] = w_index_q[rank]
        rank_w_index_q_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = w_index_q_scale[rank]
        rank_w_index_score: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = w_index_score[rank]
        l2_prefill_c1a_full_cp(
            hc_input[rank], hc_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], w_attn_norm[rank],
            wq_a[rank], rank_wq_a_scale, wq_norm[rank], wq_b[rank], rank_wq_b_scale,
            wkv[rank], rank_wkv_scale, wkv_norm[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], rank_wo_b_scale,
            rank_w_compressor_kv, w_compressor_norm[rank],
            rank_w_index_k, w_index_norm[rank],
            rank_w_index_q, rank_w_index_q_scale, rank_w_index_score,
            q_rope_cos[rank], q_rope_sin[rank],
            q_window_indices[rank],
            q_request_ids[rank], q_compressed_lens[rank], q_token_count[rank],
            kv_rope_cos[rank], kv_rope_sin[rank], kv_write_slots[rank], kv_token_count[rank],
            window_cache[rank],
            compressed_rope_cos[rank], compressed_rope_sin[rank], compressed_write_slots[rank],
            compressed_cache[rank],
            index_cache[rank], index_cache_scale[rank],
            index_block_table[rank],
            cp_hidden_window, cp_hidden_arrived, kv_restore_indices[rank], rank, attention_epoch,
            kv_hidden[rank],
            hc_output[rank], hc_next_pre_mix[rank],
            q_topk_indices[rank],
            q_pool_block_ids[rank], q_pool_block_counts[rank],
            device=rank,
        )


def golden_c1a_prefill(tensors, *, mode="full"):
    """Evaluate complete requests and scatter reference outputs to CP ranks."""
    from compressor import golden_c1a_compressor
    from o_proj import golden_o_proj
    from qkv_proj_rope import golden_qkv_proj_rope, golden_rope_interleave
    from quant import quantize_mxfp8_cache
    from sparse_attention import golden_gather_sparse_attention
    from quant import _publish_quantized_cache

    if mode not in ('full', 'reindex', 'reuse'):
        raise ValueError(f"unsupported C1A mode: {mode}")
    ranks, tokens = tensors["kv_hidden"].shape[:2]
    for base in range(0, ranks, CP_SIZE):
        active = int(tensors["kv_token_count"][base, 0])
        rows = prefill_head_tail_rows([active], CP_SIZE)
        hc_input = torch.zeros(tokens, M.hc_mult, M.hidden_size)
        hc_pre_mix = torch.zeros(tokens, M.hc_mult)
        local_fields = {}
        for name in ("q_window_indices", "q_request_ids", "q_compressed_lens", "q_compressed_indices"):
            if name in tensors:
                local_fields[name] = tensors[name].new_zeros((tokens, *tensors[name].shape[2:]))
        if mode == "reindex":
            for name in ("q_pool_block_ids", "q_pool_block_counts"):
                local_fields[name] = tensors[name].new_zeros((tokens, *tensors[name].shape[2:]))
        for cp_rank, indices in enumerate(rows):
            rank, count = base + cp_rank, indices.numel()
            hc_input[indices.long()] = tensors["hc_input"][rank, :count]
            hc_pre_mix[indices.long()] = tensors["hc_pre_mix"][rank, :count]
            for name, value in local_fields.items():
                value[indices.long()] = tensors[name][rank, :count]
        pre_mix, post_mix, residual_mix = golden_hc_mixes(
            hc_input, tensors["hc_attn_fn"][base], tensors["hc_attn_scale"][base], tensors["hc_attn_base"][base],
        )
        hidden = golden_hc_head_norm(hc_input, hc_pre_mix, tensors["w_attn_norm"][base])
        output = torch.zeros(tokens, M.hidden_size, dtype=torch.bfloat16)
        result = {}
        if active:
            a = {name: value[base] for name, value in tensors.items() if isinstance(value, torch.Tensor) and value.ndim > 0}
            x = hidden[:active]
            cos, sin = a["kv_rope_cos"][:active], a["kv_rope_sin"][:active]
            query, window_kv, query_latent = golden_qkv_proj_rope(
                x, a["wq_a"], a["wq_a_scale"], a["wq_norm"], a["wq_b"], a["wq_b_scale"],
                a["wkv"], a["wkv_scale"], a["wkv_norm"], cos, sin, sequential_query=True,
            )
            window, window_scale = unpack_quantized_cache(a["window_cache"], False)
            window, window_scale = _publish_quantized_cache(
                window, window_scale, window_kv, a["kv_write_slots"][:active], quantize_mxfp8_cache,
            )
            compressed, compressed_scale = unpack_quantized_cache(a["compressed_cache"], True)
            index, index_scale = a.get("index_cache"), a.get("index_cache_scale")
            topk = local_fields.get("q_compressed_indices")
            if topk is not None:
                topk = topk[:active]
            if mode == "full":
                latent = golden_c1a_compressor(x, a["w_compressor_kv"], a["w_compressor_norm"])
                slots = a["compressed_write_slots"][:active]
                compressed_cos = a["compressed_rope_cos"][:active]
                compressed_sin = a["compressed_rope_sin"][:active]
                rope_width = compressed_cos.shape[-1] * 2
                tail = golden_rope_interleave(latent[..., -rope_width:], compressed_cos, compressed_sin)
                rotated = torch.cat((latent[..., :-rope_width], tail), dim=-1)
                compressed, compressed_scale = _publish_quantized_cache(
                    compressed, compressed_scale, rotated, slots,
                    lambda value: quantize_mxfp4_cache(value, 16, "bf16", round_to_even=True),
                )
                keys = golden_index_key(latent, a["w_index_k"], a["w_index_norm"], compressed_cos, compressed_sin)
                index, index_scale = _publish_quantized_cache(
                    index, index_scale, keys, slots,
                    lambda value: quantize_mxfp4_cache(value, 32, "e8m0", round_to_even=True),
                )
            if mode != "reuse":
                keys = dequantize_mxfp4_cache(index, index_scale, 32, "e8m0").to(query.dtype)
                pool_args = {}
                if mode == "reindex":
                    pool_args = {
                        "pool_block_ids": local_fields["q_pool_block_ids"][:active],
                        "pool_block_counts": local_fields["q_pool_block_counts"][:active],
                    }
                scores, topk = golden_prefill_indexer(
                    x, query_latent, local_fields["q_request_ids"][:active], keys, a["index_block_table"],
                    local_fields["q_compressed_lens"][:active], a["w_index_q"], a["w_index_q_scale"], a["w_index_score"],
                    cos, sin, c2a=False, topk=INDEX_TOPK, **pool_args,
                )
                if mode == "full":
                    pool_width = a["index_block_table"].shape[-1] * 128
                    scores = torch.nn.functional.pad(scores, (0, pool_width - scores.shape[-1]), value=-torch.inf)
                    pool_ids, pool_counts = golden_select_candidate_block_pool(
                        scores, local_fields["q_compressed_lens"][:active],
                    )
                    result.update(pool_block_ids=pool_ids, pool_block_counts=pool_counts)
            attended = golden_gather_sparse_attention(
                query, local_fields["q_window_indices"][:active], pack_quantized_cache(window, window_scale),
                topk, pack_quantized_cache(compressed, compressed_scale), a["attn_sink"], compressed_keys=INDEX_TOPK,
            )
            projected = golden_o_proj(
                attended, cos, sin, a["wo_a"], a["wo_b"], a["wo_b_scale"],
                wo_a_scale=a["wo_a_scale"], output_dtype=torch.float32, sequential_projection=True,
            )
            output[:active] = projected.bfloat16()
            result.update(
                window_cache=window, window_cache_scale=window_scale,
                compressed_cache=compressed, compressed_cache_scale=compressed_scale,
                index_cache=index, index_cache_scale=index_scale, topk_indices=topk,
            )
        hc_output = golden_hc_post(output, hc_input, post_mix, residual_mix)
        for cp_rank, indices in enumerate(rows):
            rank = base + cp_rank
            tensors["kv_hidden"][rank].copy_(hidden)
            for name, reference in (("hc_output", hc_output), ("hc_next_pre_mix", pre_mix)):
                tensors[name][rank].zero_()
                tensors[name][rank, :indices.numel()] = reference[indices.long()]
            for name in ("window_cache", "compressed_cache", "index_cache", "index_cache_scale", "state_cache"):
                if name not in tensors or name not in result or result[name] is None:
                    continue
                reference = result[name]
                if name in ("window_cache", "compressed_cache"):
                    reference = pack_quantized_cache(reference, result[name + "_scale"])
                tensors[name][rank].view(torch.uint8).copy_(reference.view(torch.uint8))
            if mode != "reuse":
                tensors["q_topk_indices"][rank].fill_(-1)
                if active:
                    tensors["q_topk_indices"][rank, :indices.numel()] = result["topk_indices"][indices.long()]
            if mode == "full":
                tensors["q_pool_block_ids"][rank].fill_(-1)
                tensors["q_pool_block_counts"][rank].zero_()
                if active:
                    tensors["q_pool_block_ids"][rank, :indices.numel()] = result["pool_block_ids"][indices.long()]
                    tensors["q_pool_block_counts"][rank, :indices.numel()] = result["pool_block_counts"][indices.long()]


def build_c1a_specs(args, entry, *, mode="full"):
    """Build C1A operands and lazy DP/CP fixtures for all supported modes."""
    import math

    from golden import ScalarSpec, TensorSpec
    from quant import quantize_grouped_mxfp8_weight, quantize_mxfp8_cache, quantize_mxfp8_upward_reference
    from utils import packed_query_lengths, window_metadata

    if mode not in ("full", "reindex", "reuse"):
        raise ValueError(f"unsupported C1A mode: {mode}")
    query_lengths, starts = prefill_batch_layout(args)
    counts = [sum(group) for group in query_lengths]
    rows = [prefill_head_tail_rows([active], CP_SIZE) for active in counts]
    tokens = args.tokens
    ranks = CP_SIZE * args.dp
    local_tokens = tokens if CP_SIZE == 1 else max(1, max(r.numel() for group in rows for r in group))
    if local_tokens * CP_SIZE > PREFILL_MAX_TOKENS:
        raise ValueError("padded CP gather exceeds PREFILL_MAX_TOKENS")
    fixture_lengths = [prefill_fixture_lengths(group, tokens) for group in query_lengths]
    window_pages = max(prefill_window_pages(ns, ps) for ns, ps in zip(fixture_lengths, starts))
    max_end = max(p + n for ps, ns in zip(starts, fixture_lengths) for p, n in zip(ps, ns))
    table_pages = max(1, (max_end + 127) // 128)
    requests = max(map(len, query_lengths))
    batch = requests
    compressed_pages = requests * table_pages + 1
    pool_capacity = index_pool_block_capacity(table_pages * 128)

    values = {}

    def initialize(name):
        if not values:
            shards = []
            for dp_rank, active in enumerate(counts):
                lengths = query_lengths[dp_rank]
                seed = args.seed + dp_rank * CP_SIZE
                case = args.case
                query_lens = prefill_fixture_lengths(lengths, tokens)
                start_pos = starts[dp_rank]
                generator = torch.Generator().manual_seed(seed + 3000)
                input_requests = len(query_lens)
                if input_requests < 1 or tokens < 1:
                    raise ValueError('requests and tokens must be positive')
                gen = torch.Generator().manual_seed(seed)

                def rand(*shape):
                    return torch.randn(*shape, generator=gen)

                def weight(k, n):
                    payload, scales = quantize_mxfp8_upward_reference(rand(n, k) / math.sqrt(k))
                    return (payload.T.contiguous(), pack_mx_b_scale(scales.T.contiguous()).view(torch.float8_e8m0fnu))
                qa, qas = weight(D, Q_LORA)
                qb, qbs = weight(Q_LORA, LOCAL_H * HEAD_DIM)
                kv, kvs = weight(D, HEAD_DIM)
                ob, obs = weight(LOCAL_O_WIDTH, D)
                qa = pack_fp8_nz(qa)
                qb = pack_fp8_nz(qb)
                kv = pack_fp8_nz(kv)
                ob = pack_fp8_nz(ob)
                inputs = dict(
                    wq_a=qa, wq_a_scale=qas, q_norm_weight=(rand(Q_LORA) * 0.1 + 1).to(torch.bfloat16),
                    wq_b=qb, wq_b_scale=qbs,
                    wkv=kv, wkv_scale=kvs, kv_norm_weight=(rand(HEAD_DIM) * 0.1 + 1).to(torch.bfloat16),
                    attn_sink=rand(LOCAL_H) * 2,
                )
                wo_a = (rand(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(torch.bfloat16)
                inputs["wo_a"], inputs["wo_a_scale"] = quantize_grouped_mxfp8_weight(wo_a)
                inputs.update(wo_b=ob, wo_b_scale=obs)
                gen = torch.Generator().manual_seed(seed + 1000)
                input_lengths = torch.tensor(packed_query_lengths(tokens, input_requests, query_lens), dtype=torch.int64)
                prefixes = torch.tensor(start_pos, dtype=torch.int64)
                ids = torch.repeat_interleave(torch.arange(input_requests), input_lengths)
                positions = torch.cat([torch.arange(int(p), int(p + n)) for p, n in zip(prefixes, input_lengths)])
                logical_capacity = int((prefixes + input_lengths + 129).max()) // 128 + 1
                table = torch.full((input_requests, logical_capacity), -1, dtype=torch.int32)
                mappings = []
                for request, (prefix, length) in enumerate(zip(prefixes.tolist(), input_lengths.tolist())):
                    start = max(0, prefix - 127) // 128
                    end = (prefix + length + 128) // 128
                    mappings.extend(((request, block) for block in range(start, end + 1)))
                input_pages = torch.randperm(len(mappings) + 1, generator=gen)[:len(mappings)]
                for (request, block), page in zip(mappings, input_pages):
                    table[request, block] = page
                slots, indices, _ = window_metadata(positions, ids, table)
                if slots.unique().numel() != tokens:
                    raise ValueError("prefill fixture slots must be unique")
                for row, (request, position) in enumerate(zip(ids.tolist(), positions.tolist())):
                    history = list(range(max(0, position - 127), position + 1))
                    expected = [int(table[request, p // 128]) * 128 + p % 128 for p in history]
                    if indices[row, :len(expected)].tolist() != expected or not (indices[row, len(expected):] == -1).all():
                        raise ValueError("prefill fixture indices do not match the paged window")
                cache_values = torch.randn(len(mappings) + 1, 128, 1, HEAD_DIM, generator=gen).bfloat16()
                cache, scales = quantize_mxfp8_cache(cache_values)
                inputs.update(window_slots=slots, window_indices=indices, window_cache=pack_quantized_cache(cache, scales))
                inputs.update(make_hc_inputs(tokens, seed + 2000, case))
                for source, target in (
                    ("x_hc", "hc_input"), ("pre_mix", "hc_pre_mix"), ("attn_norm_weight", "w_attn_norm"),
                    ("q_norm_weight", "wq_norm"), ("kv_norm_weight", "wkv_norm"),
                    ("window_slots", "kv_write_slots"), ("window_indices", "q_window_indices"),
                ):
                    inputs[target] = inputs.pop(source)
                positions = torch.cat([torch.arange(p, p + n, dtype=torch.int32) for p, n in zip(start_pos, query_lens)])
                request_ids = torch.arange(input_requests, dtype=torch.int32)
                request_ids = torch.repeat_interleave(request_ids, torch.tensor(query_lens))
                compressed_lengths = positions + 1
                input_table_pages = max(1, (max(p + n for p, n in zip(start_pos, query_lens)) + 127) // 128)
                pages = input_requests * input_table_pages + 1
                table = torch.randperm(pages, generator=generator)[:input_requests * input_table_pages].int()
                table = table.reshape(input_requests, input_table_pages)
                cos, sin = precompute_rope_tables(max(start_pos) + tokens + 1, compressed_attention=True)
                inputs["kv_rope_cos"], inputs["kv_rope_sin"] = cos[positions.long()], sin[positions.long()]
                inputs["compressed_rope_cos"] = cos[positions.long()]
                inputs["compressed_rope_sin"] = sin[positions.long()]
                inputs["compressed_write_slots"] = paged_slots(positions, request_ids, table, 128, 1, True)
                inputs["q_request_ids"], inputs["q_compressed_lens"] = request_ids, compressed_lengths
                inputs["kv_request_ids"], inputs["kv_position_ids"] = request_ids, positions
                inputs["kv_query_start"] = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32)
                inputs["index_block_table"] = table
                compressed = torch.randn(pages, 128, 1, M.head_dim, generator=generator).bfloat16()
                payload, scales = quantize_mxfp4_cache(compressed, 16, "bf16")
                inputs["compressed_cache"] = pack_quantized_cache(payload, scales)
                index = torch.randn(pages, 128, 1, M.index_head_dim, generator=generator).bfloat16()
                inputs["index_cache"], scales = quantize_mxfp4_cache(index, 32, "e8m0")
                inputs["index_cache_scale"] = scales.view(torch.float8_e8m0fnu)
                for weight_name, inn, out in (
                    ("w_compressor_kv", M.hidden_size, M.head_dim), ("w_compressor_gate", M.hidden_size, M.head_dim),
                    ("w_index_k", M.head_dim, M.index_head_dim), ("w_index_score", M.hidden_size, M.index_n_heads),
                ):
                    weight = (torch.randn(inn, out, generator=generator) / inn**0.5).bfloat16()
                    inputs[weight_name] = pack_bf16_nz(weight)
                inputs["w_compressor_norm"] = torch.ones(M.head_dim, dtype=torch.bfloat16)
                inputs["w_index_norm"] = torch.ones(M.index_head_dim, dtype=torch.bfloat16)
                index_q = torch.randn(M.q_lora_rank, M.index_n_heads * M.index_head_dim, generator=generator)
                inputs["w_index_q"] = pack_fp8_nz((index_q / M.q_lora_rank**0.5).to(torch.float8_e4m3fn))
                scale_codes = torch.full((M.q_lora_rank // 32, M.index_n_heads * M.index_head_dim), 127, dtype=torch.uint8)
                inputs["w_index_q_scale"] = pack_mx_b_scale(scale_codes).view(torch.float8_e8m0fnu)
                input_pool_capacity = index_pool_block_capacity(input_table_pages * 128)
                pool_ids = torch.full((tokens, input_pool_capacity), -1, dtype=torch.int32)
                pool_counts = torch.zeros(tokens, dtype=torch.int32)
                selected = torch.full((tokens, M.index_topk), -1, dtype=torch.int32)
                for row, length in enumerate(compressed_lengths.tolist()):
                    blocks = torch.arange((length + 7) // 8, dtype=torch.int32)
                    if mode == "reindex":
                        blocks = blocks[::2]
                    blocks = blocks[:input_pool_capacity]
                    pool_ids[row, :blocks.numel()] = blocks
                    pool_counts[row] = blocks.numel()
                    logical = torch.arange(length, dtype=torch.int64)[:M.index_topk]
                    physical_pages = table[int(request_ids[row]), logical // 128]
                    selected[row, :logical.numel()] = physical_pages * 128 + (logical % 128).int()
                inputs["q_pool_block_ids"], inputs["q_pool_block_counts"] = pool_ids, pool_counts
                inputs["q_compressed_indices"] = selected

                shapes = {
                    "window_cache": (window_pages, *inputs["window_cache"].shape[1:]),
                    "compressed_cache": (compressed_pages, *inputs["compressed_cache"].shape[1:]),
                    "index_cache": (compressed_pages, *inputs["index_cache"].shape[1:]),
                    "index_cache_scale": (compressed_pages, *inputs["index_cache_scale"].shape[1:]),
                    "index_block_table": (batch, table_pages),
                    "q_pool_block_ids": (tokens, pool_capacity),
                }
                for key, shape in shapes.items():
                    value = inputs[key]
                    fill = -1 if key.endswith(("table", "ids")) else 0
                    dtype = value.dtype
                    if dtype == torch.float8_e8m0fnu:
                        value = value.view(torch.uint8)
                    padded = value.new_full(shape, fill)
                    padded[tuple(slice(0, n) for n in value.shape)] = value
                    inputs[key] = padded.view(dtype)
                ends = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
                inputs["kv_query_start"] = torch.full((batch + 1,), active, dtype=torch.int32)
                inputs["kv_query_start"][0] = 0
                inputs["kv_query_start"][1:len(lengths) + 1] = ends
                inputs["kv_token_count"] = torch.tensor([active], dtype=torch.int32)
                inputs["kv_write_slots"][active:] = -1
                inputs["compressed_write_slots"][active:] = -1
                rows = prefill_head_tail_rows([active], CP_SIZE)
                restore = torch.full((tokens,), -1, dtype=torch.int32)
                for rank, indices in enumerate(rows):
                    restore[indices.long()] = rank * local_tokens + torch.arange(indices.numel(), dtype=torch.int32)
                inputs["kv_restore_indices"] = restore
                inputs["q_rope_cos"], inputs["q_rope_sin"] = inputs["kv_rope_cos"], inputs["kv_rope_sin"]
                for indices in rows:
                    shard = dict(inputs)
                    shard["q_token_count"] = torch.tensor([indices.numel()], dtype=torch.int32)
                    for key, value in inputs.items():
                        if key not in ("hc_input", "hc_pre_mix") and not key.startswith("q_"):
                            continue
                        fill = 17 if key == "hc_input" else -1 if key.endswith(("indices", "ids")) else 0
                        if key in ("q_rope_cos", "q_rope_sin"):
                            fill = float("nan")
                        out = value.new_full((local_tokens, *value.shape[1:]), fill)
                        out[:indices.numel()] = value[indices.long()]
                        shard[key] = out
                    shards.append(shard)
            for key in entry.param_names:
                if key not in shards[0]:
                    continue
                dtype = shards[0][key].dtype
                pieces = [s[key] for s in shards]
                if dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                    pieces = [v.view(torch.uint8) for v in pieces]
                values[key] = torch.stack(pieces).view(dtype)
        return values[name].clone()

    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    specs = [
        TensorSpec("hc_input", [ranks, local_tokens, HC_MULT, D], torch.float32),
        TensorSpec("hc_pre_mix", [ranks, local_tokens, HC_MULT], torch.float32),
        TensorSpec("hc_attn_fn", [ranks, MIX_HC, HC_DIM], torch.float32),
        TensorSpec("hc_attn_scale", [ranks, 3], torch.float32),
        TensorSpec("hc_attn_base", [ranks, MIX_HC], torch.float32),
        TensorSpec("w_attn_norm", [ranks, D], bf),
        TensorSpec("wq_a", [ranks, D, Q_LORA], fp),
        TensorSpec("wq_a_scale", [ranks, D // 32, Q_LORA], mx),
        TensorSpec("wq_norm", [ranks, Q_LORA], bf),
        TensorSpec("wq_b", [ranks, Q_LORA, LOCAL_H * HEAD_DIM], fp),
        TensorSpec("wq_b_scale", [ranks, Q_LORA // 32, LOCAL_H * HEAD_DIM], mx),
        TensorSpec("wkv", [ranks, D, HEAD_DIM], fp),
        TensorSpec("wkv_scale", [ranks, D // 32, HEAD_DIM], mx),
        TensorSpec("wkv_norm", [ranks, HEAD_DIM], bf),
        TensorSpec("attn_sink", [ranks, LOCAL_H], torch.float32),
        TensorSpec("wo_a", [ranks, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], fp),
        TensorSpec("wo_a_scale", [ranks, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], mx),
        TensorSpec("wo_b", [ranks, LOCAL_O_WIDTH, D], fp),
        TensorSpec("wo_b_scale", [ranks, LOCAL_O_WIDTH // 32, D], mx),
        TensorSpec("w_compressor_kv", [ranks, D, HEAD_DIM], bf),
        TensorSpec("w_compressor_norm", [ranks, HEAD_DIM], bf),
        TensorSpec("w_index_k", [ranks, HEAD_DIM, INDEX_DIM], bf),
        TensorSpec("w_index_norm", [ranks, INDEX_DIM], bf),
        TensorSpec("w_index_q", [ranks, Q_LORA, INDEX_H * INDEX_DIM], fp),
        TensorSpec("w_index_q_scale", [ranks, Q_LORA // 32, INDEX_H * INDEX_DIM], mx),
        TensorSpec("w_index_score", [ranks, D, INDEX_H], bf),
        TensorSpec("q_rope_cos", [ranks, local_tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("q_rope_sin", [ranks, local_tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("q_window_indices", [ranks, local_tokens, 128], torch.int32),
        TensorSpec("q_request_ids", [ranks, local_tokens], torch.int32),
        TensorSpec("q_compressed_lens", [ranks, local_tokens], torch.int32),
        TensorSpec("q_compressed_indices", [ranks, local_tokens, INDEX_TOPK], torch.int32),
        TensorSpec("q_token_count", [ranks, 1], torch.int32),
        TensorSpec("kv_rope_cos", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("kv_rope_sin", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("kv_write_slots", [ranks, tokens], torch.int64),
        TensorSpec("kv_token_count", [ranks, 1], torch.int32),
        TensorSpec("window_cache", [ranks, window_pages, 128, 1, WINDOW_CACHE_BYTES], torch.uint8),
        TensorSpec("compressed_rope_cos", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("compressed_rope_sin", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("compressed_write_slots", [ranks, tokens], torch.int64),
        TensorSpec("compressed_cache", [ranks, compressed_pages, 128, 1, COMPRESSED_CACHE_BYTES], torch.uint8),
        TensorSpec("index_cache", [ranks, compressed_pages, 128, 1, INDEX_DIM // 2], torch.uint8),
        TensorSpec("index_cache_scale", [ranks, compressed_pages, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], mx),
        TensorSpec("index_block_table", [ranks, requests, table_pages], torch.int32),
        TensorSpec("kv_restore_indices", [ranks, tokens], torch.int32),
        TensorSpec("kv_hidden", [ranks, tokens, D], bf),
        TensorSpec("hc_output", [ranks, local_tokens, HC_MULT, D], torch.float32),
        TensorSpec("hc_next_pre_mix", [ranks, local_tokens, HC_MULT], torch.float32),
        TensorSpec("q_topk_indices", [ranks, local_tokens, INDEX_TOPK], torch.int32),
        TensorSpec("q_pool_block_ids", [ranks, local_tokens, pool_capacity], torch.int32),
        TensorSpec("q_pool_block_counts", [ranks, local_tokens], torch.int32),
    ]
    outputs = {"kv_hidden", "hc_output", "hc_next_pre_mix", "q_topk_indices"}
    if mode == "full":
        outputs.update(("q_pool_block_ids", "q_pool_block_counts"))
    for spec in specs:
        spec.resident = "stacked"
        if spec.name not in outputs:
            spec.init_value = lambda name=spec.name: initialize(name)
    specs.append(ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True, benchmark_step=None))
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in entry.param_names]


if __name__ == "__main__":
    import argparse

    from config import SUPPORTED_CP_SIZES
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run
    from pypto.ir import DistributedConfig

    from utils import parse_prefill_groups

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C1A full")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=lambda value: [int(part) for part in value.split(",")], help="comma-separated device IDs")
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=SUPPORTED_CP_SIZES)
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2, 4, 8])
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--dp-tokens", type=lambda value: [int(part) for part in value.split(",")], help="active token counts per DP group")
    parser.add_argument("--requests", type=int, default=1, help="request count per DP when query-lens is omitted")
    parser.add_argument("--query-lens", type=parse_prefill_groups, help="request lengths, e.g. '17,15;32'")
    parser.add_argument("--start-pos", type=parse_prefill_groups, help="request prefix lengths, e.g. '127,0;256'")
    parser.add_argument("--case", default="normal", choices=["normal", "prefix"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args()
    devices = list(range(CP_SIZE * args.dp)) if args.device is None else args.device
    if len(devices) != CP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("device IDs must be distinct nonnegative integers, with count CP * DP")
    if not 1 <= args.tokens <= PREFILL_MAX_TOKENS:
        parser.error(f"tokens must be in [1, {PREFILL_MAX_TOKENS}]")
    if args.dp_tokens is not None:
        if len(args.dp_tokens) != args.dp or any(n < 0 or n > args.tokens for n in args.dp_tokens):
            parser.error("dp-tokens must contain one count per DP group, each in [0, tokens]")
    torch.set_num_threads(8)
    entry = l3_prefill_c1a_full if CP_SIZE == 1 else l3_prefill_c1a_full_cp
    result = run(
        fn=entry,
        specs=build_c1a_specs(args, entry, mode="full"),
        golden_fn=functools.partial(golden_c1a_prefill, mode="full"),
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            ring_heap=PREFILL_ATTN_RING_HEAP,
        ),
        compare_fn={
            "kv_hidden": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "hc_next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "hc_output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "window_cache": packed_cache_compare(
                mapped_pool_ratio_allclose(
                    "kv_write_slots", pool_name="window_cache",
                    mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.01,
                ),
                mapped_pool_ratio_allclose(
                    "kv_write_slots", pool_name="window_cache scale",
                    mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.0,
                ),
                fp4=False,
            ),
            "compressed_cache": packed_cache_compare(
                mapped_pool_ratio_allclose(
                    "compressed_write_slots", pool_name="compressed_cache",
                    mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.01,
                ),
                mapped_pool_ratio_allclose(
                    "compressed_write_slots", pool_name="compressed_cache scale",
                    mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.01,
                ),
                fp4=True,
            ),
            "index_cache": mapped_pool_ratio_allclose(
                "compressed_write_slots", pool_name="index_cache",
                mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.01,
            ),
            "index_cache_scale": mapped_pool_ratio_allclose(
                "compressed_write_slots", pool_name="index_cache_scale",
                mapping_shape=(len(devices), args.tokens), block_size=128, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "q_pool_block_ids": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "q_pool_block_counts": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "q_topk_indices": functools.partial(
                compare_compressed_topk, ratio=1, mode="full", cutoff_atol=5e-5, cutoff_rtol=0.03,
            ),
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
