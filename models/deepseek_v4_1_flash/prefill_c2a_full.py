# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""C2A full prefill."""


import functools

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from config import (
    FLASH as M, RANKS_DYN, LOCAL_T_DYN, T_DYN, ORI_BLOCKS_DYN, B_DYN, TABLE_DYN, INDEX_BLOCKS_DYN,
    INDEX_CACHE_GROUP, STATE_BLOCKS_DYN, STATE_CAPACITY, Q_START_DYN, CP_SIZE, PREFILL_MAX_TOKENS,
    WINDOW_CACHE_BYTES, CMP_BLOCKS_DYN, COMPRESSED_CACHE_BYTES,
)
from compressor import c2a_compressor_project, c2a_compressor_publish, STATE_WIDTH
from hc import (
    golden_hc_head_norm, golden_hc_mixes, golden_hc_post, hc_head_norm, hc_project, hc_sinkhorn, zero_bf16_padding,
)
from o_proj import o_proj_hc_post, retire_prefill_signals
from prefill_c1a_full import (
    compressed_prefill_shapes, make_compressed_prefill_initializer, compare_compressed_topk,
)
from prefill_cp_allgather import prefill_cp_input_allgather
from prefill_indexer import (
    golden_prefill_indexer, golden_index_key, golden_select_candidate_block_pool, prefill_indexer,
    publish_c2a_index_key,
)
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj, window_publish
from quant import pack_quantized_cache, unpack_quantized_cache
from sparse_attention import golden_compressed_attention, golden_prefill_c1a_attention, attend_packed
from utils import prefill_batch_layout
from utils import prefill_head_tail_rows


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


@pl.jit.inline(auto_scope=False)
def prefill_c2a_full(
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
    w_compressor_gate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
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
    kv_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
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
        compressor_kv = pl.create_tensor([kv_capacity, HEAD_DIM], dtype=pl.FP32)
        compressor_score = pl.create_tensor([kv_capacity, HEAD_DIM], dtype=pl.FP32)
        c2a_compressor_project(
            kv_hidden, w_compressor_kv, w_compressor_gate, compressor_kv, compressor_score,
            kv_token_count, n_tile=128,
        )
        compressed_latent = pl.create_tensor([kv_capacity, HEAD_DIM], dtype=pl.BF16)
        compressed_published = c2a_compressor_publish(
            compressor_kv, compressor_score, kv_query_start, kv_position_ids, kv_request_ids,
            state_block_table, state_cache, w_compressor_norm, compressed_latent, kv_token_count, normalized_ready,
            compressed_rope_cos, compressed_rope_sin, compressed_write_slots, compressed_cache,
        )
        compressed_ready = publish_c2a_index_key(
            compressed_latent, w_index_k, w_index_norm, compressed_rope_cos, compressed_rope_sin,
            compressed_write_slots, index_cache, index_cache_scale, kv_token_count, compressed_published, normalized_ready,
        )
        chunk_hidden = pl.create_tensor([Q_TILE, D], dtype=pl.BF16)
        chunk_rope_cos = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_rope_sin = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_window_indices = pl.create_tensor([Q_TILE, 128], dtype=pl.INT32)
        chunk_request_ids = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        chunk_compressed_lens = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        chunk_topk_indices = pl.create_tensor([Q_TILE, INDEX_TOPK], dtype=pl.INT32)
        q_latent = pl.create_tensor([Q_TILE, Q_LORA], dtype=pl.BF16)
        q = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)

        chunk_done = normalized_ready
        for start in pl.range(0, kv_token_count, Q_TILE):
            active = pl.min(Q_TILE, kv_token_count - start)
            # The prior chunk must finish reading every reusable scratch buffer.
            with pl.spmd(WORKER_TILE, name_hint="prefill_c2a_full_stage", deps=[chunk_done]) as stage_tid:
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
                chunk_topk_indices, chunk_compressed_lens, chunk_topk_indices, active, index_inputs_ready,
                pool=False, c2a=True,
            )
            attention_ready = pl.system.task_dummy(deps=[stage_tid, kv_ready, index_ready])
            attend_packed(
                q, chunk_window_indices, window_cache, chunk_topk_indices, compressed_cache,
                attn_sink, attended, active, attention_ready,
            )
            _chunk_output, projection_done = o_proj_hc_post(
                attended, wo_a, wo_a_scale, wo_b, wo_b_scale, chunk_rope_cos, chunk_rope_sin,
                hc_input[start:start + active, :, :], post_mix[start:start + active, :],
                residual_mix[start:start + active, :, :], hc_output[start:start + active, :, :], active,
                oa_row_tile=128, ob_row_tile=128,
            )

            with pl.spmd(WORKER_TILE, name_hint="prefill_c2a_full_collect", deps=[projection_done]) as collect_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    q_topk_indices[start + row:start + row + 1, :] = chunk_topk_indices[row:row + 1, :]
            chunk_done = collect_tid
    if kv_token_count < q_capacity:
        with pl.spmd(32, name_hint="prefill_c2a_full_zero_padding"):
            worker = pl.tile.get_block_idx()
            for row in pl.range(kv_token_count + worker, q_capacity, 32):
                zero_output = pl.full([1, HC_MULT * D], dtype=pl.FP32, value=0.0)
                hc_output[row:row + 1, :, :] = pl.reshape(zero_output, [1, HC_MULT, D])
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                hc_next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
                q_topk_indices[row:row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
    return hc_output, hc_next_pre_mix


@pl.jit
def l2_prefill_c2a_full(
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
    w_compressor_gate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
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
    kv_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
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
):
    hc_input.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    kv_query_start.bind_dynamic(0, Q_START_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    kv_tokenactive = pl.read(kv_token_count, [0])
    prefill_c2a_full(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, w_attn_norm,
        wq_a, wq_a_scale, wq_norm, wq_b, wq_b_scale,
        wkv, wkv_scale, wkv_norm, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        w_compressor_kv, w_compressor_gate, w_compressor_norm,
        w_index_k, w_index_norm,
        w_index_q, w_index_q_scale, w_index_score,
        q_window_indices,
        q_request_ids, q_compressed_lens,
        kv_rope_cos, kv_rope_sin, kv_write_slots, kv_tokenactive,
        window_cache,
        compressed_rope_cos, compressed_rope_sin, compressed_write_slots,
        kv_request_ids, kv_position_ids, kv_query_start,
        state_block_table, state_cache,
        compressed_cache,
        index_cache, index_cache_scale,
        index_block_table,
        kv_hidden,
        hc_output, hc_next_pre_mix,
        q_topk_indices,
    )
    return window_cache, state_cache, compressed_cache, index_cache, index_cache_scale, kv_hidden, hc_output, hc_next_pre_mix, q_topk_indices


@pl.jit.host
def l3_prefill_c2a_full(
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
    w_compressor_gate: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.BF16],
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
    kv_request_ids: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    kv_position_ids: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    kv_query_start: pl.Tensor[[RANKS_DYN, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[RANKS_DYN, B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[RANKS_DYN, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
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
):
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    kv_query_start.bind_dynamic(1, Q_START_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    for rank in pl.range(pld.world_size()):
        rank_wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        rank_wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        rank_wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        rank_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        rank_w_compressor_kv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = w_compressor_kv[rank]
        rank_w_compressor_gate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = w_compressor_gate[rank]
        rank_w_index_k: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = w_index_k[rank]
        rank_w_index_q: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ] = w_index_q[rank]
        rank_w_index_q_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = w_index_q_scale[rank]
        rank_w_index_score: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = w_index_score[rank]
        l2_prefill_c2a_full(
            hc_input[rank], hc_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], w_attn_norm[rank],
            wq_a[rank], rank_wq_a_scale, wq_norm[rank], wq_b[rank], rank_wq_b_scale,
            wkv[rank], rank_wkv_scale, wkv_norm[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], rank_wo_b_scale,
            rank_w_compressor_kv, rank_w_compressor_gate, w_compressor_norm[rank],
            rank_w_index_k, w_index_norm[rank],
            rank_w_index_q, rank_w_index_q_scale, rank_w_index_score,
            q_window_indices[rank],
            q_request_ids[rank], q_compressed_lens[rank],
            kv_rope_cos[rank], kv_rope_sin[rank], kv_write_slots[rank], kv_token_count[rank],
            window_cache[rank],
            compressed_rope_cos[rank], compressed_rope_sin[rank], compressed_write_slots[rank],
            kv_request_ids[rank], kv_position_ids[rank], kv_query_start[rank],
            state_block_table[rank], state_cache[rank],
            compressed_cache[rank],
            index_cache[rank], index_cache_scale[rank],
            index_block_table[rank],
            kv_hidden[rank],
            hc_output[rank], hc_next_pre_mix[rank],
            q_topk_indices[rank],
            device=rank,
        )

@pl.jit.inline(auto_scope=False)
def prefill_c2a_full_cp(
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
    w_compressor_gate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
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
    kv_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
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
        with pl.spmd(WORKER_TILE, name_hint="prefill_c2a_full_cp_begin", deps=[gathered]):
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
        compressor_kv = pl.create_tensor([kv_capacity, HEAD_DIM], dtype=pl.FP32)
        compressor_score = pl.create_tensor([kv_capacity, HEAD_DIM], dtype=pl.FP32)
        c2a_compressor_project(
            kv_hidden, w_compressor_kv, w_compressor_gate, compressor_kv, compressor_score,
            kv_token_count, n_tile=128,
        )
        compressed_latent = pl.create_tensor([kv_capacity, HEAD_DIM], dtype=pl.BF16)
        compressed_published = c2a_compressor_publish(
            compressor_kv, compressor_score, kv_query_start, kv_position_ids, kv_request_ids,
            state_block_table, state_cache, w_compressor_norm, compressed_latent, kv_token_count, gathered,
            compressed_rope_cos, compressed_rope_sin, compressed_write_slots, compressed_cache,
        )
        compressed_ready = publish_c2a_index_key(
            compressed_latent, w_index_k, w_index_norm, compressed_rope_cos, compressed_rope_sin,
            compressed_write_slots, index_cache, index_cache_scale, kv_token_count, compressed_published, gathered,
        )
        publication[0] = pl.system.task_dummy(deps=[kv_ready, compressed_ready])
        chunk_hidden = pl.create_tensor([Q_TILE, D], dtype=pl.BF16)
        chunk_rope_cos = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_rope_sin = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_window_indices = pl.create_tensor([Q_TILE, 128], dtype=pl.INT32)
        chunk_request_ids = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        chunk_compressed_lens = pl.create_tensor([Q_TILE], dtype=pl.INT32)
        chunk_topk_indices = pl.create_tensor([Q_TILE, INDEX_TOPK], dtype=pl.INT32)
        q_latent = pl.create_tensor([Q_TILE, Q_LORA], dtype=pl.BF16)
        q = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)

        chunk_done = gathered
        for start in pl.range(0, q_token_count, Q_TILE):
            active = pl.min(Q_TILE, q_token_count - start)
            # The prior chunk must finish reading every reusable scratch buffer.
            with pl.spmd(WORKER_TILE, name_hint="prefill_c2a_full_cp_stage", deps=[chunk_done]) as stage_tid:
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
                chunk_topk_indices, chunk_compressed_lens, chunk_topk_indices, active, index_inputs_ready,
                pool=False, c2a=True,
            )
            attention_ready = pl.system.task_dummy(deps=[stage_tid, kv_ready, index_ready])
            attend_packed(
                q, chunk_window_indices, window_cache, chunk_topk_indices, compressed_cache,
                attn_sink, attended, active, attention_ready,
            )
            _chunk_output, projection_done = o_proj_hc_post(
                attended, wo_a, wo_a_scale, wo_b, wo_b_scale, chunk_rope_cos, chunk_rope_sin,
                hc_input[start:start + active, :, :], post_mix[start:start + active, :],
                residual_mix[start:start + active, :, :], hc_output[start:start + active, :, :], active,
                oa_row_tile=128, ob_row_tile=128,
            )

            with pl.spmd(WORKER_TILE, name_hint="prefill_c2a_full_cp_collect", deps=[projection_done]) as collect_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    q_topk_indices[start + row:start + row + 1, :] = chunk_topk_indices[row:row + 1, :]
            chunk_done = collect_tid
    if q_token_count < q_capacity:
        with pl.spmd(32, name_hint="prefill_c2a_full_cp_zero_padding", deps=[publication[0]]):
            worker = pl.tile.get_block_idx()
            for row in pl.range(q_token_count + worker, q_capacity, 32):
                zero_output = pl.full([1, HC_MULT * D], dtype=pl.FP32, value=0.0)
                hc_output[row:row + 1, :, :] = pl.reshape(zero_output, [1, HC_MULT, D])
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                hc_next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
                q_topk_indices[row:row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
    return hc_output, hc_next_pre_mix


@pl.jit
def l2_prefill_c2a_full_cp(
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
    w_compressor_gate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
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
    kv_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_query_start: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
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
):
    hc_input.bind_dynamic(0, LOCAL_T_DYN)
    kv_rope_cos.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    kv_query_start.bind_dynamic(0, Q_START_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    state_cache.bind_dynamic(0, STATE_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    q_tokenactive = pl.read(q_token_count, [0])
    kv_tokenactive = pl.read(kv_token_count, [0])
    prefill_c2a_full_cp(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, w_attn_norm,
        wq_a, wq_a_scale, wq_norm, wq_b, wq_b_scale,
        wkv, wkv_scale, wkv_norm, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        w_compressor_kv, w_compressor_gate, w_compressor_norm,
        w_index_k, w_index_norm,
        w_index_q, w_index_q_scale, w_index_score,
        q_rope_cos, q_rope_sin,
        q_window_indices,
        q_request_ids, q_compressed_lens, q_tokenactive,
        kv_rope_cos, kv_rope_sin, kv_write_slots, kv_tokenactive,
        window_cache,
        compressed_rope_cos, compressed_rope_sin, compressed_write_slots,
        kv_request_ids, kv_position_ids, kv_query_start,
        state_block_table, state_cache,
        compressed_cache,
        index_cache, index_cache_scale,
        index_block_table,
        cp_hidden_window, cp_hidden_arrived, kv_restore_indices, global_rank // CP_SIZE * CP_SIZE, global_rank % CP_SIZE, attention_epoch,
        kv_hidden,
        hc_output, hc_next_pre_mix,
        q_topk_indices,
    )
    retire_prefill_signals(hc_output, cp_hidden_arrived, global_rank // CP_SIZE * CP_SIZE, global_rank % CP_SIZE, group_size=CP_SIZE)
    return window_cache, state_cache, compressed_cache, index_cache, index_cache_scale, kv_hidden, hc_output, hc_next_pre_mix, q_topk_indices


@pl.jit.host
def l3_prefill_c2a_full_cp(
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
    w_compressor_gate: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.BF16],
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
    kv_request_ids: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    kv_position_ids: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    kv_query_start: pl.Tensor[[RANKS_DYN, Q_START_DYN], pl.INT32],
    state_block_table: pl.Tensor[[RANKS_DYN, B_DYN, 1], pl.INT32],
    state_cache: pl.InOut[pl.Tensor[[RANKS_DYN, STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
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
):
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, LOCAL_T_DYN)
    kv_rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    kv_query_start.bind_dynamic(1, Q_START_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    state_cache.bind_dynamic(1, STATE_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
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
        rank_w_compressor_gate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ] = w_compressor_gate[rank]
        rank_w_index_k: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ] = w_index_k[rank]
        rank_w_index_q: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ] = w_index_q[rank]
        rank_w_index_q_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = w_index_q_scale[rank]
        rank_w_index_score: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ] = w_index_score[rank]
        l2_prefill_c2a_full_cp(
            hc_input[rank], hc_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], w_attn_norm[rank],
            wq_a[rank], rank_wq_a_scale, wq_norm[rank], wq_b[rank], rank_wq_b_scale,
            wkv[rank], rank_wkv_scale, wkv_norm[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], rank_wo_b_scale,
            rank_w_compressor_kv, rank_w_compressor_gate, w_compressor_norm[rank],
            rank_w_index_k, w_index_norm[rank],
            rank_w_index_q, rank_w_index_q_scale, rank_w_index_score,
            q_rope_cos[rank], q_rope_sin[rank],
            q_window_indices[rank],
            q_request_ids[rank], q_compressed_lens[rank], q_token_count[rank],
            kv_rope_cos[rank], kv_rope_sin[rank], kv_write_slots[rank], kv_token_count[rank],
            window_cache[rank],
            compressed_rope_cos[rank], compressed_rope_sin[rank], compressed_write_slots[rank],
            kv_request_ids[rank], kv_position_ids[rank], kv_query_start[rank],
            state_block_table[rank], state_cache[rank],
            compressed_cache[rank],
            index_cache[rank], index_cache_scale[rank],
            index_block_table[rank],
            cp_hidden_window, cp_hidden_arrived, kv_restore_indices[rank], rank, attention_epoch,
            kv_hidden[rank],
            hc_output[rank], hc_next_pre_mix[rank],
            q_topk_indices[rank],
            device=rank,
        )

def build_c2a_full_specs(args, entry):
    from golden import ScalarSpec, TensorSpec

    shapes = compressed_prefill_shapes(args, ratio=2)
    ranks, tokens, local_tokens, pages, compressed_pages, table_pages, pool_capacity = shapes
    requests = max(map(len, prefill_batch_layout(args)[0]))
    initialize = make_compressed_prefill_initializer(args, entry, ratio=2, mode="full")
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
        TensorSpec("w_compressor_gate", [ranks, D, HEAD_DIM], bf),
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
        TensorSpec("q_token_count", [ranks, 1], torch.int32),
        TensorSpec("kv_rope_cos", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("kv_rope_sin", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("kv_write_slots", [ranks, tokens], torch.int64),
        TensorSpec("kv_token_count", [ranks, 1], torch.int32),
        TensorSpec("window_cache", [ranks, pages, 128, 1, WINDOW_CACHE_BYTES], torch.uint8),
        TensorSpec("compressed_rope_cos", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("compressed_rope_sin", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("compressed_write_slots", [ranks, tokens], torch.int64),
        TensorSpec("kv_request_ids", [ranks, tokens], torch.int32),
        TensorSpec("kv_position_ids", [ranks, tokens], torch.int32),
        TensorSpec("kv_query_start", [ranks, requests + 1], torch.int32),
        TensorSpec("state_block_table", [ranks, requests, 1], torch.int32),
        TensorSpec("state_cache", [ranks, requests + 1, STATE_CAPACITY, STATE_WIDTH], torch.float32),
        TensorSpec("compressed_cache", [ranks, compressed_pages, 128, 1, COMPRESSED_CACHE_BYTES], torch.uint8),
        TensorSpec("index_cache", [ranks, compressed_pages, 128, 1, INDEX_DIM // 2], torch.uint8),
        TensorSpec("index_cache_scale", [ranks, compressed_pages, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], mx),
        TensorSpec("index_block_table", [ranks, requests, table_pages], torch.int32),
        TensorSpec("kv_restore_indices", [ranks, tokens], torch.int32),
        TensorSpec("kv_hidden", [ranks, tokens, D], bf),
        TensorSpec("hc_output", [ranks, local_tokens, HC_MULT, D], torch.float32),
        TensorSpec("hc_next_pre_mix", [ranks, local_tokens, HC_MULT], torch.float32),
        TensorSpec("q_topk_indices", [ranks, local_tokens, INDEX_TOPK], torch.int32),
    ]
    for spec in specs:
        spec.resident = "stacked"
        if spec.name not in ("kv_hidden", "hc_output", "hc_next_pre_mix", "q_topk_indices"):
            spec.init_value = lambda name=spec.name: initialize(name)
    specs.append(ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True, benchmark_step=None))
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in entry.param_names]


def golden_c2a_prefill(tensors, *, mode):
    """Evaluate complete requests and scatter reference outputs to CP ranks."""
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
            fields = {name: tensors[name][base] for name in (
                "wq_a", "wq_a_scale", "wq_b", "wq_b_scale", "wkv", "wkv_scale", "attn_sink",
                "wo_a", "wo_a_scale", "wo_b", "wo_b_scale",
            )}
            fields.update(x=hidden[:active], q_norm_weight=tensors["wq_norm"][base], kv_norm_weight=tensors["wkv_norm"][base])
            for source, target in (
                ("kv_rope_cos", "rope_cos"), ("kv_rope_sin", "rope_sin"), ("kv_write_slots", "window_slots"),
                ("compressed_rope_cos", "compressed_rope_cos"), ("compressed_rope_sin", "compressed_rope_sin"),
                ("compressed_write_slots", "compressed_slots"), ("kv_position_ids", "position_ids"),
            ):
                fields[target] = tensors[source][base, :active] if source in tensors else None
            for source, target in (
                ("q_window_indices", "window_indices"), ("q_request_ids", "request_ids"),
                ("q_compressed_lens", "compressed_lens"), ("q_compressed_indices", "compressed_indices"),
            ):
                fields[target] = local_fields[source][:active] if source in local_fields else None
            for source, target in (
                ("w_compressor_kv", "compressor_wkv"), ("w_compressor_gate", "compressor_wgate"),
                ("w_compressor_norm", "compressor_norm_weight"), ("w_index_k", "index_wk"),
                ("w_index_norm", "index_norm_weight"), ("w_index_q", "index_wq_b"),
                ("w_index_q_scale", "index_wq_b_scale"), ("w_index_score", "index_weights_proj"),
                ("state_cache", "state_cache"), ("state_block_table", "state_block_table"),
                ("kv_query_start", "query_start_loc"), ("index_cache", "index_cache"),
                ("index_cache_scale", "index_cache_scale"), ("index_block_table", "index_block_table"),
            ):
                fields[target] = tensors[source][base] if source in tensors else None
            for name, fp4 in (("window_cache", False), ("compressed_cache", True)):
                fields[name], fields[name + "_scale"] = unpack_quantized_cache(tensors[name][base], fp4)
            indexer = functools.partial(golden_prefill_indexer, c2a=True)
            result = golden_compressed_attention(
                **fields, mode=mode, ratio=2, candidate_mask=None,
                indexer_fn=indexer, index_key_fn=golden_index_key, pool_selector_fn=golden_select_candidate_block_pool,
                attention_fn=golden_prefill_c1a_attention, sequential_query=True,
                round_cache_to_even=False, output_dtype=torch.float32,
            )
            output[:active] = result["output"].bfloat16()
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


if __name__ == "__main__":
    import argparse

    from config import SUPPORTED_CP_SIZES
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run
    from utils import packed_cache_compare
    from pypto.ir import DistributedConfig

    from utils import parse_prefill_groups

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C2A full")
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
    entry = l3_prefill_c2a_full if CP_SIZE == 1 else l3_prefill_c2a_full_cp
    result = run(
        fn=entry,
        specs=build_c2a_full_specs(args, entry),
        golden_fn=functools.partial(golden_c2a_prefill, mode="full"),
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
            "state_cache": ratio_allclose(atol=1e-3, rtol=1e-3),
            "q_topk_indices": functools.partial(
                compare_compressed_topk, ratio=2, mode="full", cutoff_atol=5e-5, cutoff_rtol=0.03,
            ),
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
