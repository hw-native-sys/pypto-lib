# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C1A reuse prefill."""


import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    FLASH as M, RANKS_DYN, LOCAL_T_DYN, T_DYN, ORI_BLOCKS_DYN,
    CP_SIZE, PREFILL_MAX_TOKENS, WINDOW_CACHE_BYTES, CMP_BLOCKS_DYN, COMPRESSED_CACHE_BYTES,
)
from sparse_attention import attend_packed
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj, window_publish
from o_proj import o_proj_hc_post, retire_prefill_signals
from hc import hc_head_norm, hc_project, hc_sinkhorn, zero_bf16_padding
from prefill_cp_allgather import prefill_cp_input_allgather


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
INDEX_TOPK = M.index_topk


# tiling
Q_TILE = 1024
WORKER_TILE = 64


@pl.jit.inline(auto_scope=False)
def prefill_c1a_reuse(
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
    q_window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    q_compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Scalar[pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    # Outputs.
    kv_hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    hc_output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
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
        chunk_hidden = pl.create_tensor([Q_TILE, D], dtype=pl.BF16)
        chunk_rope_cos = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_rope_sin = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_window_indices = pl.create_tensor([Q_TILE, 128], dtype=pl.INT32)
        chunk_compressed_indices = pl.create_tensor([Q_TILE, INDEX_TOPK], dtype=pl.INT32)
        q_latent = pl.create_tensor([Q_TILE, Q_LORA], dtype=pl.BF16)
        q = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)

        chunk_done = normalized_ready
        for start in pl.range(0, kv_token_count, Q_TILE):
            active = pl.min(Q_TILE, kv_token_count - start)
            # The prior chunk must finish reading every reusable scratch buffer.
            with pl.spmd(WORKER_TILE, name_hint="prefill_c1a_reuse_stage", deps=[chunk_done]) as stage_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    source_row = start + row
                    chunk_hidden[row:row + 1, :] = kv_hidden[source_row:source_row + 1, :]
                    chunk_rope_cos[row:row + 1, :] = kv_rope_cos[source_row:source_row + 1, :]
                    chunk_rope_sin[row:row + 1, :] = kv_rope_sin[source_row:source_row + 1, :]
                    chunk_window_indices[row:row + 1, :] = q_window_indices[source_row:source_row + 1, :]
                    chunk_compressed_indices[row:row + 1, :] = q_compressed_indices[source_row:source_row + 1, :]
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
            attention_ready = pl.system.task_dummy(deps=[stage_tid, kv_ready])
            attend_packed(
                q, chunk_window_indices, window_cache, chunk_compressed_indices, compressed_cache,
                attn_sink, attended, active, attention_ready,
            )
            _chunk_output, projection_done = o_proj_hc_post(
                attended, wo_a, wo_a_scale, wo_b, wo_b_scale, chunk_rope_cos, chunk_rope_sin,
                hc_input[start:start + active, :, :], post_mix[start:start + active, :],
                residual_mix[start:start + active, :, :], hc_output[start:start + active, :, :], active,
                oa_row_tile=128, ob_row_tile=128,
            )

            chunk_done = projection_done
    if kv_token_count < q_capacity:
        with pl.spmd(32, name_hint="prefill_c1a_reuse_zero_padding"):
            worker = pl.tile.get_block_idx()
            for row in pl.range(kv_token_count + worker, q_capacity, 32):
                zero_output = pl.full([1, HC_MULT * D], dtype=pl.FP32, value=0.0)
                hc_output[row:row + 1, :, :] = pl.reshape(zero_output, [1, HC_MULT, D])
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                hc_next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
    return hc_output, hc_next_pre_mix


@pl.jit
def l2_prefill_c1a_reuse(
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
    q_window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    q_compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
):
    hc_input.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    kv_tokenactive = pl.read(kv_token_count, [0])
    prefill_c1a_reuse(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, w_attn_norm,
        wq_a, wq_a_scale, wq_norm, wq_b, wq_b_scale,
        wkv, wkv_scale, wkv_norm, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        q_window_indices,
        q_compressed_indices,
        kv_rope_cos, kv_rope_sin, kv_write_slots, kv_tokenactive,
        window_cache,
        compressed_cache,
        kv_hidden,
        hc_output, hc_next_pre_mix,
    )
    return window_cache, kv_hidden, hc_output, hc_next_pre_mix


@pl.jit.host
def l3_prefill_c1a_reuse(
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
    q_window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    q_compressed_indices: pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT], pl.FP32]],
):
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    for rank in pl.range(pld.world_size()):
        rank_wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        rank_wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        rank_wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        rank_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        l2_prefill_c1a_reuse(
            hc_input[rank], hc_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], w_attn_norm[rank],
            wq_a[rank], rank_wq_a_scale, wq_norm[rank], wq_b[rank], rank_wq_b_scale,
            wkv[rank], rank_wkv_scale, wkv_norm[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], rank_wo_b_scale,
            q_window_indices[rank],
            q_compressed_indices[rank],
            kv_rope_cos[rank], kv_rope_sin[rank], kv_write_slots[rank], kv_token_count[rank],
            window_cache[rank],
            compressed_cache[rank],
            kv_hidden[rank],
            hc_output[rank], hc_next_pre_mix[rank],
            device=rank,
        )

@pl.jit.inline(auto_scope=False)
def prefill_c1a_reuse_cp(
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
    # Local Q metadata.
    q_rope_cos: pl.Tensor[[LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_window_indices: pl.Tensor[[LOCAL_T_DYN, 128], pl.INT32],
    q_compressed_indices: pl.Tensor[[LOCAL_T_DYN, INDEX_TOPK], pl.INT32],
    q_token_count: pl.Scalar[pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Scalar[pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
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
        with pl.spmd(WORKER_TILE, name_hint="prefill_c1a_reuse_cp_begin", deps=[gathered]):
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
        publication[0] = kv_ready
        chunk_hidden = pl.create_tensor([Q_TILE, D], dtype=pl.BF16)
        chunk_rope_cos = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_rope_sin = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_window_indices = pl.create_tensor([Q_TILE, 128], dtype=pl.INT32)
        chunk_compressed_indices = pl.create_tensor([Q_TILE, INDEX_TOPK], dtype=pl.INT32)
        q_latent = pl.create_tensor([Q_TILE, Q_LORA], dtype=pl.BF16)
        q = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)

        chunk_done = gathered
        for start in pl.range(0, q_token_count, Q_TILE):
            active = pl.min(Q_TILE, q_token_count - start)
            # The prior chunk must finish reading every reusable scratch buffer.
            with pl.spmd(WORKER_TILE, name_hint="prefill_c1a_reuse_cp_stage", deps=[chunk_done]) as stage_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    source_row = start + row
                    chunk_hidden[row:row + 1, :] = q_hidden[source_row:source_row + 1, :]
                    chunk_rope_cos[row:row + 1, :] = q_rope_cos[source_row:source_row + 1, :]
                    chunk_rope_sin[row:row + 1, :] = q_rope_sin[source_row:source_row + 1, :]
                    chunk_window_indices[row:row + 1, :] = q_window_indices[source_row:source_row + 1, :]
                    chunk_compressed_indices[row:row + 1, :] = q_compressed_indices[source_row:source_row + 1, :]
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
            attention_ready = pl.system.task_dummy(deps=[stage_tid, kv_ready])
            attend_packed(
                q, chunk_window_indices, window_cache, chunk_compressed_indices, compressed_cache,
                attn_sink, attended, active, attention_ready,
            )
            _chunk_output, projection_done = o_proj_hc_post(
                attended, wo_a, wo_a_scale, wo_b, wo_b_scale, chunk_rope_cos, chunk_rope_sin,
                hc_input[start:start + active, :, :], post_mix[start:start + active, :],
                residual_mix[start:start + active, :, :], hc_output[start:start + active, :, :], active,
                oa_row_tile=128, ob_row_tile=128,
            )

            chunk_done = projection_done
    if q_token_count < q_capacity:
        with pl.spmd(32, name_hint="prefill_c1a_reuse_cp_zero_padding", deps=[publication[0]]):
            worker = pl.tile.get_block_idx()
            for row in pl.range(q_token_count + worker, q_capacity, 32):
                zero_output = pl.full([1, HC_MULT * D], dtype=pl.FP32, value=0.0)
                hc_output[row:row + 1, :, :] = pl.reshape(zero_output, [1, HC_MULT, D])
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                hc_next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
    return hc_output, hc_next_pre_mix


@pl.jit
def l2_prefill_c1a_reuse_cp(
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
    # Local Q metadata.
    q_rope_cos: pl.Tensor[[LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_window_indices: pl.Tensor[[LOCAL_T_DYN, 128], pl.INT32],
    q_compressed_indices: pl.Tensor[[LOCAL_T_DYN, INDEX_TOPK], pl.INT32],
    q_token_count: pl.Tensor[[1], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
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
):
    hc_input.bind_dynamic(0, LOCAL_T_DYN)
    kv_rope_cos.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    q_tokenactive = pl.read(q_token_count, [0])
    kv_tokenactive = pl.read(kv_token_count, [0])
    prefill_c1a_reuse_cp(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, w_attn_norm,
        wq_a, wq_a_scale, wq_norm, wq_b, wq_b_scale,
        wkv, wkv_scale, wkv_norm, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        q_rope_cos, q_rope_sin,
        q_window_indices,
        q_compressed_indices, q_tokenactive,
        kv_rope_cos, kv_rope_sin, kv_write_slots, kv_tokenactive,
        window_cache,
        compressed_cache,
        cp_hidden_window, cp_hidden_arrived, kv_restore_indices, global_rank // CP_SIZE * CP_SIZE, global_rank % CP_SIZE, attention_epoch,
        kv_hidden,
        hc_output, hc_next_pre_mix,
    )
    retire_prefill_signals(hc_output, cp_hidden_arrived, global_rank // CP_SIZE * CP_SIZE, global_rank % CP_SIZE, group_size=CP_SIZE)
    return window_cache, kv_hidden, hc_output, hc_next_pre_mix


@pl.jit.host
def l3_prefill_c1a_reuse_cp(
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
    # Local Q metadata.
    q_rope_cos: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_rope_sin: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, ROPE_DIM // 2], pl.FP32],
    q_window_indices: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, 128], pl.INT32],
    q_compressed_indices: pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, INDEX_TOPK], pl.INT32],
    q_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Compressed and index caches.
    compressed_cache: pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    kv_restore_indices: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT], pl.FP32]],
):
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, LOCAL_T_DYN)
    kv_rope_cos.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    cp_hidden_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    cp_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        cp_hidden_window = pld.window(cp_hidden_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        cp_hidden_arrived = pld.window(cp_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        rank_wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        rank_wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        rank_wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        rank_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        l2_prefill_c1a_reuse_cp(
            hc_input[rank], hc_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], w_attn_norm[rank],
            wq_a[rank], rank_wq_a_scale, wq_norm[rank], wq_b[rank], rank_wq_b_scale,
            wkv[rank], rank_wkv_scale, wkv_norm[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], rank_wo_b_scale,
            q_rope_cos[rank], q_rope_sin[rank],
            q_window_indices[rank],
            q_compressed_indices[rank], q_token_count[rank],
            kv_rope_cos[rank], kv_rope_sin[rank], kv_write_slots[rank], kv_token_count[rank],
            window_cache[rank],
            compressed_cache[rank],
            cp_hidden_window, cp_hidden_arrived, kv_restore_indices[rank], rank, attention_epoch,
            kv_hidden[rank],
            hc_output[rank], hc_next_pre_mix[rank],
            device=rank,
        )

def build_c1a_reuse_specs(args, entry):
    import torch
    from golden import ScalarSpec, TensorSpec
    from prefill_c1a_full import compressed_prefill_shapes, make_compressed_prefill_initializer

    shapes = compressed_prefill_shapes(args, ratio=1)
    ranks, tokens, local_tokens, pages, compressed_pages, table_pages, pool_capacity = shapes
    initialize = make_compressed_prefill_initializer(args, entry, ratio=1, mode="reuse")
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
        TensorSpec("q_rope_cos", [ranks, local_tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("q_rope_sin", [ranks, local_tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("q_window_indices", [ranks, local_tokens, 128], torch.int32),
        TensorSpec("q_compressed_indices", [ranks, local_tokens, INDEX_TOPK], torch.int32),
        TensorSpec("q_token_count", [ranks, 1], torch.int32),
        TensorSpec("kv_rope_cos", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("kv_rope_sin", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("kv_write_slots", [ranks, tokens], torch.int64),
        TensorSpec("kv_token_count", [ranks, 1], torch.int32),
        TensorSpec("window_cache", [ranks, pages, 128, 1, WINDOW_CACHE_BYTES], torch.uint8),
        TensorSpec("compressed_cache", [ranks, compressed_pages, 128, 1, COMPRESSED_CACHE_BYTES], torch.uint8),
        TensorSpec("kv_restore_indices", [ranks, tokens], torch.int32),
        TensorSpec("kv_hidden", [ranks, tokens, D], bf),
        TensorSpec("hc_output", [ranks, local_tokens, HC_MULT, D], torch.float32),
        TensorSpec("hc_next_pre_mix", [ranks, local_tokens, HC_MULT], torch.float32),
    ]
    for spec in specs:
        spec.resident = "stacked"
        if spec.name not in ("kv_hidden", "hc_output", "hc_next_pre_mix"):
            spec.init_value = lambda name=spec.name: initialize(name)
    specs.append(ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True, benchmark_step=None))
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in entry.param_names]


if __name__ == "__main__":
    import argparse
    import functools
    import torch

    from config import SUPPORTED_CP_SIZES
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run
    from prefill_c1a_full import golden_c1a_prefill
    from utils import packed_cache_compare
    from pypto.ir import DistributedConfig

    from utils import parse_prefill_groups

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C1A reuse")
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
    entry = l3_prefill_c1a_reuse if CP_SIZE == 1 else l3_prefill_c1a_reuse_cp
    result = run(
        fn=entry,
        specs=build_c1a_reuse_specs(args, entry),
        golden_fn=functools.partial(golden_c1a_prefill, mode="reuse"),
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
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
