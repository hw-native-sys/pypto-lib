# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Context-parallel SWA prefill."""


import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    FLASH as M, RANKS_DYN, LOCAL_T_DYN, T_DYN, ORI_BLOCKS_DYN,
    CP_SIZE, PREFILL_MAX_TOKENS, WINDOW_CACHE_BYTES,
)
from sparse_attention import gather_sparse_attention
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj, window_publish
from o_proj import o_proj_hc_post, retire_prefill_signals
from hc import hc_pre, hc_normalize, zero_bf16_padding
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


# tiling
Q_TILE = 1024
WORKER_TILE = 64


@pl.jit.inline(auto_scope=False)
def prefill_swa(
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
    # Query indices.
    q_window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    # Token metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Scalar[pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    # Outputs.
    kv_hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    hc_output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
):
    """Compute SWA on one rank with shared normalized Q/KV inputs."""
    q_capacity = pl.tensor.dim(hc_input, 0)
    post_mix = pl.create_tensor([q_capacity, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([q_capacity, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_collapsed = pl.create_tensor([q_capacity, D], dtype=pl.BF16)
    hc_pre(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        hc_collapsed, hc_next_pre_mix, post_mix, residual_mix,
    )
    normalized_ready = hc_normalize(hc_collapsed, w_attn_norm, kv_hidden, q_capacity)
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
        q_latent = pl.create_tensor([Q_TILE, Q_LORA], dtype=pl.BF16)
        q = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)

        chunk_done = normalized_ready
        for start in pl.range(0, kv_token_count, Q_TILE):
            active = pl.min(Q_TILE, kv_token_count - start)
            # The prior chunk must finish reading every reusable scratch buffer.
            with pl.spmd(WORKER_TILE, name_hint="prefill_swa_stage", deps=[chunk_done]) as stage_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    source_row = start + row
                    chunk_hidden[row:row + 1, :] = kv_hidden[source_row:source_row + 1, :]
                    chunk_rope_cos[row:row + 1, :] = kv_rope_cos[source_row:source_row + 1, :]
                    chunk_rope_sin[row:row + 1, :] = kv_rope_sin[source_row:source_row + 1, :]
                    chunk_window_indices[row:row + 1, :] = q_window_indices[source_row:source_row + 1, :]
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
            gather_sparse_attention(
                q, chunk_window_indices, window_cache, chunk_window_indices, window_cache,
                attn_sink, attended, active, attention_ready, compressed_keys=0,
            )
            _chunk_output, projection_done = o_proj_hc_post(
                attended, wo_a, wo_a_scale, wo_b, wo_b_scale, chunk_rope_cos, chunk_rope_sin,
                hc_input[start:start + active, :, :], post_mix[start:start + active, :],
                residual_mix[start:start + active, :, :], hc_output[start:start + active, :, :], active,
                oa_row_tile=128, ob_row_tile=128,
            )

            chunk_done = projection_done
    if kv_token_count < q_capacity:
        with pl.spmd(32, name_hint="prefill_swa_zero_padding"):
            worker = pl.tile.get_block_idx()
            for row in pl.range(kv_token_count + worker, q_capacity, 32):
                zero_output = pl.full([1, HC_MULT * D], dtype=pl.FP32, value=0.0)
                hc_output[row:row + 1, :, :] = pl.reshape(zero_output, [1, HC_MULT, D])
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                hc_next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
    return hc_output, hc_next_pre_mix


@pl.jit
def l2_prefill_swa(
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
    # Query indices.
    q_window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    # Token metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
):
    hc_input.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    active_tokens = pl.read(kv_token_count, [0])
    prefill_swa(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, w_attn_norm,
        wq_a, wq_a_scale, wq_norm, wq_b, wq_b_scale,
        wkv, wkv_scale, wkv_norm, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        q_window_indices,
        kv_rope_cos, kv_rope_sin, kv_write_slots, active_tokens, window_cache,
        kv_hidden, hc_output, hc_next_pre_mix,
    )
    return kv_hidden, hc_output, hc_next_pre_mix, window_cache


@pl.jit.host
def l3_prefill_swa(
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
    # Query indices.
    q_window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    # Token metadata and cache.
    kv_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, HC_MULT], pl.FP32]],
):
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    for rank in pl.range(pld.world_size()):
        rank_wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        rank_wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        rank_wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        rank_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        l2_prefill_swa(
            hc_input[rank], hc_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], w_attn_norm[rank],
            wq_a[rank], rank_wq_a_scale, wq_norm[rank], wq_b[rank], rank_wq_b_scale,
            wkv[rank], rank_wkv_scale, wkv_norm[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], rank_wo_b_scale,
            q_window_indices[rank],
            kv_rope_cos[rank], kv_rope_sin[rank], kv_write_slots[rank], kv_token_count[rank], window_cache[rank],
            kv_hidden[rank], hc_output[rank], hc_next_pre_mix[rank],
            device=rank,
        )


@pl.jit.inline(auto_scope=False)
def prefill_swa_cp(
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
    q_token_count: pl.Scalar[pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Scalar[pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
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
    """Compute local queries against replicated KV caches.

    CP peers gather head-tail rows and share complete weights, metadata and cache pages.
    Cache write slots are unique, and attention epochs serialize cache and scratch reuse.
    """
    q_capacity = pl.tensor.dim(hc_input, 0)
    q_hidden = pl.create_tensor([q_capacity, D], dtype=pl.BF16)
    post_mix = pl.create_tensor([q_capacity, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([q_capacity, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_collapsed = pl.create_tensor([q_capacity, D], dtype=pl.BF16)
    hc_pre(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        hc_collapsed, hc_next_pre_mix, post_mix, residual_mix,
    )
    hc_normalize(hc_collapsed, w_attn_norm, q_hidden, q_capacity)
    zero_bf16_padding(q_hidden, q_token_count)
    kv_hidden, gathered = prefill_cp_input_allgather(
        q_hidden, cp_hidden_window, cp_hidden_arrived, kv_restore_indices, kv_hidden,
        cp_group_base, cp_rank, attention_epoch, group_size=CP_SIZE,
    )
    if kv_token_count > 0:
        kv_capacity = pl.tensor.dim(kv_hidden, 0)
        kv_projection = pl.create_tensor([kv_capacity, HEAD_DIM], dtype=pl.BF16)
        # Gate every projection tile before scratch storage is reused by a new epoch.
        with pl.spmd(WORKER_TILE, name_hint="prefill_swa_begin", deps=[gathered]):
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
        chunk_hidden = pl.create_tensor([Q_TILE, D], dtype=pl.BF16)
        chunk_rope_cos = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_rope_sin = pl.create_tensor([Q_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_window_indices = pl.create_tensor([Q_TILE, 128], dtype=pl.INT32)
        q_latent = pl.create_tensor([Q_TILE, Q_LORA], dtype=pl.BF16)
        q = pl.create_tensor([Q_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)

        chunk_done = gathered
        for start in pl.range(0, q_token_count, Q_TILE):
            active = pl.min(Q_TILE, q_token_count - start)
            # The prior chunk must finish reading every reusable scratch buffer.
            with pl.spmd(WORKER_TILE, name_hint="prefill_swa_stage", deps=[chunk_done]) as stage_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    source_row = start + row
                    chunk_hidden[row:row + 1, :] = q_hidden[source_row:source_row + 1, :]
                    chunk_rope_cos[row:row + 1, :] = q_rope_cos[source_row:source_row + 1, :]
                    chunk_rope_sin[row:row + 1, :] = q_rope_sin[source_row:source_row + 1, :]
                    chunk_window_indices[row:row + 1, :] = q_window_indices[source_row:source_row + 1, :]
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
            gather_sparse_attention(
                q, chunk_window_indices, window_cache, chunk_window_indices, window_cache,
                attn_sink, attended, active, attention_ready, compressed_keys=0,
            )
            _chunk_output, projection_done = o_proj_hc_post(
                attended, wo_a, wo_a_scale, wo_b, wo_b_scale, chunk_rope_cos, chunk_rope_sin,
                hc_input[start:start + active, :, :], post_mix[start:start + active, :],
                residual_mix[start:start + active, :, :], hc_output[start:start + active, :, :], active,
                oa_row_tile=128, ob_row_tile=128,
            )

            chunk_done = projection_done
    if q_token_count < q_capacity:
        with pl.spmd(32, name_hint="cp_mix_zero_padding", deps=[gathered]):
            worker = pl.tile.get_block_idx()
            for row in pl.range(q_token_count + worker, q_capacity, 32):
                zero_output = pl.full([1, HC_MULT * D], dtype=pl.FP32, value=0.0)
                hc_output[row:row + 1, :, :] = pl.reshape(zero_output, [1, HC_MULT, D])
                zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                hc_next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
    return hc_output, hc_next_pre_mix


@pl.jit
def l2_prefill_swa_cp(
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
    q_token_count: pl.Tensor[[1], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
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
    """Run one CP rank and retire communication signals."""
    hc_input.bind_dynamic(0, LOCAL_T_DYN)
    kv_hidden.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    q_tokens = pl.read(q_token_count, [0])
    kv_tokens = pl.read(kv_token_count, [0])
    cp_group_base = global_rank // CP_SIZE * CP_SIZE
    cp_rank = global_rank % CP_SIZE
    prefill_swa_cp(
        hc_input, hc_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, w_attn_norm,
        wq_a, wq_a_scale, wq_norm, wq_b, wq_b_scale,
        wkv, wkv_scale, wkv_norm, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        q_rope_cos, q_rope_sin, q_window_indices, q_tokens,
        kv_rope_cos, kv_rope_sin, kv_write_slots, kv_tokens, window_cache,
        cp_hidden_window, cp_hidden_arrived, kv_restore_indices, cp_group_base, cp_rank, attention_epoch,
        kv_hidden, hc_output, hc_next_pre_mix,
    )
    retire_prefill_signals(hc_output, cp_hidden_arrived, cp_group_base, cp_rank, group_size=CP_SIZE)
    return kv_hidden, hc_output, hc_next_pre_mix, window_cache


@pl.jit.host
def l3_prefill_swa_cp(
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
    q_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    # Global KV metadata and cache.
    kv_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    kv_write_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    kv_token_count: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    # CP communication and row restoration.
    kv_restore_indices: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    # Outputs.
    kv_hidden: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    hc_output: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT, D], pl.FP32]],
    hc_next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, LOCAL_T_DYN, HC_MULT], pl.FP32]],
):
    """Allocate CP communication windows and dispatch ranks."""
    hc_input.bind_dynamic(0, RANKS_DYN)
    hc_input.bind_dynamic(1, LOCAL_T_DYN)
    kv_hidden.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    cp_hidden_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    cp_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    for global_rank in pl.range(pld.world_size()):
        cp_hidden_window = pld.window(cp_hidden_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        cp_hidden_arrived = pld.window(cp_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        # Rank-local projection scales use MX_B_NN layout.
        rank_wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[global_rank]
        rank_wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[global_rank]
        rank_wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[global_rank]
        rank_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[global_rank]
        l2_prefill_swa_cp(
            hc_input[global_rank], hc_pre_mix[global_rank],
            hc_attn_fn[global_rank], hc_attn_scale[global_rank], hc_attn_base[global_rank], w_attn_norm[global_rank],
            wq_a[global_rank], rank_wq_a_scale, wq_norm[global_rank], wq_b[global_rank], rank_wq_b_scale,
            wkv[global_rank], rank_wkv_scale, wkv_norm[global_rank], attn_sink[global_rank],
            wo_a[global_rank], wo_a_scale[global_rank], wo_b[global_rank], rank_wo_b_scale,
            q_rope_cos[global_rank], q_rope_sin[global_rank],
            q_window_indices[global_rank], q_token_count[global_rank],
            kv_rope_cos[global_rank], kv_rope_sin[global_rank],
            kv_write_slots[global_rank], kv_token_count[global_rank], window_cache[global_rank],
            cp_hidden_window, cp_hidden_arrived, kv_restore_indices[global_rank], global_rank, attention_epoch,
            kv_hidden[global_rank], hc_output[global_rank], hc_next_pre_mix[global_rank],
            device=global_rank,
        )


def build_swa_specs(args, entry):
    """Build independent DP request batches and optional single-request CP shards."""
    import math

    import torch
    from golden import ScalarSpec, TensorSpec
    from hc import make_hc_inputs
    from quant import (
        pack_fp8_nz, pack_mx_b_scale, pack_quantized_cache,
        quantize_mxfp8_cache, quantize_mxfp8_upward_reference, quantize_grouped_mxfp8_weight,
    )
    from utils import (
        prefill_head_tail_rows, prefill_batch_layout, prefill_fixture_lengths, prefill_window_pages,
        window_metadata,
    )

    query_lengths, starts = prefill_batch_layout(args)
    ranks = CP_SIZE * args.dp
    tokens = args.tokens
    counts = [sum(group) for group in query_lengths]
    row_groups = [prefill_head_tail_rows([active], CP_SIZE) for active in counts]
    local_tokens = max(1, max(rows.numel() for group in row_groups for rows in group))
    if CP_SIZE == 1:
        local_tokens = tokens
    if local_tokens * CP_SIZE > PREFILL_MAX_TOKENS:
        raise ValueError("padded CP gather exceeds PREFILL_MAX_TOKENS")
    fixture_lengths = [prefill_fixture_lengths(group, tokens) for group in query_lengths]
    pages = max(prefill_window_pages(ns, ps) for ns, ps in zip(fixture_lengths, starts))
    values = {}

    def initialize(name):
        if not values:
            rank_inputs = []
            for dp_rank, group_rows in enumerate(row_groups):
                seed = args.seed + dp_rank * CP_SIZE
                case = "mixed" if args.case == "normal" else args.case
                lengths = fixture_lengths[dp_rank]
                if len(lengths) < 1 or tokens < 1:
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
                    wo_b=ob, wo_b_scale=obs,
                )
                wo_a = (rand(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(torch.bfloat16)
                inputs['wo_a'], inputs['wo_a_scale'] = quantize_grouped_mxfp8_weight(wo_a)
                gen = torch.Generator().manual_seed(seed + 1000)
                input_lengths = torch.tensor(lengths, dtype=torch.int64)
                prefixes = torch.tensor(starts[dp_rank], dtype=torch.int64)
                ids = torch.repeat_interleave(torch.arange(len(lengths)), input_lengths)
                positions = torch.cat([torch.arange(int(p), int(p + n)) for p, n in zip(prefixes, input_lengths)])
                logical_capacity = int((prefixes + input_lengths + 129).max()) // 128 + 1
                table = torch.full((len(lengths), logical_capacity), -1, dtype=torch.int32)
                mappings = []
                for request, (prefix, length) in enumerate(zip(prefixes.tolist(), input_lengths.tolist())):
                    start = max(0, prefix - 127) // 128
                    end = (prefix + length + 128) // 128
                    mappings.extend(((request, block) for block in range(start, end + 1)))
                input_pages = torch.randperm(len(mappings) + 1, generator=gen)[:len(mappings)]
                for (request, block), page in zip(mappings, input_pages):
                    table[request, block] = page
                slots, indices, _ = window_metadata(positions, ids, table)
                freq = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
                angles = positions.float()[:, None] * freq
                cache_values = torch.randn(len(mappings) + 1, 128, 1, HEAD_DIM, generator=gen).bfloat16()
                cache, scales = quantize_mxfp8_cache(cache_values)
                inputs.update(
                    rope_cos=angles.cos(), rope_sin=angles.sin(),
                    window_slots=slots, window_indices=indices,
                    window_cache=pack_quantized_cache(cache, scales),
                )
                print(f'[FIXTURE] tokens={tokens} requests={len(lengths)} lengths={lengths} prefixes={prefixes.tolist()}')
                cache = inputs["window_cache"]
                padding = cache.new_zeros((pages - cache.shape[0], *cache.shape[1:]))
                inputs["window_cache"] = torch.cat((cache, padding))
                inputs.update(make_hc_inputs(tokens, seed + 2000, case))
                for source, target in (
                    ("x_hc", "hc_input"), ("pre_mix", "hc_pre_mix"), ("attn_norm_weight", "w_attn_norm"),
                    ("q_norm_weight", "wq_norm"), ("kv_norm_weight", "wkv_norm"),
                    ("rope_cos", "kv_rope_cos"), ("rope_sin", "kv_rope_sin"), ("window_slots", "kv_write_slots"),
                ):
                    inputs[target] = inputs.pop(source)
                active = counts[dp_rank]
                inputs["kv_write_slots"][active:] = -1
                inputs["kv_token_count"] = torch.tensor([active], dtype=torch.int32)
                restore = torch.full((tokens,), -1, dtype=torch.int32)
                for cp_rank, rows in enumerate(group_rows):
                    restore[rows.long()] = cp_rank * local_tokens + torch.arange(rows.numel(), dtype=torch.int32)
                inputs["kv_restore_indices"] = restore
                for rows in group_rows:
                    shard = dict(inputs)
                    shard["q_token_count"] = torch.tensor([rows.numel()], dtype=torch.int32)
                    for source, target, fill in (
                        ("hc_input", "hc_input", 17.0), ("hc_pre_mix", "hc_pre_mix", 0),
                        ("kv_rope_cos", "q_rope_cos", float("nan")), ("kv_rope_sin", "q_rope_sin", float("nan")),
                        ("window_indices", "q_window_indices", -1),
                    ):
                        value = inputs[source].new_full((local_tokens, *inputs[source].shape[1:]), fill)
                        value[:rows.numel()] = inputs[source][rows.long()]
                        shard[target] = value
                    rank_inputs.append(shard)
            for key in entry.param_names:
                if key in rank_inputs[0]:
                    dtype = rank_inputs[0][key].dtype
                    shards = [rank[key] for rank in rank_inputs]
                    if dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                        shards = [shard.view(torch.uint8) for shard in shards]
                    values[key] = torch.stack(shards).view(dtype)
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
        TensorSpec("q_rope_cos", [ranks, local_tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("q_rope_sin", [ranks, local_tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("q_window_indices", [ranks, local_tokens, 128], torch.int32),
        TensorSpec("q_token_count", [ranks, 1], torch.int32),
        TensorSpec("kv_rope_cos", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("kv_rope_sin", [ranks, tokens, ROPE_DIM // 2], torch.float32),
        TensorSpec("kv_write_slots", [ranks, tokens], torch.int64),
        TensorSpec("kv_token_count", [ranks, 1], torch.int32),
        TensorSpec("window_cache", [ranks, pages, 128, 1, WINDOW_CACHE_BYTES], torch.uint8),
        TensorSpec("kv_restore_indices", [ranks, tokens], torch.int32),
    ]
    for spec in specs:
        spec.resident = "stacked"
        spec.init_value = lambda name=spec.name: initialize(name)
    specs += [
        ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True, benchmark_step=None),
        TensorSpec("kv_hidden", [ranks, tokens, D], bf, resident="stacked"),
        TensorSpec("hc_output", [ranks, local_tokens, HC_MULT, D], torch.float32, resident="stacked"),
        TensorSpec("hc_next_pre_mix", [ranks, local_tokens, HC_MULT], torch.float32, resident="stacked"),
    ]
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in entry.param_names]


def golden_swa(tensors):
    """Compute full-request reference outputs and split them across CP ranks."""
    import torch
    from hc import golden_hc_head_norm, golden_hc_mixes, golden_hc_post
    from quant import pack_quantized_cache, unpack_quantized_cache
    from sparse_attention import golden_gather_sparse_attention
    from qkv_proj_rope import golden_qkv_proj_rope
    from o_proj import golden_o_proj
    from quant import quantize_mxfp8_cache
    from quant import _publish_quantized_cache
    from utils import prefill_head_tail_rows

    ranks, tokens = tensors["kv_hidden"].shape[:2]
    for base in range(0, ranks, CP_SIZE):
        active = int(tensors["kv_token_count"][base, 0])
        row_groups = prefill_head_tail_rows([active], CP_SIZE)
        hc_input = torch.zeros(tokens, HC_MULT, D)
        hc_pre_mix = torch.zeros(tokens, HC_MULT)
        window_indices = torch.full((tokens, 128), -1, dtype=torch.int32)
        for cp_rank, rows in enumerate(row_groups):
            rank = base + cp_rank
            count = rows.numel()
            hc_input[rows.long()] = tensors["hc_input"][rank, :count]
            hc_pre_mix[rows.long()] = tensors["hc_pre_mix"][rank, :count]
            window_indices[rows.long()] = tensors["q_window_indices"][rank, :count]
        pre_mix, post_mix, residual_mix = golden_hc_mixes(
            hc_input, tensors["hc_attn_fn"][base], tensors["hc_attn_scale"][base], tensors["hc_attn_base"][base],
        )
        kv_hidden = golden_hc_head_norm(hc_input, hc_pre_mix, tensors["w_attn_norm"][base])
        attn_output = torch.zeros(tokens, D, dtype=torch.bfloat16)
        cache, scale = unpack_quantized_cache(tensors["window_cache"][base], False)
        if active:
            fields = {name: tensors[name][base] for name in (
                "wq_a", "wq_a_scale", "wq_b", "wq_b_scale", "wkv", "wkv_scale", "attn_sink",
                "wo_a", "wo_a_scale", "wo_b", "wo_b_scale",
            )}
            fields.update(
                x=kv_hidden[:active], q_norm_weight=tensors["wq_norm"][base], kv_norm_weight=tensors["wkv_norm"][base],
                rope_cos=tensors["kv_rope_cos"][base, :active], rope_sin=tensors["kv_rope_sin"][base, :active],
                window_slots=tensors["kv_write_slots"][base, :active], window_indices=window_indices[:active],
                window_cache=cache, window_cache_scale=scale,
            )
            query, window_kv, _ = golden_qkv_proj_rope(
                fields["x"], fields["wq_a"], fields["wq_a_scale"], fields["q_norm_weight"],
                fields["wq_b"], fields["wq_b_scale"], fields["wkv"], fields["wkv_scale"], fields["kv_norm_weight"],
                fields["rope_cos"], fields["rope_sin"], sequential_query=True,
            )
            cache, scale = _publish_quantized_cache(
                fields["window_cache"], fields["window_cache_scale"], window_kv, fields["window_slots"], quantize_mxfp8_cache,
            )
            packed_cache = pack_quantized_cache(cache, scale)
            indices = fields["window_indices"]
            attended = golden_gather_sparse_attention(
                query, indices, packed_cache, indices, packed_cache, fields["attn_sink"], compressed_keys=0,
            )
            projected = golden_o_proj(
                attended, fields["rope_cos"], fields["rope_sin"], fields["wo_a"], fields["wo_b"], fields["wo_b_scale"],
                wo_a_scale=fields["wo_a_scale"], sequential_projection=True,
            )
            attn_output[:active] = projected
        hc_output = golden_hc_post(attn_output, hc_input, post_mix, residual_mix)
        packed_cache = pack_quantized_cache(cache, scale)
        for cp_rank, rows in enumerate(row_groups):
            rank = base + cp_rank
            tensors["window_cache"][rank].copy_(packed_cache)
            tensors["kv_hidden"][rank].copy_(kv_hidden)
            for name, reference in (("hc_next_pre_mix", pre_mix), ("hc_output", hc_output)):
                tensors[name][rank].zero_()
                tensors[name][rank, :rows.numel()] = reference[rows.long()]


if __name__ == "__main__":
    import argparse
    import torch

    from config import SUPPORTED_CP_SIZES
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run
    from quant import packed_cache_compare
    from pypto.ir import DistributedConfig

    from utils import parse_prefill_groups

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill SWA")
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
    entry = l3_prefill_swa if CP_SIZE == 1 else l3_prefill_swa_cp
    result = run(
        fn=entry,
        specs=build_swa_specs(args, entry),
        golden_fn=golden_swa,
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
            "hc_next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "kv_hidden": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "hc_output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "window_cache": packed_cache_compare(
                mapped_pool_ratio_allclose(
                    "kv_write_slots", pool_name="window_cache",
                    mapping_shape=(len(devices), args.tokens),
                    block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.01,
                ),
                mapped_pool_ratio_allclose(
                    "kv_write_slots", pool_name="window cache scale",
                    mapping_shape=(len(devices), args.tokens),
                    block_size=128, leading_rank_axis=True,
                    atol=0.0, rtol=0.0, max_error_ratio=0.0,
                ),
                fp4=False,
            ),
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
