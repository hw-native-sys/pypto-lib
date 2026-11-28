# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""Prefill a chunk and publish its DSpark draft context KV on each rank."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import BLOCK_SIZE
from dspark_drafter import l2_dspark_prefill_context_kv_rank
from prefill_fwd import (
    BIAS_DYN,
    B_DYN,
    C,
    C1A_TABLE_DYN,
    C2A_SOURCE_COUNT,
    C2A_TABLE_DYN,
    CAPACITY,
    COMPRESSED_CACHE_BYTES,
    CP_SIZE,
    D,
    DSPARK_DRAFT_LAYERS,
    DSPARK_MAIN_IN,
    ENGRAM_K,
    ENGRAM_SHARD_ROWS,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    INDEX_DIM,
    INDEX_H,
    INDEX_SOURCE_COUNT,
    KV_OUT,
    KV_SOURCE_COUNT,
    LM_HEAD_TP_SIZE,
    LOCAL_CAPACITY,
    LOCAL_EXPERTS,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    MIX_HC,
    MOE_INTER,
    MOE_WORKSPACE_BYTES,
    MX_GROUP,
    MX_PACKED_LANE_COLS,
    MX_W1_PACKED_ROWS,
    MX_W2_PACKED_ROWS,
    MX_W3_PACKED_ROWS,
    N_EXPERTS,
    N_HASH_COLS,
    ORI_BLOCKS_DYN,
    O_GROUP_IN,
    O_LORA,
    PAGES,
    PREFILL_MAX_TOKENS,
    Q_LORA,
    Q_START_DYN,
    RANKS,
    ROPE_DIM,
    ROPE_ROWS_DYN,
    STATE_BLOCKS_DYN,
    STATE_WIDTH,
    VOCAB,
    WEIGHT_BANK_SIZE,
    WINDOW_CACHE_BYTES,
    WINDOW_PAGES,
    l2_prefill_fwd,
)
from prefill_fwd import PREFILL_TENSORS as TARGET_PREFILL_TENSORS


PREFILL_TENSORS = TARGET_PREFILL_TENSORS + (
    "draft_main_proj_weight",
    "draft_main_proj_scale",
    "draft_main_norm_weight",
    "draft_num_tokens",
    "draft_context_slots",
    "draft_context_rope_cos",
    "draft_context_rope_sin",
    "draft_wkv",
    "draft_wkv_scale",
    "draft_kv_norm_weight",
    "draft_window_cache",
)


@pl.jit.host
def l3_prefill_fwd_dspark(
    embedding: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    engram_ids: pl.Tensor[[RANKS, 2 * CAPACITY * N_HASH_COLS, 1], pl.INT32],
    engram_wkv: pl.Tensor[[RANKS, 2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[RANKS, 2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[RANKS, 2, HC_MULT, D], pl.FP32],
    embedding_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    c2a_compressor_wkv: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[RANKS, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[RANKS, KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_weight: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[RANKS, BIAS_DYN], pl.FP32],
    routed_w1: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (MOE_INTER // MX_GROUP)), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (LOCAL_EXPERTS * (D // MX_GROUP)), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[RANKS, WEIGHT_BANK_SIZE * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    counts: pl.Tensor[[RANKS, 2], pl.INT32],
    replay_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    freqs_cos: pl.Tensor[[RANKS, 2, ROPE_ROWS_DYN, 32], pl.FP32],
    freqs_sin: pl.Tensor[[RANKS, 2, ROPE_ROWS_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT64],
    window_restore_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[RANKS, 2, CAPACITY, 128], pl.INT32],
    c2a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    c1a_compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    token_to_req_indices: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    query_start_loc: pl.Tensor[[RANKS, 2, Q_START_DYN], pl.INT32],
    c2a_index_block_table: pl.Tensor[[RANKS, B_DYN, C2A_TABLE_DYN], pl.INT32],
    c1a_index_block_table: pl.Tensor[[RANKS, B_DYN, C1A_TABLE_DYN], pl.INT32],
    state_block_table: pl.Tensor[[RANKS, B_DYN, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS, 40, WINDOW_PAGES, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS, 5, PAGES, 64, 1, COMPRESSED_CACHE_BYTES], pl.UINT8]],
    index_cache: pl.InOut[pl.Tensor[[RANKS, 5, PAGES, 64, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[RANKS, 5, PAGES, 64, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[RANKS, 3, STATE_BLOCKS_DYN, C.STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    final_norm_weight: pl.Tensor[[RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[RANKS, D, VOCAB // LM_HEAD_TP_SIZE], pl.BF16],
    logit_rows: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_temperatures: pl.Tensor[[RANKS, 16], pl.FP32],
    sampling_top_ks: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_top_ps: pl.Tensor[[RANKS, 16], pl.FP32],
    sampling_seeds: pl.Tensor[[RANKS, 16], pl.INT32],
    sampling_positions: pl.Tensor[[RANKS, 16], pl.INT32],
    x_hc_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.FP32]],
    pre_mix_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    dspark_target_hidden: pl.Out[pl.Tensor[[RANKS, CAPACITY, DSPARK_MAIN_IN], pl.BF16]],
    logits: pl.Out[pl.Tensor[[RANKS, 16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[RANKS, 16, 8], pl.INT32]],
    draft_main_proj_weight: pl.Tensor[[RANKS, DSPARK_MAIN_IN, D], pl.FP8E4M3FN, pl.NZ],
    draft_main_proj_scale: pl.Tensor[[RANKS, DSPARK_MAIN_IN // MX_GROUP, D], pl.FP8E8M0],
    draft_main_norm_weight: pl.Tensor[[RANKS, D], pl.BF16],
    draft_num_tokens: pl.Tensor[[RANKS, 1], pl.INT32],
    draft_context_slots: pl.Tensor[[RANKS, DSPARK_DRAFT_LAYERS, CAPACITY], pl.INT64],
    draft_context_rope_cos: pl.Tensor[[RANKS, CAPACITY, ROPE_DIM // 2], pl.FP32],
    draft_context_rope_sin: pl.Tensor[[RANKS, CAPACITY, ROPE_DIM // 2], pl.FP32],
    draft_wkv: pl.Tensor[[RANKS, DSPARK_DRAFT_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    draft_wkv_scale: pl.Tensor[[RANKS, DSPARK_DRAFT_LAYERS, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    draft_kv_norm_weight: pl.Tensor[[RANKS, DSPARK_DRAFT_LAYERS, HEAD_DIM], pl.BF16],
    draft_window_cache: pl.InOut[pl.Tensor[[RANKS, DSPARK_DRAFT_LAYERS, ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
):
    """Append a packed CED chunk. Each DP group owns its requests and caches across calls.

    Pass encoder then decoder metadata with absolute positions and source-local cache slots.
    For full replay, use identical encoder and decoder counts and identity replay rows padded with -1.
    Supply base then compressed RoPE tables covering the retained history. Logit rows index decoder replay.
    Keep cache pages and request slots until completion. Restore or reset cache state before retrying.
    """
    correction_bias.bind_dynamic(1, BIAS_DYN)
    freqs_cos.bind_dynamic(2, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, ROPE_ROWS_DYN)
    compressed_cache.bind_dynamic(2, PAGES)
    index_cache.bind_dynamic(2, PAGES)
    index_scales.bind_dynamic(2, PAGES)
    state_cache.bind_dynamic(2, STATE_BLOCKS_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(2, C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, B_DYN)
    c1a_index_block_table.bind_dynamic(2, C1A_TABLE_DYN)
    query_start_loc.bind_dynamic(2, Q_START_DYN)
    engram_scale_buffer = pld.alloc_window_buffer(
        [2 * ENGRAM_SHARD_ROWS, 8], dtype=pl.UINT8,
        memory_source="host", init="engram_scale",
    )
    engram_payload_buffer = pld.alloc_window_buffer(
        [2 * ENGRAM_SHARD_ROWS, 256], dtype=pl.UINT8,
        memory_source="host", init="engram_payload",
    )
    moe_workspace_buffer = pld.alloc_window_buffer([MOE_WORKSPACE_BYTES], dtype=pl.UINT8)
    attention_window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    attention_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    hc_window_buffer = pld.alloc_window_buffer([LOCAL_CAPACITY, HC_DIM + 16], dtype=pl.FP32)
    hc_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    head_hidden_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
    head_hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    head_logits_buffer = pld.alloc_window_buffer([16, VOCAB], dtype=pl.BF16)
    head_logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        engram_scales = pld.window(engram_scale_buffer, [2 * ENGRAM_SHARD_ROWS, 8], dtype=pl.UINT8)
        engram_payload = pld.window(engram_payload_buffer, [2 * ENGRAM_SHARD_ROWS, 256], dtype=pl.UINT8)
        moe_workspace = pld.window(moe_workspace_buffer, [MOE_WORKSPACE_BYTES], dtype=pl.UINT8)
        attention_window = pld.window(attention_window_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        attention_arrived = pld.window(attention_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        hc_window = pld.window(hc_window_buffer, [LOCAL_CAPACITY, HC_DIM + 16], dtype=pl.FP32)
        hc_arrived = pld.window(hc_arrived_buffer, [CP_SIZE, 1], dtype=pl.INT32)
        head_hidden = pld.window(head_hidden_buffer, [LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
        head_hidden_done = pld.window(head_hidden_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        head_logits = pld.window(head_logits_buffer, [16, VOCAB], dtype=pl.BF16)
        head_logits_done = pld.window(head_logits_done_buffer, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_prefill_fwd(
            embedding[rank],
            engram_ids[rank], engram_scales, engram_payload,
            engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            embedding_ids[rank], initial_pre[rank], identity_rows[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale[rank], q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank],
            wkv[rank], wkv_scale[rank], kv_norm_weight[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale[rank],
            c2a_compressor_wkv[rank], c2a_compressor_wgate[rank], c1a_compressor_wkv[rank],
            compressor_norm_weight[rank],
            index_wk[rank], index_norm_weight[rank],
            index_wq_b[rank], index_wq_b_scale[rank], index_weights_proj[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank], gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            counts[rank], replay_rows[rank],
            position_ids[rank], freqs_cos[rank], freqs_sin[rank],
            window_slots[rank], window_restore_slots[rank], window_indices[rank],
            c2a_compressed_slots[rank], c1a_compressed_slots[rank],
            token_to_req_indices[rank], query_start_loc[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank], state_block_table[rank],
            window_cache[rank], compressed_cache[rank],
            index_cache[rank], index_scales[rank], state_cache[rank],
            final_norm_weight[rank], lm_head_weight[rank], logit_rows[rank],
            sampling_temperatures[rank], sampling_top_ks[rank], sampling_top_ps[rank],
            sampling_seeds[rank], sampling_positions[rank],
            x_hc_out[rank], pre_mix_out[rank], dspark_target_hidden[rank],
            logits[rank], sampled_ids[rank],
            moe_workspace,
            attention_window, attention_arrived, hc_window, hc_arrived,
            head_hidden, head_hidden_done, head_logits, head_logits_done,
            rank, device=rank,
        )
        # Fuse draft context KV materialization into the terminal prefill
        # composite. The dependency is carried by dspark_target_hidden, so
        # the context write starts after the target layers publish their taps.
        context_slots = draft_context_slots[rank]
        context_rope_cos = draft_context_rope_cos[rank]
        context_rope_sin = draft_context_rope_sin[rank]
        l2_dspark_prefill_context_kv_rank(
            dspark_target_hidden[rank], draft_main_proj_weight[rank], draft_main_proj_scale[rank],
            draft_main_norm_weight[rank], draft_num_tokens[rank], context_slots,
            context_rope_cos, context_rope_sin, draft_wkv[rank],
            draft_wkv_scale[rank], draft_kv_norm_weight[rank], draft_window_cache[rank],
            device=rank,
        )
