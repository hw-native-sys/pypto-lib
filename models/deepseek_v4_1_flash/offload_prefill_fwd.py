# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""CED prefill with routed experts supplied from a host-backed slot bank."""
import pypto.language as pl
import pypto.language.distributed as pld
import config as C
from prefill_fwd import (
    BIAS_DYN, C1A_TABLE_DYN, C2A_SOURCE_COUNT, C2A_TABLE_DYN, CAPACITY, CP_SIZE, D, DSPARK_MAIN_IN,
    HC_DIM, HC_MULT, HEAD_DIM, INDEX_DIM, INDEX_H, INDEX_SOURCE_COUNT,
    KV_SOURCE_COUNT, LOCAL_CAPACITY, LOCAL_H, LOCAL_O_GROUPS, LOCAL_O_WIDTH, MIX_HC,
    MOE_INTER, MX_GROUP, N_LAYERS, O_GROUP_IN, O_LORA, PAGES, Q_LORA, RANKS, RECV_MAX, TOPK, VOCAB,
    WINDOW_PAGES,
)
from config import (
    LM_HEAD_TP_SIZE, B_DYN,
    PREFILL_MAX_TOKENS, Q_START_DYN, STATE_BLOCKS_DYN, MOE_TOKENS,
    WINDOW_CACHE_BYTES, COMPRESSED_CACHE_BYTES,
)

from rope_tables import ROPE_ROWS_DYN
from engram import ENGRAM_K, KV_OUT

from compressor import STATE_WIDTH
from expert_routed import MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS

from expert_context import EXPERT_SLOTS, CACHE_ROWS, CACHE_COLS, EXPERT_LAYERS
from offload_prefill_moe import AUX_WIDTH, ROUTE_WIDTH

LOCAL_EXPERTS = C.N_LOCAL_EXPERTS
N_EXPERTS = RANKS * LOCAL_EXPERTS
ROUTED_BANK_ROWS = EXPERT_SLOTS
ROUTED_LAYER_ROWS = EXPERT_SLOTS
ROUTED_LAYER_STRIDE = 0


from prefill_fwd import l2_prefill_fwd


@pl.jit.host
def l3_prefill_fwd(
    embedding: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    engram_lookup: pl.Tensor[[RANKS, 2, CAPACITY, ENGRAM_K], pl.BF16],
    engram_wkv: pl.Tensor[[RANKS, 2, ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    engram_wkv_scale: pl.Tensor[[RANKS, 2 * (ENGRAM_K // 32), KV_OUT], pl.FP8E8M0],
    engram_weight: pl.Tensor[[RANKS, 2, HC_MULT, D], pl.FP32],
    embedding_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    hc_attn_fn: pl.Tensor[[RANKS, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS, N_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS, N_LAYERS * MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS, N_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[RANKS, N_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS, N_LAYERS * Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS, N_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS, N_LAYERS * (Q_LORA // MX_GROUP), LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS, N_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS, N_LAYERS * HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS, N_LAYERS * LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS, N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS, N_LAYERS * LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS, N_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS, N_LAYERS * (LOCAL_O_WIDTH // MX_GROUP), D], pl.FP8E8M0],
    c2a_compressor_wkv: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c2a_compressor_wgate: pl.Tensor[[RANKS, C2A_SOURCE_COUNT, D, HEAD_DIM], pl.BF16],
    c1a_compressor_wkv: pl.Tensor[[RANKS, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[RANKS, KV_SOURCE_COUNT * HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT * (Q_LORA // MX_GROUP), INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[RANKS, INDEX_SOURCE_COUNT, D, INDEX_H], pl.BF16],
    hc_ffn_fn: pl.Tensor[[RANKS, N_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[RANKS, N_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[RANKS, N_LAYERS * MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[RANKS, N_LAYERS * D], pl.BF16],
    gate_weight: pl.Tensor[[RANKS, N_LAYERS * N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[RANKS, BIAS_DYN], pl.FP32],
    routed_w1: pl.InOut[pl.Tensor[[RANKS, ROUTED_BANK_ROWS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8]],
    routed_w1_scale: pl.InOut[pl.Tensor[[RANKS, ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0]],
    routed_w2: pl.InOut[pl.Tensor[[RANKS, ROUTED_BANK_ROWS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8]],
    routed_w2_scale: pl.InOut[pl.Tensor[[RANKS, ROUTED_BANK_ROWS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0]],
    routed_w3: pl.InOut[pl.Tensor[[RANKS, ROUTED_BANK_ROWS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8]],
    routed_w3_scale: pl.InOut[pl.Tensor[[RANKS, ROUTED_BANK_ROWS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0]],
    shared_w1: pl.Tensor[[RANKS, N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[RANKS, N_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[RANKS, N_LAYERS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[RANKS, N_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[RANKS, N_LAYERS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
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
    expert_state: pl.InOut[pl.Tensor[[RANKS, CACHE_ROWS, CACHE_COLS], pl.INT32]],
    expert_descriptors: pl.Tensor[[RANKS, EXPERT_LAYERS, 6, 4], pl.INT64],
):
    """Append one chunk with EP = DP * CP and return global state/logits.

    CAPACITY bounds current rows to at most 4096; the dynamic page extent
    independently sizes retained history. For full causal continuation replay
    all current rows:
    counts[:, 0] == counts[:, 1], replay_rows and identity_rows are current-row
    identities, padded with -1. Positions, lengths and physical slots are
    absolute within the request. Both stages' window indices include committed
    history; the compressed families have independent local page tables.

    embedding holds caller-gathered BF16 rows; embedding_ids indexes that
    rank-local slab, not the model vocabulary. initial_pre is the current
    rows' delayed mix (identity for new tokens). position_ids, window_slots,
    window_indices, token_to_req_indices and query_start_loc have stage order
    encoder, decoder. Each stage has packed query_start_loc and absolute
    position_ids; token_to_req_indices indexes request-table rows.
    C2A/C1A index_block_table arguments have independent widths and source-local
    physical page IDs. KV and index keys share their family's mapping. C2A pages
    contain 64 stored rows covering 128 original tokens; C1A and SWA pages
    contain 128 stored rows covering 128 original tokens.
    Visible lengths are derived once as floor((position+1)/ratio), including
    completed current groups: C2A follows encoder rows and C1A decoder replay rows.
    freqs_cos/sin are resident full FP32 half-width tables with profile order
    base, compressed. Their row capacity is independent of current CAPACITY
    and covers all active absolute positions. The graph gathers encoder base,
    encoder compressed, decoder compressed and C2A group-first rows once.
    C1A publication reuses encoder compressed rows before replay selection.
    c2a_compressed_slots and c1a_compressed_slots both follow encoder rows;
    C1A publishes all encoder rows before decoder replay selection. Slots are
    source-local physical rows; -1 disables publication. C2A publishes only at
    completed pair positions. state_block_table maps requests to source-local
    state blocks shared across three independent C2A source slices; -1 is inactive.
    logit_rows indexes the decoder replay slab; -1 disables that output row.

    Preserve all seven InOut cache tensors across chunks and decode. Their
    source/page dimensions can be flattened into decode pool views.
    Residual/pre-mix outputs describe these rows after layer 39. The DSpark
    output packs HC-mean hidden states from layers 37, 38 and 39 per row.
    New tokens
    start from their own embedding and identity pre-mix; historical HC state
    is represented by the committed KV/index/compressor caches.

    Zero-count ranks participate in EP communication without publishing cache
    rows. Wait for runtime completion before advancing lengths, launching a
    dependent call or reusing buffers. Use persistent=True with
    reset_persistent_windows=False; signals are retired at forward completion.
    After request completion, release pages and reset the request slot in all C2A sources before
    assigning them to another request; failed calls require cache restoration
    or reset before retry. Engram consumes host lookup rows, with zero rows
    suppressing its update. correction_bias accepts the text layout
    [RANKS, N_LAYERS * N_EXPERTS] or flattened per-token biases
    [RANKS, N_LAYERS * CAPACITY * N_EXPERTS] in encoder input-row order.
    Decoder replay and slab ownership gather the corresponding token bias.
    """
    correction_bias.bind_dynamic(1, BIAS_DYN)
    freqs_cos.bind_dynamic(2, ROPE_ROWS_DYN)
    freqs_sin.bind_dynamic(2, ROPE_ROWS_DYN)
    compressed_cache.bind_dynamic(2, PAGES)
    state_cache.bind_dynamic(2, STATE_BLOCKS_DYN)
    state_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(1, B_DYN)
    c2a_index_block_table.bind_dynamic(2, C2A_TABLE_DYN)
    c1a_index_block_table.bind_dynamic(1, B_DYN)
    c1a_index_block_table.bind_dynamic(2, C1A_TABLE_DYN)
    query_start_loc.bind_dynamic(2, Q_START_DYN)
    recv_meta_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
    recv_weights_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buffer = pld.alloc_window_buffer([RANKS, 128], dtype=pl.INT32)
    data_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    routed_output_buffer = pld.alloc_window_buffer([MOE_TOKENS * TOPK, D], dtype=pl.BF16)
    combine_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    attention_window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    attention_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    hc_window_buffer = pld.alloc_window_buffer([LOCAL_CAPACITY, HC_DIM + 16], dtype=pl.FP32)
    hc_arrived_buffer = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    head_hidden_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE * 16, D], dtype=pl.BF16)
    head_hidden_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    head_logits_buffer = pld.alloc_window_buffer([16, VOCAB], dtype=pl.BF16)
    head_logits_done_buffer = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buffer, [RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buffer, [LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buffer, [LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buffer, [LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buffer, [LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buffer, [RANKS, 128], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buffer, [MOE_TOKENS * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
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
            engram_lookup[rank], engram_wkv[rank], engram_wkv_scale[rank], engram_weight[rank],
            embedding_ids[rank],
            initial_pre[rank],
            identity_rows[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], wq_a_scale[rank], q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank],
            wkv[rank], wkv_scale[rank], kv_norm_weight[rank],
            attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale[rank],
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
            counts[rank],
            replay_rows[rank],
            position_ids[rank],
            freqs_cos[rank], freqs_sin[rank],
            window_slots[rank],
            window_restore_slots[rank],
            window_indices[rank],
            c2a_compressed_slots[rank],
            c1a_compressed_slots[rank],
            token_to_req_indices[rank],
            query_start_loc[rank],
            c2a_index_block_table[rank], c1a_index_block_table[rank],
            state_block_table[rank],
            window_cache[rank],
            compressed_cache[rank],
            index_cache[rank],
            index_scales[rank],
            state_cache[rank],
            final_norm_weight[rank],
            lm_head_weight[rank],
            logit_rows[rank],
            sampling_temperatures[rank], sampling_top_ks[rank], sampling_top_ps[rank],
            sampling_seeds[rank], sampling_positions[rank],
            x_hc_out[rank], pre_mix_out[rank], dspark_target_hidden[rank],
            logits[rank],
            sampled_ids[rank],
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            attention_window,
            attention_arrived,
            hc_window, hc_arrived,
            head_hidden, head_hidden_done,
            head_logits,
            head_logits_done,
            rank, expert_state=expert_state[rank],
            expert_descriptors=expert_descriptors[rank], expert_slots=EXPERT_SLOTS, offload=True, device=rank,
        )
