# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared draft and Markov work for native request cycles."""

import pypto.language as pl
import pypto.language.distributed as pld
from .drafter.device_metadata import BANK_COUNT
from .drafter.entry import drafter_inline
from .markov.entry import markov_inline
from .markov.entry import D, DONE_VALUE, DSPARK_MOE_TOKENS, MAX_LOGIT_ROWS, VOCAB, lm_head, normalize_head_hidden
from .markov.tp8_score import m8_confidence_outputs
from .drafter.device_drafter_tp8 import device_drafter_tp8
from .markov.head8 import publish_base_rows, run_head8_steps
from .target_state import eight_rank_reuse_gate


@pl.jit.inline(auto_scope=False)
def prepare_draft_markov(
    d_main_proj_weight: pl.Tensor[[4096, 12288], pl.BF16],
    d_main_norm_weight: pl.Tensor[[4096], pl.BF16],
    d_last_sampled: pl.Out[pl.Tensor[[1], pl.INT64]],
    d_embedding_weight: pl.Tensor[[129280, 4096], pl.BF16],
    d_context_group_position_ids_bank: pl.Tensor[[BANK_COUNT, 512], pl.INT32],
    d_context_group_slot_mapping_bank: pl.Tensor[[BANK_COUNT, 3, 512], pl.INT64],
    d_block_tables: pl.Tensor[[3, 1, 32768], pl.INT32],
    d_query_group_position_ids_bank: pl.Tensor[[BANK_COUNT, 80], pl.INT32],
    d_query_group_slot_mapping_bank: pl.Tensor[[BANK_COUNT, 3, 80], pl.INT64],
    d_context_group_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 512, 64], pl.BF16],
    d_context_group_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 512, 64], pl.BF16],
    d_query_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 20, 64], pl.BF16],
    d_query_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 20, 64], pl.BF16],
    d_query_group_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 80, 64], pl.BF16],
    d_query_group_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 80, 64], pl.BF16],
    d_hc_attn_fn: pl.Tensor[[72, 16384], pl.FP32],
    d_hc_attn_scale: pl.Tensor[[9], pl.FP32],
    d_hc_attn_base: pl.Tensor[[72], pl.FP32],
    d_attn_norm_w: pl.Tensor[[12288], pl.BF16],
    d_wq_a: pl.Tensor[[3, 4096, 1024], pl.BF16, pl.NZ],
    d_wq_b: pl.Tensor[[3, 1024, 32768], pl.INT8, pl.NZ],
    d_wq_b_scale: pl.Tensor[[98304], pl.FP32],
    d_wkv: pl.Tensor[[12288, 512], pl.BF16],
    d_gamma_cq: pl.Tensor[[3072], pl.BF16],
    d_gamma_ckv: pl.Tensor[[1536], pl.BF16],
    d_kv_caches: pl.InOut[pl.Tensor[[3, 512, 32, 1, 512], pl.BF16]],
    d_attn_sink: pl.Tensor[[192], pl.FP32],
    d_wo_a: pl.Tensor[[6, 1024, 4096], pl.BF16, pl.NZ],
    d_wo_b: pl.Tensor[[6, 4096, 1024], pl.INT8, pl.NZ],
    d_wo_b_scale: pl.Tensor[[12288], pl.FP32],
    d_hc_ffn_fn: pl.Tensor[[72, 16384], pl.FP32],
    d_hc_ffn_scale: pl.Tensor[[9], pl.FP32],
    d_hc_ffn_base: pl.Tensor[[72], pl.FP32],
    d_ffn_norm_w: pl.Tensor[[12288], pl.BF16],
    d_gate_w: pl.Tensor[[768, 4096], pl.FP32],
    d_gate_bias: pl.Tensor[[768], pl.FP32],
    d_tid2eid: pl.Tensor[[387840, 6], pl.INT32],
    d_routed_w1: pl.Tensor[[192, 2048, 4096], pl.INT8, pl.NZ],
    d_routed_w1_scale: pl.Tensor[[192, 2048], pl.FP32],
    d_routed_w3: pl.Tensor[[192, 2048, 4096], pl.INT8, pl.NZ],
    d_routed_w3_scale: pl.Tensor[[192, 2048], pl.FP32],
    d_routed_w2: pl.Tensor[[192, 4096, 2048], pl.INT8, pl.NZ],
    d_routed_w2_scale: pl.Tensor[[192, 4096], pl.FP32],
    d_shared_w1: pl.Tensor[[3, 2048, 4096], pl.INT8, pl.NZ],
    d_shared_w1_scale: pl.Tensor[[6144], pl.FP32],
    d_shared_w3: pl.Tensor[[3, 2048, 4096], pl.INT8, pl.NZ],
    d_shared_w3_scale: pl.Tensor[[6144], pl.FP32],
    d_shared_w2: pl.Tensor[[3, 4096, 2048], pl.INT8, pl.NZ],
    d_shared_w2_scale: pl.Tensor[[12288], pl.FP32],
    d_hc_head_fn: pl.Tensor[[4, 16384], pl.FP32],
    d_hc_head_scale: pl.Tensor[[1], pl.FP32],
    d_hc_head_base: pl.Tensor[[4], pl.FP32],
    d_initial_hidden: pl.Out[pl.Tensor[[32, 4, 4096], pl.FP32]],
    d_intermediate_hidden: pl.Out[pl.Tensor[[3, 32, 4, 4096], pl.FP32]],
    d_head_hidden: pl.Out[pl.Tensor[[1, 5, 4096], pl.BF16]],
    d_hidden_gather_window: pld.DistributedTensor[[8192, 4096], pl.BF16],
    d_hidden_gather_signal: pld.DistributedTensor[[4, 1], pl.INT32],
    d_attention_window: pld.DistributedTensor[[256, 4096], pl.BF16],
    d_attention_signal: pld.DistributedTensor[[4, 1], pl.INT32],
    d_o_window: pld.DistributedTensor[[128, 4096], pl.BF16],
    d_o_signal: pld.DistributedTensor[[4, 1], pl.INT32],
    d_recv_meta: pld.DistributedTensor[[4, 64], pl.INT32],
    d_recv_x: pld.DistributedTensor[[8192, 4096], pl.INT8],
    d_recv_aux: pld.DistributedTensor[[8192, 8], pl.FP32],
    d_recv_route: pld.DistributedTensor[[8192, 8], pl.INT32],
    d_arrived: pld.DistributedTensor[[4, 1], pl.INT32],
    d_data_arrived: pld.DistributedTensor[[4, 1], pl.INT32],
    d_routed_y_buf: pld.DistributedTensor[[192, 4096], pl.BF16],
    d_combine_arrived: pld.DistributedTensor[[4, 1], pl.INT32],
    m_final_norm_weight: pl.Tensor[[4096], pl.BF16],
    m_lm_head_weight: pl.Tensor[[32320, 4096], pl.BF16, pl.NZ],
    m_logit_row_indices: pl.Tensor[[32], pl.INT32],
    m_num_sampled: pl.Tensor[[1], pl.INT32],
    m_next_prefill_tokens: pl.Tensor[[1], pl.INT64],
    m_markov_w1: pl.Tensor[[129280, 256], pl.BF16],
    m_markov_w2: pl.Tensor[[129280, 256], pl.BF16],
    m_confidence_head_weight: pl.Tensor[[1, 4352], pl.FP32],
    m_hidden_window: pld.DistributedTensor[[128, 4096], pl.BF16],
    m_hidden_done: pld.DistributedTensor[[4, 1], pl.INT32],
    m_logits_window: pld.DistributedTensor[[32, 129280], pl.FP32],
    m_logits_done: pld.DistributedTensor[[4, 1], pl.INT32],
    s_history_ring: pl.InOut[pl.Tensor[[136, 12288], pl.BF16]],
    s_request_state: pl.InOut[pl.Tensor[[4], pl.INT64]],
    m_draft_token_ids_bank: pl.Tensor[[1, 5], pl.INT32],
    m_confidence_probs_bank: pl.Tensor[[1, 5], pl.FP32],
    s_my_rank: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor[[5], pl.INT32], pl.Tensor[[1, 5], pl.FP32], pl.Scalar[pl.TASK_ID], pl.Scalar[pl.TASK_ID]]:
    d_ring_view = s_history_ring[0:128, :]
    (d_ready_head, d_head_ready, d_local_cleanup) = drafter_inline(
        d_ring_view,
        d_main_proj_weight,
        d_main_norm_weight,
        s_request_state,
        d_last_sampled,
        d_embedding_weight,
        d_context_group_position_ids_bank,
        d_context_group_slot_mapping_bank,
        d_block_tables,
        d_query_group_position_ids_bank,
        d_query_group_slot_mapping_bank,
        d_context_group_freqs_cos_bank,
        d_context_group_freqs_sin_bank,
        d_query_freqs_cos_bank,
        d_query_freqs_sin_bank,
        d_query_group_freqs_cos_bank,
        d_query_group_freqs_sin_bank,
        d_hc_attn_fn,
        d_hc_attn_scale,
        d_hc_attn_base,
        d_attn_norm_w,
        d_wq_a,
        d_wq_b,
        d_wq_b_scale,
        d_wkv,
        d_gamma_cq,
        d_gamma_ckv,
        d_kv_caches,
        d_attn_sink,
        d_wo_a,
        d_wo_b,
        d_wo_b_scale,
        d_hc_ffn_fn,
        d_hc_ffn_scale,
        d_hc_ffn_base,
        d_ffn_norm_w,
        d_gate_w,
        d_gate_bias,
        d_tid2eid,
        d_routed_w1,
        d_routed_w1_scale,
        d_routed_w3,
        d_routed_w3_scale,
        d_routed_w2,
        d_routed_w2_scale,
        d_shared_w1,
        d_shared_w1_scale,
        d_shared_w3,
        d_shared_w3_scale,
        d_shared_w2,
        d_shared_w2_scale,
        d_hc_head_fn,
        d_hc_head_scale,
        d_hc_head_base,
        d_initial_hidden,
        d_intermediate_hidden,
        d_head_hidden,
        d_hidden_gather_window,
        d_hidden_gather_signal,
        d_attention_window,
        d_attention_signal,
        d_o_window,
        d_o_signal,
        d_recv_meta,
        d_recv_x,
        d_recv_aux,
        d_recv_route,
        d_arrived,
        d_data_arrived,
        d_routed_y_buf,
        d_combine_arrived,
        0,
        s_my_rank,
        s_my_rank,
        prior_cycle_gate,
    )
    (m_ready_ids, m_ready_confidence, m_token_output_ready) = markov_inline(
        d_ready_head,
        m_final_norm_weight,
        m_lm_head_weight,
        m_logit_row_indices,
        m_num_sampled,
        d_last_sampled,
        m_next_prefill_tokens,
        m_markov_w1,
        m_markov_w2,
        m_confidence_head_weight,
        m_draft_token_ids_bank,
        m_confidence_probs_bank,
        m_hidden_window,
        m_hidden_done,
        m_logits_window,
        m_logits_done,
        0,
        s_my_rank,
        d_head_ready,
    )
    bridge_ids_view = pl.reshape(m_ready_ids, [5])
    return bridge_ids_view, m_ready_confidence, d_local_cleanup, m_token_output_ready


@pl.jit.inline(auto_scope=False)
def prepare_draft_markov8(
    d_main_proj_weight: pl.Tensor[[4096, 12288], pl.BF16],
    d_main_norm_weight: pl.Tensor[[4096], pl.BF16],
    d_last_sampled: pl.Out[pl.Tensor[[1], pl.INT64]],
    d_embedding_weight: pl.Tensor[[129280, 4096], pl.BF16],
    d_context_group_position_ids_bank: pl.Tensor[[BANK_COUNT, 512], pl.INT32],
    d_context_group_slot_mapping_bank: pl.Tensor[[BANK_COUNT, 3, 512], pl.INT64],
    d_block_tables: pl.Tensor[[3, 1, 32768], pl.INT32],
    d_query_group_position_ids_bank: pl.Tensor[[BANK_COUNT, 80], pl.INT32],
    d_query_group_slot_mapping_bank: pl.Tensor[[BANK_COUNT, 3, 80], pl.INT64],
    d_context_group_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 512, 64], pl.BF16],
    d_context_group_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 512, 64], pl.BF16],
    d_query_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 20, 64], pl.BF16],
    d_query_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 20, 64], pl.BF16],
    d_query_group_freqs_cos_bank: pl.Tensor[[BANK_COUNT, 80, 64], pl.BF16],
    d_query_group_freqs_sin_bank: pl.Tensor[[BANK_COUNT, 80, 64], pl.BF16],
    d_hc_attn_fn: pl.Tensor[[72, 16384], pl.FP32],
    d_hc_attn_scale: pl.Tensor[[9], pl.FP32],
    d_hc_attn_base: pl.Tensor[[72], pl.FP32],
    d_attn_norm_w: pl.Tensor[[12288], pl.BF16],
    d_wq_a: pl.Tensor[[3, 4096, 1024], pl.BF16, pl.NZ],
    d_wq_b: pl.Tensor[[3, 1024, 4096], pl.INT8, pl.NZ],
    d_wq_b_scale: pl.Tensor[[3, 4096], pl.FP32],
    d_wkv: pl.Tensor[[12288, 512], pl.BF16],
    d_gamma_cq: pl.Tensor[[3072], pl.BF16],
    d_gamma_ckv: pl.Tensor[[1536], pl.BF16],
    d_kv_caches: pl.InOut[pl.Tensor[[3, 512, 32, 1, 512], pl.BF16]],
    d_attn_sink: pl.Tensor[[3, 8], pl.FP32],
    d_wo_a: pl.Tensor[[3, 1024, 4096], pl.BF16, pl.NZ],
    d_wo_b: pl.Tensor[[3, 4096, 1024], pl.INT8, pl.NZ],
    d_wo_b_scale: pl.Tensor[[12288], pl.FP32],
    d_hc_ffn_fn: pl.Tensor[[72, 16384], pl.FP32],
    d_hc_ffn_scale: pl.Tensor[[9], pl.FP32],
    d_hc_ffn_base: pl.Tensor[[72], pl.FP32],
    d_ffn_norm_w: pl.Tensor[[12288], pl.BF16],
    d_gate_w: pl.Tensor[[768, 4096], pl.FP32],
    d_gate_bias: pl.Tensor[[768], pl.FP32],
    d_tid2eid: pl.Tensor[[387840, 6], pl.INT32],
    d_routed_w1: pl.Tensor[[3, 257, 256, 4096], pl.INT8, pl.NZ],
    d_routed_w1_scale: pl.Tensor[[3, 257, 256], pl.FP32],
    d_routed_w3: pl.Tensor[[3, 257, 256, 4096], pl.INT8, pl.NZ],
    d_routed_w3_scale: pl.Tensor[[3, 257, 256], pl.FP32],
    d_routed_w2: pl.Tensor[[3, 257, 512, 2048], pl.INT8, pl.NZ],
    d_routed_w2_scale: pl.Tensor[[3, 257, 512], pl.FP32],
    d_shared_w1: pl.Tensor[[3, 256, 4096], pl.INT8, pl.NZ],
    d_shared_w1_scale: pl.Tensor[[3, 256], pl.FP32],
    d_shared_w3: pl.Tensor[[3, 256, 4096], pl.INT8, pl.NZ],
    d_shared_w3_scale: pl.Tensor[[3, 256], pl.FP32],
    d_shared_w2: pl.Tensor[[3, 512, 2048], pl.INT8, pl.NZ],
    d_shared_w2_scale: pl.Tensor[[3, 512], pl.FP32],
    d_hc_head_fn: pl.Tensor[[4, 16384], pl.FP32],
    d_hc_head_scale: pl.Tensor[[1], pl.FP32],
    d_hc_head_base: pl.Tensor[[4], pl.FP32],
    d_initial_hidden: pl.Out[pl.Tensor[[32, 4, 4096], pl.FP32]],
    d_intermediate_hidden: pl.Out[pl.Tensor[[3, 32, 4, 4096], pl.FP32]],
    d_head_hidden: pl.Out[pl.Tensor[[1, 5, 4096], pl.BF16]],
    d_tp8_o_window: pld.DistributedTensor[[80, 4096], pl.FP32],
    d_tp8_o_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    d_tp8_max_window: pld.DistributedTensor[[640, 8], pl.FP32],
    d_tp8_max_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    d_tp8_int8_window: pld.DistributedTensor[[640, 256], pl.INT8],
    d_tp8_int8_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    d_tp8_output_window: pld.DistributedTensor[[80, 512], pl.BF16],
    d_tp8_output_flags: pld.DistributedTensor[[8, 32], pl.INT32],
    m_final_norm_weight: pl.Tensor[[4096], pl.BF16],
    m_lm_head_weight: pl.Tensor[[16384, 4096], pl.BF16, pl.NZ],
    m_logit_row_indices: pl.Tensor[[32], pl.INT32],
    m_num_sampled: pl.Tensor[[1], pl.INT32],
    m_next_prefill_tokens: pl.Tensor[[1], pl.INT64],
    m_markov_w1: pl.Tensor[[129280, 256], pl.BF16],
    m_markov_w2: pl.Tensor[[16384, 256], pl.BF16],
    m_confidence_head_weight: pl.Tensor[[1, 4352], pl.FP32],
    m_hidden_window: pld.DistributedTensor[[32, 4096], pl.BF16],
    m8_seed: pld.DistributedTensor[[2, 8], pl.INT64],
    m8_ready: pld.DistributedTensor[[2, 32], pl.INT32],
    m8_pairs: pld.DistributedTensor[[80, 16], pl.FP32],
    m8_arrivals: pld.DistributedTensor[[80, 32], pl.INT32],
    m8_epoch: pl.Tensor[[1], pl.INT32],
    m8_slot: pl.Scalar[pl.INDEX],
    m8_metadata_ready: pl.Scalar[pl.TASK_ID],
    s_history_ring: pl.InOut[pl.Tensor[[136, 12288], pl.BF16]],
    s_request_state: pl.InOut[pl.Tensor[[4], pl.INT64]],
    m_draft_token_ids_bank: pl.Tensor[[1, 5], pl.INT32],
    m_confidence_probs_bank: pl.Tensor[[1, 5], pl.FP32],
    s_my_rank: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor[[5], pl.INT32], pl.Tensor[[1, 5], pl.FP32], pl.Scalar[pl.TASK_ID], pl.Scalar[pl.TASK_ID]]:
    d_ring_view = s_history_ring[0:128, :]
    (d_ready_head, d_head_ready, d_local_cleanup) = device_drafter_tp8(
        d_ring_view,
        d_main_proj_weight,
        d_main_norm_weight,
        s_request_state,
        d_last_sampled,
        d_embedding_weight,
        d_context_group_position_ids_bank,
        d_context_group_slot_mapping_bank,
        d_block_tables,
        d_query_group_position_ids_bank,
        d_query_group_slot_mapping_bank,
        d_context_group_freqs_cos_bank,
        d_context_group_freqs_sin_bank,
        d_query_freqs_cos_bank,
        d_query_freqs_sin_bank,
        d_query_group_freqs_cos_bank,
        d_query_group_freqs_sin_bank,
        d_hc_attn_fn,
        d_hc_attn_scale,
        d_hc_attn_base,
        d_attn_norm_w,
        d_wq_a,
        d_wq_b,
        d_wq_b_scale,
        d_wkv,
        d_gamma_cq,
        d_gamma_ckv,
        d_kv_caches,
        d_attn_sink,
        d_wo_a,
        d_wo_b,
        d_wo_b_scale,
        d_hc_ffn_fn,
        d_hc_ffn_scale,
        d_hc_ffn_base,
        d_ffn_norm_w,
        d_gate_w,
        d_gate_bias,
        d_tid2eid,
        d_routed_w1,
        d_routed_w1_scale,
        d_routed_w3,
        d_routed_w3_scale,
        d_routed_w2,
        d_routed_w2_scale,
        d_shared_w1,
        d_shared_w1_scale,
        d_shared_w3,
        d_shared_w3_scale,
        d_shared_w2,
        d_shared_w2_scale,
        d_hc_head_fn,
        d_hc_head_scale,
        d_hc_head_base,
        d_initial_hidden,
        d_intermediate_hidden,
        d_head_hidden,
        d_tp8_o_window,
        d_tp8_o_flags,
        d_tp8_max_window,
        d_tp8_max_flags,
        d_tp8_int8_window,
        d_tp8_int8_flags,
        d_tp8_output_window,
        d_tp8_output_flags,
        s_my_rank,
        prior_cycle_gate,
    )
    if s_my_rank == 0:
        normalized = pl.create_tensor([DSPARK_MOE_TOKENS, D], dtype=pl.BF16)
        norm_ready = normalize_head_hidden(d_ready_head, m_final_norm_weight, normalized, d_head_ready)
        publish_base_rows(normalized, m_logit_row_indices, d_last_sampled, m_num_sampled,
                          m_next_prefill_tokens, m8_epoch, m_hidden_window, m8_seed,
                          m8_ready, m8_slot, norm_ready)
    all_ids = pl.create_tensor([1, 5], dtype=pl.INT32)
    scored = run_head8_steps(m_lm_head_weight, m_hidden_window,
                              m_markov_w1, m_markov_w2, all_ids, m8_epoch,
                              m8_seed, m8_ready, m8_pairs, m8_arrivals, s_my_rank, m8_slot, m8_metadata_ready)
    m_token_output_ready = m8_confidence_outputs(d_ready_head, m_markov_w1, m_confidence_head_weight,
                                                all_ids, m_draft_token_ids_bank, m_confidence_probs_bank,
                                                m8_seed, m8_slot, s_my_rank, scored)
    bridge_ids_view = pl.reshape(m_draft_token_ids_bank, [5])
    return bridge_ids_view, m_confidence_probs_bank, d_local_cleanup, m_token_output_ready


@pl.jit.inline(auto_scope=False)
def populate_local_draft_history(
    target_taps: pl.Tensor[[6,12288],pl.BF16],
    input_state: pl.Tensor[[4],pl.INT64],
    next_state: pl.Tensor[[4],pl.INT64],
    ring: pl.InOut[pl.Tensor[[136,12288],pl.BF16]],
    rank: pl.Scalar[pl.INT32],
    state_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Assemble the same committed rows locally after the existing state exchange."""
    with pl.spmd(6*24,name_hint="draft_tp8_commit_peer_history",deps=[state_ready]) as retired:
        if rank != 0:
            pl.system.cacheinvalid()
            item=pl.tile.get_block_idx()
            row=item//24
            col=item%24*512
            previous=pl.read(input_state,[1])
            committed=pl.read(next_state,[1])-previous
            if row<committed:
                idx=(previous+1+row)%128
                feat=pl.load(target_taps,[row,col],[1,512])
                pl.store(feat,[idx,col],ring)
    return retired


@pl.jit.inline(auto_scope=False)
def target_cycle_banks(
    s_control: pl.Tensor[[2, 16], pl.INT64],
    s_emitted: pl.Tensor[[2, 6], pl.INT64],
    s_sampled_ids: pl.Tensor[[2, 6, 8], pl.INT32],
    s_dspark_target_hidden_out: pl.Tensor[[2, 6, 12288], pl.BF16],
    native_cycle: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor[[16], pl.INT64], pl.Tensor[[6], pl.INT64],
           pl.Tensor[[6, 8], pl.INT32], pl.Tensor[[6, 12288], pl.BF16]]:
    s_control_slice = s_control[native_cycle : native_cycle + 1, :]
    s_control_bank: pl.Tensor[[16], pl.INT64] = pl.reshape(s_control_slice, [16])
    s_emitted_slice = s_emitted[native_cycle : native_cycle + 1, :]
    s_emitted_bank: pl.Tensor[[6], pl.INT64] = pl.reshape(s_emitted_slice, [6])
    s_sampled_ids_slice = s_sampled_ids[native_cycle : native_cycle + 1, :, :]
    s_sampled_ids_bank: pl.Tensor[[6, 8], pl.INT32] = pl.reshape(s_sampled_ids_slice, [6, 8])
    s_dspark_target_hidden_out_slice = s_dspark_target_hidden_out[native_cycle : native_cycle + 1, :, :]
    s_dspark_target_hidden_out_bank: pl.Tensor[[6, 12288], pl.BF16] = pl.reshape(
        s_dspark_target_hidden_out_slice, [6, 12288]
    )
    return s_control_bank, s_emitted_bank, s_sampled_ids_bank, s_dspark_target_hidden_out_bank


@pl.jit.inline(auto_scope=False)
def markov_cycle_banks(
    m_draft_token_ids: pl.Tensor[[2, 1, 5], pl.INT32],
    m_confidence_probs: pl.Tensor[[2, 1, 5], pl.FP32],
    native_cycle: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor[[1, 5], pl.INT32], pl.Tensor[[1, 5], pl.FP32]]:
    m_draft_token_ids_slice = m_draft_token_ids[native_cycle : native_cycle + 1, :, :]
    m_draft_token_ids_bank: pl.Tensor[[1, 5], pl.INT32] = pl.reshape(m_draft_token_ids_slice, [1, 5])
    m_confidence_probs_slice = m_confidence_probs[native_cycle : native_cycle + 1, :, :]
    m_confidence_probs_bank: pl.Tensor[[1, 5], pl.FP32] = pl.reshape(m_confidence_probs_slice, [1, 5])
    return m_draft_token_ids_bank, m_confidence_probs_bank


@pl.jit.inline(auto_scope=False)
def full_cycle_reuse_gate(
    s_bridge_epoch: pl.Tensor[[1], pl.INT32],
    s_cycle_arrival_signal: pld.DistributedTensor[[8, 32], pl.INT32],
    s_my_rank: pl.Scalar[pl.INT32],
    d_local_cleanup: pl.Scalar[pl.TASK_ID],
    m_token_output_ready: pl.Scalar[pl.TASK_ID],
    s6_local_reuse_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    local_cycle_exit = pl.system.task_dummy(deps=[d_local_cleanup, m_token_output_ready, s6_local_reuse_ready])
    prior_cycle_gate = eight_rank_reuse_gate(
        s_bridge_epoch, s_cycle_arrival_signal, s_my_rank, local_cycle_exit
    )
    return prior_cycle_gate
