# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

import pypto.language.distributed as pld

from .hc_head import hc_head

from .lm_head import (
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    SAMPLED_IDS_PAD,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
    lm_head_with_sampling,
)

from .rmsnorm import rms_norm

from .decode_swa import (
    B,
    BLOCK_SIZE,
    D,
    HC_DIM,
    HC_MULT,
    H,
    HEAD_DIM,
    MAX_SEQ_LEN,
    MIX_HC,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    ORI_TABLE_MAX_BLOCKS,
    Q_LORA,
    ROPE_HEAD_DIM,
    T,
    WIN as SWA_WIN,
    attention_swa,
    prepare_swa_metadata,
    SWA_ROPE_ROWS,
)

from .decode_hca import (
    CMP_MAX_BLOCKS as HCA_CMP_MAX_BLOCKS,
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    COMPRESS_STATE_BLOCK_SIZE as HCA_COMPRESS_STATE_BLOCK_SIZE,
    COMPRESS_STATE_DIM as HCA_COMPRESS_STATE_DIM,
    COMPRESS_STATE_MAX_BLOCKS as HCA_COMPRESS_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    attention_hca,
    prepare_hca_metadata,
    HCA_ROPE_ROWS,
)

from .decode_csa import (
    CMP_MAX_BLOCKS as CSA_CMP_MAX_BLOCKS,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    IDX_CACHE_MAX_BLOCKS as CSA_IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM as CSA_IDX_HEAD_DIM,
    IDX_N_HEADS as CSA_IDX_N_HEADS,
    INNER_OUT_DIM as CSA_INNER_OUT_DIM,
    INNER_STATE_BLOCK_SIZE as CSA_INNER_STATE_BLOCK_SIZE,
    INNER_STATE_DIM as CSA_INNER_STATE_DIM,
    INNER_STATE_MAX_BLOCKS as CSA_INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAIN_STATE_BLOCK_SIZE as CSA_MAIN_STATE_BLOCK_SIZE,
    MAIN_STATE_DIM as CSA_MAIN_STATE_DIM,
    MAIN_STATE_MAX_BLOCKS as CSA_MAIN_STATE_MAX_BLOCKS,
    OPROJ_REDUCE_ROWS as CSA_OPROJ_REDUCE_ROWS,
    attention_csa,
    prepare_csa_metadata,
    CSA_ROPE_ROWS,
    IDX_ROPE_ROWS,
    clear_csa_oproj_signals,
    TOK_Q_ROWS,
    TOK_O_COLS,
)

from .config import FLASH as MODEL_CONFIG, _parse_int_argv

from .decode_prepare import N_CACHE_GROUPS, build_decode_metadata, pack_x_hc_rows

from .tp_embedding import VOCAB_PER_TP as EMBED_VOCAB_PER_TP, clear_embedding_signals, tp_lookup_embedding

from .target_bridge import device_input_bridge

from .target_commit import device_commit_inline

from .target_state import publish_next_request_state

from .moe_rsag import (
    MOE_INTER,
    N_BANK,
    N_EXPERTS,
    N_RANKS,
    REDUCE_WINDOW_ROWS,
    REDUCE_SHARD_D,
    TOPK,
    VOCAB,
    clear_moe_signals,
    moe,
)

assert HCA_CMP_MAX_BLOCKS == CSA_CMP_MAX_BLOCKS

FWD_ORI_BLOCK_NUM_DYN = pl.dynamic("NATIVE_S_FWD_ORI_BLOCK_NUM_DYN")

FWD_HCA_CMP_BLOCK_NUM_DYN = pl.dynamic("NATIVE_S_FWD_HCA_CMP_BLOCK_NUM_DYN")

FWD_CSA_CMP_BLOCK_NUM_DYN = pl.dynamic("NATIVE_S_FWD_CSA_CMP_BLOCK_NUM_DYN")

FWD_IDX_BLOCK_NUM_DYN = pl.dynamic("NATIVE_S_FWD_IDX_BLOCK_NUM_DYN")

FWD_HCA_STATE_BLOCK_NUM_DYN = pl.dynamic("NATIVE_S_FWD_HCA_STATE_BLOCK_NUM_DYN")

FWD_CSA_STATE_BLOCK_NUM_DYN = pl.dynamic("NATIVE_S_FWD_CSA_STATE_BLOCK_NUM_DYN")

FWD_INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("NATIVE_S_FWD_INNER_STATE_BLOCK_NUM_DYN")

MODEL_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers

FWD_NUM_LAYERS = 43

DSPARK_TARGET_LAYER_IDS = (40, 41, 42)

DSPARK_MAIN_HIDDEN_DIM = 3 * D

MOE_BANKS = _parse_int_argv("--moe-banks", 1)

CSA_NUM_LAYERS = 21

HCA_NUM_LAYERS = 20

HCA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // HCA_COMPRESS_RATIO

CSA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // CSA_COMPRESS_RATIO

FWD_LAST_LAYER = FWD_NUM_LAYERS - 1

LM_HEAD_COMM_EPOCH = 1

assert MODEL_NUM_LAYERS == 43, "DeepSeek-V4 Flash hidden layer count changed"


@pl.jit.inline(auto_scope=False)
def decode_fwd(
    hc_attn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[FWD_NUM_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    # The CSA layers shard the projection one group per card; SWA and HCA keep
    # reading the replicated wo_a / wo_b above.
    wo_a_shard: pl.Tensor[[FWD_NUM_LAYERS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[FWD_NUM_LAYERS * D, O_LORA], pl.INT8],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32
        ]
    ],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[[FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]
    ],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[[FWD_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]
    ],
    hca_cmp_kv: pl.InOut[
        pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    csa_cmp_kv: pl.InOut[
        pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    idx_kv_cache: pl.InOut[
        pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[MOE_BANKS * N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[MOE_BANKS * N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[MOE_BANKS * N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[MOE_BANKS * N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[MOE_BANKS * N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[MOE_BANKS * N_BANK, D], pl.FP32],
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[B, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[B, HCA_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[T, SWA_WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    input_ids: pl.Tensor[[T], pl.INT64],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_temperatures: pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    dspark_target_hidden_out: pl.Out[pl.Tensor[[T, DSPARK_MAIN_HIDDEN_DIM], pl.BF16]],
    x_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, REDUCE_SHARD_D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
    gather_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, REDUCE_SHARD_D], pl.BF16],
    gather_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
    oproj_reduce_window: pld.DistributedTensor[[CSA_OPROJ_REDUCE_ROWS, D], pl.FP32],
    oproj_reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    oproj_sync_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    tok_q_window: pld.DistributedTensor[[TOK_Q_ROWS, HEAD_DIM], pl.BF16],
    tok_q_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    tok_o_window: pld.DistributedTensor[[T, TOK_O_COLS], pl.BF16],
    tok_o_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    swa_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    swa_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    compressed_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [1, 0, 0]
    )
    compressed_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [1, 0, 0]
    )
    swa_freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        swa_cos_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )
    swa_freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        swa_sin_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )
    compressed_freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        compressed_cos_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )
    compressed_freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        compressed_sin_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )
    fwd_swa_rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    fwd_swa_rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    fwd_swa_q_rope_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_swa_q_rope_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_swa_q_rope_swap_idx = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.INT32)
    fwd_swa_out_rope_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_swa_out_rope_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_swa_out_rope_swap_idx = pl.create_tensor([SWA_ROPE_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    prepare_swa_metadata(
        swa_freqs_cos,
        swa_freqs_sin,
        position_ids,
        fwd_swa_rope_cos_t,
        fwd_swa_rope_sin_t,
        fwd_swa_q_rope_cos_il,
        fwd_swa_q_rope_sin_signed,
        fwd_swa_q_rope_swap_idx,
        fwd_swa_out_rope_cos_il,
        fwd_swa_out_rope_sin_signed,
        fwd_swa_out_rope_swap_idx,
    )
    fwd_csa_rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    fwd_csa_rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    fwd_csa_step_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_csa_step_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_csa_cmp_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_csa_cmp_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_csa_q_rope_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_csa_q_rope_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_csa_q_rope_swap_idx = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.INT32)
    fwd_csa_out_rope_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_csa_out_rope_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_csa_out_rope_swap_idx = pl.create_tensor([CSA_ROPE_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    fwd_csa_idx_rope_swap_idx = pl.create_tensor([IDX_ROPE_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    prepare_csa_metadata(
        compressed_freqs_cos,
        compressed_freqs_sin,
        position_ids,
        fwd_csa_rope_cos_t,
        fwd_csa_rope_sin_t,
        fwd_csa_step_cos_il,
        fwd_csa_step_sin_signed,
        fwd_csa_cmp_cos_il,
        fwd_csa_cmp_sin_signed,
        fwd_csa_q_rope_cos_il,
        fwd_csa_q_rope_sin_signed,
        fwd_csa_q_rope_swap_idx,
        fwd_csa_out_rope_cos_il,
        fwd_csa_out_rope_sin_signed,
        fwd_csa_out_rope_swap_idx,
        fwd_csa_idx_rope_swap_idx,
    )
    fwd_hca_rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    fwd_hca_rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    fwd_hca_cmp_cos_il = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_hca_cmp_sin_signed = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_hca_cmp_seq_lens = pl.create_tensor([T], dtype=pl.INT32)
    fwd_hca_q_rope_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_hca_q_rope_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_hca_q_rope_swap_idx = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.INT32)
    fwd_hca_out_rope_cos_il = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_hca_out_rope_sin_signed = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.FP32)
    fwd_hca_out_rope_swap_idx = pl.create_tensor([HCA_ROPE_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    prepare_hca_metadata(
        compressed_freqs_cos,
        compressed_freqs_sin,
        position_ids,
        kv_seq_lens,
        fwd_hca_rope_cos_t,
        fwd_hca_rope_sin_t,
        fwd_hca_cmp_cos_il,
        fwd_hca_cmp_sin_signed,
        fwd_hca_cmp_seq_lens,
        fwd_hca_q_rope_cos_il,
        fwd_hca_q_rope_sin_signed,
        fwd_hca_q_rope_swap_idx,
        fwd_hca_out_rope_cos_il,
        fwd_hca_out_rope_sin_signed,
        fwd_hca_out_rope_swap_idx,
    )
    nt = pl.cast(0, pl.INT32)
    for owner_rank in pl.range(N_RANKS):
        nt = pl.max(nt, pl.read(num_tokens_per_owner, [owner_rank]))
    hc_attn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0]
    )
    hc_attn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [0 * 3])
    hc_attn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [0 * MIX_HC])
    attn_norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [0 * D])
    wq_a_l0: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [0 * D, 0])
    wq_b_l0: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
        wq_b, [Q_LORA, H * HEAD_DIM], [0 * Q_LORA, 0]
    )
    wq_b_scale_l0: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
        wq_b_scale, [H * HEAD_DIM], [0 * H * HEAD_DIM]
    )
    wkv_l0: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [0 * D, 0])
    gamma_cq_l0: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [0 * Q_LORA])
    gamma_ckv_l0: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [0 * HEAD_DIM])
    ori_block_num = pl.tensor.dim(kv_cache, 0) // FWD_NUM_LAYERS
    hca_state_block_num = pl.tensor.dim(hca_compress_state, 0) // HCA_NUM_LAYERS
    csa_state_block_num = pl.tensor.dim(csa_compress_state, 0) // CSA_NUM_LAYERS
    inner_state_block_num = pl.tensor.dim(csa_inner_compress_state, 0) // CSA_NUM_LAYERS
    hca_cmp_block_num = pl.tensor.dim(hca_cmp_kv, 0) // HCA_NUM_LAYERS
    csa_cmp_block_num = pl.tensor.dim(csa_cmp_kv, 0) // CSA_NUM_LAYERS
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0) // CSA_NUM_LAYERS
    kv_cache_l0 = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [0 * ori_block_num, 0, 0, 0])
    attn_sink_l0: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [0 * H])
    wo_a_l0: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [0 * O_GROUPS, 0, 0]
    )
    wo_a_shard_l0: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a_shard, [1, O_LORA, O_GROUP_IN], [0, 0, 0]
    )
    wo_b_shard_l0: pl.Tensor[[D, O_LORA], pl.INT8] = pl.slice(wo_b_shard, [D, O_LORA], [0 * D, 0])
    wo_b_l0: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [0 * D, 0])
    wo_b_scale_l0: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [0 * D])
    hc_ffn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0]
    )
    hc_ffn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [0 * 3])
    hc_ffn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [0 * MIX_HC])
    norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [0 * D])
    gate_w_l0: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS, D], [0 * N_EXPERTS, 0])
    gate_bias_l0: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS], [0 * N_EXPERTS])
    tid2eid_l0: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [0 * VOCAB, 0])
    routed_w1_l0: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
        routed_w1, [N_BANK, MOE_INTER, D], [(0 % MOE_BANKS) * N_BANK, 0, 0]
    )
    routed_w1_scale_l0: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
        routed_w1_scale, [N_BANK, MOE_INTER], [(0 % MOE_BANKS) * N_BANK, 0]
    )
    routed_w3_l0: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
        routed_w3, [N_BANK, MOE_INTER, D], [(0 % MOE_BANKS) * N_BANK, 0, 0]
    )
    routed_w3_scale_l0: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
        routed_w3_scale, [N_BANK, MOE_INTER], [(0 % MOE_BANKS) * N_BANK, 0]
    )
    routed_w2_l0: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8] = pl.slice(
        routed_w2, [N_BANK, D, MOE_INTER], [(0 % MOE_BANKS) * N_BANK, 0, 0]
    )
    routed_w2_scale_l0: pl.Tensor[[N_BANK, D], pl.FP32] = pl.slice(
        routed_w2_scale, [N_BANK, D], [(0 % MOE_BANKS) * N_BANK, 0]
    )
    hc_attn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0]
    )
    hc_attn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [1 * 3])
    hc_attn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [1 * MIX_HC])
    attn_norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [1 * D])
    wq_a_l1: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [1 * D, 0])
    wq_b_l1: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
        wq_b, [Q_LORA, H * HEAD_DIM], [1 * Q_LORA, 0]
    )
    wq_b_scale_l1: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
        wq_b_scale, [H * HEAD_DIM], [1 * H * HEAD_DIM]
    )
    wkv_l1: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [1 * D, 0])
    gamma_cq_l1: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [1 * Q_LORA])
    gamma_ckv_l1: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [1 * HEAD_DIM])
    kv_cache_l1 = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [1 * ori_block_num, 0, 0, 0])
    attn_sink_l1: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [1 * H])
    wo_a_l1: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [1 * O_GROUPS, 0, 0]
    )
    wo_a_shard_l1: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a_shard, [1, O_LORA, O_GROUP_IN], [1, 0, 0]
    )
    wo_b_shard_l1: pl.Tensor[[D, O_LORA], pl.INT8] = pl.slice(wo_b_shard, [D, O_LORA], [1 * D, 0])
    wo_b_l1: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [1 * D, 0])
    wo_b_scale_l1: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [1 * D])
    hc_ffn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0]
    )
    hc_ffn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [1 * 3])
    hc_ffn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [1 * MIX_HC])
    norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [1 * D])
    gate_w_l1: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS, D], [1 * N_EXPERTS, 0])
    gate_bias_l1: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS], [1 * N_EXPERTS])
    tid2eid_l1: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [1 * VOCAB, 0])
    routed_w1_l1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
        routed_w1, [N_BANK, MOE_INTER, D], [(1 % MOE_BANKS) * N_BANK, 0, 0]
    )
    routed_w1_scale_l1: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
        routed_w1_scale, [N_BANK, MOE_INTER], [(1 % MOE_BANKS) * N_BANK, 0]
    )
    routed_w3_l1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
        routed_w3, [N_BANK, MOE_INTER, D], [(1 % MOE_BANKS) * N_BANK, 0, 0]
    )
    routed_w3_scale_l1: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
        routed_w3_scale, [N_BANK, MOE_INTER], [(1 % MOE_BANKS) * N_BANK, 0]
    )
    routed_w2_l1: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8] = pl.slice(
        routed_w2, [N_BANK, D, MOE_INTER], [(1 % MOE_BANKS) * N_BANK, 0, 0]
    )
    routed_w2_scale_l1: pl.Tensor[[N_BANK, D], pl.FP32] = pl.slice(
        routed_w2_scale, [N_BANK, D], [(1 % MOE_BANKS) * N_BANK, 0]
    )
    x_attn0: pl.Tensor[[T, HC_MULT, D], pl.FP32] = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    x_attn1: pl.Tensor[[T, HC_MULT, D], pl.FP32] = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32] = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    with pl.scope():
        attention_swa(
            x_hc,
            gate_w_l0,
            hc_attn_fn_l0,
            hc_attn_scale_l0,
            hc_attn_base_l0,
            attn_norm_w_l0,
            wq_a_l0,
            wq_b_l0,
            wq_b_scale_l0,
            wkv_l0,
            gamma_cq_l0,
            gamma_ckv_l0,
            fwd_swa_rope_cos_t,
            fwd_swa_rope_sin_t,
            kv_cache_l0,
            swa_slot_mapping,
            swa_indices,
            swa_lens,
            position_ids,
            attn_sink_l0,
            wo_a_shard_l0,
            wo_b_shard_l0,
            wo_b_scale_l0,
            x_attn0,
            oproj_reduce_window,
            oproj_reduce_signal,
            oproj_sync_signal,
            my_rank,
            pl.cast(1, pl.INT32),
            fwd_swa_q_rope_cos_il,
            fwd_swa_q_rope_sin_signed,
            fwd_swa_q_rope_swap_idx,
            fwd_swa_out_rope_cos_il,
            fwd_swa_out_rope_sin_signed,
            fwd_swa_out_rope_swap_idx,
        )
    with pl.scope():
        moe(
            x_attn0,
            hc_ffn_fn_l0,
            hc_ffn_scale_l0,
            hc_ffn_base_l0,
            norm_w_l0,
            gate_w_l0,
            gate_bias_l0,
            tid2eid_l0,
            input_ids,
            routed_w1_l0,
            routed_w1_scale_l0,
            routed_w3_l0,
            routed_w3_scale_l0,
            routed_w2_l0,
            routed_w2_scale_l0,
            hidden,
            reduce_window,
            reduce_signal,
            gather_window,
            gather_signal,
            pl.cast(0, pl.INT32),
            nt,
            my_rank,
            pl.cast(1, pl.INT32),
        )
    with pl.scope():
        attention_swa(
            hidden,
            gate_w_l1,
            hc_attn_fn_l1,
            hc_attn_scale_l1,
            hc_attn_base_l1,
            attn_norm_w_l1,
            wq_a_l1,
            wq_b_l1,
            wq_b_scale_l1,
            wkv_l1,
            gamma_cq_l1,
            gamma_ckv_l1,
            fwd_swa_rope_cos_t,
            fwd_swa_rope_sin_t,
            kv_cache_l1,
            swa_slot_mapping,
            swa_indices,
            swa_lens,
            position_ids,
            attn_sink_l1,
            wo_a_shard_l1,
            wo_b_shard_l1,
            wo_b_scale_l1,
            x_attn1,
            oproj_reduce_window,
            oproj_reduce_signal,
            oproj_sync_signal,
            my_rank,
            pl.cast(2, pl.INT32),
            fwd_swa_q_rope_cos_il,
            fwd_swa_q_rope_sin_signed,
            fwd_swa_q_rope_swap_idx,
            fwd_swa_out_rope_cos_il,
            fwd_swa_out_rope_sin_signed,
            fwd_swa_out_rope_swap_idx,
        )
    with pl.scope():
        moe(
            x_attn1,
            hc_ffn_fn_l1,
            hc_ffn_scale_l1,
            hc_ffn_base_l1,
            norm_w_l1,
            gate_w_l1,
            gate_bias_l1,
            tid2eid_l1,
            input_ids,
            routed_w1_l1,
            routed_w1_scale_l1,
            routed_w3_l1,
            routed_w3_scale_l1,
            routed_w2_l1,
            routed_w2_scale_l1,
            hidden,
            reduce_window,
            reduce_signal,
            gather_window,
            gather_signal,
            pl.cast(1, pl.INT32),
            nt,
            my_rank,
            pl.cast(2, pl.INT32),
        )
    for loop_i in pl.range(HCA_NUM_LAYERS):
        csa_layer: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 2, pl.INT32)
        hca_layer: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 3, pl.INT32)
        csa_moe_epoch: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 3, pl.INT32)
        # 1-based o-projection call id. Every attention layer all-reduces its
        # projection, so this counts all of them -- the two leading SWA layers
        # take 1 and 2 and the last CSA layer below takes 2 * HCA_NUM_LAYERS + 3,
        # the same numbering the MoE epoch beside it uses.
        csa_oproj_epoch: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 3, pl.INT32)
        # One CSA layer per iteration, so the token-shard windows advance by one
        # while oproj's epoch advances by two (it is shared with the HCA layer).
        csa_tok_epoch: pl.Scalar[pl.INT32] = pl.cast(loop_i + 1, pl.INT32)
        hca_oproj_epoch: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 4, pl.INT32)
        hca_moe_epoch: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 4, pl.INT32)
        x_attn_csa: pl.Tensor[[T, HC_MULT, D], pl.FP32] = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
        x_attn_hca: pl.Tensor[[T, HC_MULT, D], pl.FP32] = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
        hidden_mid: pl.Tensor[[T, HC_MULT, D], pl.FP32] = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
        hc_attn_fn_csa: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [csa_layer * MIX_HC, 0]
        )
        hc_attn_scale_csa: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [csa_layer * 3])
        hc_attn_base_csa: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [csa_layer * MIX_HC]
        )
        attn_norm_w_csa: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [csa_layer * D])
        wq_a_csa: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [csa_layer * D, 0])
        wq_b_csa: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [csa_layer * Q_LORA, 0]
        )
        wq_b_scale_csa: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [csa_layer * H * HEAD_DIM]
        )
        wkv_csa: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [csa_layer * D, 0])
        gamma_cq_csa: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [csa_layer * Q_LORA])
        gamma_ckv_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [csa_layer * HEAD_DIM]
        )
        kv_cache_csa = pl.slice(
            kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer * ori_block_num, 0, 0, 0]
        )
        attn_sink_csa: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [csa_layer * H])
        wo_a_csa: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
            wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [csa_layer * O_GROUPS, 0, 0]
        )
        wo_b_csa: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(
            wo_b, [D, O_GROUPS * O_LORA], [csa_layer * D, 0]
        )
        wo_b_scale_csa: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [csa_layer * D])
        wo_a_shard_csa: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
            wo_a_shard, [1, O_LORA, O_GROUP_IN], [csa_layer, 0, 0]
        )
        wo_b_shard_csa: pl.Tensor[[D, O_LORA], pl.INT8] = pl.slice(
            wo_b_shard, [D, O_LORA], [csa_layer * D, 0]
        )
        csa_cmp_wkv_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
            csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [loop_i * CSA_MAIN_OUT_DIM, 0]
        )
        csa_cmp_wgate_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
            csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [loop_i * CSA_MAIN_OUT_DIM, 0]
        )
        csa_cmp_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(
            csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [loop_i * CSA_COMPRESS_RATIO, 0]
        )
        csa_cmp_norm_w_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            csa_cmp_norm_w, [HEAD_DIM], [loop_i * HEAD_DIM]
        )
        csa_compress_state_csa = pl.slice(
            csa_compress_state,
            [csa_state_block_num, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
            [loop_i * csa_state_block_num, 0, 0],
        )
        csa_idx_wq_b_csa: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8] = pl.slice(
            csa_idx_wq_b, [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [loop_i * Q_LORA, 0]
        )
        csa_idx_wq_b_scale_csa: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32] = pl.slice(
            csa_idx_wq_b_scale,
            [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
            [loop_i * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
        )
        csa_weights_proj_csa: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16] = pl.slice(
            csa_weights_proj, [D, CSA_IDX_N_HEADS], [loop_i * D, 0]
        )
        csa_hadamard_idx_csa: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(
            csa_hadamard_idx, [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], [loop_i * CSA_IDX_HEAD_DIM, 0]
        )
        csa_inner_wkv_csa: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(
            csa_inner_wkv, [CSA_INNER_OUT_DIM, D], [loop_i * CSA_INNER_OUT_DIM, 0]
        )
        csa_inner_wgate_csa: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(
            csa_inner_wgate, [CSA_INNER_OUT_DIM, D], [loop_i * CSA_INNER_OUT_DIM, 0]
        )
        csa_inner_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32] = pl.slice(
            csa_inner_ape, [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], [loop_i * CSA_COMPRESS_RATIO, 0]
        )
        csa_inner_norm_w_csa: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(
            csa_inner_norm_w, [CSA_IDX_HEAD_DIM], [loop_i * CSA_IDX_HEAD_DIM]
        )
        csa_inner_compress_state_csa = pl.slice(
            csa_inner_compress_state,
            [inner_state_block_num, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
            [loop_i * inner_state_block_num, 0, 0],
        )
        cmp_kv_csa = pl.slice(
            csa_cmp_kv,
            [csa_cmp_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            [loop_i * csa_cmp_block_num, 0, 0, 0],
        )
        idx_kv_cache_csa = pl.slice(
            idx_kv_cache,
            [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
            [loop_i * idx_block_num, 0, 0, 0],
        )
        idx_kv_scale_csa = pl.slice(
            idx_kv_scale, [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], [loop_i * idx_block_num, 0, 0, 0]
        )
        hc_ffn_fn_csa: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [csa_layer * MIX_HC, 0]
        )
        hc_ffn_scale_csa: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [csa_layer * 3])
        hc_ffn_base_csa: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [csa_layer * MIX_HC])
        norm_w_csa: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [csa_layer * D])
        gate_w_csa: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_w, [N_EXPERTS, D], [csa_layer * N_EXPERTS, 0]
        )
        gate_bias_csa: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            gate_bias, [N_EXPERTS], [csa_layer * N_EXPERTS]
        )
        tid2eid_csa: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(
            tid2eid, [VOCAB, TOPK], [csa_layer * VOCAB, 0]
        )
        routed_w1_csa: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w1, [N_BANK, MOE_INTER, D], [(csa_layer % MOE_BANKS) * N_BANK, 0, 0]
        )
        routed_w1_scale_csa: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
            routed_w1_scale, [N_BANK, MOE_INTER], [(csa_layer % MOE_BANKS) * N_BANK, 0]
        )
        routed_w3_csa: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w3, [N_BANK, MOE_INTER, D], [(csa_layer % MOE_BANKS) * N_BANK, 0, 0]
        )
        routed_w3_scale_csa: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
            routed_w3_scale, [N_BANK, MOE_INTER], [(csa_layer % MOE_BANKS) * N_BANK, 0]
        )
        routed_w2_csa: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8] = pl.slice(
            routed_w2, [N_BANK, D, MOE_INTER], [(csa_layer % MOE_BANKS) * N_BANK, 0, 0]
        )
        routed_w2_scale_csa: pl.Tensor[[N_BANK, D], pl.FP32] = pl.slice(
            routed_w2_scale, [N_BANK, D], [(csa_layer % MOE_BANKS) * N_BANK, 0]
        )
        with pl.scope():
            attention_csa(
                hidden,
                gate_w_csa,
                hc_attn_fn_csa,
                hc_attn_scale_csa,
                hc_attn_base_csa,
                attn_norm_w_csa,
                wq_a_csa,
                wq_b_csa,
                wq_b_scale_csa,
                wkv_csa,
                gamma_cq_csa,
                gamma_ckv_csa,
                fwd_csa_rope_cos_t,
                fwd_csa_rope_sin_t,
                fwd_csa_step_cos_il,
                fwd_csa_step_sin_signed,
                fwd_csa_cmp_cos_il,
                fwd_csa_cmp_sin_signed,
                csa_cmp_wkv_csa,
                csa_cmp_wgate_csa,
                csa_cmp_ape_csa,
                csa_cmp_norm_w_csa,
                csa_compress_state_csa,
                csa_compress_state_block_table,
                csa_idx_wq_b_csa,
                csa_idx_wq_b_scale_csa,
                csa_weights_proj_csa,
                csa_hadamard_idx_csa,
                csa_inner_wkv_csa,
                csa_inner_wgate_csa,
                csa_inner_ape_csa,
                csa_inner_norm_w_csa,
                csa_inner_compress_state_csa,
                csa_inner_compress_state_block_table,
                kv_cache_csa,
                cmp_kv_csa,
                csa_cmp_block_table,
                idx_kv_cache_csa,
                idx_kv_scale_csa,
                idx_block_table,
                ori_slot_mapping,
                swa_indices,
                swa_lens,
                csa_cmp_slot_mapping,
                csa_idx_slot_mapping,
                csa_state_slot_mapping,
                csa_inner_state_slot_mapping,
                position_ids,
                kv_seq_lens,
                attn_sink_csa,
                wo_a_shard_csa,
                wo_b_shard_csa,
                wo_b_scale_csa,
                x_attn_csa,
                oproj_reduce_window,
                oproj_reduce_signal,
                oproj_sync_signal,
                tok_q_window,
                tok_q_signal,
                tok_o_window,
                tok_o_signal,
                my_rank,
                csa_oproj_epoch,
                csa_tok_epoch,
                fwd_csa_q_rope_cos_il,
                fwd_csa_q_rope_sin_signed,
                fwd_csa_q_rope_swap_idx,
                fwd_csa_out_rope_cos_il,
                fwd_csa_out_rope_sin_signed,
                fwd_csa_out_rope_swap_idx,
                fwd_csa_idx_rope_swap_idx,
            )
        with pl.scope():
            moe(
                x_attn_csa,
                hc_ffn_fn_csa,
                hc_ffn_scale_csa,
                hc_ffn_base_csa,
                norm_w_csa,
                gate_w_csa,
                gate_bias_csa,
                tid2eid_csa,
                input_ids,
                routed_w1_csa,
                routed_w1_scale_csa,
                routed_w3_csa,
                routed_w3_scale_csa,
                routed_w2_csa,
                routed_w2_scale_csa,
                hidden_mid,
                reduce_window,
                reduce_signal,
                gather_window,
                gather_signal,
                csa_layer,
                nt,
                my_rank,
                csa_moe_epoch,
            )
        if csa_layer == DSPARK_TARGET_LAYER_IDS[0]:
            # Capture layer40 before the alternating workspace is reused.
            for token in pl.spmd(T, name_hint="lowlat_store_dspark_l40"):
                target_hc_l40 = hidden_mid[token : token + 1, 0:HC_MULT, 0:D]
                target_rows_l40 = pl.reshape(target_hc_l40, [HC_MULT, D])
                target_sum_l40 = pl.col_sum(target_rows_l40)
                target_mean_l40 = pl.mul(target_sum_l40, 1.0 / HC_MULT)
                target_bf16_l40 = pl.cast(target_mean_l40, pl.BF16, mode="rint")
                dspark_target_hidden_out[token : token + 1, 0:D] = target_bf16_l40
        hc_attn_fn_hca: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [hca_layer * MIX_HC, 0]
        )
        hc_attn_scale_hca: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [hca_layer * 3])
        hc_attn_base_hca: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [hca_layer * MIX_HC]
        )
        attn_norm_w_hca: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [hca_layer * D])
        wq_a_hca: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [hca_layer * D, 0])
        wq_b_hca: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [hca_layer * Q_LORA, 0]
        )
        wq_b_scale_hca: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [hca_layer * H * HEAD_DIM]
        )
        wkv_hca: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [hca_layer * D, 0])
        gamma_cq_hca: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [hca_layer * Q_LORA])
        gamma_ckv_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [hca_layer * HEAD_DIM]
        )
        kv_cache_hca = pl.slice(
            kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [hca_layer * ori_block_num, 0, 0, 0]
        )
        attn_sink_hca: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [hca_layer * H])
        wo_a_hca: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
            wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [hca_layer * O_GROUPS, 0, 0]
        )
        wo_a_shard_hca: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
            wo_a_shard, [1, O_LORA, O_GROUP_IN], [hca_layer, 0, 0]
        )
        wo_b_shard_hca: pl.Tensor[[D, O_LORA], pl.INT8] = pl.slice(
            wo_b_shard, [D, O_LORA], [hca_layer * D, 0]
        )
        wo_b_hca: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(
            wo_b, [D, O_GROUPS * O_LORA], [hca_layer * D, 0]
        )
        wo_b_scale_hca: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [hca_layer * D])
        hca_cmp_wkv_hca: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
            hca_cmp_wkv, [HCA_MAIN_OUT_DIM, D], [loop_i * HCA_MAIN_OUT_DIM, 0]
        )
        hca_cmp_wgate_hca: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
            hca_cmp_wgate, [HCA_MAIN_OUT_DIM, D], [loop_i * HCA_MAIN_OUT_DIM, 0]
        )
        hca_cmp_ape_hca: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(
            hca_cmp_ape, [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], [loop_i * HCA_COMPRESS_RATIO, 0]
        )
        hca_cmp_norm_w_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            hca_cmp_norm_w, [HEAD_DIM], [loop_i * HEAD_DIM]
        )
        hca_compress_state_hca = pl.slice(
            hca_compress_state,
            [hca_state_block_num, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
            [loop_i * hca_state_block_num, 0, 0],
        )
        cmp_kv_hca = pl.slice(
            hca_cmp_kv,
            [hca_cmp_block_num, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            [loop_i * hca_cmp_block_num, 0, 0, 0],
        )
        hc_ffn_fn_hca: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [hca_layer * MIX_HC, 0]
        )
        hc_ffn_scale_hca: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [hca_layer * 3])
        hc_ffn_base_hca: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [hca_layer * MIX_HC])
        norm_w_hca: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [hca_layer * D])
        gate_w_hca: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
            gate_w, [N_EXPERTS, D], [hca_layer * N_EXPERTS, 0]
        )
        gate_bias_hca: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
            gate_bias, [N_EXPERTS], [hca_layer * N_EXPERTS]
        )
        tid2eid_hca: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(
            tid2eid, [VOCAB, TOPK], [hca_layer * VOCAB, 0]
        )
        routed_w1_hca: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w1, [N_BANK, MOE_INTER, D], [(hca_layer % MOE_BANKS) * N_BANK, 0, 0]
        )
        routed_w1_scale_hca: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
            routed_w1_scale, [N_BANK, MOE_INTER], [(hca_layer % MOE_BANKS) * N_BANK, 0]
        )
        routed_w3_hca: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w3, [N_BANK, MOE_INTER, D], [(hca_layer % MOE_BANKS) * N_BANK, 0, 0]
        )
        routed_w3_scale_hca: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
            routed_w3_scale, [N_BANK, MOE_INTER], [(hca_layer % MOE_BANKS) * N_BANK, 0]
        )
        routed_w2_hca: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8] = pl.slice(
            routed_w2, [N_BANK, D, MOE_INTER], [(hca_layer % MOE_BANKS) * N_BANK, 0, 0]
        )
        routed_w2_scale_hca: pl.Tensor[[N_BANK, D], pl.FP32] = pl.slice(
            routed_w2_scale, [N_BANK, D], [(hca_layer % MOE_BANKS) * N_BANK, 0]
        )
        with pl.scope():
            attention_hca(
                hidden_mid,
                gate_w_hca,
                hc_attn_fn_hca,
                hc_attn_scale_hca,
                hc_attn_base_hca,
                attn_norm_w_hca,
                wq_a_hca,
                wq_b_hca,
                wq_b_scale_hca,
                wkv_hca,
                gamma_cq_hca,
                gamma_ckv_hca,
                fwd_hca_rope_cos_t,
                fwd_hca_rope_sin_t,
                fwd_hca_cmp_cos_il,
                fwd_hca_cmp_sin_signed,
                fwd_hca_cmp_seq_lens,
                hca_cmp_wkv_hca,
                hca_cmp_wgate_hca,
                hca_cmp_ape_hca,
                hca_cmp_norm_w_hca,
                hca_compress_state_hca,
                hca_compress_state_block_table,
                kv_cache_hca,
                cmp_kv_hca,
                hca_cmp_block_table,
                ori_slot_mapping,
                swa_indices,
                swa_lens,
                hca_cmp_slot_mapping,
                hca_state_slot_mapping,
                position_ids,
                kv_seq_lens,
                attn_sink_hca,
                wo_a_shard_hca,
                wo_b_shard_hca,
                wo_b_scale_hca,
                x_attn_hca,
                oproj_reduce_window,
                oproj_reduce_signal,
                oproj_sync_signal,
                my_rank,
                hca_oproj_epoch,
                fwd_hca_q_rope_cos_il,
                fwd_hca_q_rope_sin_signed,
                fwd_hca_q_rope_swap_idx,
                fwd_hca_out_rope_cos_il,
                fwd_hca_out_rope_sin_signed,
                fwd_hca_out_rope_swap_idx,
            )
        with pl.scope():
            moe(
                x_attn_hca,
                hc_ffn_fn_hca,
                hc_ffn_scale_hca,
                hc_ffn_base_hca,
                norm_w_hca,
                gate_w_hca,
                gate_bias_hca,
                tid2eid_hca,
                input_ids,
                routed_w1_hca,
                routed_w1_scale_hca,
                routed_w3_hca,
                routed_w3_scale_hca,
                routed_w2_hca,
                routed_w2_scale_hca,
                hidden,
                reduce_window,
                reduce_signal,
                gather_window,
                gather_signal,
                hca_layer,
                nt,
                my_rank,
                hca_moe_epoch,
            )
        if hca_layer == DSPARK_TARGET_LAYER_IDS[1]:
            # Capture layer41 before the final CSA consumes this workspace.
            for token in pl.spmd(T, name_hint="lowlat_store_dspark_l41"):
                target_hc_l41 = hidden[token : token + 1, 0:HC_MULT, 0:D]
                target_rows_l41 = pl.reshape(target_hc_l41, [HC_MULT, D])
                target_sum_l41 = pl.col_sum(target_rows_l41)
                target_mean_l41 = pl.mul(target_sum_l41, 1.0 / HC_MULT)
                target_bf16_l41 = pl.cast(target_mean_l41, pl.BF16, mode="rint")
                dspark_target_hidden_out[token : token + 1, D : 2 * D] = target_bf16_l41
    # FWD index (14), not CSA_LAST_LAYER (6): the *_last slices below index per-FWD-layer
    # stacked weights; csa-stacked weights use the literal (CSA_NUM_LAYERS-1).
    csa_layer_last: pl.Scalar[pl.INT32] = pl.cast(FWD_LAST_LAYER, pl.INT32)
    last_moe_epoch: pl.Scalar[pl.INT32] = pl.cast(2 * HCA_NUM_LAYERS + 3, pl.INT32)
    x_attn_last: pl.Tensor[[T, HC_MULT, D], pl.FP32] = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    hc_attn_fn_last: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_attn_fn, [MIX_HC, HC_DIM], [csa_layer_last * MIX_HC, 0]
    )
    hc_attn_scale_last: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [csa_layer_last * 3])
    hc_attn_base_last: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_attn_base, [MIX_HC], [csa_layer_last * MIX_HC]
    )
    attn_norm_w_last: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [csa_layer_last * D])
    wq_a_last: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [csa_layer_last * D, 0])
    wq_b_last: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
        wq_b, [Q_LORA, H * HEAD_DIM], [csa_layer_last * Q_LORA, 0]
    )
    wq_b_scale_last: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
        wq_b_scale, [H * HEAD_DIM], [csa_layer_last * H * HEAD_DIM]
    )
    wkv_last: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [csa_layer_last * D, 0])
    gamma_cq_last: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [csa_layer_last * Q_LORA])
    gamma_ckv_last: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        gamma_ckv, [HEAD_DIM], [csa_layer_last * HEAD_DIM]
    )
    kv_cache_last = pl.slice(
        kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer_last * ori_block_num, 0, 0, 0]
    )
    attn_sink_last: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [csa_layer_last * H])
    wo_a_last: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [csa_layer_last * O_GROUPS, 0, 0]
    )
    wo_b_last: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(
        wo_b, [D, O_GROUPS * O_LORA], [csa_layer_last * D, 0]
    )
    wo_b_scale_last: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [csa_layer_last * D])
    wo_a_shard_last: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a_shard, [1, O_LORA, O_GROUP_IN], [csa_layer_last, 0, 0]
    )
    wo_b_shard_last: pl.Tensor[[D, O_LORA], pl.INT8] = pl.slice(
        wo_b_shard, [D, O_LORA], [csa_layer_last * D, 0]
    )
    last_oproj_epoch: pl.Scalar[pl.INT32] = pl.cast(2 * HCA_NUM_LAYERS + 3, pl.INT32)
    csa_cmp_wkv_last: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
        csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [(CSA_NUM_LAYERS - 1) * CSA_MAIN_OUT_DIM, 0]
    )
    csa_cmp_wgate_last: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
        csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [(CSA_NUM_LAYERS - 1) * CSA_MAIN_OUT_DIM, 0]
    )
    csa_cmp_ape_last: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(
        csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [(CSA_NUM_LAYERS - 1) * CSA_COMPRESS_RATIO, 0]
    )
    csa_cmp_norm_w_last: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
        csa_cmp_norm_w, [HEAD_DIM], [(CSA_NUM_LAYERS - 1) * HEAD_DIM]
    )
    csa_compress_state_last = pl.slice(
        csa_compress_state,
        [csa_state_block_num, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        [(CSA_NUM_LAYERS - 1) * csa_state_block_num, 0, 0],
    )
    csa_idx_wq_b_last: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8] = pl.slice(
        csa_idx_wq_b, [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [(CSA_NUM_LAYERS - 1) * Q_LORA, 0]
    )
    csa_idx_wq_b_scale_last: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32] = pl.slice(
        csa_idx_wq_b_scale,
        [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
        [(CSA_NUM_LAYERS - 1) * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
    )
    csa_weights_proj_last: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16] = pl.slice(
        csa_weights_proj, [D, CSA_IDX_N_HEADS], [(CSA_NUM_LAYERS - 1) * D, 0]
    )
    csa_hadamard_idx_last: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(
        csa_hadamard_idx, [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], [(CSA_NUM_LAYERS - 1) * CSA_IDX_HEAD_DIM, 0]
    )
    csa_inner_wkv_last: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(
        csa_inner_wkv, [CSA_INNER_OUT_DIM, D], [(CSA_NUM_LAYERS - 1) * CSA_INNER_OUT_DIM, 0]
    )
    csa_inner_wgate_last: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(
        csa_inner_wgate, [CSA_INNER_OUT_DIM, D], [(CSA_NUM_LAYERS - 1) * CSA_INNER_OUT_DIM, 0]
    )
    csa_inner_ape_last: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32] = pl.slice(
        csa_inner_ape, [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], [(CSA_NUM_LAYERS - 1) * CSA_COMPRESS_RATIO, 0]
    )
    csa_inner_norm_w_last: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16] = pl.slice(
        csa_inner_norm_w, [CSA_IDX_HEAD_DIM], [(CSA_NUM_LAYERS - 1) * CSA_IDX_HEAD_DIM]
    )
    csa_inner_compress_state_last = pl.slice(
        csa_inner_compress_state,
        [inner_state_block_num, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        [(CSA_NUM_LAYERS - 1) * inner_state_block_num, 0, 0],
    )
    cmp_kv_last = pl.slice(
        csa_cmp_kv,
        [csa_cmp_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
        [(CSA_NUM_LAYERS - 1) * csa_cmp_block_num, 0, 0, 0],
    )
    idx_kv_cache_last = pl.slice(
        idx_kv_cache,
        [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
        [(CSA_NUM_LAYERS - 1) * idx_block_num, 0, 0, 0],
    )
    idx_kv_scale_last = pl.slice(
        idx_kv_scale,
        [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1],
        [(CSA_NUM_LAYERS - 1) * idx_block_num, 0, 0, 0],
    )
    hc_ffn_fn_last: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
        hc_ffn_fn, [MIX_HC, HC_DIM], [csa_layer_last * MIX_HC, 0]
    )
    hc_ffn_scale_last: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [csa_layer_last * 3])
    hc_ffn_base_last: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
        hc_ffn_base, [MIX_HC], [csa_layer_last * MIX_HC]
    )
    norm_w_last: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [csa_layer_last * D])
    gate_w_last: pl.Tensor[[N_EXPERTS, D], pl.FP32] = pl.slice(
        gate_w, [N_EXPERTS, D], [csa_layer_last * N_EXPERTS, 0]
    )
    gate_bias_last: pl.Tensor[[N_EXPERTS], pl.FP32] = pl.slice(
        gate_bias, [N_EXPERTS], [csa_layer_last * N_EXPERTS]
    )
    tid2eid_last: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(
        tid2eid, [VOCAB, TOPK], [csa_layer_last * VOCAB, 0]
    )
    routed_w1_last: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
        routed_w1, [N_BANK, MOE_INTER, D], [(csa_layer_last % MOE_BANKS) * N_BANK, 0, 0]
    )
    routed_w1_scale_last: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
        routed_w1_scale, [N_BANK, MOE_INTER], [(csa_layer_last % MOE_BANKS) * N_BANK, 0]
    )
    routed_w3_last: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8] = pl.slice(
        routed_w3, [N_BANK, MOE_INTER, D], [(csa_layer_last % MOE_BANKS) * N_BANK, 0, 0]
    )
    routed_w3_scale_last: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32] = pl.slice(
        routed_w3_scale, [N_BANK, MOE_INTER], [(csa_layer_last % MOE_BANKS) * N_BANK, 0]
    )
    routed_w2_last: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8] = pl.slice(
        routed_w2, [N_BANK, D, MOE_INTER], [(csa_layer_last % MOE_BANKS) * N_BANK, 0, 0]
    )
    routed_w2_scale_last: pl.Tensor[[N_BANK, D], pl.FP32] = pl.slice(
        routed_w2_scale, [N_BANK, D], [(csa_layer_last % MOE_BANKS) * N_BANK, 0]
    )
    with pl.scope():
        attention_csa(
            hidden,
            gate_w_last,
            hc_attn_fn_last,
            hc_attn_scale_last,
            hc_attn_base_last,
            attn_norm_w_last,
            wq_a_last,
            wq_b_last,
            wq_b_scale_last,
            wkv_last,
            gamma_cq_last,
            gamma_ckv_last,
            fwd_csa_rope_cos_t,
            fwd_csa_rope_sin_t,
            fwd_csa_step_cos_il,
            fwd_csa_step_sin_signed,
            fwd_csa_cmp_cos_il,
            fwd_csa_cmp_sin_signed,
            csa_cmp_wkv_last,
            csa_cmp_wgate_last,
            csa_cmp_ape_last,
            csa_cmp_norm_w_last,
            csa_compress_state_last,
            csa_compress_state_block_table,
            csa_idx_wq_b_last,
            csa_idx_wq_b_scale_last,
            csa_weights_proj_last,
            csa_hadamard_idx_last,
            csa_inner_wkv_last,
            csa_inner_wgate_last,
            csa_inner_ape_last,
            csa_inner_norm_w_last,
            csa_inner_compress_state_last,
            csa_inner_compress_state_block_table,
            kv_cache_last,
            cmp_kv_last,
            csa_cmp_block_table,
            idx_kv_cache_last,
            idx_kv_scale_last,
            idx_block_table,
            ori_slot_mapping,
            swa_indices,
            swa_lens,
            csa_cmp_slot_mapping,
            csa_idx_slot_mapping,
            csa_state_slot_mapping,
            csa_inner_state_slot_mapping,
            position_ids,
            kv_seq_lens,
            attn_sink_last,
            wo_a_shard_last,
            wo_b_shard_last,
            wo_b_scale_last,
            x_attn_last,
            oproj_reduce_window,
            oproj_reduce_signal,
            oproj_sync_signal,
            tok_q_window,
            tok_q_signal,
            tok_o_window,
            tok_o_signal,
            my_rank,
            last_oproj_epoch,
            pl.cast(HCA_NUM_LAYERS + 1, pl.INT32),
            fwd_csa_q_rope_cos_il,
            fwd_csa_q_rope_sin_signed,
            fwd_csa_q_rope_swap_idx,
            fwd_csa_out_rope_cos_il,
            fwd_csa_out_rope_sin_signed,
            fwd_csa_out_rope_swap_idx,
            fwd_csa_idx_rope_swap_idx,
        )
    with pl.scope():
        moe(
            x_attn_last,
            hc_ffn_fn_last,
            hc_ffn_scale_last,
            hc_ffn_base_last,
            norm_w_last,
            gate_w_last,
            gate_bias_last,
            tid2eid_last,
            input_ids,
            routed_w1_last,
            routed_w1_scale_last,
            routed_w3_last,
            routed_w3_scale_last,
            routed_w2_last,
            routed_w2_scale_last,
            pre_hc_hidden_out,
            reduce_window,
            reduce_signal,
            gather_window,
            gather_signal,
            csa_layer_last,
            nt,
            my_rank,
            last_moe_epoch,
        )
    # Auxiliary taps are the HC mean, never the learned final-output head.
    for token in pl.spmd(T, name_hint="lowlat_store_dspark_l42"):
        target_hc_l42 = pre_hc_hidden_out[token : token + 1, 0:HC_MULT, 0:D]
        target_rows_l42 = pl.reshape(target_hc_l42, [HC_MULT, D])
        target_sum_l42 = pl.col_sum(target_rows_l42)
        target_mean_l42 = pl.mul(target_sum_l42, 1.0 / HC_MULT)
        target_bf16_l42 = pl.cast(target_mean_l42, pl.BF16, mode="rint")
        dspark_target_hidden_out[token : token + 1, 2 * D : 3 * D] = target_bf16_l42
    s6_moe_cleanup_tid = clear_moe_signals(pre_hc_hidden_out, reduce_signal, gather_signal)
    s6_oproj_cleanup_tid = clear_csa_oproj_signals(
        pre_hc_hidden_out, oproj_reduce_signal, oproj_sync_signal, tok_q_signal, tok_o_signal
    )
    x_head: pl.Tensor[[T, D], pl.BF16] = pl.create_tensor([T, D], dtype=pl.BF16)
    lm_cleanup_carrier = pl.array.create(1, pl.TASK_ID)
    with pl.scope():
        hc_head(pre_hc_hidden_out, hc_head_fn, hc_head_scale, hc_head_base, x_head)
        rms_norm(x_head, final_norm_w, x_out)
        _s6_logits, _s6_sampled, s6_lm_cleanup_tid = lm_head_with_sampling(
            x_out,
            lm_head_weight,
            logit_row_indices,
            sampling_temperatures,
            sampling_top_ks,
            sampling_seeds,
            sampling_positions,
            logits,
            sampled_ids,
            lm_head_hidden_window,
            lm_head_hidden_done,
            lm_head_logits_window,
            lm_head_logits_done,
            my_rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE,
            my_rank % LM_HEAD_TP_SIZE,
            LM_HEAD_COMM_EPOCH,
        )
        lm_cleanup_carrier[0] = s6_lm_cleanup_tid
    s6_lm_cleanup_tid_out = lm_cleanup_carrier[0]
    return x_out, s6_moe_cleanup_tid, s6_oproj_cleanup_tid, s6_lm_cleanup_tid_out


@pl.jit.inline(auto_scope=False)
def s6_prepared(
    embed_weight: pl.Tensor[[EMBED_VOCAB_PER_TP, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[FWD_NUM_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    wo_a_shard: pl.Tensor[[FWD_NUM_LAYERS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[FWD_NUM_LAYERS * D, O_LORA], pl.INT8],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32
        ]
    ],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[[FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]
    ],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[[FWD_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]
    ],
    hca_cmp_kv: pl.InOut[
        pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    csa_cmp_kv: pl.InOut[
        pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    idx_kv_cache: pl.InOut[
        pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[MOE_BANKS * N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[MOE_BANKS * N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[MOE_BANKS * N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[MOE_BANKS * N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[MOE_BANKS * N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[MOE_BANKS * N_BANK, D], pl.FP32],
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Out[pl.Tensor[[T], pl.INT32]],
    kv_seq_lens: pl.Out[pl.Tensor[[B], pl.INT32]],
    hca_compress_state_block_table: pl.Tensor[[B, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[B, HCA_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    block_counts: pl.Tensor[[B, N_CACHE_GROUPS], pl.INT32],
    input_ids: pl.Out[pl.Tensor[[T], pl.INT64]],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_temperatures: pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    dspark_target_hidden_out: pl.Out[pl.Tensor[[T, DSPARK_MAIN_HIDDEN_DIM], pl.BF16]],
    x_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, REDUCE_SHARD_D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
    gather_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, REDUCE_SHARD_D], pl.BF16],
    gather_signal: pld.DistributedTensor[[N_RANKS, 32], pl.INT32],
    oproj_reduce_window: pld.DistributedTensor[[CSA_OPROJ_REDUCE_ROWS, D], pl.FP32],
    oproj_reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    oproj_sync_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    tok_q_window: pld.DistributedTensor[[TOK_Q_ROWS, HEAD_DIM], pl.BF16],
    tok_q_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    tok_o_window: pld.DistributedTensor[[T, TOK_O_COLS], pl.BF16],
    tok_o_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    embed_window: pld.DistributedTensor[[T, D], pl.BF16],
    embed_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    bridge_markov_ids: pl.Tensor[[5], pl.INT32],
    bridge_epoch: pl.Out[pl.Tensor[[1], pl.INT32]],
    bridge_seen_epoch: pl.Out[pl.Tensor[[1], pl.INT32]],
    bridge_ids_window: pld.DistributedTensor[[8], pl.INT64],
    bridge_ready_signal: pld.DistributedTensor[[1], pl.INT32],
    state_window: pld.DistributedTensor[[4], pl.INT64],
    state_signal: pld.DistributedTensor[[1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    history_ring: pl.InOut[pl.Tensor[[136, 12288], pl.BF16]],
    request_state: pl.InOut[pl.Tensor[[4], pl.INT64]],
    control: pl.Out[pl.Tensor[[16], pl.INT64]],
    emitted: pl.Out[pl.Tensor[[6], pl.INT64]],
    request_snapshot: pl.Tensor[[4], pl.INT64],
):
    bridge_local_ids = pl.create_tensor([T], dtype=pl.INT64)
    bridge_local_ids, bridge_input_ready = device_input_bridge(
        input_ids,
        bridge_markov_ids,
        bridge_epoch,
        bridge_local_ids,
        bridge_seen_epoch,
        bridge_ids_window,
        bridge_ready_signal,
        my_rank,
        pl.const(N_RANKS, pl.INT32),
    )
    ori_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    swa_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    swa_indices = pl.create_tensor([T, SWA_WIN], dtype=pl.INT32)
    swa_lens = pl.create_tensor([T], dtype=pl.INT32)
    hca_cmp_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    hca_state_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    csa_cmp_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    csa_idx_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    csa_state_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    csa_inner_state_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    build_decode_metadata(
        position_ids,
        kv_seq_lens,
        block_table,
        hca_cmp_block_table,
        csa_cmp_block_table,
        idx_block_table,
        hca_compress_state_block_table,
        csa_compress_state_block_table,
        csa_inner_compress_state_block_table,
        block_counts,
        ori_slot_mapping,
        swa_slot_mapping,
        swa_indices,
        swa_lens,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
    )
    embed_hidden = pl.create_tensor([T, D], dtype=pl.BF16)
    tp_lookup_embedding(
        bridge_local_ids,
        embed_weight,
        embed_hidden,
        embed_window,
        embed_signal,
        my_rank,
        pl.const(1, pl.INT32),
        defer_clear=True,
    )
    x_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    pack_x_hc_rows(embed_hidden, x_hc)
    result, s6_moe_cleanup_tid, s6_oproj_cleanup_tid, s6_lm_cleanup_tid = decode_fwd(
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        kv_cache,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        wo_a_shard,
        wo_b_shard,
        hca_cmp_wkv,
        hca_cmp_wgate,
        hca_cmp_ape,
        hca_cmp_norm_w,
        hca_compress_state,
        csa_cmp_wkv,
        csa_cmp_wgate,
        csa_cmp_ape,
        csa_cmp_norm_w,
        csa_compress_state,
        csa_idx_wq_b,
        csa_idx_wq_b_scale,
        csa_weights_proj,
        csa_hadamard_idx,
        csa_inner_wkv,
        csa_inner_wgate,
        csa_inner_ape,
        csa_inner_norm_w,
        csa_inner_compress_state,
        hca_cmp_kv,
        csa_cmp_kv,
        idx_kv_cache,
        idx_kv_scale,
        hc_ffn_fn,
        hc_ffn_scale,
        hc_ffn_base,
        norm_w,
        gate_w,
        gate_bias,
        tid2eid,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        freqs_cos,
        freqs_sin,
        x_hc,
        position_ids,
        kv_seq_lens,
        hca_compress_state_block_table,
        csa_compress_state_block_table,
        csa_inner_compress_state_block_table,
        hca_cmp_block_table,
        csa_cmp_block_table,
        idx_block_table,
        ori_slot_mapping,
        swa_slot_mapping,
        swa_indices,
        swa_lens,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
        bridge_local_ids,
        hc_head_fn,
        hc_head_scale,
        hc_head_base,
        final_norm_w,
        lm_head_weight,
        logit_row_indices,
        sampling_temperatures,
        sampling_top_ks,
        sampling_seeds,
        sampling_positions,
        pre_hc_hidden_out,
        dspark_target_hidden_out,
        x_out,
        logits,
        sampled_ids,
        reduce_window,
        reduce_signal,
        gather_window,
        gather_signal,
        oproj_reduce_window,
        oproj_reduce_signal,
        oproj_sync_signal,
        tok_q_window,
        tok_q_signal,
        tok_o_window,
        tok_o_signal,
        lm_head_hidden_window,
        lm_head_hidden_done,
        lm_head_logits_window,
        lm_head_logits_done,
        num_tokens_per_owner,
        my_rank,
    )
    commit_ack_tid = bridge_input_ready
    if my_rank == 0:
        history_ring, control, emitted, commit_ack_tid = device_commit_inline(
            bridge_markov_ids,
            sampled_ids,
            dspark_target_hidden_out,
            request_snapshot,
            history_ring,
            control,
            emitted,
        )
    else:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="nonowner_commit_outputs") as nonowner_ack_tid:
            pl.write(control, [0], pl.const(0, pl.INT64))
            pl.write(control, [1], pl.const(0, pl.INT64))
            pl.write(control, [2], pl.const(0, pl.INT64))
            pl.write(control, [3], pl.const(0, pl.INT64))
            pl.write(control, [4], pl.const(0, pl.INT64))
            pl.write(control, [5], pl.const(0, pl.INT64))
            pl.write(control, [6], pl.const(0, pl.INT64))
            pl.write(control, [7], pl.const(0, pl.INT64))
            pl.write(control, [8], pl.const(0, pl.INT64))
            pl.write(control, [9], pl.const(0, pl.INT64))
            pl.write(control, [10], pl.const(0, pl.INT64))
            pl.write(control, [11], pl.const(0, pl.INT64))
            pl.write(control, [12], pl.const(0, pl.INT64))
            pl.write(control, [13], pl.const(0, pl.INT64))
            pl.write(control, [14], pl.const(0, pl.INT64))
            pl.write(control, [15], pl.const(0, pl.INT64))
            for token in pl.range(6):
                pl.write(emitted, [token], pl.const(-1, pl.INT64))
        commit_ack_tid = nonowner_ack_tid
    request_state, next_state_ready = publish_next_request_state(
        control,
        request_state,
        bridge_epoch,
        state_window,
        state_signal,
        my_rank,
        pl.const(N_RANKS, pl.INT32),
        commit_ack_tid,
    )
    _s6_result, s6_embed_cleanup_tid = clear_embedding_signals(result, embed_signal)
    s6_local_reuse_ready = pl.system.task_dummy(
        deps=[
            next_state_ready,
            s6_moe_cleanup_tid,
            s6_oproj_cleanup_tid,
            s6_lm_cleanup_tid,
            s6_embed_cleanup_tid,
        ]
    )
    return (result, commit_ack_tid, next_state_ready, s6_local_reuse_ready)
