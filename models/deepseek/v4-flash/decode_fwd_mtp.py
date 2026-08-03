# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=8
# ci: no-sim
"""DeepSeek-V4 Flash decode-to-MTP benchmark with device-side token handoff and full MTP logits."""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from golden import ScalarSpec, TensorSpec, run_jit
from pypto.ir.distributed_compiled_program import DistributedConfig

import decode_fwd as main_decode
import decode_mtp as mtp_decode
from decode_fwd import decode_fwd
from decode_input_pack import pack_mtp_inputs
from decode_mtp import mtp_decode_layer_logits
from decode_routing import ROUTING_MODES, routing_mode_value


# Dynamic shape variables.
MAIN_EMBED_VOCAB_DYN = main_decode.EMBED_VOCAB_DYN
MTP_EMBED_VOCAB_DYN = mtp_decode.EMBED_VOCAB_DYN
FWD_ORI_BLOCK_NUM_DYN = main_decode.FWD_ORI_BLOCK_NUM_DYN
FWD_CMP_BLOCK_NUM_DYN = main_decode.FWD_CMP_BLOCK_NUM_DYN
FWD_IDX_BLOCK_NUM_DYN = main_decode.FWD_IDX_BLOCK_NUM_DYN
FWD_HCA_STATE_BLOCK_NUM_DYN = main_decode.FWD_HCA_STATE_BLOCK_NUM_DYN
FWD_CSA_STATE_BLOCK_NUM_DYN = main_decode.FWD_CSA_STATE_BLOCK_NUM_DYN
FWD_INNER_STATE_BLOCK_NUM_DYN = main_decode.FWD_INNER_STATE_BLOCK_NUM_DYN
MTP_ORI_BLOCK_NUM_DYN = mtp_decode.ORI_BLOCK_NUM_DYN

# model config
N_RANKS = main_decode.N_RANKS
FWD_NUM_LAYERS = main_decode.FWD_NUM_LAYERS
CSA_NUM_LAYERS = main_decode.CSA_NUM_LAYERS
HCA_NUM_LAYERS = main_decode.HCA_NUM_LAYERS
B = main_decode.B
T = main_decode.T
D = main_decode.D
H = main_decode.H
HEAD_DIM = main_decode.HEAD_DIM
ROPE_HEAD_DIM = main_decode.ROPE_HEAD_DIM
HC_MULT = main_decode.HC_MULT
HC_DIM = main_decode.HC_DIM
MIX_HC = main_decode.MIX_HC
Q_LORA = main_decode.Q_LORA
O_GROUPS = main_decode.O_GROUPS
O_LORA = main_decode.O_LORA
O_GROUP_IN = main_decode.O_GROUP_IN
BLOCK_SIZE = main_decode.BLOCK_SIZE
MAX_SEQ_LEN = main_decode.MAX_SEQ_LEN
ORI_TABLE_MAX_BLOCKS = main_decode.ORI_TABLE_MAX_BLOCKS
N_CACHE_GROUPS = main_decode.N_CACHE_GROUPS
HCA_MAIN_OUT_DIM = main_decode.HCA_MAIN_OUT_DIM
HCA_COMPRESS_RATIO = main_decode.HCA_COMPRESS_RATIO
HCA_COMPRESS_STATE_BLOCK_SIZE = main_decode.HCA_COMPRESS_STATE_BLOCK_SIZE
HCA_COMPRESS_STATE_DIM = main_decode.HCA_COMPRESS_STATE_DIM
HCA_COMPRESS_STATE_MAX_BLOCKS = main_decode.HCA_COMPRESS_STATE_MAX_BLOCKS
CSA_MAIN_OUT_DIM = main_decode.CSA_MAIN_OUT_DIM
CSA_COMPRESS_RATIO = main_decode.CSA_COMPRESS_RATIO
CSA_MAIN_STATE_BLOCK_SIZE = main_decode.CSA_MAIN_STATE_BLOCK_SIZE
CSA_MAIN_STATE_DIM = main_decode.CSA_MAIN_STATE_DIM
CSA_MAIN_STATE_MAX_BLOCKS = main_decode.CSA_MAIN_STATE_MAX_BLOCKS
CSA_IDX_N_HEADS = main_decode.CSA_IDX_N_HEADS
CSA_IDX_HEAD_DIM = main_decode.CSA_IDX_HEAD_DIM
CSA_INNER_OUT_DIM = main_decode.CSA_INNER_OUT_DIM
CSA_INNER_STATE_BLOCK_SIZE = main_decode.CSA_INNER_STATE_BLOCK_SIZE
CSA_INNER_STATE_DIM = main_decode.CSA_INNER_STATE_DIM
CSA_INNER_STATE_MAX_BLOCKS = main_decode.CSA_INNER_STATE_MAX_BLOCKS
CSA_CMP_MAX_BLOCKS = main_decode.CSA_CMP_MAX_BLOCKS
CSA_IDX_CACHE_MAX_BLOCKS = main_decode.CSA_IDX_CACHE_MAX_BLOCKS
N_EXPERTS_GLOBAL = main_decode.N_EXPERTS_GLOBAL
N_LOCAL = main_decode.N_LOCAL
N_ROUTES = main_decode.N_ROUTES
RECV_MAX = main_decode.RECV_MAX
MOE_INTER = main_decode.MOE_INTER
MOE_VOCAB = main_decode.VOCAB
MOE_TOPK = main_decode.TOPK
MTP_MOE_VOCAB = mtp_decode.MOE_VOCAB
MTP_MOE_TOPK = mtp_decode.MOE_TOPK
AUX_PAD = main_decode.AUX_PAD
IDX_PAD = main_decode.IDX_PAD
GROUP_LOGIT_ROWS = main_decode.GROUP_LOGIT_ROWS
MAX_LOGIT_ROWS = main_decode.MAX_LOGIT_ROWS
LM_HEAD_TP_SIZE = main_decode.LM_HEAD_TP_SIZE
LM_HEAD_VOCAB = main_decode.LM_HEAD_VOCAB
VOCAB_PER_TP = main_decode.VOCAB_PER_TP
SAMPLED_IDS_PAD = main_decode.SAMPLED_IDS_PAD

# fixture config
DECODE_START_POS = main_decode.DECODE_START_POS
ORI_BLOCK_NUM = main_decode.ORI_BLOCK_NUM
CSA_CMP_BLOCK_NUM = main_decode.CSA_CMP_BLOCK_NUM
CSA_IDX_CACHE_BLOCK_NUM = main_decode.CSA_IDX_CACHE_BLOCK_NUM
HCA_COMPRESS_STATE_BLOCK_NUM = main_decode.HCA_COMPRESS_STATE_BLOCK_NUM
CSA_MAIN_STATE_BLOCK_NUM = main_decode.CSA_MAIN_STATE_BLOCK_NUM
CSA_INNER_STATE_BLOCK_NUM = main_decode.CSA_INNER_STATE_BLOCK_NUM


@pl.jit.host
def l3_decode_fwd_mtp(
    embed_weight: pl.Tensor[[N_RANKS, MAIN_EMBED_VOCAB_DYN, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    hc_ffn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_VOCAB, MOE_TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[N_RANKS, B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    kv_seq_lens: pl.Tensor[[N_RANKS, B], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, B, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    cmp_block_table: pl.Tensor[[N_RANKS, B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[N_RANKS, B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    block_counts: pl.Tensor[[N_RANKS, B, N_CACHE_GROUPS], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
    mtp_tail_token_pool: pl.InOut[pl.Tensor[[N_RANKS, B], pl.INT64]],
    mtp_tail_position_pool: pl.InOut[pl.Tensor[[N_RANKS, B], pl.INT32]],
    mtp_input_ids: pl.Out[pl.Tensor[[N_RANKS, T], pl.INT64]],
    mtp_position_ids: pl.Out[pl.Tensor[[N_RANKS, T], pl.INT32]],
    mtp_tail_pre_hc_pool: pl.InOut[pl.Tensor[[N_RANKS, B, HC_MULT, D], pl.FP32]],
    mtp_accepted_counts: pl.Tensor[[N_RANKS, B], pl.INT32],
    mtp_tail_slot_ids: pl.Tensor[[N_RANKS, B], pl.INT32],
    mtp_enorm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_hnorm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_e_proj_w: pl.Tensor[[N_RANKS, D, D], pl.INT8],
    mtp_e_proj_w_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_e_proj_smooth: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_h_proj_w: pl.Tensor[[N_RANKS, D, D], pl.INT8],
    mtp_h_proj_w_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_h_proj_smooth: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    mtp_hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    mtp_hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    mtp_attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    mtp_wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    mtp_wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    mtp_wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    mtp_wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    mtp_gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    mtp_gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    mtp_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, MTP_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    mtp_ori_block_table: pl.Tensor[[N_RANKS, B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
    mtp_attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    mtp_wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    mtp_wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    mtp_wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    mtp_hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    mtp_hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    mtp_moe_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    mtp_gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    mtp_gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    mtp_tid2eid: pl.Tensor[[N_RANKS, MTP_MOE_VOCAB, MTP_MOE_TOPK], pl.INT32],
    mtp_routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    mtp_routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    mtp_routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    mtp_routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    mtp_routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    mtp_routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    mtp_shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    mtp_shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    mtp_shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    mtp_shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    mtp_shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    mtp_shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    mtp_hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    mtp_hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    mtp_final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    mtp_hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    mtp_next_pre_hc_hidden: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    mtp_logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    mtp_logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    routing_mode: pl.Scalar[pl.INT32],
):
    main_recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    main_recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    main_recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    main_recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    main_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    main_data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    main_routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    main_combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    main_lm_head_hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    main_lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    main_lm_head_logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
    main_lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        main_recv_meta = pld.window(main_recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        main_recv_x = pld.window(main_recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        main_recv_aux = pld.window(main_recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        main_recv_route = pld.window(main_recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        main_arrived = pld.window(main_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        main_data_arrived = pld.window(main_data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        main_routed_y_buf = pld.window(main_routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        main_combine_arrived = pld.window(main_combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        main_lm_head_hidden_window = pld.window(
            main_lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16
        )
        main_lm_head_hidden_done = pld.window(
            main_lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        main_lm_head_logits_window = pld.window(
            main_lm_head_logits_window_buf, [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32
        )
        main_lm_head_logits_done = pld.window(
            main_lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        decode_fwd(
            embed_weight[r], hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r], attn_norm_w[r], wq_a[r],
            wq_b[r], wq_b_scale[r], wkv[r], gamma_cq[r], gamma_ckv[r], kv_cache[r], attn_sink[r],
            wo_a[r], wo_b[r], wo_b_scale[r], hca_cmp_wkv[r], hca_cmp_wgate[r], hca_cmp_ape[r],
            hca_cmp_norm_w[r], hca_compress_state[r], csa_cmp_wkv[r], csa_cmp_wgate[r],
            csa_cmp_ape[r], csa_cmp_norm_w[r], csa_compress_state[r], csa_idx_wq_b[r],
            csa_idx_wq_b_scale[r], csa_weights_proj[r], csa_hadamard_idx[r], csa_inner_wkv[r],
            csa_inner_wgate[r], csa_inner_ape[r], csa_inner_norm_w[r], csa_inner_compress_state[r],
            cmp_kv[r], idx_kv_cache[r], idx_kv_scale[r],
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r], norm_w[r], gate_w[r], gate_bias[r],
            tid2eid[r], routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r], shared_w1[r], shared_w1_scale[r], shared_w3[r],
            shared_w3_scale[r], shared_w2[r], shared_w2_scale[r], freqs_cos[r], freqs_sin[r],
            block_table[r], position_ids[r],
            kv_seq_lens[r], hca_compress_state_block_table[r], csa_compress_state_block_table[r],
            csa_inner_compress_state_block_table[r], cmp_block_table[r], idx_block_table[r],
            block_counts[r], input_ids[r], hc_head_fn[r], hc_head_scale[r], hc_head_base[r], final_norm_w[r],
            lm_head_weight[r], logit_row_indices[r],
            pre_hc_hidden_out[r], hidden_out[r], logits[r], sampled_ids[r],
            main_recv_meta, main_recv_x, main_recv_aux, main_recv_route,
            main_arrived, main_data_arrived, main_routed_y_buf, main_combine_arrived,
            main_lm_head_hidden_window, main_lm_head_hidden_done,
            main_lm_head_logits_window, main_lm_head_logits_done,
            num_tokens_per_owner, r, routing_mode,
            device=r,
        )

    for r in pl.range(pld.world_size()):
        pack_mtp_inputs(
            sampled_ids[r], position_ids[r],
            mtp_accepted_counts[r], mtp_tail_slot_ids[r],
            mtp_tail_token_pool[r], mtp_tail_position_pool[r],
            mtp_input_ids[r], mtp_position_ids[r],
            device=r,
        )

    mtp_recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    mtp_recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    mtp_recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    mtp_recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    mtp_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    mtp_data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    mtp_routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    mtp_combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    mtp_lm_head_hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    mtp_lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    mtp_lm_head_logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
    mtp_lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for r in pl.range(pld.world_size()):
        mtp_recv_meta = pld.window(mtp_recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        mtp_recv_x = pld.window(mtp_recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        mtp_recv_aux = pld.window(mtp_recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        mtp_recv_route = pld.window(mtp_recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        mtp_arrived = pld.window(mtp_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        mtp_data_arrived = pld.window(mtp_data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        mtp_routed_y_buf = pld.window(mtp_routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        mtp_combine_arrived = pld.window(mtp_combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        mtp_lm_head_hidden_window = pld.window(
            mtp_lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16
        )
        mtp_lm_head_hidden_done = pld.window(
            mtp_lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        mtp_lm_head_logits_window = pld.window(
            mtp_lm_head_logits_window_buf, [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32
        )
        mtp_lm_head_logits_done = pld.window(
            mtp_lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        mtp_decode_layer_logits(
            embed_weight[r], pre_hc_hidden_out[r], mtp_tail_pre_hc_pool[r],
            mtp_accepted_counts[r], mtp_tail_slot_ids[r], mtp_position_ids[r],
            mtp_enorm_w[r], mtp_hnorm_w[r],
            mtp_e_proj_w[r], mtp_e_proj_w_scale[r], mtp_e_proj_smooth[r],
            mtp_h_proj_w[r], mtp_h_proj_w_scale[r], mtp_h_proj_smooth[r],
            mtp_hc_attn_fn[r], mtp_hc_attn_scale[r], mtp_hc_attn_base[r],
            mtp_attn_norm_w[r],
            mtp_wq_a[r], mtp_wq_b[r], mtp_wq_b_scale[r],
            mtp_wkv[r], mtp_gamma_cq[r], mtp_gamma_ckv[r],
            freqs_cos[r], freqs_sin[r],
            mtp_kv_cache[r], mtp_ori_block_table[r],
            mtp_attn_sink[r], mtp_wo_a[r], mtp_wo_b[r], mtp_wo_b_scale[r],
            mtp_hc_ffn_fn[r], mtp_hc_ffn_scale[r], mtp_hc_ffn_base[r],
            mtp_moe_norm_w[r], mtp_gate_w[r], mtp_gate_bias[r],
            mtp_tid2eid[r], mtp_input_ids[r], input_ids[r],
            mtp_routed_w1[r], mtp_routed_w1_scale[r], mtp_routed_w3[r], mtp_routed_w3_scale[r],
            mtp_routed_w2[r], mtp_routed_w2_scale[r],
            mtp_shared_w1[r], mtp_shared_w1_scale[r], mtp_shared_w3[r], mtp_shared_w3_scale[r],
            mtp_shared_w2[r], mtp_shared_w2_scale[r],
            mtp_hc_head_fn[r], mtp_hc_head_scale[r], mtp_hc_head_base[r], mtp_final_norm_w[r],
            lm_head_weight[r], mtp_logit_row_indices[r],
            mtp_hidden_out[r], mtp_next_pre_hc_hidden[r], mtp_logits[r],
            mtp_recv_meta, mtp_recv_x, mtp_recv_aux, mtp_recv_route,
            mtp_arrived, mtp_data_arrived, mtp_routed_y_buf, mtp_combine_arrived,
            mtp_lm_head_hidden_window, mtp_lm_head_hidden_done,
            mtp_lm_head_logits_window, mtp_lm_head_logits_done,
            r, num_tokens, routing_mode,
            device=r,
        )


MTP_REUSED_NAMES = {
    "embed_weight",
    "main_pre_hc_hidden",
    "position_ids",
    "freqs_cos",
    "freqs_sin",
    "input_ids",
    "routing_input_ids",
    "lm_head_weight",
    "sampled_ids",
}

MTP_NAME_OVERRIDES = {
    "norm_w": "mtp_moe_norm_w",
    "mtp_hc_head_fn": "mtp_hc_head_fn",
    "mtp_hc_head_scale": "mtp_hc_head_scale",
    "mtp_hc_head_base": "mtp_hc_head_base",
    "mtp_norm_w": "mtp_final_norm_w",
    "hidden_out": "mtp_hidden_out",
    "next_pre_hc_hidden": "mtp_next_pre_hc_hidden",
    "logits": "mtp_logits",
    "logit_row_indices": "mtp_logit_row_indices",
}


def _mtp_spec_name(name):
    return MTP_NAME_OVERRIDES.get(name, f"mtp_{name}")


def _rename_mtp_spec(spec):
    import dataclasses

    return dataclasses.replace(spec, name=_mtp_spec_name(spec.name))


def _ranked_fixture(value):
    return value.unsqueeze(0).expand(N_RANKS, *value.shape).contiguous()


def build_tensor_specs(
    start_pos=DECODE_START_POS,
    num_tokens=T,
    ori_block_num=ORI_BLOCK_NUM,
    cmp_block_num=CSA_CMP_BLOCK_NUM,
    idx_block_num=CSA_IDX_CACHE_BLOCK_NUM,
    hca_state_block_num=HCA_COMPRESS_STATE_BLOCK_NUM,
    csa_state_block_num=CSA_MAIN_STATE_BLOCK_NUM,
    inner_state_block_num=CSA_INNER_STATE_BLOCK_NUM,
    *,
    routing_mode="model",
):
    main_specs = main_decode.build_tensor_specs(
        start_pos=start_pos,
        num_tokens=num_tokens,
        ori_block_num=ori_block_num,
        cmp_block_num=cmp_block_num,
        idx_block_num=idx_block_num,
        hca_state_block_num=hca_state_block_num,
        csa_state_block_num=csa_state_block_num,
        inner_state_block_num=inner_state_block_num,
        routing_mode=routing_mode,
    )
    mtp_specs = mtp_decode.build_tensor_specs(
        start_pos=start_pos,
        num_tokens=num_tokens,
        ori_block_num=ori_block_num,
        routing_mode=routing_mode,
    )

    accepted_counts = torch.tensor([1, 2, 1, 2], dtype=torch.int32)
    tail_slot_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    tail_token_pool = torch.arange(B, dtype=torch.int64)
    tail_position_pool = torch.arange(B, dtype=torch.int32) + start_pos - 1

    if main_specs[-1].name != "routing_mode":
        raise ValueError(f"expected trailing main routing_mode, got {main_specs[-1].name}")
    specs = list(main_specs[:-1])
    specs.extend(
        [
            TensorSpec(
                "mtp_tail_token_pool",
                [N_RANKS, B],
                torch.int64,
                init_value=lambda: _ranked_fixture(tail_token_pool),
                resident="stacked",
            ),
            TensorSpec(
                "mtp_tail_position_pool",
                [N_RANKS, B],
                torch.int32,
                init_value=lambda: _ranked_fixture(tail_position_pool),
                resident="stacked",
            ),
            TensorSpec("mtp_input_ids", [N_RANKS, T], torch.int64, resident="stacked"),
            TensorSpec("mtp_position_ids", [N_RANKS, T], torch.int32, resident="stacked"),
        ]
    )
    for spec in mtp_specs:
        if isinstance(spec, TensorSpec) and spec.name not in MTP_REUSED_NAMES:
            renamed_spec = _rename_mtp_spec(spec)
            if spec.name == "accepted_counts":
                renamed_spec.init_value = lambda: _ranked_fixture(accepted_counts)
            elif spec.name == "tail_slot_ids":
                renamed_spec.init_value = lambda: _ranked_fixture(tail_slot_ids)
            specs.append(renamed_spec)
    specs.append(ScalarSpec("num_tokens", torch.int32, num_tokens))
    specs.append(ScalarSpec("routing_mode", torch.int32, routing_mode_value(routing_mode)))

    for spec in specs:
        if isinstance(spec, TensorSpec):
            spec.is_output = spec.name == "mtp_logits"

    resident_edges = (
        "pre_hc_hidden_out",
        "sampled_ids",
        "mtp_tail_token_pool",
        "mtp_tail_position_pool",
        "mtp_input_ids",
        "mtp_position_ids",
        "mtp_tail_pre_hc_pool",
        "mtp_logits",
    )
    for spec in specs:
        if isinstance(spec, TensorSpec) and spec.name in resident_edges:
            spec.resident = "stacked"

    names = [spec.name for spec in specs]
    assert len(names) == len(set(names))
    assert len(specs) == 140
    return specs


def validate_topology(*, ep, tp, device_ids):
    if ep != 8:
        raise ValueError(f"comparison requires ep=8, got {ep}")
    if tp != 4:
        raise ValueError(f"comparison requires tp=4, got {tp}")
    if ep % tp != 0:
        raise ValueError(f"ep must be divisible by tp, got ep={ep}, tp={tp}")
    if len(device_ids) < ep:
        raise ValueError(f"need at least {ep} devices, got {device_ids}")


def golden_finite_smoke(_tensors):
    return None


def finite_tensor_compare(actual, _expected, **_context):
    finite = torch.isfinite(actual)
    if bool(finite.all()):
        return True, ""
    invalid = int((~finite).sum().item())
    return False, f"{invalid}/{actual.numel()} values are non-finite"


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Flash decode-to-MTP EP8/TP4 benchmark.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8])
    parser.add_argument("--tp", type=int, default=LM_HEAD_TP_SIZE, choices=[2, 4, 8, 16])
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)))
    parser.add_argument("--start-pos", type=int, default=DECODE_START_POS)
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--ori-block-num", type=int, default=ORI_BLOCK_NUM)
    parser.add_argument("--cmp-block-num", type=int, default=CSA_CMP_BLOCK_NUM)
    parser.add_argument("--idx-block-num", type=int, default=CSA_IDX_CACHE_BLOCK_NUM)
    parser.add_argument("--hca-state-block-num", type=int, default=HCA_COMPRESS_STATE_BLOCK_NUM)
    parser.add_argument("--csa-state-block-num", type=int, default=CSA_MAIN_STATE_BLOCK_NUM)
    parser.add_argument("--inner-state-block-num", type=int, default=CSA_INNER_STATE_BLOCK_NUM)
    parser.add_argument("--routing-mode", choices=sorted(ROUTING_MODES), default="model")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(device_id) for device_id in args.device.split(",")]
    validate_topology(ep=args.ep, tp=args.tp, device_ids=device_ids)
    if N_RANKS != args.ep:
        raise ValueError(f"import-time N_RANKS must match --ep, got {N_RANKS} vs {args.ep}")
    if LM_HEAD_TP_SIZE != args.tp:
        raise ValueError(f"import-time LM_HEAD_TP_SIZE must match --tp, got {LM_HEAD_TP_SIZE} vs {args.tp}")

    specs = build_tensor_specs(
        start_pos=args.start_pos,
        num_tokens=args.num_tokens,
        ori_block_num=args.ori_block_num,
        cmp_block_num=args.cmp_block_num,
        idx_block_num=args.idx_block_num,
        hca_state_block_num=args.hca_state_block_num,
        csa_state_block_num=args.csa_state_block_num,
        inner_state_block_num=args.inner_state_block_num,
        routing_mode=args.routing_mode,
    )
    result = run_jit(
        fn=l3_decode_fwd_mtp,
        specs=specs,
        golden_fn=golden_finite_smoke,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=False,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_scope_stats=args.enable_scope_stats,
        ),
        compare_fn={"mtp_logits": finite_tensor_compare},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
