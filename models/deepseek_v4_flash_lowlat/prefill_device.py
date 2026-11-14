# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark batch-one TP8/DP1 prefill device forward."""

import pypto.language as pl
import pypto.language.distributed as pld

from models.deepseek_v4_flash_lowlat.prefill_moe import (
    D, HC_DIM, HC_MULT, MIX_HC, MOE_INTER, N_BANK, N_EXPERTS, N_RANKS,
    REDUCE_WINDOW_ROWS, T, TOPK, VOCAB, prefill_moe_tp, clear_prefill_moe_tp_signals,
)
N_LOCAL = N_BANK
N_EXPERTS_GLOBAL = N_EXPERTS

from config import FLASH as MODEL_CONFIG
from prefill_swa import (
    prefill_attention_swa_cp,
)
from prefill_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    CMP_STORAGE_BLOCK_SIZE as HCA_CMP_STORAGE_BLOCK_SIZE,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    SPARSE_CMP_MAX_BLOCKS as HCA_CMP_MAX_BLOCKS,
    prefill_attention_hca_cp,
)
from prefill_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    H,
    HEAD_DIM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS as CSA_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    prefill_attention_csa_cp,
)
from prefill_cp_token_allgather import (
    PREFILL_GROUP_CAP,
    TP_SIZE,
)
from prefill_metadata import (
    CHECKPOINT_TILES_DYN, CHECKPOINTS_PER_TILE,
    QUERY_START_LOC_DYN, REQUESTS_DYN, lower_local_request_ids,
)
from prefill_o_proj import (
    O_PROJ_LOCAL_GROUPS,
    O_PROJ_SCRATCH_D,
    O_PROJ_SCRATCH_GROUPS,
    O_PROJ_SCRATCH_INPUT,
    O_PROJ_SCRATCH_RANK,
    O_PROJ_WO_A_WINDOW_COLS,
    O_PROJ_WO_A_WINDOW_ROWS,
    O_PROJ_WO_B_WINDOW_COLS,
    O_PROJ_WO_B_WINDOW_ROWS,
    retire_o_proj_weight_signals,
)
from hc_head import hc_head
from lm_head import (
    GRAMMAR_SEGMENT_WORDS,
    GREEDY_GRID_ROWS,
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    SAMPLED_IDS_PAD,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
    greedy_sample,
    lm_head,
)
from rmsnorm import rms_norm


# Dynamic shape variables.
FWD_TOKENS_DYN = pl.dynamic("PREFILL_FWD_TOKENS_DYN")
FWD_GROUP_TOKENS_DYN = pl.dynamic("PREFILL_FWD_GROUP_TOKENS_DYN")
FWD_ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
FWD_HCA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_CMP_BLOCK_NUM_DYN")
FWD_CSA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_CMP_BLOCK_NUM_DYN")
FWD_IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
FWD_HCA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")
FWD_CSA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
FWD_INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")

# model config
MODEL_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
FWD_NUM_LAYERS = 43
CSA_NUM_LAYERS = 21
HCA_NUM_LAYERS = 20
CSA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE
HCA_COMPRESS_STATE_DIM = 2 * HCA_MAIN_OUT_DIM
CSA_COMPRESS_STATE_DIM = 2 * CSA_MAIN_OUT_DIM
CSA_INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
# Layer schedule: SWA 0-1, CSA even 2-42, and HCA odd 3-41.

# Runtime ring heaps by scope depth, halved to fit the EP8 density default
# (32 experts/rank) -- matches pypto-serving's own halved EP8 CI profile.
PREFILL_RING_HEAP = (1 * 1024 * 1024 * 1024, 1 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
LM_HEAD_COMM_EPOCH = 1

TARGET_LAYER_IDS = (40, 41, 42)  # dspark_target_layer_ids
MAIN_HIDDEN_DIM = len(TARGET_LAYER_IDS) * D

if MODEL_NUM_LAYERS != FWD_NUM_LAYERS:
    raise ValueError("DeepSeek-V4 Flash hidden layer count changed")
if len(TARGET_LAYER_IDS) != 3 or TARGET_LAYER_IDS != tuple(
    range(FWD_NUM_LAYERS - len(TARGET_LAYER_IDS), FWD_NUM_LAYERS)
):
    raise ValueError(f"DSpark target layers must be the final three layers: {TARGET_LAYER_IDS}")
if N_RANKS % TP_SIZE:
    raise ValueError(f"EP world size {N_RANKS} must be divisible by CP group size {TP_SIZE}")
if N_RANKS % LM_HEAD_TP_SIZE:
    raise ValueError(f"EP world size {N_RANKS} must be divisible by LM-head TP={LM_HEAD_TP_SIZE}")
if LM_HEAD_VOCAB != MODEL_CONFIG.vocab_size:
    raise ValueError(f"LM-head vocab={LM_HEAD_VOCAB} does not match model vocab={MODEL_CONFIG.vocab_size}")


@pl.jit.incore
def _copy_target_hc_row(
    source: pl.Tensor,
    target: pl.Out[pl.Tensor],
):
    """Preserve one auxiliary HC row before the next layer overwrites it."""
    token = pl.tile.get_block_idx()
    row = pl.load(source, [token, 0, 0], [1, HC_MULT, D])
    target = pl.store(row, [token, 0, 0], target)
    return target


@pl.jit(auto_scope=False)
def prefill_tp8_fwd(
    x_hc: pl.InOut[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    state_checkpoint_ends: pl.Tensor[[REQUESTS_DYN, CHECKPOINT_TILES_DYN, CHECKPOINTS_PER_TILE], pl.INT32],
    hc_attn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[FWD_NUM_LAYERS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[FWD_NUM_LAYERS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_kv: pl.InOut[pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    swa_freqs_cos: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    swa_freqs_sin: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    compressed_freqs_cos: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    compressed_freqs_sin: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_freqs_cos: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_freqs_sin: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_freqs_cos: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_freqs_sin: pl.Tensor[[FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[REQUESTS_DYN, HCA_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[REQUESTS_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32],
    o_proj_wo_a_full: pl.Out[pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16]],
    o_proj_wo_b_full: pl.Out[pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_D, O_LORA], pl.INT8]],
    attn_stage: pl.InOut[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    dspark_target_hidden: pl.Out[pl.Tensor[[FWD_TOKENS_DYN, MAIN_HIDDEN_DIM], pl.BF16]],
    x_out: pl.Out[pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    grammar_mask: pl.Tensor[[MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    terminal: pl.Scalar[pl.INT32],
    tp_reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    tp_reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run prefill; a zero query_start_loc terminal marks an empty DP group.

    Query boundaries must agree across each TP/CP group. Empty groups retain
    the physical token shape and participate in every EP MoE wave.
    """
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    state_checkpoint_ends.bind_dynamic(0, REQUESTS_DYN)
    state_checkpoint_ends.bind_dynamic(1, CHECKPOINT_TILES_DYN)
    hca_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    csa_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    ori_block_table.bind_dynamic(0, REQUESTS_DYN)
    hca_cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    csa_cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    idx_block_table.bind_dynamic(0, REQUESTS_DYN)
    dspark_target_hidden.bind_dynamic(0, FWD_TOKENS_DYN)
    swa_freqs_cos.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    swa_freqs_sin.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    compressed_freqs_cos.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    compressed_freqs_sin.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    hca_cmp_freqs_cos.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    hca_cmp_freqs_sin.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_cmp_freqs_cos.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_cmp_freqs_sin.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    group_base = my_rank // TP_SIZE * TP_SIZE
    tp_rank = my_rank % TP_SIZE
    local_tokens = pl.tensor.dim(position_ids_local, 0)
    request_count = pl.tensor.dim(query_start_loc, 0) - 1
    # Query boundaries must agree across the TP/CP group.
    group_tokens = pl.read(query_start_loc, [request_count])
    local_request_ids = pl.create_tensor([local_tokens], dtype=pl.INT32)
    lower_local_request_ids(query_start_loc, local_request_ids, tp_rank * local_tokens)
    ori_block_num = pl.tensor.dim(kv_cache, 0) // FWD_NUM_LAYERS
    hca_cmp_block_num = pl.tensor.dim(hca_cmp_kv, 0) // HCA_NUM_LAYERS
    csa_cmp_block_num = pl.tensor.dim(csa_cmp_kv, 0) // CSA_NUM_LAYERS
    hca_state_block_num = pl.tensor.dim(hca_compress_state, 0) // HCA_NUM_LAYERS
    csa_state_block_num = pl.tensor.dim(csa_compress_state, 0) // CSA_NUM_LAYERS
    inner_state_block_num = pl.tensor.dim(csa_inner_compress_state, 0) // CSA_NUM_LAYERS
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0) // CSA_NUM_LAYERS

    stage_token = pl.create_tensor([1], dtype=pl.INT32)
    layer_completion = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_order_state_init"):
        pl.write(stage_token, [0], pl.cast(0, pl.INT32))
        pl.write(layer_completion, [0], pl.cast(0, pl.INT32))

    # Layer 0: SWA.
    with pl.scope():
        layer_l0 = pl.const(0, pl.INT32)
        mix_start_l0 = layer_l0 * MIX_HC
        scale_start_l0 = layer_l0 * 3
        d_start_l0 = layer_l0 * D
        q_lora_start_l0 = layer_l0 * Q_LORA
        q_head_start_l0 = layer_l0 * H * HEAD_DIM
        head_dim_start_l0 = layer_l0 * HEAD_DIM
        attn_head_start_l0 = layer_l0 * H
        o_proj_group_start_l0 = layer_l0 * O_PROJ_LOCAL_GROUPS
        expert_start_l0 = layer_l0 * N_EXPERTS_GLOBAL
        vocab_start_l0 = layer_l0 * VOCAB
        local_expert_start_l0 = layer_l0 * N_LOCAL

        hc_attn_fn_l0 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_l0, 0])
        hc_attn_scale_l0 = pl.slice(hc_attn_scale, [3], [scale_start_l0])
        hc_attn_base_l0 = pl.slice(hc_attn_base, [MIX_HC], [mix_start_l0])
        attn_norm_w_l0 = pl.slice(attn_norm_w, [D], [d_start_l0])
        wq_a_l0 = wq_a[layer_l0]
        wq_b_l0 = wq_b[layer_l0]
        wq_b_scale_l0 = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_l0])
        wkv_l0 = pl.slice(wkv, [D, HEAD_DIM], [d_start_l0, 0])
        gamma_cq_l0 = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_l0])
        gamma_ckv_l0 = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_l0])
        kv_cache_l0 = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [layer_l0 * ori_block_num, 0, 0, 0])
        attn_sink_l0 = pl.slice(attn_sink, [H], [attn_head_start_l0])
        wo_a_l0 = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_l0, 0, 0])
        wo_b_l0 = pl.slice(wo_b, [O_PROJ_LOCAL_GROUPS, D, O_LORA], [o_proj_group_start_l0, 0, 0])
        wo_b_scale_l0 = pl.slice(wo_b_scale, [D], [d_start_l0])

        hc_ffn_fn_l0 = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_l0, 0])
        hc_ffn_scale_l0 = pl.slice(hc_ffn_scale, [3], [scale_start_l0])
        hc_ffn_base_l0 = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_l0])
        norm_w_l0 = pl.slice(norm_w, [D], [d_start_l0])
        gate_w_l0 = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_l0, 0])
        gate_bias_l0 = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_l0])
        tid2eid_l0 = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_l0, 0])
        routed_w1_l0 = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_l0, 0, 0])
        routed_w1_scale_l0 = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_l0, 0])
        routed_w3_l0 = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_l0, 0, 0])
        routed_w3_scale_l0 = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_l0, 0])
        routed_w2_l0 = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_l0, 0, 0])
        routed_w2_scale_l0 = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_l0, 0])

        with pl.scope():
            attn_stage_step = attn_stage
            gather_signal_step = gather_signal
            o_proj_wo_b_full_step = o_proj_wo_b_full
            if group_tokens > 0:
                prefill_attention_swa_cp(
                    x_hc,
                    hc_attn_fn_l0, hc_attn_scale_l0, hc_attn_base_l0,
                    attn_norm_w_l0, wq_a_l0, wq_b_l0, wq_b_scale_l0,
                    wkv_l0, gamma_cq_l0, gamma_ckv_l0,
                    swa_freqs_cos, swa_freqs_sin,
                    kv_cache_l0, ori_block_table, ori_slot_mapping_full,
                    position_ids_local, position_ids_full, local_request_ids,
                    attn_sink_l0, wo_a_l0, wo_b_l0, wo_b_scale_l0,
                    o_proj_wo_a_full, o_proj_wo_b_full_step,
                    attn_stage_step,
                    gather_window, gather_signal_step,
                    o_proj_wo_a_window, o_proj_wo_b_window,
                    o_proj_weight_ready, o_proj_weight_consumed,
                    layer_completion,
                    group_base, tp_rank, layer_l0 + pl.const(1, pl.INT32),
                )

        with pl.scope():
            prefill_moe_tp(
                attn_stage, hc_ffn_fn_l0, hc_ffn_scale_l0, hc_ffn_base_l0,
                norm_w_l0, gate_w_l0, gate_bias_l0, tid2eid_l0, input_ids,
                routed_w1_l0, routed_w1_scale_l0, routed_w3_l0, routed_w3_scale_l0,
                routed_w2_l0, routed_w2_scale_l0, x_hc, stage_token, layer_completion,
                tp_reduce_window, tp_reduce_signal, layer_l0, group_tokens, my_rank,
                layer_l0 * pl.cast((group_tokens + T - 1) // T, pl.INT32),
            )

    # Layer 1: SWA.
    with pl.scope():
        layer_l1 = pl.const(1, pl.INT32)
        mix_start_l1 = layer_l1 * MIX_HC
        scale_start_l1 = layer_l1 * 3
        d_start_l1 = layer_l1 * D
        q_lora_start_l1 = layer_l1 * Q_LORA
        q_head_start_l1 = layer_l1 * H * HEAD_DIM
        head_dim_start_l1 = layer_l1 * HEAD_DIM
        attn_head_start_l1 = layer_l1 * H
        o_proj_group_start_l1 = layer_l1 * O_PROJ_LOCAL_GROUPS
        expert_start_l1 = layer_l1 * N_EXPERTS_GLOBAL
        vocab_start_l1 = layer_l1 * VOCAB
        local_expert_start_l1 = layer_l1 * N_LOCAL

        hc_attn_fn_l1 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_l1, 0])
        hc_attn_scale_l1 = pl.slice(hc_attn_scale, [3], [scale_start_l1])
        hc_attn_base_l1 = pl.slice(hc_attn_base, [MIX_HC], [mix_start_l1])
        attn_norm_w_l1 = pl.slice(attn_norm_w, [D], [d_start_l1])
        wq_a_l1 = wq_a[layer_l1]
        wq_b_l1 = wq_b[layer_l1]
        wq_b_scale_l1 = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_l1])
        wkv_l1 = pl.slice(wkv, [D, HEAD_DIM], [d_start_l1, 0])
        gamma_cq_l1 = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_l1])
        gamma_ckv_l1 = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_l1])
        kv_cache_l1 = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [layer_l1 * ori_block_num, 0, 0, 0])
        attn_sink_l1 = pl.slice(attn_sink, [H], [attn_head_start_l1])
        wo_a_l1 = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_l1, 0, 0])
        wo_b_l1 = pl.slice(wo_b, [O_PROJ_LOCAL_GROUPS, D, O_LORA], [o_proj_group_start_l1, 0, 0])
        wo_b_scale_l1 = pl.slice(wo_b_scale, [D], [d_start_l1])

        hc_ffn_fn_l1 = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_l1, 0])
        hc_ffn_scale_l1 = pl.slice(hc_ffn_scale, [3], [scale_start_l1])
        hc_ffn_base_l1 = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_l1])
        norm_w_l1 = pl.slice(norm_w, [D], [d_start_l1])
        gate_w_l1 = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_l1, 0])
        gate_bias_l1 = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_l1])
        tid2eid_l1 = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_l1, 0])
        routed_w1_l1 = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_l1, 0, 0])
        routed_w1_scale_l1 = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_l1, 0])
        routed_w3_l1 = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_l1, 0, 0])
        routed_w3_scale_l1 = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_l1, 0])
        routed_w2_l1 = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_l1, 0, 0])
        routed_w2_scale_l1 = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_l1, 0])

        with pl.scope():
            attn_stage_step = attn_stage
            gather_signal_step = gather_signal
            o_proj_wo_b_full_step = o_proj_wo_b_full
            if group_tokens > 0:
                prefill_attention_swa_cp(
                    x_hc,
                    hc_attn_fn_l1, hc_attn_scale_l1, hc_attn_base_l1,
                    attn_norm_w_l1, wq_a_l1, wq_b_l1, wq_b_scale_l1,
                    wkv_l1, gamma_cq_l1, gamma_ckv_l1,
                    swa_freqs_cos, swa_freqs_sin,
                    kv_cache_l1, ori_block_table, ori_slot_mapping_full,
                    position_ids_local, position_ids_full, local_request_ids,
                    attn_sink_l1, wo_a_l1, wo_b_l1, wo_b_scale_l1,
                    o_proj_wo_a_full, o_proj_wo_b_full_step,
                    attn_stage_step,
                    gather_window, gather_signal_step,
                    o_proj_wo_a_window, o_proj_wo_b_window,
                    o_proj_weight_ready, o_proj_weight_consumed,
                    layer_completion,
                    group_base, tp_rank, layer_l1 + pl.const(1, pl.INT32),
                )

        with pl.scope():
            prefill_moe_tp(
                attn_stage, hc_ffn_fn_l1, hc_ffn_scale_l1, hc_ffn_base_l1,
                norm_w_l1, gate_w_l1, gate_bias_l1, tid2eid_l1, input_ids,
                routed_w1_l1, routed_w1_scale_l1, routed_w3_l1, routed_w3_scale_l1,
                routed_w2_l1, routed_w2_scale_l1, x_hc, stage_token, layer_completion,
                tp_reduce_window, tp_reduce_signal, layer_l1, group_tokens, my_rank,
                layer_l1 * pl.cast((group_tokens + T - 1) // T, pl.INT32),
            )

    group_rows = pl.tensor.dim(x_hc, 0)
    local_tokens = pl.tensor.dim(position_ids_local, 0)
    local_start = pl.cast(tp_rank, pl.INDEX) * pl.cast(local_tokens, pl.INDEX)
    target_l41_start = pl.cast(local_tokens, pl.INDEX)
    target_hc_rows = 2 * local_tokens
    target_hc_stack = pl.create_tensor([target_hc_rows, HC_MULT, D], dtype=pl.FP32)

    # Layers 2-41: CSA/HCA pairs.
    for pair_order in pl.range(HCA_NUM_LAYERS):
        attention_order = pl.cast(pair_order, pl.INT32)

        with pl.scope():
            csa_layer = attention_order * 2 + pl.const(2, pl.INT32)
            mix_start_csa = csa_layer * MIX_HC
            scale_start_csa = csa_layer * 3
            d_start_csa = csa_layer * D
            q_lora_start_csa = csa_layer * Q_LORA
            q_head_start_csa = csa_layer * H * HEAD_DIM
            head_dim_start_csa = csa_layer * HEAD_DIM
            attn_head_start_csa = csa_layer * H
            o_proj_group_start_csa = csa_layer * O_PROJ_LOCAL_GROUPS
            expert_start_csa = csa_layer * N_EXPERTS_GLOBAL
            vocab_start_csa = csa_layer * VOCAB
            local_expert_start_csa = csa_layer * N_LOCAL

            hc_attn_fn_csa = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_csa, 0])
            hc_attn_scale_csa = pl.slice(hc_attn_scale, [3], [scale_start_csa])
            hc_attn_base_csa = pl.slice(hc_attn_base, [MIX_HC], [mix_start_csa])
            attn_norm_w_csa = pl.slice(attn_norm_w, [D], [d_start_csa])
            wq_a_csa = wq_a[csa_layer]
            wq_b_csa = wq_b[csa_layer]
            wq_b_scale_csa = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_csa])
            wkv_csa = pl.slice(wkv, [D, HEAD_DIM], [d_start_csa, 0])
            gamma_cq_csa = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_csa])
            gamma_ckv_csa = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_csa])
            kv_cache_csa = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer * ori_block_num, 0, 0, 0])
            attn_sink_csa = pl.slice(attn_sink, [H], [attn_head_start_csa])
            wo_a_csa = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_csa, 0, 0])
            wo_b_csa = pl.slice(wo_b, [O_PROJ_LOCAL_GROUPS, D, O_LORA], [o_proj_group_start_csa, 0, 0])
            wo_b_scale_csa = pl.slice(wo_b_scale, [D], [d_start_csa])

            hc_ffn_fn_csa = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_csa, 0])
            hc_ffn_scale_csa = pl.slice(hc_ffn_scale, [3], [scale_start_csa])
            hc_ffn_base_csa = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_csa])
            norm_w_csa = pl.slice(norm_w, [D], [d_start_csa])
            gate_w_csa = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_csa, 0])
            gate_bias_csa = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_csa])
            tid2eid_csa = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_csa, 0])
            routed_w1_csa = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_csa, 0, 0])
            routed_w1_scale_csa = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_csa, 0])
            routed_w3_csa = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_csa, 0, 0])
            routed_w3_scale_csa = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_csa, 0])
            routed_w2_csa = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_csa, 0, 0])
            routed_w2_scale_csa = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_csa, 0])

            csa_compress_start_csa = attention_order * CSA_MAIN_OUT_DIM
            csa_ape_start_csa = attention_order * CSA_COMPRESS_RATIO
            csa_norm_start_csa = attention_order * HEAD_DIM
            csa_idx_head_start_csa = attention_order * IDX_HEAD_DIM
            csa_idx_q_lora_start_csa = attention_order * Q_LORA
            csa_idx_q_head_start_csa = attention_order * IDX_N_HEADS * IDX_HEAD_DIM
            csa_inner_start_csa = attention_order * INNER_OUT_DIM
            csa_state_start_csa = attention_order * csa_state_block_num
            csa_inner_state_start_csa = attention_order * inner_state_block_num
            csa_cmp_start_csa = attention_order * csa_cmp_block_num
            csa_idx_start_csa = attention_order * idx_block_num

            csa_cmp_wkv_csa = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [csa_compress_start_csa, 0])
            csa_cmp_wgate_csa = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [csa_compress_start_csa, 0])
            csa_cmp_ape_csa = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [csa_ape_start_csa, 0])
            csa_cmp_norm_w_csa = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [csa_norm_start_csa])
            csa_compress_state_csa = pl.slice(
                csa_compress_state,
                [csa_state_block_num, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
                [csa_state_start_csa, 0, 0],
            )
            csa_hadamard_idx_csa = pl.slice(csa_hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [csa_idx_head_start_csa, 0])
            csa_idx_wq_b_csa = pl.slice(csa_idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [csa_idx_q_lora_start_csa, 0])
            csa_idx_wq_b_scale_csa = pl.slice(csa_idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [csa_idx_q_head_start_csa])
            csa_weights_proj_csa = csa_weights_proj[attention_order]
            csa_inner_wkv_csa = pl.slice(csa_inner_wkv, [INNER_OUT_DIM, D], [csa_inner_start_csa, 0])
            csa_inner_wgate_csa = pl.slice(csa_inner_wgate, [INNER_OUT_DIM, D], [csa_inner_start_csa, 0])
            csa_inner_ape_csa = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [csa_ape_start_csa, 0])
            csa_inner_norm_w_csa = pl.slice(csa_inner_norm_w, [IDX_HEAD_DIM], [csa_idx_head_start_csa])
            csa_inner_compress_state_csa = pl.slice(
                csa_inner_compress_state,
                [inner_state_block_num, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM],
                [csa_inner_state_start_csa, 0, 0],
            )
            csa_cmp_kv_csa = pl.slice(
                csa_cmp_kv,
                [csa_cmp_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                [csa_cmp_start_csa, 0, 0, 0],
            )
            idx_kv_cache_csa = pl.slice(
                idx_kv_cache,
                [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
                [csa_idx_start_csa, 0, 0, 0],
            )
            idx_kv_scale_csa = pl.slice(
                idx_kv_scale,
                [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1],
                [csa_idx_start_csa, 0, 0, 0],
            )

            with pl.scope():
                attn_stage_step = attn_stage
                gather_signal_step = gather_signal
                o_proj_wo_b_full_step = o_proj_wo_b_full
                if group_tokens > 0:
                    prefill_attention_csa_cp(
                        x_hc,
                        query_start_loc, state_checkpoint_ends,
                        hc_attn_fn_csa, hc_attn_scale_csa, hc_attn_base_csa,
                        attn_norm_w_csa, wq_a_csa, wq_b_csa, wq_b_scale_csa,
                        wkv_csa, gamma_cq_csa, gamma_ckv_csa,
                        compressed_freqs_cos, compressed_freqs_sin,
                        csa_cmp_freqs_cos, csa_cmp_freqs_sin,
                        csa_cmp_wkv_csa, csa_cmp_wgate_csa, csa_cmp_ape_csa, csa_cmp_norm_w_csa,
                        csa_compress_state_csa, csa_compress_state_block_table,
                        csa_hadamard_idx_csa,
                        csa_idx_wq_b_csa, csa_idx_wq_b_scale_csa, csa_weights_proj_csa,
                        csa_inner_wkv_csa, csa_inner_wgate_csa, csa_inner_ape_csa, csa_inner_norm_w_csa,
                        csa_inner_compress_state_csa, csa_inner_compress_state_block_table,
                        kv_cache_csa, ori_block_table, ori_slot_mapping_full,
                        csa_cmp_kv_csa, csa_cmp_block_table,
                        idx_kv_cache_csa, idx_kv_scale_csa, idx_block_table,
                        position_ids_local, position_ids_full, local_request_ids,
                        csa_cmp_slot_mapping_full, csa_idx_slot_mapping_full,
                        csa_state_slot_mapping_full, csa_inner_state_slot_mapping_full,
                        attn_sink_csa, wo_a_csa, wo_b_csa, wo_b_scale_csa,
                        o_proj_wo_a_full, o_proj_wo_b_full_step,
                        attn_stage_step,
                        gather_window, gather_signal_step,
                        o_proj_wo_a_window, o_proj_wo_b_window,
                        o_proj_weight_ready, o_proj_weight_consumed,
                        layer_completion,
                        group_base, tp_rank, csa_layer + pl.const(1, pl.INT32),
                    )

            with pl.scope():
                prefill_moe_tp(
                    attn_stage, hc_ffn_fn_csa, hc_ffn_scale_csa, hc_ffn_base_csa,
                    norm_w_csa, gate_w_csa, gate_bias_csa, tid2eid_csa, input_ids,
                    routed_w1_csa, routed_w1_scale_csa, routed_w3_csa, routed_w3_scale_csa,
                    routed_w2_csa, routed_w2_scale_csa, x_hc, stage_token, layer_completion,
                    tp_reduce_window, tp_reduce_signal, csa_layer, group_tokens, my_rank,
                    csa_layer * pl.cast((group_tokens + T - 1) // T, pl.INT32),
                )
                if group_tokens > 0:
                    if pair_order == HCA_NUM_LAYERS - 1:
                        target_x_hc_l40 = pl.slice(
                            x_hc,
                            [local_tokens, HC_MULT, D],
                            [local_start, 0, 0],
                        )
                        target_hc_l40 = pl.slice(
                            target_hc_stack,
                            [local_tokens, HC_MULT, D],
                            [0, 0, 0],
                        )
                        with pl.spmd(local_tokens, name_hint="prefill_fwd_capture_target_hc_l40"):
                            target_hc_l40 = _copy_target_hc_row(target_x_hc_l40, target_hc_l40)

        with pl.scope():
            hca_layer = attention_order * 2 + pl.const(3, pl.INT32)
            mix_start_hca = hca_layer * MIX_HC
            scale_start_hca = hca_layer * 3
            d_start_hca = hca_layer * D
            q_lora_start_hca = hca_layer * Q_LORA
            q_head_start_hca = hca_layer * H * HEAD_DIM
            head_dim_start_hca = hca_layer * HEAD_DIM
            attn_head_start_hca = hca_layer * H
            o_proj_group_start_hca = hca_layer * O_PROJ_LOCAL_GROUPS
            expert_start_hca = hca_layer * N_EXPERTS_GLOBAL
            vocab_start_hca = hca_layer * VOCAB
            local_expert_start_hca = hca_layer * N_LOCAL

            hc_attn_fn_hca = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_hca, 0])
            hc_attn_scale_hca = pl.slice(hc_attn_scale, [3], [scale_start_hca])
            hc_attn_base_hca = pl.slice(hc_attn_base, [MIX_HC], [mix_start_hca])
            attn_norm_w_hca = pl.slice(attn_norm_w, [D], [d_start_hca])
            wq_a_hca = wq_a[hca_layer]
            wq_b_hca = wq_b[hca_layer]
            wq_b_scale_hca = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_hca])
            wkv_hca = pl.slice(wkv, [D, HEAD_DIM], [d_start_hca, 0])
            gamma_cq_hca = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_hca])
            gamma_ckv_hca = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_hca])
            kv_cache_hca = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [hca_layer * ori_block_num, 0, 0, 0])
            attn_sink_hca = pl.slice(attn_sink, [H], [attn_head_start_hca])
            wo_a_hca = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_hca, 0, 0])
            wo_b_hca = pl.slice(wo_b, [O_PROJ_LOCAL_GROUPS, D, O_LORA], [o_proj_group_start_hca, 0, 0])
            wo_b_scale_hca = pl.slice(wo_b_scale, [D], [d_start_hca])

            hc_ffn_fn_hca = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_hca, 0])
            hc_ffn_scale_hca = pl.slice(hc_ffn_scale, [3], [scale_start_hca])
            hc_ffn_base_hca = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_hca])
            norm_w_hca = pl.slice(norm_w, [D], [d_start_hca])
            gate_w_hca = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_hca, 0])
            gate_bias_hca = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_hca])
            tid2eid_hca = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_hca, 0])
            routed_w1_hca = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_hca, 0, 0])
            routed_w1_scale_hca = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_hca, 0])
            routed_w3_hca = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_hca, 0, 0])
            routed_w3_scale_hca = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_hca, 0])
            routed_w2_hca = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_hca, 0, 0])
            routed_w2_scale_hca = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_hca, 0])

            hca_compress_start_hca = attention_order * HCA_MAIN_OUT_DIM
            hca_ape_start_hca = attention_order * HCA_COMPRESS_RATIO
            hca_norm_start_hca = attention_order * HEAD_DIM
            hca_state_start_hca = attention_order * hca_state_block_num
            hca_cmp_start_hca = attention_order * hca_cmp_block_num
            hca_cmp_wkv_hca = pl.slice(hca_cmp_wkv, [HCA_MAIN_OUT_DIM, D], [hca_compress_start_hca, 0])
            hca_cmp_wgate_hca = pl.slice(hca_cmp_wgate, [HCA_MAIN_OUT_DIM, D], [hca_compress_start_hca, 0])
            hca_cmp_ape_hca = pl.slice(hca_cmp_ape, [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], [hca_ape_start_hca, 0])
            hca_cmp_norm_w_hca = pl.slice(hca_cmp_norm_w, [HEAD_DIM], [hca_norm_start_hca])
            hca_compress_state_hca = pl.slice(
                hca_compress_state,
                [hca_state_block_num, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
                [hca_state_start_hca, 0, 0],
            )
            hca_cmp_kv_hca = pl.slice(
                hca_cmp_kv,
                [hca_cmp_block_num, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                [hca_cmp_start_hca, 0, 0, 0],
            )

            with pl.scope():
                attn_stage_step = attn_stage
                gather_signal_step = gather_signal
                o_proj_wo_b_full_step = o_proj_wo_b_full
                if group_tokens > 0:
                    prefill_attention_hca_cp(
                        x_hc,
                        query_start_loc, state_checkpoint_ends, local_request_ids,
                        hc_attn_fn_hca, hc_attn_scale_hca, hc_attn_base_hca,
                        attn_norm_w_hca, wq_a_hca, wq_b_hca, wq_b_scale_hca,
                        wkv_hca, gamma_cq_hca, gamma_ckv_hca,
                        compressed_freqs_cos, compressed_freqs_sin,
                        hca_cmp_freqs_cos, hca_cmp_freqs_sin,
                        hca_cmp_wkv_hca, hca_cmp_wgate_hca, hca_cmp_ape_hca, hca_cmp_norm_w_hca,
                        hca_compress_state_hca, hca_compress_state_block_table,
                        kv_cache_hca, ori_slot_mapping_full, ori_block_table,
                        hca_cmp_kv_hca, hca_cmp_block_table,
                        position_ids_local, position_ids_full,
                        hca_cmp_slot_mapping_full, hca_state_slot_mapping_full,
                        attn_sink_hca, wo_a_hca, wo_b_hca, wo_b_scale_hca,
                        o_proj_wo_a_full, o_proj_wo_b_full_step,
                        attn_stage_step,
                        gather_window, gather_signal_step,
                        o_proj_wo_a_window, o_proj_wo_b_window,
                        o_proj_weight_ready, o_proj_weight_consumed,
                        layer_completion,
                        group_base, tp_rank, hca_layer + pl.const(1, pl.INT32),
                    )

            with pl.scope():
                prefill_moe_tp(
                    attn_stage, hc_ffn_fn_hca, hc_ffn_scale_hca, hc_ffn_base_hca,
                    norm_w_hca, gate_w_hca, gate_bias_hca, tid2eid_hca, input_ids,
                    routed_w1_hca, routed_w1_scale_hca, routed_w3_hca, routed_w3_scale_hca,
                    routed_w2_hca, routed_w2_scale_hca, x_hc, stage_token, layer_completion,
                    tp_reduce_window, tp_reduce_signal, hca_layer, group_tokens, my_rank,
                    hca_layer * pl.cast((group_tokens + T - 1) // T, pl.INT32),
                )
                if group_tokens > 0:
                    if pair_order == HCA_NUM_LAYERS - 1:
                        target_x_hc_l41 = pl.slice(
                            x_hc,
                            [local_tokens, HC_MULT, D],
                            [local_start, 0, 0],
                        )
                        target_hc_l41 = pl.slice(
                            target_hc_stack,
                            [local_tokens, HC_MULT, D],
                            [target_l41_start, 0, 0],
                        )
                        with pl.spmd(local_tokens, name_hint="prefill_fwd_capture_target_hc_l41"):
                            target_hc_l41 = _copy_target_hc_row(target_x_hc_l41, target_hc_l41)

    # Layer 42: CSA order 20.
    with pl.scope():
        layer_last = pl.const(42, pl.INT32)
        csa_order_last = pl.const(20, pl.INT32)
        mix_start_last = layer_last * MIX_HC
        scale_start_last = layer_last * 3
        d_start_last = layer_last * D
        q_lora_start_last = layer_last * Q_LORA
        q_head_start_last = layer_last * H * HEAD_DIM
        head_dim_start_last = layer_last * HEAD_DIM
        attn_head_start_last = layer_last * H
        o_proj_group_start_last = layer_last * O_PROJ_LOCAL_GROUPS
        expert_start_last = layer_last * N_EXPERTS_GLOBAL
        vocab_start_last = layer_last * VOCAB
        local_expert_start_last = layer_last * N_LOCAL

        hc_attn_fn_last = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start_last, 0])
        hc_attn_scale_last = pl.slice(hc_attn_scale, [3], [scale_start_last])
        hc_attn_base_last = pl.slice(hc_attn_base, [MIX_HC], [mix_start_last])
        attn_norm_w_last = pl.slice(attn_norm_w, [D], [d_start_last])
        wq_a_last = wq_a[layer_last]
        wq_b_last = wq_b[layer_last]
        wq_b_scale_last = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start_last])
        wkv_last = pl.slice(wkv, [D, HEAD_DIM], [d_start_last, 0])
        gamma_cq_last = pl.slice(gamma_cq, [Q_LORA], [q_lora_start_last])
        gamma_ckv_last = pl.slice(gamma_ckv, [HEAD_DIM], [head_dim_start_last])
        kv_cache_last = pl.slice(kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [layer_last * ori_block_num, 0, 0, 0])
        attn_sink_last = pl.slice(attn_sink, [H], [attn_head_start_last])
        wo_a_last = pl.slice(wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start_last, 0, 0])
        wo_b_last = pl.slice(wo_b, [O_PROJ_LOCAL_GROUPS, D, O_LORA], [o_proj_group_start_last, 0, 0])
        wo_b_scale_last = pl.slice(wo_b_scale, [D], [d_start_last])

        hc_ffn_fn_last = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start_last, 0])
        hc_ffn_scale_last = pl.slice(hc_ffn_scale, [3], [scale_start_last])
        hc_ffn_base_last = pl.slice(hc_ffn_base, [MIX_HC], [mix_start_last])
        norm_w_last = pl.slice(norm_w, [D], [d_start_last])
        gate_w_last = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start_last, 0])
        gate_bias_last = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start_last])
        tid2eid_last = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start_last, 0])
        routed_w1_last = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start_last, 0, 0])
        routed_w1_scale_last = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start_last, 0])
        routed_w3_last = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start_last, 0, 0])
        routed_w3_scale_last = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start_last, 0])
        routed_w2_last = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start_last, 0, 0])
        routed_w2_scale_last = pl.slice(routed_w2_scale, [N_LOCAL, D], [local_expert_start_last, 0])

        csa_compress_start_last = csa_order_last * CSA_MAIN_OUT_DIM
        csa_ape_start_last = csa_order_last * CSA_COMPRESS_RATIO
        csa_norm_start_last = csa_order_last * HEAD_DIM
        csa_idx_head_start_last = csa_order_last * IDX_HEAD_DIM
        csa_idx_q_lora_start_last = csa_order_last * Q_LORA
        csa_idx_q_head_start_last = csa_order_last * IDX_N_HEADS * IDX_HEAD_DIM
        csa_inner_start_last = csa_order_last * INNER_OUT_DIM
        csa_state_start_last = csa_order_last * csa_state_block_num
        csa_inner_state_start_last = csa_order_last * inner_state_block_num
        csa_cmp_start_last = csa_order_last * csa_cmp_block_num
        csa_idx_start_last = csa_order_last * idx_block_num

        csa_cmp_wkv_last = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [csa_compress_start_last, 0])
        csa_cmp_wgate_last = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [csa_compress_start_last, 0])
        csa_cmp_ape_last = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [csa_ape_start_last, 0])
        csa_cmp_norm_w_last = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [csa_norm_start_last])
        csa_compress_state_last = pl.slice(
            csa_compress_state,
            [csa_state_block_num, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
            [csa_state_start_last, 0, 0],
        )
        csa_hadamard_idx_last = pl.slice(csa_hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [csa_idx_head_start_last, 0])
        csa_idx_wq_b_last = pl.slice(csa_idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [csa_idx_q_lora_start_last, 0])
        csa_idx_wq_b_scale_last = pl.slice(csa_idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [csa_idx_q_head_start_last])
        csa_weights_proj_last = csa_weights_proj[csa_order_last]
        csa_inner_wkv_last = pl.slice(csa_inner_wkv, [INNER_OUT_DIM, D], [csa_inner_start_last, 0])
        csa_inner_wgate_last = pl.slice(csa_inner_wgate, [INNER_OUT_DIM, D], [csa_inner_start_last, 0])
        csa_inner_ape_last = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [csa_ape_start_last, 0])
        csa_inner_norm_w_last = pl.slice(csa_inner_norm_w, [IDX_HEAD_DIM], [csa_idx_head_start_last])
        csa_inner_compress_state_last = pl.slice(
            csa_inner_compress_state,
            [inner_state_block_num, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM],
            [csa_inner_state_start_last, 0, 0],
        )
        csa_cmp_kv_last = pl.slice(
            csa_cmp_kv,
            [csa_cmp_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            [csa_cmp_start_last, 0, 0, 0],
        )
        idx_kv_cache_last = pl.slice(
            idx_kv_cache,
            [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
            [csa_idx_start_last, 0, 0, 0],
        )
        idx_kv_scale_last = pl.slice(
            idx_kv_scale,
            [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1],
            [csa_idx_start_last, 0, 0, 0],
        )

        with pl.scope():
            attn_stage_step = attn_stage
            gather_signal_step = gather_signal
            o_proj_wo_b_full_step = o_proj_wo_b_full
            if group_tokens > 0:
                prefill_attention_csa_cp(
                    x_hc,
                    query_start_loc, state_checkpoint_ends,
                    hc_attn_fn_last, hc_attn_scale_last, hc_attn_base_last,
                    attn_norm_w_last, wq_a_last, wq_b_last, wq_b_scale_last,
                    wkv_last, gamma_cq_last, gamma_ckv_last,
                    compressed_freqs_cos, compressed_freqs_sin,
                    csa_cmp_freqs_cos, csa_cmp_freqs_sin,
                    csa_cmp_wkv_last, csa_cmp_wgate_last, csa_cmp_ape_last, csa_cmp_norm_w_last,
                    csa_compress_state_last, csa_compress_state_block_table,
                    csa_hadamard_idx_last,
                    csa_idx_wq_b_last, csa_idx_wq_b_scale_last, csa_weights_proj_last,
                    csa_inner_wkv_last, csa_inner_wgate_last, csa_inner_ape_last, csa_inner_norm_w_last,
                    csa_inner_compress_state_last, csa_inner_compress_state_block_table,
                    kv_cache_last, ori_block_table, ori_slot_mapping_full,
                    csa_cmp_kv_last, csa_cmp_block_table,
                    idx_kv_cache_last, idx_kv_scale_last, idx_block_table,
                    position_ids_local, position_ids_full, local_request_ids,
                    csa_cmp_slot_mapping_full, csa_idx_slot_mapping_full,
                    csa_state_slot_mapping_full, csa_inner_state_slot_mapping_full,
                    attn_sink_last, wo_a_last, wo_b_last, wo_b_scale_last,
                    o_proj_wo_a_full, o_proj_wo_b_full_step,
                    attn_stage_step,
                    gather_window, gather_signal_step,
                    o_proj_wo_a_window, o_proj_wo_b_window,
                    o_proj_weight_ready, o_proj_weight_consumed,
                    layer_completion,
                    group_base, tp_rank, layer_last + pl.const(1, pl.INT32),
                )

        with pl.scope():
            prefill_moe_tp(
                attn_stage, hc_ffn_fn_last, hc_ffn_scale_last, hc_ffn_base_last,
                norm_w_last, gate_w_last, gate_bias_last, tid2eid_last, input_ids,
                routed_w1_last, routed_w1_scale_last, routed_w3_last, routed_w3_scale_last,
                routed_w2_last, routed_w2_scale_last, x_hc, stage_token, layer_completion,
                tp_reduce_window, tp_reduce_signal, layer_last, group_tokens, my_rank,
                layer_last * pl.cast((group_tokens + T - 1) // T, pl.INT32),
            )

    moe_retired_tid = clear_prefill_moe_tp_signals(stage_token, tp_reduce_signal)
    with pl.scope():
        if group_tokens > 0:
            retire_o_proj_weight_signals(
                layer_completion,
                o_proj_weight_ready, o_proj_weight_consumed,
                group_base, tp_rank, pl.const(43, pl.INT32),
            )

    # Auxiliary taps use HC means; only the main output uses the learned head.
    with pl.scope():
        if group_tokens > 0:
            for token in pl.spmd(local_tokens, name_hint="prefill_fwd_store_target_hidden"):
                target_l41_row = target_l41_start + token
                target_l42_row = local_start + token
                target_hc_l40_row = target_hc_stack[token : token + 1, 0 : HC_MULT, 0:D]
                target_hidden_l40 = pl.mul(pl.col_sum(pl.reshape(target_hc_l40_row, [HC_MULT, D])), 1.0 / HC_MULT)
                dspark_target_hidden[token : token + 1, 0:D] = pl.cast(target_hidden_l40, pl.BF16, mode="rint")
                target_hc_l41_row = target_hc_stack[
                    target_l41_row : target_l41_row + 1, 0 : HC_MULT, 0:D,
                ]
                target_hidden_l41 = pl.mul(pl.col_sum(pl.reshape(target_hc_l41_row, [HC_MULT, D])), 1.0 / HC_MULT)
                dspark_target_hidden[token : token + 1, D : 2 * D] = pl.cast(
                    target_hidden_l41, pl.BF16, mode="rint",
                )
                target_hc_l42_row = x_hc[target_l42_row : target_l42_row + 1, 0 : HC_MULT, 0:D]
                target_hidden_l42 = pl.mul(pl.col_sum(pl.reshape(target_hc_l42_row, [HC_MULT, D])), 1.0 / HC_MULT)
                dspark_target_hidden[token : token + 1, 2 * D : 3 * D] = pl.cast(
                    target_hidden_l42, pl.BF16, mode="rint",
                )
        else:
            for token in pl.spmd(local_tokens, name_hint="prefill_fwd_inactive_target_hidden"):
                for head in pl.range(MAIN_HIDDEN_DIM // D):
                    zero_target_hidden = pl.full([1, D], dtype=pl.BF16, value=0.0)
                    dspark_target_hidden[token : token + 1, head * D : (head + 1) * D] = zero_target_hidden
            for token in pl.spmd(group_rows, name_hint="prefill_fwd_inactive_hidden"):
                x_hc[token : token + 1, :, :] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)

    if terminal > 0:
        # All head-group members participate, including idle attention owners.
        layer42_hidden = pl.create_tensor([group_rows, D], dtype=pl.BF16)
        layer42_hidden = hc_head(x_hc, hc_head_fn, hc_head_scale, hc_head_base, layer42_hidden)
        final_norm_tid = rms_norm(layer42_hidden, final_norm_w, x_out)
        # Retire final MoE epochs before every owner's head collectives.
        head_ready_tid = pl.system.task_dummy(deps=[final_norm_tid, moe_retired_tid])
        lm_head(
            x_out, lm_head_weight, logit_row_indices, logits,
            lm_head_hidden_window, lm_head_hidden_done,
            lm_head_logits_window, lm_head_logits_done,
            my_rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE, my_rank % LM_HEAD_TP_SIZE,
            pl.const(LM_HEAD_COMM_EPOCH, pl.INT32), head_ready_tid,
        )
        greedy_sample(logits, logit_row_indices, grammar_mask, sampled_ids)
    return x_out


# DSA-CP layer-major multi-wave forward.
@pl.jit.host
def l3_prefill_tp8_fwd(
    x_hc: pl.InOut[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    query_start_loc: pl.Tensor[[N_RANKS, QUERY_START_LOC_DYN], pl.INT32],
    state_checkpoint_ends: pl.Tensor[[N_RANKS, REQUESTS_DYN, CHECKPOINT_TILES_DYN, CHECKPOINTS_PER_TILE], pl.INT32],
    hc_attn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    swa_freqs_cos: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    swa_freqs_sin: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    compressed_freqs_cos: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    compressed_freqs_sin: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_freqs_cos: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_freqs_sin: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_freqs_cos: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_freqs_sin: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, HCA_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[N_RANKS, REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hc_ffn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32],
    o_proj_wo_a_full: pl.Out[pl.Tensor[[N_RANKS, O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16]],
    o_proj_wo_b_full: pl.Out[pl.Tensor[[N_RANKS, O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_D, O_LORA], pl.INT8]],
    attn_stage: pl.InOut[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
    dspark_target_hidden: pl.Out[pl.Tensor[[N_RANKS, FWD_TOKENS_DYN, MAIN_HIDDEN_DIM], pl.BF16]],
    x_out: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    grammar_mask: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, GREEDY_GRID_ROWS, GRAMMAR_SEGMENT_WORDS], pl.INT16],
    sampled_ids: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    terminal: pl.Scalar[pl.INT32],
):
    """Run layer-major DSA-CP over a caller-padded physical token extent.

    Packed request boundaries are provided by a monotonic ``query_start_loc``
    that starts at zero and ends at total logical length ``N``. Its request
    count must match the leading dimension of every request-indexed block table,
    and metadata must be identical within a TP group. Each TP group supplies
    ``P = align_up(N, TP_SIZE)`` full rows and each rank supplies
    ``L = P // TP_SIZE`` local rows. Callers zero padded ``x_hc`` and
    ``input_ids``, use non-aliasing synthetic positions and ``-1`` cache/state
    mappings for padding, and restrict live ``logit_row_indices`` to ``[0, N)``.
    The device schedule, including MoE collectives, runs over physical P/L.
    """
    x_hc.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    query_start_loc.bind_dynamic(1, QUERY_START_LOC_DYN)
    state_checkpoint_ends.bind_dynamic(1, REQUESTS_DYN)
    state_checkpoint_ends.bind_dynamic(2, CHECKPOINT_TILES_DYN)
    hca_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    ori_block_table.bind_dynamic(1, REQUESTS_DYN)
    hca_cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    csa_cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    idx_block_table.bind_dynamic(1, REQUESTS_DYN)
    dspark_target_hidden.bind_dynamic(1, FWD_TOKENS_DYN)
    x_out.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    attn_stage.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    swa_freqs_cos.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    swa_freqs_sin.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    compressed_freqs_cos.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    compressed_freqs_sin.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_cmp_freqs_cos.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_cmp_freqs_sin.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_cmp_freqs_cos.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_cmp_freqs_sin.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    position_ids_local.bind_dynamic(1, FWD_TOKENS_DYN)
    input_ids.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    ori_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    position_ids_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)

    tp_reduce_window_buf = pld.alloc_window_buffer([REDUCE_WINDOW_ROWS, D], dtype=pl.FP32)
    tp_reduce_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_wo_a_window_buf = pld.alloc_window_buffer([O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16)
    o_proj_wo_b_window_buf = pld.alloc_window_buffer([O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8)
    o_proj_weight_ready_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_weight_consumed_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    lm_head_hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    lm_head_logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
    lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        tp_reduce_window = pld.window(tp_reduce_window_buf, [REDUCE_WINDOW_ROWS, D], dtype=pl.FP32)
        tp_reduce_signal = pld.window(tp_reduce_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_wo_a_window = pld.window(
            o_proj_wo_a_window_buf, [O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16,
        )
        o_proj_wo_b_window = pld.window(
            o_proj_wo_b_window_buf, [O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8,
        )
        o_proj_weight_ready = pld.window(o_proj_weight_ready_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_weight_consumed = pld.window(o_proj_weight_consumed_buf, [TP_SIZE, 1], dtype=pl.INT32)
        lm_head_hidden_window = pld.window(lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        lm_head_hidden_done = pld.window(lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        lm_head_logits_window = pld.window(lm_head_logits_window_buf, [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
        lm_head_logits_done = pld.window(lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        prefill_tp8_fwd(
            x_hc[r],
            query_start_loc[r], state_checkpoint_ends[r],
            hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r],
            attn_norm_w[r], wq_a[r], wq_b[r], wq_b_scale[r],
            wkv[r], gamma_cq[r], gamma_ckv[r],
            kv_cache[r], attn_sink[r], wo_a[r], wo_b[r], wo_b_scale[r],
            hca_cmp_kv[r], csa_cmp_kv[r],
            hca_cmp_wkv[r], hca_cmp_wgate[r], hca_cmp_ape[r],
            hca_cmp_norm_w[r], hca_compress_state[r],
            csa_cmp_wkv[r], csa_cmp_wgate[r], csa_cmp_ape[r],
            csa_cmp_norm_w[r], csa_compress_state[r],
            csa_hadamard_idx[r], csa_idx_wq_b[r],
            csa_idx_wq_b_scale[r], csa_weights_proj[r],
            csa_inner_wkv[r], csa_inner_wgate[r],
            csa_inner_ape[r], csa_inner_norm_w[r],
            csa_inner_compress_state[r], idx_kv_cache[r], idx_kv_scale[r],
            hca_compress_state_block_table[r],
            csa_compress_state_block_table[r],
            csa_inner_compress_state_block_table[r],
            swa_freqs_cos[r], swa_freqs_sin[r],
            compressed_freqs_cos[r], compressed_freqs_sin[r],
            hca_cmp_freqs_cos[r], hca_cmp_freqs_sin[r],
            csa_cmp_freqs_cos[r], csa_cmp_freqs_sin[r],
            ori_block_table[r], hca_cmp_block_table[r],
            csa_cmp_block_table[r], idx_block_table[r],
            ori_slot_mapping_full[r], position_ids_local[r],
            position_ids_full[r], input_ids[r],
            hca_cmp_slot_mapping_full[r], hca_state_slot_mapping_full[r],
            csa_cmp_slot_mapping_full[r], csa_idx_slot_mapping_full[r],
            csa_state_slot_mapping_full[r], csa_inner_state_slot_mapping_full[r],
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r],
            routed_w1[r], routed_w1_scale[r],
            routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            o_proj_wo_a_full[r], o_proj_wo_b_full[r],
            attn_stage[r],
            hc_head_fn[r], hc_head_scale[r], hc_head_base[r],
            final_norm_w[r], lm_head_weight[r], logit_row_indices[r],
            dspark_target_hidden[r],
            x_out[r], logits[r], grammar_mask[r], sampled_ids[r], terminal,
            tp_reduce_window, tp_reduce_signal,
            gather_window, gather_signal,
            o_proj_wo_a_window, o_proj_wo_b_window,
            o_proj_weight_ready, o_proj_weight_consumed,
            lm_head_hidden_window, lm_head_hidden_done,
            lm_head_logits_window, lm_head_logits_done,
            r,
            device=r,
        )
    return x_out, logits, sampled_ids, dspark_target_hidden
