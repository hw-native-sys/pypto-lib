# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""GLM-5.3-Flash shared chunk-capable attention and feed-forward layer composition."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__" and not any(tok == "--ep" or tok.startswith("--ep=") for tok in sys.argv):
    sys.argv += ["--ep", "2"]

import argparse
from dataclasses import dataclass

import pypto.language as pl
import pypto.language.distributed as pld
import torch

import models.glm5_3_flash.config as C
from models.glm5_3_flash.all_reduce import tp_all_reduce
from models.glm5_3_flash.attention import (
    MlaTensors, mla_cache_write, mla_epilog_w8a8, mla_prolog_w8a8, prefill_sparse_attn, synthetic_mla,
    topk_attention_mask,
)
from models.glm5_3_flash.config import (
    BLOCK_SIZE, DENSE_INTER, EP_SIZE, FLASH, HC_DIM, HC_MULT, INDEX_DIM, INDEX_H, INDEX_KPOOL,
    INDEX_STATE_BLOCK_SIZE, INDEX_STATE_WIDTH, KDA_CONV_K, KDA_DIM, KPOOL_SELECT_K, KV_LORA, LOCAL_H,
    LOCAL_KDA_H, LOCAL_KDA_QKV_DIM, MIX_HC, MOE_INTER, N_EXPERTS, N_LOCAL_EXPERTS, Q_LORA, QK_DIM, TOPK,
    TOPK_INDEX_WIDTH, TP_SIZE, V_DIM, AttentionKind, D, Glm53FlashConfig, Glm53LayerConfig, MlpKind,
)
from models.glm5_3_flash.dense_mlp import (
    DenseTensors, dense_mlp, dense_mlp_forward, swiglu_limited, synthetic_dense,
)
from models.glm5_3_flash.expert_routed import (
    ExpertProvider, SyntheticExpertProvider, build_checkpoint_moe_specs, expert_routed, moe_forward,
)
from models.glm5_3_flash.expert_shared import (
    SH_M_TILE, SH_ROW_PAD, expert_shared, golden_expert_shared_fn, moe_layer_add,
)
from models.glm5_3_flash.gate import (
    GATE_M_TILE, SCORE_PAD, TOPK_PAD, RouterTensors, gate, make_correction_bias, moe_gate, router_forward,
)
from models.glm5_3_flash.indexer import (
    POOL_SCALE_WIDTH, POOL_VALID_WIDTH, SORT_TILE, IndexerTensors, indexer_cache_write, indexer_expand,
    indexer_pool_write, indexer_proj, indexer_score, indexer_topk, reference_indexer_topk,
    synthetic_indexer,
)
from models.glm5_3_flash.kda import (
    BETA_PAD, CHUNK_TILE, CONV_DIM, STATE_LEN, KdaTensors, kda_conv_prefill, kda_forward, kda_output,
    kda_projection, prefill_kda, stack_conv_weight, synthetic_kda,
)
from models.glm5_3_flash.lm_head import FP32_HEAD_VOCAB_PAD, lm_head_logits, lm_head_mtp_logits
from models.glm5_3_flash.mhc import (
    MhcTensors, combine_streams, hc_mixes, hc_pre, hc_seed, mhc_head, mhc_mapping, mhc_mixes, mhc_post,
    mhc_pre, plain_residual_post, plain_residual_pre, seed_hc_streams, synthetic_mhc,
)
from models.glm5_3_flash.mtp import mtp_projection
from models.glm5_3_flash.prefill_moe import (
    AUX_PAD, COUNT_PAD, GROUPED_CAP, N_RANKS, ROUTES_PER_SRC, ROW_TILE, TOTAL_CAP, T, golden_prefill_moe,
    prefill_moe_dispatch, prefill_moe_return, prefill_ring_heap,
)
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX
from models.glm5_3_flash.rmsnorm import golden_rmsnorm_quant, reference_rms_norm, rmsnorm, rmsnorm_quant
from models.glm5_3_flash.utils import _assert_close, _prefill_model, _randn, gen_routed_weight, pack_int8_nz

MAX_BATCH_PER_DP = 32
PREFILL_MAX_TOKENS = 8192
_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--capacity", type=int, default=2048)
_shape_parser.add_argument("--max-seq-len", type=int)
_shape_parser.add_argument("--cache-tokens", type=int)
_shape_parser.add_argument("--tokens", type=int)
_shape_options = _shape_parser.parse_known_args()[0]
CAPACITY = _shape_options.capacity
MAX_SEQ_LEN = CAPACITY if _shape_options.max_seq_len is None else _shape_options.max_seq_len
_default_cache_tokens = min(
    16 * ((MAX_SEQ_LEN + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE) * C.BLOCK_SIZE, C.FLASH.max_position_embeddings,
)
CACHE_TOKENS = _default_cache_tokens if _shape_options.cache_tokens is None else _shape_options.cache_tokens
T_MOE = ((CAPACITY + C.TP_SIZE * 16 - 1) // (C.TP_SIZE * 16)) * 16
RANKS = EP_SIZE
MODEL = _prefill_model()
NUM_LAYERS = MODEL.num_hidden_layers  # backbone layers; the loop bound and nothing else
KDA_NUM_LAYERS = len(MODEL.kda_layer_ids)  # linear-attention stack depth
DSA_NUM_LAYERS = len(MODEL.dsa_layer_ids) + 1  # sparse-attention cache stack depth
DENSE_NUM_LAYERS = MODEL.first_k_dense_replace  # dense-MLP stack depth
MOE_NUM_LAYERS = NUM_LAYERS - DENSE_NUM_LAYERS  # MoE layers on the backbone
MLA_STACK_LAYERS = len(MODEL.indexer_layer_ids)
MOE_STACK_LAYERS = MODEL.num_checkpoint_layers - DENSE_NUM_LAYERS
NORM_STACK_LAYERS = MODEL.num_checkpoint_layers
ROUTER_STACK_LAYERS = MOE_STACK_LAYERS
BATCH = 16  # first-light packed batch: 16 prompts, one batch for all ranks
LOGIT_ROWS = MAX_BATCH_PER_DP  # 32; covers BATCH with headroom (design S3.4)
QUERY_ROWS_DYN = pl.dynamic("GLM53_QUERY_ROWS_DYN")
LOCAL_DENSE_INTER = DENSE_INTER // TP_SIZE
TABLE_MAX = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
LATENT_BLOCKS = CACHE_TOKENS // BLOCK_SIZE
LATENT_ROWS = LATENT_BLOCKS * BLOCK_SIZE
POOL_STRIDE = (
    ((MAX_SEQ_LEN + INDEX_KPOOL - 1) // INDEX_KPOOL + INDEX_STATE_BLOCK_SIZE - 1)
    // INDEX_STATE_BLOCK_SIZE
    * INDEX_STATE_BLOCK_SIZE
)
POOL_AXIS = (
    (max(LATENT_ROWS // INDEX_KPOOL, CAPACITY // INDEX_KPOOL + BATCH) + 2 * SORT_TILE - 2) // SORT_TILE
) * SORT_TILE
POOL_TABLE_ROWS = LATENT_ROWS // INDEX_KPOOL + BLOCK_SIZE // INDEX_KPOOL  # final raw-page span is scratch
KDA_STATE_BLOCKS = min(TABLE_MAX, (CAPACITY + BLOCK_SIZE - 1) // BLOCK_SIZE + 2)
KDA_STATE_PAGES = BATCH * (KDA_STATE_BLOCKS + 2)
KDA_STATE_COLUMNS = 2 + (CAPACITY + CHUNK_TILE - 1) // CHUNK_TILE
KDA_ROWS = BATCH + KDA_STATE_PAGES  # private working rows followed by grouped checkpoints
POOL_TABLE_MAX = (MAX_SEQ_LEN + INDEX_KPOOL * INDEX_STATE_BLOCK_SIZE - 1) // (INDEX_KPOOL * INDEX_STATE_BLOCK_SIZE)
CACHE_LAYOUTS = {
    "mla_latent_cache": ((DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA), torch.bfloat16),
    "index_raw_cache": ((DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH), torch.float32),
    "index_pool_cache": ((DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM), torch.int8),
    "index_pool_scale": ((DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH), torch.float32),
    "kda_conv_state": ((KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM), torch.bfloat16),
    "kda_recurrent_state": ((KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM), torch.float32),
}

RESIDENT_ROUTES_PER_SRC = T_MOE * TOPK
RESIDENT_TOTAL_CAP = N_RANKS * RESIDENT_ROUTES_PER_SRC
RESIDENT_GROUPED_CAP = RESIDENT_TOTAL_CAP + N_LOCAL_EXPERTS * (ROW_TILE - 1)


FORWARD_ABI_VERSION = 8
FORWARD_PROGRAMS = ("prefill", "decode")
_COMMON_PARAMETERS = (
    'embedding_rows', 'query_start_loc', 'kda_state_rows', 'active_tokens', 'kda_tri_incl',
    'kda_tri_strict', 'kda_head_onehot', 'kda_eye_wide', 'kda_w_q', 'kda_w_k', 'kda_w_v', 'kda_conv_q',
    'kda_conv_k', 'kda_conv_v', 'kda_w_f_a', 'kda_w_f_b', 'kda_w_g_a', 'kda_w_g_b', 'kda_w_b',
    'kda_dt_bias', 'kda_a_log', 'kda_o_norm', 'kda_w_o', 'mhc_attn_fn', 'mhc_attn_scale',
    'mhc_attn_base', 'mhc_ffn_fn', 'mhc_ffn_scale', 'mhc_ffn_base', 'norm_input_norm', 'norm_post_norm',
    'dense_w_gate_up', 'dense_w_gate_up_scale', 'dense_w_down', 'dense_w_down_scale', 'kda_conv_state',
    'kda_recurrent_state', 'ws_attn_pre', 'ws_attn_post', 'ws_attn_res', 'ws_collapsed', 'ws_normed',
    'ws_attn_partial', 'ws_sublayer_out', 'ws_x_hc_mid', 'ws_ffn_pre', 'ws_ffn_post', 'ws_ffn_res',
    'ws_ffn_input', 'ws_ffn_partial', 'ws_ffn_out', 'kda_out', 'mixed_qkv', 'decay', 'beta', 'kda_q',
    'kda_k', 'kda_v', 'dense_x_int8', 'request_ids', 'mla_slots', 'index_slots', 'seg_start',
    'pool_count_tok', 'tail_start', 'tail_count', 'kv_len_tok', 'pool_token_slots', 'pool_slots',
    'pool_blocks', 'block_table', 'mla_w_q_a', 'mla_w_q_a_scale', 'mla_q_a_norm', 'mla_w_q_b',
    'mla_w_q_b_scale', 'mla_w_kv_a', 'mla_w_kv_a_scale', 'mla_kv_a_norm', 'mla_w_k', 'mla_w_v',
    'mla_w_o', 'mla_w_o_scale', 'idx_w_q_b', 'idx_w_k', 'idx_k_norm_w', 'idx_k_norm_b', 'idx_w_weights',
    'idx_w_compress_gate', 'idx_compress_ape', 'hadamard', 'mla_latent_cache', 'index_raw_cache',
    'index_pool_cache', 'index_pool_scale', 'dsa_query', 'attn_out', 'index_q', 'index_k',
    'index_scores', 'selected_pools', 'selected_valid', 'router_gate_w', 'router_gate_bias',
    'moe_w_gate_up', 'moe_w_gate_up_scale', 'moe_w_down', 'moe_w_down_scale', 'moe_shared_w_gate_up',
    'moe_shared_w_gate_up_scale', 'moe_shared_w_down', 'moe_shared_w_down_scale', 'logit_row_indices',
    'head_hidden', 'ws_x_hc_pair',
)
_QUERY_PARAMETERS = (
    'embedding_rows', 'ws_attn_pre', 'ws_attn_post', 'ws_attn_res', 'ws_collapsed', 'ws_normed',
    'ws_attn_partial', 'ws_sublayer_out', 'ws_x_hc_mid', 'ws_ffn_pre', 'ws_ffn_post', 'ws_ffn_res',
    'ws_ffn_input', 'ws_ffn_partial', 'ws_ffn_out', 'kda_out', 'mixed_qkv', 'decay', 'beta', 'kda_q',
    'kda_k', 'kda_v', 'dense_x_int8', 'request_ids', 'mla_slots', 'index_slots', 'seg_start',
    'pool_count_tok', 'tail_start', 'tail_count', 'kv_len_tok', 'dsa_query', 'attn_out', 'index_q',
    'index_k', 'index_scores', 'selected_pools', 'selected_valid',
)
_HEAD_PARAMETERS = (
    'final_norm', 'lm_head_weight', 'logits',
)
_TENSOR_LAYOUTS = {
    'embedding_rows': ((CAPACITY, D), torch.bfloat16),
    'query_start_loc': ((BATCH + 1,), torch.int32),
    'kda_state_rows': ((BATCH, KDA_STATE_COLUMNS), torch.int32),
    'active_tokens': ((1,), torch.int32),
    'kda_tri_incl': ((CHUNK_TILE, CHUNK_TILE), torch.float32),
    'kda_tri_strict': ((CHUNK_TILE, CHUNK_TILE), torch.float32),
    'kda_head_onehot': ((LOCAL_KDA_H, BETA_PAD), torch.float32),
    'kda_eye_wide': ((CHUNK_TILE, KDA_DIM), torch.float32),
    'kda_w_q': ((KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D), torch.bfloat16),
    'kda_w_k': ((KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D), torch.bfloat16),
    'kda_w_v': ((KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D), torch.bfloat16),
    'kda_conv_q': ((KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K), torch.bfloat16),
    'kda_conv_k': ((KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K), torch.bfloat16),
    'kda_conv_v': ((KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K), torch.bfloat16),
    'kda_w_f_a': ((KDA_NUM_LAYERS, KDA_DIM, D), torch.bfloat16),
    'kda_w_f_b': ((KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM), torch.bfloat16),
    'kda_w_g_a': ((KDA_NUM_LAYERS, KDA_DIM, D), torch.bfloat16),
    'kda_w_g_b': ((KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM), torch.bfloat16),
    'kda_w_b': ((KDA_NUM_LAYERS, LOCAL_KDA_H, D), torch.bfloat16),
    'kda_dt_bias': ((KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM), torch.float32),
    'kda_a_log': ((KDA_NUM_LAYERS, LOCAL_KDA_H), torch.float32),
    'kda_o_norm': ((KDA_NUM_LAYERS, KDA_DIM), torch.bfloat16),
    'kda_w_o': ((KDA_NUM_LAYERS, D, LOCAL_KDA_QKV_DIM), torch.bfloat16),
    'mhc_attn_fn': ((NUM_LAYERS, MIX_HC, HC_DIM), torch.bfloat16),
    'mhc_attn_scale': ((NUM_LAYERS, 3), torch.float32),
    'mhc_attn_base': ((NUM_LAYERS, MIX_HC), torch.float32),
    'mhc_ffn_fn': ((NUM_LAYERS, MIX_HC, HC_DIM), torch.bfloat16),
    'mhc_ffn_scale': ((NUM_LAYERS, 3), torch.float32),
    'mhc_ffn_base': ((NUM_LAYERS, MIX_HC), torch.float32),
    'norm_input_norm': ((NORM_STACK_LAYERS, D), torch.bfloat16),
    'norm_post_norm': ((NORM_STACK_LAYERS, D), torch.bfloat16),
    'dense_w_gate_up': ((DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER, D), torch.int8),
    'dense_w_gate_up_scale': ((DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER), torch.float32),
    'dense_w_down': ((DENSE_NUM_LAYERS, D, LOCAL_DENSE_INTER), torch.int8),
    'dense_w_down_scale': ((DENSE_NUM_LAYERS, D), torch.float32),
    'kda_conv_state': ((KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM), torch.bfloat16),
    'kda_recurrent_state': ((KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM), torch.float32),
    'ws_attn_pre': ((CAPACITY, HC_MULT), torch.float32),
    'ws_attn_post': ((CAPACITY, HC_MULT), torch.float32),
    'ws_attn_res': ((CAPACITY, HC_MULT, HC_MULT), torch.float32),
    'ws_collapsed': ((CAPACITY, D), torch.bfloat16),
    'ws_normed': ((CAPACITY, D), torch.bfloat16),
    'ws_attn_partial': ((CAPACITY, D), torch.float32),
    'ws_sublayer_out': ((CAPACITY, D), torch.bfloat16),
    'ws_x_hc_mid': ((CAPACITY, HC_MULT, D), torch.bfloat16),
    'ws_ffn_pre': ((CAPACITY, HC_MULT), torch.float32),
    'ws_ffn_post': ((CAPACITY, HC_MULT), torch.float32),
    'ws_ffn_res': ((CAPACITY, HC_MULT, HC_MULT), torch.float32),
    'ws_ffn_input': ((CAPACITY, D), torch.bfloat16),
    'ws_ffn_partial': ((CAPACITY, D), torch.float32),
    'ws_ffn_out': ((CAPACITY, D), torch.bfloat16),
    'kda_out': ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    'mixed_qkv': ((CAPACITY, 3 * LOCAL_KDA_QKV_DIM), torch.bfloat16),
    'decay': ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.float32),
    'beta': ((CAPACITY, BETA_PAD), torch.float32),
    'kda_q': ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    'kda_k': ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    'kda_v': ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    'dense_x_int8': ((CAPACITY, D), torch.int8),
    'request_ids': ((CAPACITY,), torch.int32),
    'mla_slots': ((CAPACITY,), torch.int32),
    'index_slots': ((CAPACITY,), torch.int32),
    'seg_start': ((CAPACITY,), torch.int32),
    'pool_count_tok': ((CAPACITY,), torch.int32),
    'tail_start': ((CAPACITY,), torch.int32),
    'tail_count': ((CAPACITY,), torch.int32),
    'kv_len_tok': ((CAPACITY,), torch.int32),
    'pool_token_slots': ((POOL_AXIS, INDEX_KPOOL), torch.int32),
    'pool_slots': ((POOL_AXIS,), torch.int32),
    'pool_blocks': ((POOL_AXIS,), torch.int32),
    'block_table': ((BATCH, TABLE_MAX), torch.int32),
    'mla_w_q_a': ((MLA_STACK_LAYERS, Q_LORA, D), torch.int8),
    'mla_w_q_a_scale': ((MLA_STACK_LAYERS, Q_LORA), torch.float32),
    'mla_q_a_norm': ((MLA_STACK_LAYERS, Q_LORA), torch.bfloat16),
    'mla_w_q_b': ((MLA_STACK_LAYERS, LOCAL_H * QK_DIM, Q_LORA), torch.int8),
    'mla_w_q_b_scale': ((MLA_STACK_LAYERS, LOCAL_H * QK_DIM), torch.float32),
    'mla_w_kv_a': ((MLA_STACK_LAYERS, KV_LORA, D), torch.int8),
    'mla_w_kv_a_scale': ((MLA_STACK_LAYERS, KV_LORA), torch.float32),
    'mla_kv_a_norm': ((MLA_STACK_LAYERS, KV_LORA), torch.bfloat16),
    'mla_w_k': ((MLA_STACK_LAYERS, LOCAL_H, QK_DIM, KV_LORA), torch.bfloat16),
    'mla_w_v': ((MLA_STACK_LAYERS, LOCAL_H, V_DIM, KV_LORA), torch.bfloat16),
    'mla_w_o': ((MLA_STACK_LAYERS, D, LOCAL_H * V_DIM), torch.int8),
    'mla_w_o_scale': ((MLA_STACK_LAYERS, D), torch.float32),
    'idx_w_q_b': ((MLA_STACK_LAYERS, Q_LORA, INDEX_H * INDEX_DIM), torch.bfloat16),
    'idx_w_k': ((MLA_STACK_LAYERS, D, INDEX_DIM), torch.bfloat16),
    'idx_k_norm_w': ((MLA_STACK_LAYERS, INDEX_DIM), torch.bfloat16),
    'idx_k_norm_b': ((MLA_STACK_LAYERS, INDEX_DIM), torch.bfloat16),
    'idx_w_weights': ((MLA_STACK_LAYERS, D, INDEX_H), torch.bfloat16),
    'idx_w_compress_gate': ((MLA_STACK_LAYERS, D, INDEX_DIM), torch.bfloat16),
    'idx_compress_ape': ((MLA_STACK_LAYERS, INDEX_KPOOL, INDEX_DIM), torch.bfloat16),
    'hadamard': ((INDEX_DIM, INDEX_DIM), torch.bfloat16),
    'mla_latent_cache': ((DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA), torch.bfloat16),
    'index_raw_cache': ((DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH), torch.float32),
    'index_pool_cache': ((DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM), torch.int8),
    'index_pool_scale': ((DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH), torch.float32),
    'dsa_query': ((CAPACITY, LOCAL_H, QK_DIM), torch.bfloat16),
    'attn_out': ((CAPACITY, LOCAL_H, V_DIM), torch.bfloat16),
    'index_q': ((CAPACITY, INDEX_H, INDEX_DIM), torch.bfloat16),
    'index_k': ((CAPACITY, INDEX_DIM), torch.bfloat16),
    'index_scores': ((CAPACITY, POOL_AXIS), torch.float32),
    'selected_pools': ((CAPACITY, KPOOL_SELECT_K), torch.int32),
    'selected_valid': ((CAPACITY, KPOOL_SELECT_K), torch.int32),
    'router_gate_w': ((ROUTER_STACK_LAYERS, N_EXPERTS, D), torch.bfloat16),
    'router_gate_bias': ((ROUTER_STACK_LAYERS, N_EXPERTS), torch.float32),
    'moe_w_gate_up': ((MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D), torch.int8),
    'moe_w_gate_up_scale': ((MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER), torch.float32),
    'moe_w_down': ((MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D, MOE_INTER), torch.int8),
    'moe_w_down_scale': ((MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D), torch.float32),
    'moe_shared_w_gate_up': ((MOE_STACK_LAYERS, 2 * MOE_INTER, D), torch.int8),
    'moe_shared_w_gate_up_scale': ((MOE_STACK_LAYERS, 2 * MOE_INTER), torch.float32),
    'moe_shared_w_down': ((MOE_STACK_LAYERS, D, MOE_INTER), torch.int8),
    'moe_shared_w_down_scale': ((MOE_STACK_LAYERS, D), torch.float32),
    'logit_row_indices': ((LOGIT_ROWS,), torch.int32),
    'head_hidden': ((LOGIT_ROWS, D), torch.bfloat16),
    'ws_x_hc_pair': ((2 * CAPACITY, HC_MULT, D), torch.bfloat16),
    'final_norm': ((D,), torch.bfloat16),
    'lm_head_weight': ((FP32_HEAD_VOCAB_PAD, D), torch.float32),
    'logits': ((BATCH, FP32_HEAD_VOCAB_PAD), torch.float32),
}


def _check_phase(kind: str) -> None:
    if kind not in FORWARD_PROGRAMS:
        raise ValueError(f"unknown GLM forward phase {kind!r}")


def forward_parameter_names(kind: str) -> tuple[str, ...]:
    """Return tensor parameter names in the complete NPU entry's positional order."""
    _check_phase(kind)
    return _COMMON_PARAMETERS + _HEAD_PARAMETERS


def forward_query_tensor_names() -> tuple[str, ...]:
    """Return tensors whose first per-rank axis is the phase query capacity."""
    return _QUERY_PARAMETERS


def forward_tensor_layouts(kind: str):
    """Return ordered per-rank logical shapes and Torch dtypes, without a rank axis."""
    query_capacity = forward_geometry(kind)["query_capacity"]
    layouts = {}
    for name in forward_parameter_names(kind):
        shape, dtype = _TENSOR_LAYOUTS[name]
        if name in _QUERY_PARAMETERS:
            shape = (query_capacity, *shape[1:])
        layouts[name] = (shape, dtype)
    return layouts


def forward_geometry(kind: str) -> dict[str, int]:
    """Return frozen phase input, owner, stream and request capacities."""
    _check_phase(kind)
    return {
        "version": FORWARD_ABI_VERSION,
        "query_capacity": CAPACITY if kind == "prefill" else BATCH,
        "owner_capacity": T_MOE,
        "stream_capacity": CAPACITY,
        "request_capacity": BATCH,
        "head_buffer_rows": LOGIT_ROWS,
        "tp": TP_SIZE,
        "ep": EP_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "cache_tokens": CACHE_TOKENS,
    }


def forward_epoch_guard(index: int) -> int:
    """Validate a zero-based completed-call index for one prepared phase program."""
    if type(index) is not int or index < 0:
        raise ValueError("forward epoch must be a nonnegative integer")
    if (index + 1) * 2 * NUM_LAYERS > (2**31 - 1) // 2:
        raise OverflowError("collective epochs exhausted; close and reopen the GLM runner")
    return index


@pl.jit.inline
def _complete_forward_stage(
    streams: pl.Tensor[[QUERY_ROWS_DYN, HC_MULT, D], pl.BF16],
) -> pl.Scalar[pl.TASK_ID]:
    """Complete all query-row stream producers before constructing the next stage."""
    rows = pl.tensor.dim(streams, 0)
    stage_tokens = pl.create_tensor([rows, 1, 16], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="forward_stage_complete", allow_early_resolve=False) as stage_tid:
        for row in pl.range(rows):
            sample = pl.slice(streams, [1, 1, 16], [row, 0, 0])
            stage_tokens[row:row + 1, 0:1, 0:16] = pl.cast(sample, pl.FP32)
    _completed = pl.read(stage_tokens, [0, 0, 0])
    return stage_tid



@pl.jit(auto_scope=False)
def l2_prefill_layer(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_weight: pl.Tensor[[D], pl.BF16],
    gate_weight: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    w_gate_up: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w_gate_up: pl.Tensor[[2 * MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w_gate_up_scale: pl.Tensor[[2 * MOE_INTER], pl.FP32],
    shared_w_down: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w_down_scale: pl.Tensor[[D], pl.FP32],
    y: pl.Out[pl.Tensor[[T, D], pl.FP32]],
    lane_x: pld.DistributedTensor[[TOTAL_CAP, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[TOTAL_CAP, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    count_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[ROUTES_PER_SRC, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Run EP routed experts and a complete shared expert over this rank's tokens."""
    x_scale = pl.create_tensor([T, 1], dtype=pl.FP32)

    # Block head: normalized router and the INT8 view both branches reuse.
    x_int8 = pl.create_tensor([T, D], dtype=pl.INT8)
    route_indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    route_weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    moe_gate(
        x_mixed, norm_weight, gate_weight, correction_bias, route_weights, route_indices, x_int8, x_scale,
        num_tokens,
    )

    # Routed branch: count/pack this rank's routes, exchange them, run the local
    # experts over the regrouped slab, scatter back and sum the TOPK slots.
    slab_src = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.INT32)
    slab_route = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.INT32)
    slab_weight = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.FP32)
    slab_scale = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.FP32)
    slab_x = pl.create_tensor([GROUPED_CAP, D], dtype=pl.INT8)
    expert_offsets = pl.create_tensor([N_LOCAL_EXPERTS + 1], dtype=pl.INT32)
    expert_counts = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
    recv_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL_EXPERTS], dtype=pl.INT32)
    send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32)
    dispatch_tid = prefill_moe_dispatch(
        route_indices, route_weights, x_int8, x_scale, send_counts, recv_expert_counts, expert_counts,
        expert_offsets, lane_x, lane_aux, lane_count, count_ready, x_ready, slab_x, slab_scale, slab_weight,
        slab_route, slab_src, num_tokens, my_rank, epoch,
    )
    expert_y = pl.create_tensor([GROUPED_CAP, D], dtype=pl.FP32)
    expert_routed(
        slab_x, slab_scale, w_gate_up, w_gate_up_scale,
        w_down, w_down_scale, expert_offsets, slab_weight, expert_y,
    )
    routed = pl.create_tensor([T, D], dtype=pl.FP32)
    prefill_moe_return(
        expert_y, slab_route, slab_src, expert_offsets, expert_counts, routed, slot_out, return_ready,
        num_tokens, my_rank, epoch, dispatch_tid,
    )

    # Shared branch: the complete local expert uses this rank's activation rows.
    shared_out = pl.create_tensor([T, D], dtype=pl.FP32)
    expert_shared(
        x_int8, x_scale, shared_w_gate_up, shared_w_gate_up_scale, shared_w_down, shared_w_down_scale,
        shared_out,
    )
    moe_layer_add(routed, shared_out, y, num_tokens)
    return y


@pl.jit.host
def l3_prefill_layer(
    x_mixed: pl.Tensor[[N_RANKS, T, D], pl.BF16],
    norm_weight: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_weight: pl.Tensor[[N_RANKS, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[N_RANKS, N_EXPERTS], pl.FP32],
    w_gate_up: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w_gate_up: pl.Tensor[[N_RANKS, 2 * MOE_INTER, D], pl.INT8],
    shared_w_gate_up_scale: pl.Tensor[[N_RANKS, 2 * MOE_INTER], pl.FP32],
    shared_w_down: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w_down_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    y: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
    num_ranks: pl.Scalar[pl.INT32],
):
    """Launch independent token batches with EP routed weights and full shared replicas."""
    lane_x_buf = pld.alloc_window_buffer([TOTAL_CAP, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([N_RANKS, COUNT_PAD], dtype=pl.INT32)
    count_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    x_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([ROUTES_PER_SRC, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    # Repeat the same inputs with successive EP epochs to exercise window reuse.
    # Each rank completes combine before starting its next dispatch.
    for call in pl.range(2):
        for r in pl.range(pld.world_size()):
            lane_x = pld.window(lane_x_buf, [TOTAL_CAP, D], dtype=pl.INT8)
            lane_aux = pld.window(lane_aux_buf, [TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
            lane_count = pld.window(lane_count_buf, [N_RANKS, COUNT_PAD], dtype=pl.INT32)
            count_ready = pld.window(count_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            x_ready = pld.window(x_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            slot_out = pld.window(slot_out_buf, [ROUTES_PER_SRC, D], dtype=pl.FP32)
            return_ready = pld.window(return_ready_buf, [N_RANKS, 1], dtype=pl.INT32)
            l2_prefill_layer(
                x_mixed[r], norm_weight[r], gate_weight[r], correction_bias[r], w_gate_up[r],
                w_gate_up_scale[r],
                w_down[r], w_down_scale[r],
                shared_w_gate_up[r], shared_w_gate_up_scale[r], shared_w_down[r], shared_w_down_scale[r],
                y[r], lane_x, lane_aux, lane_count, count_ready, x_ready, slot_out, return_ready, num_tokens,
                r, pl.cast(call + 1, pl.INT32), device=r,
            )


@pl.jit.inline
def zero_bf16_padding_rows(
    x: pl.Tensor[[CAPACITY, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Zero BF16 rows outside the active token prefix."""
    rows = pl.tensor.dim(x, 0)
    active = pl.cast(num_tokens, pl.INDEX)
    if active < 0:
        active = pl.cast(0, pl.INDEX)
    with pl.spmd(pl.max(rows - active, 1), name_hint="zero_bf16_padding"):
        row = active + pl.tile.get_block_idx()
        if row < rows:
            # shapes-less store: the tile's own [1, D] extent is the store extent;
            # a shapes= kwarg becomes a 4th operand the static-extent partition
            # path cannot strip (lower_tile_to_buffer_pass rank-2 check).
            pl.tile.store(pl.tile.full([1, D], dtype=pl.BF16, value=0.0), [row, 0], x)


@pl.jit.inline
def local_token_count(num_tokens: pl.Scalar[pl.INT32], my_rank: pl.Scalar[pl.INT32]):
    """This rank's share of the replicated token prefix (owner map ``t % TP_SIZE``)."""
    return (pl.cast(num_tokens, pl.INDEX) + TP_SIZE - 1 - pl.cast(my_rank, pl.INDEX)) // TP_SIZE


@pl.jit.inline
def compact_local_tokens(
    source: pl.Tensor[[CAPACITY, D], pl.BF16],
    compact: pl.Tensor[[T_MOE, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Pack this rank's owned rows into the dense local prefix the MoE consumes."""
    local_tokens = local_token_count(num_tokens, my_rank)
    with pl.spmd(T_MOE, name_hint="moe_compact"):
        row = pl.tile.get_block_idx()
        value = pl.tile.full([1, D], dtype=pl.BF16, value=0.0)
        if row < local_tokens:
            value = pl.tile.load(source, [row * TP_SIZE + pl.cast(my_rank, pl.INDEX), 0], [1, D])
        # shapes-less store: the [1, D] tile carries the extent (static-extent
        # partition path rejects the shapes kwarg's 4th operand).
        pl.tile.store(value, [row, 0], compact)


@pl.jit.inline
def scatter_local_tokens_fp32(
    compact: pl.Tensor[[T_MOE, D], pl.FP32],
    output: pl.Tensor[[CAPACITY, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Zero the full-width FP32 partial, then place this rank's owned MoE rows.

    The owner's FP32 row is the complete result and every other rank holds zeros for
    it, so the following TP reduction is the restore (template
    ``_scatter_local_tokens`` + ``widen_to_fp32`` folded into one pass).
    """
    local_tokens = local_token_count(num_tokens, my_rank)
    rows = pl.tensor.dim(output, 0)
    with pl.spmd(rows, name_hint="moe_scatter_zero") as zero_tid:
        row = pl.tile.get_block_idx()
        pl.tile.store(pl.tile.full([1, D], dtype=pl.FP32, value=0.0), [row, 0], output)
    with pl.spmd(T_MOE, name_hint="moe_scatter", deps=[zero_tid]):
        row = pl.tile.get_block_idx()
        if row < local_tokens:
            dst_row = row * TP_SIZE + pl.cast(my_rank, pl.INDEX)
            value = pl.tile.load(compact, [row, 0], [1, D])
            pl.tile.store(value, [dst_row, 0], output)


@pl.jit.inline
def gather_head_rows(
    source: pl.Tensor[[CAPACITY, D], pl.BF16],
    row_indices: pl.Tensor[[LOGIT_ROWS], pl.INT32],
    output: pl.Tensor[[LOGIT_ROWS, D], pl.BF16],
):
    """Gather per-request last rows into the padded head buffer."""
    with pl.spmd(LOGIT_ROWS, name_hint="gather_head_rows"):
        r = pl.tile.get_block_idx()
        src = pl.cast(pl.read(row_indices, [r]), pl.INDEX)
        pl.tile.store(pl.tile.load(source, [src, 0], [1, D]), [r, 0], output)


@pl.jit.inline
def dsa_attention_sublayer(
    normed: pl.Tensor[[CAPACITY, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.INT8],
    w_q_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_a_norm: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    w_q_b_scale: pl.Tensor[[LOCAL_H * QK_DIM], pl.FP32],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.INT8],
    w_kv_a_scale: pl.Tensor[[KV_LORA], pl.FP32],
    kv_a_norm: pl.Tensor[[KV_LORA], pl.BF16],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    w_v: pl.Tensor[[LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_H * V_DIM], pl.INT8],
    w_o_scale: pl.Tensor[[D], pl.FP32],
    idx_w_q_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    idx_w_k: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    idx_k_norm_w: pl.Tensor[[INDEX_DIM], pl.BF16],
    idx_k_norm_b: pl.Tensor[[INDEX_DIM], pl.BF16],
    idx_w_weights: pl.Tensor[[D, INDEX_H], pl.BF16],
    idx_w_compress_gate: pl.Tensor[[D, INDEX_DIM], pl.BF16],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    latent_cache: pl.Tensor[[LATENT_ROWS, KV_LORA], pl.BF16],
    raw_cache: pl.Tensor[[LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32],
    pool_cache: pl.Tensor[[POOL_TABLE_ROWS, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32],
    block_table: pl.Tensor[[BATCH, TABLE_MAX], pl.INT32],
    request_ids: pl.Tensor[[CAPACITY], pl.INT32],
    mla_slots: pl.Tensor[[CAPACITY], pl.INT32],
    index_slots: pl.Tensor[[CAPACITY], pl.INT32],
    pool_token_slots: pl.Tensor[[POOL_AXIS, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOL_AXIS], pl.INT32],
    pool_blocks: pl.Tensor[[POOL_AXIS], pl.INT32],
    seg_start: pl.Tensor[[CAPACITY], pl.INT32],
    pool_count_tok: pl.Tensor[[CAPACITY], pl.INT32],
    tail_start: pl.Tensor[[CAPACITY], pl.INT32],
    tail_count: pl.Tensor[[CAPACITY], pl.INT32],
    kv_len_tok: pl.Tensor[[CAPACITY], pl.INT32],
    q_resid: pl.Tensor[[CAPACITY, Q_LORA], pl.BF16],
    query: pl.Tensor[[CAPACITY, LOCAL_H, QK_DIM], pl.BF16],
    kv_latent: pl.Tensor[[CAPACITY, KV_LORA], pl.BF16],
    attn_out: pl.Tensor[[CAPACITY, LOCAL_H, V_DIM], pl.BF16],
    index_q: pl.Tensor[[CAPACITY, INDEX_H, INDEX_DIM], pl.BF16],
    index_k: pl.Tensor[[CAPACITY, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[CAPACITY, INDEX_H], pl.FP32],
    gate_scores: pl.Tensor[[CAPACITY, INDEX_DIM], pl.FP32],
    index_scores: pl.Tensor[[CAPACITY, POOL_AXIS], pl.FP32],
    selected_pools: pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32],
    topk_indices: pl.Tensor[[CAPACITY, TOPK_INDEX_WIDTH], pl.INT32],
    pool_valid: pl.Tensor[[POOL_AXIS, POOL_VALID_WIDTH], pl.INT32],
    attn_partial: pl.Tensor[[CAPACITY, D], pl.FP32],
    skip_topk: pl.Scalar[pl.INT32],
    query_extent: pl.constexpr = CAPACITY,
):
    """MLA NoPE projection -> latent cache publish -> kpool indexer chain -> sparse
    attention -> head projection, leaving the row-parallel FP32 partial.

    The indexer step orders these operations through shared buffers:
    the cache and pool writes must precede the sparse attention because the
    always-appended tail reads this dispatch's own rows.
    """
    mla_prolog_w8a8(
        normed,
        w_q_a, w_q_a_scale,
        q_a_norm,
        w_q_b, w_q_b_scale,
        w_kv_a, w_kv_a_scale,
        kv_a_norm, q_resid, query, kv_latent,
    )
    mla_cache_write(kv_latent, mla_slots, latent_cache)
    indexer_proj(
        normed, q_resid, idx_w_q_b, idx_w_k, idx_k_norm_w, idx_k_norm_b, idx_w_weights, idx_w_compress_gate,
        index_q, index_k, head_weights, gate_scores, skip_topk,
    )
    indexer_cache_write(index_k, gate_scores, index_slots, raw_cache)
    indexer_pool_write(
        raw_cache, compress_ape, hadamard, pool_token_slots, pool_slots, pool_cache, pool_scale, pool_valid,
    )
    if skip_topk == 0:
        indexer_score(
            index_q, hadamard, pool_cache, pool_scale, pool_blocks, head_weights, seg_start, pool_count_tok,
            index_scores,
        )
        indexer_topk(index_scores, seg_start, pool_count_tok, selected_pools, selected_valid)
    indexer_expand(selected_pools, selected_valid, tail_start, tail_count, kv_len_tok, topk_indices)
    prefill_sparse_attn(query, latent_cache, block_table, request_ids, w_k, w_v, topk_indices, attn_out, query_extent)
    mla_epilog_w8a8(attn_out, w_o, w_o_scale, attn_partial)


@pl.jit.inline
def kda_attention_sublayer(
    normed: pl.Tensor[[CAPACITY, D], pl.BF16],
    w_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    conv_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    conv_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    conv_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    o_norm: pl.Tensor[[KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    query_start_loc: pl.Tensor[[BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[BATCH, KDA_STATE_COLUMNS], pl.INT32],
    tri_incl: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    tri_strict: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    eye_wide: pl.Tensor[[CHUNK_TILE, KDA_DIM], pl.FP32],
    conv_w: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    mixed_qkv: pl.Tensor[[CAPACITY, 3 * LOCAL_KDA_QKV_DIM], pl.BF16],
    decay: pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[CAPACITY, BETA_PAD], pl.FP32],
    out_gate: pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    query: pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    kda_out: pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    attn_partial: pl.Tensor[[CAPACITY, D], pl.FP32],
    query_extent: pl.constexpr = CAPACITY,
):
    """KDA projection -> causal conv -> chunked delta attention -> gated output."""
    stack_conv_weight(conv_q, conv_k, conv_v, conv_w)
    kda_projection(
        normed,
        w_q, w_k, w_v, w_f_a, w_f_b, dt_bias, a_log, w_b, w_g_a, w_g_b, mixed_qkv, decay, beta, out_gate,
    )
    kda_conv_prefill(mixed_qkv, conv_w, conv_state, query_start_loc, kda_state_rows, query, key, value)
    prefill_kda(
        query, key, value, decay, beta, recurrent_state, query_start_loc, kda_state_rows, tri_incl,
        tri_strict, head_onehot, eye_wide, kda_out, query_extent, BATCH, state_extent=KDA_ROWS,
    )


@pl.jit.inline
def dense_ffn_sublayer(
    ffn_input: pl.Tensor[[CAPACITY, D], pl.BF16],
    post_norm_w: pl.Tensor[[D], pl.BF16],
    w_gate_up: pl.Tensor[[2 * LOCAL_DENSE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[2 * LOCAL_DENSE_INTER], pl.FP32],
    w_down: pl.Tensor[[D, LOCAL_DENSE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    normed: pl.Tensor[[CAPACITY, D], pl.BF16],
    x_int8: pl.Tensor[[CAPACITY, D], pl.INT8],
    x_scale: pl.Tensor[[CAPACITY, 1], pl.FP32],
    ffn_partial: pl.Tensor[[CAPACITY, D], pl.FP32],
):
    """Dense MLP with its own fused norm+quant head; TP-column-sliced weights leave
    an FP32 partial for the shared reduction."""
    rmsnorm_quant(ffn_input, post_norm_w, normed, x_int8, x_scale, num_tokens)
    dense_mlp(x_int8, x_scale, w_gate_up, w_gate_up_scale, w_down, w_down_scale, ffn_partial)


@pl.jit.inline(auto_scope=False)
def moe_ffn_sublayer(
    ffn_input: pl.Tensor[[CAPACITY, D], pl.BF16],
    post_norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    w_gate_up: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    w_gate_up_scale: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    w_down: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    w_down_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    shared_w_gate_up: pl.Tensor[[2 * MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w_gate_up_scale: pl.Tensor[[2 * MOE_INTER], pl.FP32],
    shared_w_down: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w_down_scale: pl.Tensor[[D], pl.FP32],
    lane_x: pld.DistributedTensor[[RESIDENT_TOTAL_CAP, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[RESIDENT_TOTAL_CAP, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    count_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[RESIDENT_ROUTES_PER_SRC, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    local_ffn: pl.Tensor[[T_MOE, D], pl.BF16],
    route_weights: pl.Tensor[[T_MOE, TOPK], pl.FP32],
    route_indices: pl.Tensor[[T_MOE, TOPK], pl.INT32],
    x_int8: pl.Tensor[[T_MOE, D], pl.INT8],
    x_scale: pl.Tensor[[T_MOE, 1], pl.FP32],
    send_counts: pl.Tensor[[N_RANKS, 1], pl.INT32],
    recv_expert_counts: pl.Tensor[[N_RANKS, N_LOCAL_EXPERTS], pl.INT32],
    expert_counts: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    expert_offsets: pl.Tensor[[N_LOCAL_EXPERTS + 1], pl.INT32],
    slab_x: pl.Tensor[[RESIDENT_GROUPED_CAP, D], pl.INT8],
    slab_scale: pl.Tensor[[RESIDENT_GROUPED_CAP, 1], pl.FP32],
    slab_weight: pl.Tensor[[RESIDENT_GROUPED_CAP, 1], pl.FP32],
    slab_route: pl.Tensor[[RESIDENT_GROUPED_CAP, 1], pl.INT32],
    slab_src: pl.Tensor[[RESIDENT_GROUPED_CAP, 1], pl.INT32],
    expert_y: pl.Tensor[[RESIDENT_GROUPED_CAP, D], pl.FP32],
    routed: pl.Tensor[[T_MOE, D], pl.FP32],
    shared_out: pl.Tensor[[T_MOE, D], pl.FP32],
    y_local: pl.Tensor[[T_MOE, D], pl.FP32],
    ffn_partial: pl.Tensor[[CAPACITY, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
):
    """Owner-compact -> gate (deferred norm) -> EP dispatch -> routed + shared ->
    TOPK sum -> scatter back into the full-width FP32 partial.

    Token ``t`` belongs to rank ``t % TP_SIZE`` (metadata.tp_token_owners), so the
    replicated post-attention batch is dispatched exactly once per token; the
    scatter's zero-fill makes the following TP reduction the restore.
    """
    local_tokens = pl.cast(local_token_count(num_tokens, my_rank), pl.INT32)
    compact_local_tokens(ffn_input, local_ffn, num_tokens, my_rank)
    moe_gate(local_ffn, post_norm_w, gate_w, gate_bias, route_weights, route_indices, x_int8, x_scale, local_tokens)
    dispatch_tid = prefill_moe_dispatch(
        route_indices, route_weights, x_int8, x_scale, send_counts, recv_expert_counts, expert_counts,
        expert_offsets, lane_x, lane_aux, lane_count, count_ready, x_ready, slab_x, slab_scale, slab_weight,
        slab_route, slab_src, local_tokens, my_rank, epoch,
    )
    expert_routed(
        slab_x, slab_scale, w_gate_up, w_gate_up_scale,
        w_down, w_down_scale, expert_offsets, slab_weight, expert_y,
    )
    prefill_moe_return(
        expert_y, slab_route, slab_src, expert_offsets, expert_counts, routed, slot_out, return_ready,
        local_tokens, my_rank, epoch, dispatch_tid,
    )
    if local_tokens > 0:
        expert_shared(
            x_int8, x_scale, shared_w_gate_up, shared_w_gate_up_scale, shared_w_down, shared_w_down_scale,
            shared_out,
        )
    moe_layer_add(routed, shared_out, y_local, local_tokens)
    scatter_local_tokens_fp32(y_local, ffn_partial, num_tokens, my_rank)
    return ffn_partial


@pl.jit(auto_scope=False)
def prefill_seed(
    embedding_rows: pl.Tensor[[CAPACITY, D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16]],
):
    """Seed the carried stream: broadcast the embedding rows into all four streams.

    The production orchestrator launches this once per forward, before layer 0;
    the diff harness seeds its input the same way on the host.
    """
    seed_hc_streams(embedding_rows, x_hc)
    return x_hc


@pl.jit(auto_scope=False)
def prefill_head_rows(
    x_hc: pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16],
    logit_row_indices: pl.Tensor[[LOGIT_ROWS], pl.INT32],
    head_hidden: pl.Out[pl.Tensor[[LOGIT_ROWS, D], pl.BF16]],
):
    """Collapse the mHC stream by its unweighted mean and gather the logit rows."""
    head_token_rows = pl.tensor.dim(x_hc, 0)
    head_all = pl.create_tensor([head_token_rows, D], dtype=pl.BF16)
    mhc_head(x_hc, head_all)
    gather_head_rows(head_all, logit_row_indices, head_hidden)
    return head_hidden


@pl.jit(auto_scope=False)
def prefill_kda_attention(
    # -- sublayer input: the stream entering the attention half ----------------------
    x_hc: pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16],
    # -- metadata and per-step inputs (the KDA chain's own subset) -------------------
    query_start_loc: pl.Tensor[[BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[BATCH, KDA_STATE_COLUMNS], pl.INT32],
    active_tokens: pl.Tensor[[1], pl.INT32],
    kda_tri_incl: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_tri_strict: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    kda_eye_wide: pl.Tensor[[CHUNK_TILE, KDA_DIM], pl.FP32],
    # -- resident weight stacks (the attention half's own families) ------------------
    kda_w_q: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_k: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_v: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_conv_q: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_k: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_v: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_w_f_a: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_f_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_g_a: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_g_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_H, D], pl.BF16],
    kda_dt_bias: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM], pl.FP32],
    kda_a_log: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_H], pl.FP32],
    kda_o_norm: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM], pl.BF16],
    kda_w_o: pl.Tensor[[KDA_NUM_LAYERS, D, LOCAL_KDA_QKV_DIM], pl.BF16],
    mhc_attn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_attn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    mhc_attn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    norm_input_norm: pl.Tensor[[NORM_STACK_LAYERS, D], pl.BF16],
    # -- cache pools ------------------------------------------------------------------
    kda_conv_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    # -- window (fresh per diff launch; the attention reduce is epoch 1) --------------
    tp_partial: pld.DistributedTensor[[CAPACITY, D], pl.FP32],
    tp_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    # -- boundary outputs: the attention half's sublayer boundaries -------------------
    out_attn_pre: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_attn_post: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_attn_res: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_collapsed: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_normed: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_attn_partial: pl.Out[pl.Tensor[[CAPACITY, D], pl.FP32]],
    out_sublayer_out: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_x_hc_mid: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16]],
    # The recurrence output stays a boundary buffer so the diff harness observes
    # the exact tensor kda_output consumes.
    kda_out: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    mixed_qkv: pl.InOut[pl.Tensor[[CAPACITY, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.InOut[pl.Tensor[[CAPACITY, BETA_PAD], pl.FP32]],
    kda_q: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_k: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_v: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    # -- scalars last: the ordinals are dispatch-time runtime scalars computed by
    #    the HOST (never layer arithmetic inside the graph) ---------------------------
    bb_ord: pl.Scalar[pl.INT32],
    kda_ord: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    attn_epoch: pl.Scalar[pl.INT32],
    query_extent: pl.constexpr = CAPACITY,
):
    """Run resident KDA attention with runtime layer ordinals."""
    num_tokens: pl.Scalar[pl.INT32] = pl.read(active_tokens, [0])
    t_tokens = pl.tensor.dim(x_hc, 0)


    attn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16] = mhc_attn_fn[bb_ord]
    attn_scale_l: pl.Tensor[[3], pl.FP32] = mhc_attn_scale[bb_ord]
    attn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = mhc_attn_base[bb_ord]
    input_norm_l: pl.Tensor[[D], pl.BF16] = norm_input_norm[bb_ord]
    kda_w_q_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16] = kda_w_q[kda_ord]
    kda_w_k_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16] = kda_w_k[kda_ord]
    kda_w_v_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16] = kda_w_v[kda_ord]
    kda_conv_q_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16] = kda_conv_q[kda_ord]
    kda_conv_k_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16] = kda_conv_k[kda_ord]
    kda_conv_v_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16] = kda_conv_v[kda_ord]
    kda_w_f_a_l: pl.Tensor[[KDA_DIM, D], pl.BF16] = kda_w_f_a[kda_ord]
    kda_w_f_b_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16] = kda_w_f_b[kda_ord]
    kda_w_g_a_l: pl.Tensor[[KDA_DIM, D], pl.BF16] = kda_w_g_a[kda_ord]
    kda_w_g_b_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16] = kda_w_g_b[kda_ord]
    kda_w_b_l: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16] = kda_w_b[kda_ord]
    kda_dt_bias_l: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32] = kda_dt_bias[kda_ord]
    kda_a_log_l: pl.Tensor[[LOCAL_KDA_H], pl.FP32] = kda_a_log[kda_ord]
    kda_o_norm_l: pl.Tensor[[KDA_DIM], pl.BF16] = kda_o_norm[kda_ord]
    kda_w_o_l: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16] = kda_w_o[kda_ord]
    conv_state_l = kda_conv_state[kda_ord]
    recurrent_state_l = kda_recurrent_state[kda_ord]

    out_gate = pl.create_tensor([t_tokens, LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)
    conv_w = pl.create_tensor([KDA_CONV_K, CONV_DIM], dtype=pl.BF16)
    with pl.scope():
        mhc_mixes(x_hc, attn_fn_l, attn_scale_l, attn_base_l, out_attn_pre, out_attn_post, out_attn_res)
        mhc_pre(x_hc, out_attn_pre, out_collapsed)
        rmsnorm(out_collapsed, input_norm_l, out_normed)
        kda_attention_sublayer(
            out_normed,
            kda_w_q_l, kda_w_k_l, kda_w_v_l, kda_w_f_a_l, kda_w_f_b_l, kda_dt_bias_l, kda_a_log_l, kda_w_b_l,
            kda_w_g_a_l, kda_w_g_b_l, kda_conv_q_l, kda_conv_k_l, kda_conv_v_l, kda_o_norm_l, kda_w_o_l,
            conv_state_l, recurrent_state_l, query_start_loc, kda_state_rows, kda_tri_incl, kda_tri_strict,
            kda_head_onehot, kda_eye_wide, conv_w, mixed_qkv, decay, beta, out_gate,
            kda_q, kda_k, kda_v, kda_out,
            out_attn_partial, query_extent,
        )
        kda_output(kda_out, kda_o_norm_l, out_gate, kda_w_o_l, out_attn_partial)
        tp_all_reduce(out_attn_partial, tp_partial, tp_arrived, out_sublayer_out, num_tokens, my_rank, 0, attn_epoch)
        zero_bf16_padding_rows(out_sublayer_out, num_tokens)
        mhc_post(out_sublayer_out, x_hc, out_attn_post, out_attn_res, out_x_hc_mid)

    return (
        out_attn_pre,
        out_attn_post,
        out_attn_res,
        out_collapsed,
        out_normed,
        out_attn_partial,
        out_sublayer_out,
        out_x_hc_mid,
    )


@pl.jit(auto_scope=False)
def prefill_moe_layer(
    # -- sublayer input: the stream the attention half left in the mid workspace -----
    x_hc_mid: pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16],
    active_tokens: pl.Tensor[[1], pl.INT32],
    # -- resident weight stacks (the FFN half's own families) ------------------------
    mhc_ffn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_ffn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    mhc_ffn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    norm_post_norm: pl.Tensor[[NORM_STACK_LAYERS, D], pl.BF16],
    router_gate_w: pl.Tensor[[ROUTER_STACK_LAYERS, N_EXPERTS, D], pl.BF16],
    router_gate_bias: pl.Tensor[[ROUTER_STACK_LAYERS, N_EXPERTS], pl.FP32],
    moe_w_gate_up: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    moe_w_gate_up_scale: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    moe_w_down: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    moe_w_down_scale: pl.Tensor[[MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D], pl.FP32],
    moe_shared_w_gate_up: pl.Tensor[[MOE_STACK_LAYERS, 2 * MOE_INTER, D], pl.INT8, pl.NZ],
    moe_shared_w_gate_up_scale: pl.Tensor[[MOE_STACK_LAYERS, 2 * MOE_INTER], pl.FP32],
    moe_shared_w_down: pl.Tensor[[MOE_STACK_LAYERS, D, MOE_INTER], pl.INT8, pl.NZ],
    moe_shared_w_down_scale: pl.Tensor[[MOE_STACK_LAYERS, D], pl.FP32],
    # -- windows (fresh per diff launch: the MoE barriers and this entry's
    #    single TP reduce all run at epoch 1 -- a fresh window's tp epoch IS
    #    its call count, two notifies per call) ---------------------------------------
    lane_x: pld.DistributedTensor[[RESIDENT_TOTAL_CAP, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[RESIDENT_TOTAL_CAP, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    count_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[RESIDENT_ROUTES_PER_SRC, D], pl.FP32],
    return_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    tp_partial: pld.DistributedTensor[[CAPACITY, D], pl.FP32],
    tp_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    # -- boundary outputs: the FFN half's sublayer boundaries -------------------------
    out_ffn_pre: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_ffn_post: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_ffn_res: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_ffn_input: pl.Out[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_ffn_partial: pl.Out[pl.Tensor[[CAPACITY, D], pl.FP32]],
    out_ffn_out: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_x_hc_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16]],
    # -- scalars last: the ordinals are dispatch-time runtime scalars computed by
    #    the HOST (never layer arithmetic inside the graph) ---------------------------
    bb_ord: pl.Scalar[pl.INT32],
    moe_ord: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    ffn_epoch: pl.Scalar[pl.INT32],
):
    """Run the resident MoE feed-forward sublayer with runtime layer ordinals."""
    num_tokens: pl.Scalar[pl.INT32] = pl.read(active_tokens, [0])


    post_norm_l: pl.Tensor[[D], pl.BF16] = norm_post_norm[bb_ord]
    router_gate_w_l: pl.Tensor[[N_EXPERTS, D], pl.BF16] = router_gate_w[moe_ord]
    router_gate_bias_l: pl.Tensor[[N_EXPERTS], pl.FP32] = router_gate_bias[moe_ord]
    moe_w_gate_up_l: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8] = moe_w_gate_up[moe_ord]
    moe_w_gate_up_scale_l: pl.Tensor[[N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32] = moe_w_gate_up_scale[moe_ord]
    moe_w_down_l: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8] = moe_w_down[moe_ord]
    moe_w_down_scale_l: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32] = moe_w_down_scale[moe_ord]
    moe_shared_w_gate_up_l: pl.Tensor[[2 * MOE_INTER, D], pl.INT8, pl.NZ] = moe_shared_w_gate_up[moe_ord]
    moe_shared_w_gate_up_scale_l: pl.Tensor[[2 * MOE_INTER], pl.FP32] = moe_shared_w_gate_up_scale[moe_ord]
    moe_shared_w_down_l: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ] = moe_shared_w_down[moe_ord]
    moe_shared_w_down_scale_l: pl.Tensor[[D], pl.FP32] = moe_shared_w_down_scale[moe_ord]

    y_local = pl.create_tensor([T_MOE, D], dtype=pl.FP32)
    shared_out = pl.create_tensor([T_MOE, D], dtype=pl.FP32)
    routed = pl.create_tensor([T_MOE, D], dtype=pl.FP32)
    expert_y = pl.create_tensor([RESIDENT_GROUPED_CAP, D], dtype=pl.FP32)
    slab_src = pl.create_tensor([RESIDENT_GROUPED_CAP, 1], dtype=pl.INT32)
    slab_route = pl.create_tensor([RESIDENT_GROUPED_CAP, 1], dtype=pl.INT32)
    slab_weight = pl.create_tensor([RESIDENT_GROUPED_CAP, 1], dtype=pl.FP32)
    slab_scale = pl.create_tensor([RESIDENT_GROUPED_CAP, 1], dtype=pl.FP32)
    slab_x = pl.create_tensor([RESIDENT_GROUPED_CAP, D], dtype=pl.INT8)
    expert_offsets = pl.create_tensor([N_LOCAL_EXPERTS + 1], dtype=pl.INT32)
    expert_counts = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
    recv_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL_EXPERTS], dtype=pl.INT32)
    send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32)
    moe_x_scale = pl.create_tensor([T_MOE, 1], dtype=pl.FP32)
    moe_x_int8 = pl.create_tensor([T_MOE, D], dtype=pl.INT8)
    moe_route_indices = pl.create_tensor([T_MOE, TOPK], dtype=pl.INT32)
    moe_route_weights = pl.create_tensor([T_MOE, TOPK], dtype=pl.FP32)
    local_ffn = pl.create_tensor([T_MOE, D], dtype=pl.BF16)
    with pl.scope():
        if bb_ord == NUM_LAYERS:
            plain_residual_pre(x_hc_mid, out_ffn_input, out_ffn_pre, out_ffn_post, out_ffn_res)
        else:
            ffn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16] = mhc_ffn_fn[bb_ord]
            ffn_scale_l: pl.Tensor[[3], pl.FP32] = mhc_ffn_scale[bb_ord]
            ffn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = mhc_ffn_base[bb_ord]
            mhc_mixes(x_hc_mid, ffn_fn_l, ffn_scale_l, ffn_base_l, out_ffn_pre, out_ffn_post, out_ffn_res)
            mhc_pre(x_hc_mid, out_ffn_pre, out_ffn_input)
        moe_ffn_sublayer(
            out_ffn_input,
            post_norm_l, router_gate_w_l, router_gate_bias_l, moe_w_gate_up_l, moe_w_gate_up_scale_l,
            moe_w_down_l, moe_w_down_scale_l, moe_shared_w_gate_up_l, moe_shared_w_gate_up_scale_l,
            moe_shared_w_down_l, moe_shared_w_down_scale_l, lane_x, lane_aux, lane_count, count_ready,
            x_ready, slot_out, return_ready, local_ffn, moe_route_weights, moe_route_indices, moe_x_int8,
            moe_x_scale, send_counts, recv_expert_counts, expert_counts, expert_offsets, slab_x, slab_scale,
            slab_weight, slab_route, slab_src, expert_y, routed, shared_out, y_local, out_ffn_partial,
            num_tokens, my_rank, moe_epoch,
        )
        tp_all_reduce(out_ffn_partial, tp_partial, tp_arrived, out_ffn_out, num_tokens, my_rank, 0, ffn_epoch)
        zero_bf16_padding_rows(out_ffn_out, num_tokens)
        if bb_ord == NUM_LAYERS:
            plain_residual_post(out_ffn_out, x_hc_mid, out_x_hc_out)
        else:
            mhc_post(out_ffn_out, x_hc_mid, out_ffn_post, out_ffn_res, out_x_hc_out)

    return (out_ffn_pre, out_ffn_post, out_ffn_res, out_ffn_input, out_ffn_partial, out_ffn_out, out_x_hc_out)


@pl.jit(auto_scope=False)
def prefill_dense_layer(
    # -- layer input: the stream entering the layer --------------------------------
    x_hc: pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16],
    # -- metadata and per-step inputs (the KDA chain's own subset; the dense
    #    layer runs no DSA/indexer/MoE chain, so those tensors left the ABI) --------
    query_start_loc: pl.Tensor[[BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[BATCH, KDA_STATE_COLUMNS], pl.INT32],
    active_tokens: pl.Tensor[[1], pl.INT32],
    kda_tri_incl: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_tri_strict: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP32],
    kda_head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    kda_eye_wide: pl.Tensor[[CHUNK_TILE, KDA_DIM], pl.FP32],
    # -- resident weight stacks (the dense layer's own families) --------------------
    kda_w_q: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_k: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_v: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_conv_q: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_k: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_v: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_w_f_a: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_f_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_g_a: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_g_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_b: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_H, D], pl.BF16],
    kda_dt_bias: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM], pl.FP32],
    kda_a_log: pl.Tensor[[KDA_NUM_LAYERS, LOCAL_KDA_H], pl.FP32],
    kda_o_norm: pl.Tensor[[KDA_NUM_LAYERS, KDA_DIM], pl.BF16],
    kda_w_o: pl.Tensor[[KDA_NUM_LAYERS, D, LOCAL_KDA_QKV_DIM], pl.BF16],
    mhc_attn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_attn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    mhc_attn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    mhc_ffn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_ffn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    mhc_ffn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    norm_input_norm: pl.Tensor[[NORM_STACK_LAYERS, D], pl.BF16],
    norm_post_norm: pl.Tensor[[NORM_STACK_LAYERS, D], pl.BF16],
    dense_w_gate_up: pl.Tensor[[DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER, D], pl.INT8],
    dense_w_gate_up_scale: pl.Tensor[[DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER], pl.FP32],
    dense_w_down: pl.Tensor[[DENSE_NUM_LAYERS, D, LOCAL_DENSE_INTER], pl.INT8],
    dense_w_down_scale: pl.Tensor[[DENSE_NUM_LAYERS, D], pl.FP32],
    # -- cache pools ------------------------------------------------------------------
    kda_conv_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    # -- window (fresh per launch; the dense layer's shared tp pair runs at 1 and 2) -
    tp_partial: pld.DistributedTensor[[CAPACITY, D], pl.FP32],
    tp_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    # -- boundary outputs (identical names and mutability to the whole-layer entry) --
    out_attn_pre: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_attn_post: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_attn_res: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_collapsed: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_normed: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_attn_partial: pl.Out[pl.Tensor[[CAPACITY, D], pl.FP32]],
    out_sublayer_out: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_x_hc_mid: pl.InOut[pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16]],
    out_ffn_pre: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_ffn_post: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_ffn_res: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_ffn_input: pl.Out[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_ffn_partial: pl.Out[pl.Tensor[[CAPACITY, D], pl.FP32]],
    out_ffn_out: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_x_hc_out: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16]],
    kda_out: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    mixed_qkv: pl.InOut[pl.Tensor[[CAPACITY, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.InOut[pl.Tensor[[CAPACITY, BETA_PAD], pl.FP32]],
    kda_q: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_k: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_v: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    dense_x_int8: pl.InOut[pl.Tensor[[CAPACITY, D], pl.INT8]],
    # -- scalars last: the ordinals are dispatch-time runtime scalars computed by
    #    the HOST (never layer arithmetic inside the graph) ---------------------------
    bb_ord: pl.Scalar[pl.INT32],
    kda_ord: pl.Scalar[pl.INT32],
    dense_ord: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    attn_epoch: pl.Scalar[pl.INT32],
    ffn_epoch: pl.Scalar[pl.INT32],
    query_extent: pl.constexpr = CAPACITY,
):
    """Run a resident KDA layer with a dense feed-forward sublayer."""
    num_tokens: pl.Scalar[pl.INT32] = pl.read(active_tokens, [0])
    t_tokens = pl.tensor.dim(x_hc, 0)


    attn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16] = mhc_attn_fn[bb_ord]
    attn_scale_l: pl.Tensor[[3], pl.FP32] = mhc_attn_scale[bb_ord]
    attn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = mhc_attn_base[bb_ord]
    ffn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16] = mhc_ffn_fn[bb_ord]
    ffn_scale_l: pl.Tensor[[3], pl.FP32] = mhc_ffn_scale[bb_ord]
    ffn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = mhc_ffn_base[bb_ord]
    input_norm_l: pl.Tensor[[D], pl.BF16] = norm_input_norm[bb_ord]
    post_norm_l: pl.Tensor[[D], pl.BF16] = norm_post_norm[bb_ord]
    kda_w_q_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16] = kda_w_q[kda_ord]
    kda_w_k_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16] = kda_w_k[kda_ord]
    kda_w_v_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, D], pl.BF16] = kda_w_v[kda_ord]
    kda_conv_q_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16] = kda_conv_q[kda_ord]
    kda_conv_k_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16] = kda_conv_k[kda_ord]
    kda_conv_v_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16] = kda_conv_v[kda_ord]
    kda_w_f_a_l: pl.Tensor[[KDA_DIM, D], pl.BF16] = kda_w_f_a[kda_ord]
    kda_w_f_b_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16] = kda_w_f_b[kda_ord]
    kda_w_g_a_l: pl.Tensor[[KDA_DIM, D], pl.BF16] = kda_w_g_a[kda_ord]
    kda_w_g_b_l: pl.Tensor[[LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16] = kda_w_g_b[kda_ord]
    kda_w_b_l: pl.Tensor[[LOCAL_KDA_H, D], pl.BF16] = kda_w_b[kda_ord]
    kda_dt_bias_l: pl.Tensor[[LOCAL_KDA_QKV_DIM], pl.FP32] = kda_dt_bias[kda_ord]
    kda_a_log_l: pl.Tensor[[LOCAL_KDA_H], pl.FP32] = kda_a_log[kda_ord]
    kda_o_norm_l: pl.Tensor[[KDA_DIM], pl.BF16] = kda_o_norm[kda_ord]
    kda_w_o_l: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16] = kda_w_o[kda_ord]
    conv_state_l = kda_conv_state[kda_ord]
    recurrent_state_l = kda_recurrent_state[kda_ord]
    dense_w_gate_up_l: pl.Tensor[[2 * LOCAL_DENSE_INTER, D], pl.INT8] = dense_w_gate_up[dense_ord]
    dense_w_gate_up_scale_l: pl.Tensor[[2 * LOCAL_DENSE_INTER], pl.FP32] = dense_w_gate_up_scale[dense_ord]
    dense_w_down_l: pl.Tensor[[D, LOCAL_DENSE_INTER], pl.INT8] = dense_w_down[dense_ord]
    dense_w_down_scale_l: pl.Tensor[[D], pl.FP32] = dense_w_down_scale[dense_ord]

    out_gate = pl.create_tensor([t_tokens, LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)
    conv_w = pl.create_tensor([KDA_CONV_K, CONV_DIM], dtype=pl.BF16)
    with pl.scope():
        mhc_mixes(x_hc, attn_fn_l, attn_scale_l, attn_base_l, out_attn_pre, out_attn_post, out_attn_res)
        mhc_pre(x_hc, out_attn_pre, out_collapsed)
        rmsnorm(out_collapsed, input_norm_l, out_normed)
        kda_attention_sublayer(
            out_normed,
            kda_w_q_l, kda_w_k_l, kda_w_v_l, kda_w_f_a_l, kda_w_f_b_l, kda_dt_bias_l, kda_a_log_l, kda_w_b_l,
            kda_w_g_a_l, kda_w_g_b_l, kda_conv_q_l, kda_conv_k_l, kda_conv_v_l, kda_o_norm_l, kda_w_o_l,
            conv_state_l, recurrent_state_l, query_start_loc, kda_state_rows, kda_tri_incl, kda_tri_strict,
            kda_head_onehot, kda_eye_wide, conv_w, mixed_qkv, decay, beta, out_gate,
            kda_q, kda_k, kda_v, kda_out,
            out_attn_partial, query_extent,
        )
        kda_output(kda_out, kda_o_norm_l, out_gate, kda_w_o_l, out_attn_partial)
        tp_all_reduce(out_attn_partial, tp_partial, tp_arrived, out_sublayer_out, num_tokens, my_rank, 0, attn_epoch)
        zero_bf16_padding_rows(out_sublayer_out, num_tokens)
        mhc_post(out_sublayer_out, x_hc, out_attn_post, out_attn_res, out_x_hc_mid)

    dense_x_scale = pl.create_tensor([t_tokens, 1], dtype=pl.FP32)
    dense_normed = pl.create_tensor([t_tokens, D], dtype=pl.BF16)
    with pl.scope():
        mhc_mixes(out_x_hc_mid, ffn_fn_l, ffn_scale_l, ffn_base_l, out_ffn_pre, out_ffn_post, out_ffn_res)
        mhc_pre(out_x_hc_mid, out_ffn_pre, out_ffn_input)
        dense_ffn_sublayer(
            out_ffn_input,
            post_norm_l, dense_w_gate_up_l, dense_w_gate_up_scale_l, dense_w_down_l, dense_w_down_scale_l,
            num_tokens, dense_normed, dense_x_int8, dense_x_scale, out_ffn_partial,
        )
        tp_all_reduce(out_ffn_partial, tp_partial, tp_arrived, out_ffn_out, num_tokens, my_rank, 0, ffn_epoch)
        zero_bf16_padding_rows(out_ffn_out, num_tokens)
        mhc_post(out_ffn_out, out_x_hc_mid, out_ffn_post, out_ffn_res, out_x_hc_out)

    return (
        out_attn_pre,
        out_attn_post,
        out_attn_res,
        out_collapsed,
        out_normed,
        out_attn_partial,
        out_sublayer_out,
        out_x_hc_mid,
        out_ffn_pre,
        out_ffn_post,
        out_ffn_res,
        out_ffn_input,
        out_ffn_partial,
        out_ffn_out,
        out_x_hc_out,
    )


@pl.jit(auto_scope=False)
def prefill_sparse_attention(
    # -- sublayer input: the stream entering the attention half ----------------------
    x_hc: pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16],
    # -- metadata and per-step inputs (the DSA/indexer chain's own subset) ------------
    request_ids: pl.Tensor[[CAPACITY], pl.INT32],
    mla_slots: pl.Tensor[[CAPACITY], pl.INT32],
    index_slots: pl.Tensor[[CAPACITY], pl.INT32],
    seg_start: pl.Tensor[[CAPACITY], pl.INT32],
    pool_count_tok: pl.Tensor[[CAPACITY], pl.INT32],
    tail_start: pl.Tensor[[CAPACITY], pl.INT32],
    tail_count: pl.Tensor[[CAPACITY], pl.INT32],
    kv_len_tok: pl.Tensor[[CAPACITY], pl.INT32],
    pool_token_slots: pl.Tensor[[POOL_AXIS, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOL_AXIS], pl.INT32],
    pool_blocks: pl.Tensor[[POOL_AXIS], pl.INT32],
    block_table: pl.Tensor[[BATCH, TABLE_MAX], pl.INT32],
    active_tokens: pl.Tensor[[1], pl.INT32],
    # -- resident weight stacks (the attention half's own families) ------------------
    mla_w_q_a: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA, D], pl.INT8],
    mla_w_q_a_scale: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA], pl.FP32],
    mla_q_a_norm: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA], pl.BF16],
    mla_w_q_b: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    mla_w_q_b_scale: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H * QK_DIM], pl.FP32],
    mla_w_kv_a: pl.Tensor[[MLA_STACK_LAYERS, KV_LORA, D], pl.INT8],
    mla_w_kv_a_scale: pl.Tensor[[MLA_STACK_LAYERS, KV_LORA], pl.FP32],
    mla_kv_a_norm: pl.Tensor[[MLA_STACK_LAYERS, KV_LORA], pl.BF16],
    mla_w_k: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    mla_w_v: pl.Tensor[[MLA_STACK_LAYERS, LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    mla_w_o: pl.Tensor[[MLA_STACK_LAYERS, D, LOCAL_H * V_DIM], pl.INT8],
    mla_w_o_scale: pl.Tensor[[MLA_STACK_LAYERS, D], pl.FP32],
    idx_w_q_b: pl.Tensor[[MLA_STACK_LAYERS, Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    idx_w_k: pl.Tensor[[MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_k_norm_w: pl.Tensor[[MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_k_norm_b: pl.Tensor[[MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_w_weights: pl.Tensor[[MLA_STACK_LAYERS, D, INDEX_H], pl.BF16],
    idx_w_compress_gate: pl.Tensor[[MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_compress_ape: pl.Tensor[[MLA_STACK_LAYERS, INDEX_KPOOL, INDEX_DIM], pl.BF16],
    mhc_attn_fn: pl.Tensor[[NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_attn_scale: pl.Tensor[[NUM_LAYERS, 3], pl.FP32],
    mhc_attn_base: pl.Tensor[[NUM_LAYERS, MIX_HC], pl.FP32],
    norm_input_norm: pl.Tensor[[NORM_STACK_LAYERS, D], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    # -- cache pools ------------------------------------------------------------------
    mla_latent_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    # -- window (fresh per diff launch; the attention reduce is epoch 1) --------------
    tp_partial: pld.DistributedTensor[[CAPACITY, D], pl.FP32],
    tp_arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    # -- boundary outputs: the attention half's sublayer boundaries -------------------
    out_attn_pre: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_attn_post: pl.Out[pl.Tensor[[CAPACITY, HC_MULT], pl.FP32]],
    out_attn_res: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_collapsed: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_normed: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_attn_partial: pl.Out[pl.Tensor[[CAPACITY, D], pl.FP32]],
    out_sublayer_out: pl.InOut[pl.Tensor[[CAPACITY, D], pl.BF16]],
    out_x_hc_mid: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16]],
    dsa_query: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_H, QK_DIM], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[CAPACITY, LOCAL_H, V_DIM], pl.BF16]],
    index_q: pl.InOut[pl.Tensor[[CAPACITY, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.InOut[pl.Tensor[[CAPACITY, INDEX_DIM], pl.BF16]],
    index_scores: pl.InOut[pl.Tensor[[CAPACITY, POOL_AXIS], pl.FP32]],
    selected_pools: pl.InOut[pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.InOut[pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    # -- scalars last: the ordinals are dispatch-time runtime scalars computed by
    #    the HOST (never layer arithmetic inside the graph) ---------------------------
    bb_ord: pl.Scalar[pl.INT32],
    dsa_ord: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    attn_epoch: pl.Scalar[pl.INT32],
    skip_topk: pl.Scalar[pl.INT32],
    query_extent: pl.constexpr = CAPACITY,
):
    """Run resident sparse attention with runtime layer ordinals."""
    num_tokens: pl.Scalar[pl.INT32] = pl.read(active_tokens, [0])
    t_tokens = pl.tensor.dim(x_hc, 0)
    pools_axis = pl.tensor.dim(pool_token_slots, 0)


    input_norm_l: pl.Tensor[[D], pl.BF16] = norm_input_norm[bb_ord]
    mla_w_q_a_l: pl.Tensor[[Q_LORA, D], pl.INT8] = mla_w_q_a[dsa_ord]
    mla_w_q_a_scale_l: pl.Tensor[[Q_LORA], pl.FP32] = mla_w_q_a_scale[dsa_ord]
    mla_q_a_norm_l: pl.Tensor[[Q_LORA], pl.BF16] = mla_q_a_norm[dsa_ord]
    mla_w_q_b_l: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.INT8] = mla_w_q_b[dsa_ord]
    mla_w_q_b_scale_l: pl.Tensor[[LOCAL_H * QK_DIM], pl.FP32] = mla_w_q_b_scale[dsa_ord]
    mla_w_kv_a_l: pl.Tensor[[KV_LORA, D], pl.INT8] = mla_w_kv_a[dsa_ord]
    mla_w_kv_a_scale_l: pl.Tensor[[KV_LORA], pl.FP32] = mla_w_kv_a_scale[dsa_ord]
    mla_kv_a_norm_l: pl.Tensor[[KV_LORA], pl.BF16] = mla_kv_a_norm[dsa_ord]
    mla_w_k_l: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16] = mla_w_k[dsa_ord]
    mla_w_v_l: pl.Tensor[[LOCAL_H, V_DIM, KV_LORA], pl.BF16] = mla_w_v[dsa_ord]
    mla_w_o_l: pl.Tensor[[D, LOCAL_H * V_DIM], pl.INT8] = mla_w_o[dsa_ord]
    mla_w_o_scale_l: pl.Tensor[[D], pl.FP32] = mla_w_o_scale[dsa_ord]
    idx_w_q_b_l: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.BF16] = idx_w_q_b[dsa_ord]
    idx_w_k_l: pl.Tensor[[D, INDEX_DIM], pl.BF16] = idx_w_k[dsa_ord]
    idx_k_norm_w_l: pl.Tensor[[INDEX_DIM], pl.BF16] = idx_k_norm_w[dsa_ord]
    idx_k_norm_b_l: pl.Tensor[[INDEX_DIM], pl.BF16] = idx_k_norm_b[dsa_ord]
    idx_w_weights_l: pl.Tensor[[D, INDEX_H], pl.BF16] = idx_w_weights[dsa_ord]
    idx_w_compress_gate_l: pl.Tensor[[D, INDEX_DIM], pl.BF16] = idx_w_compress_gate[dsa_ord]
    idx_compress_ape_l: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16] = idx_compress_ape[dsa_ord]
    latent_cache_l = mla_latent_cache[dsa_ord]
    raw_cache_l = index_raw_cache[dsa_ord]
    pool_cache_l = index_pool_cache[dsa_ord]
    pool_scale_l = index_pool_scale[dsa_ord]

    pool_valid = pl.create_tensor([pools_axis, POOL_VALID_WIDTH], dtype=pl.INT32)
    topk_indices = pl.create_tensor([t_tokens, TOPK_INDEX_WIDTH], dtype=pl.INT32)
    gate_scores = pl.create_tensor([t_tokens, INDEX_DIM], dtype=pl.FP32)
    head_weights = pl.create_tensor([t_tokens, INDEX_H], dtype=pl.FP32)
    kv_latent = pl.create_tensor([t_tokens, KV_LORA], dtype=pl.BF16)
    q_resid = pl.create_tensor([t_tokens, Q_LORA], dtype=pl.BF16)
    with pl.scope():
        if bb_ord == NUM_LAYERS:
            plain_residual_pre(x_hc, out_collapsed, out_attn_pre, out_attn_post, out_attn_res)
        else:
            attn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16] = mhc_attn_fn[bb_ord]
            attn_scale_l: pl.Tensor[[3], pl.FP32] = mhc_attn_scale[bb_ord]
            attn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = mhc_attn_base[bb_ord]
            mhc_mixes(x_hc, attn_fn_l, attn_scale_l, attn_base_l, out_attn_pre, out_attn_post, out_attn_res)
            mhc_pre(x_hc, out_attn_pre, out_collapsed)
        rmsnorm(out_collapsed, input_norm_l, out_normed)
        dsa_attention_sublayer(
            out_normed,
            mla_w_q_a_l, mla_w_q_a_scale_l, mla_q_a_norm_l, mla_w_q_b_l, mla_w_q_b_scale_l, mla_w_kv_a_l,
            mla_w_kv_a_scale_l, mla_kv_a_norm_l, mla_w_k_l, mla_w_v_l, mla_w_o_l, mla_w_o_scale_l,
            idx_w_q_b_l, idx_w_k_l, idx_k_norm_w_l, idx_k_norm_b_l, idx_w_weights_l, idx_w_compress_gate_l,
            idx_compress_ape_l, hadamard, latent_cache_l, raw_cache_l, pool_cache_l, pool_scale_l,
            block_table, request_ids, mla_slots,
            index_slots,
            pool_token_slots, pool_slots, pool_blocks, seg_start, pool_count_tok, tail_start, tail_count,
            kv_len_tok, q_resid, dsa_query, kv_latent, attn_out, index_q, index_k, head_weights, gate_scores,
            index_scores,
            selected_pools, selected_valid, topk_indices, pool_valid,
            out_attn_partial, skip_topk, query_extent,
        )
        tp_all_reduce(out_attn_partial, tp_partial, tp_arrived, out_sublayer_out, num_tokens, my_rank, 0, attn_epoch)
        zero_bf16_padding_rows(out_sublayer_out, num_tokens)
        if bb_ord == NUM_LAYERS:
            plain_residual_post(out_sublayer_out, x_hc, out_x_hc_mid)
        else:
            mhc_post(out_sublayer_out, x_hc, out_attn_post, out_attn_res, out_x_hc_mid)

    return (
        out_attn_pre,
        out_attn_post,
        out_attn_res,
        out_collapsed,
        out_normed,
        out_attn_partial,
        out_sublayer_out,
        out_x_hc_mid,
    )


@pl.jit.host
def l3_prefill_moe_layer(
    x_hc_mid: pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16],
    active_tokens: pl.Tensor[[RANKS, 1], pl.INT32],
    mhc_ffn_fn: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_ffn_scale: pl.Tensor[[RANKS, NUM_LAYERS, 3], pl.FP32],
    mhc_ffn_base: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC], pl.FP32],
    norm_post_norm: pl.Tensor[[RANKS, NORM_STACK_LAYERS, D], pl.BF16],
    router_gate_w: pl.Tensor[[RANKS, ROUTER_STACK_LAYERS, N_EXPERTS, D], pl.BF16],
    router_gate_bias: pl.Tensor[[RANKS, ROUTER_STACK_LAYERS, N_EXPERTS], pl.FP32],
    moe_w_gate_up: pl.Tensor[[RANKS, MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    moe_w_gate_up_scale: pl.Tensor[[RANKS, MOE_STACK_LAYERS, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    moe_w_down: pl.Tensor[[RANKS, MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    moe_w_down_scale: pl.Tensor[[RANKS, MOE_STACK_LAYERS, N_LOCAL_EXPERTS, D], pl.FP32],
    moe_shared_w_gate_up: pl.Tensor[[RANKS, MOE_STACK_LAYERS, 2 * MOE_INTER, D], pl.INT8],
    moe_shared_w_gate_up_scale: pl.Tensor[[RANKS, MOE_STACK_LAYERS, 2 * MOE_INTER], pl.FP32],
    moe_shared_w_down: pl.Tensor[[RANKS, MOE_STACK_LAYERS, D, MOE_INTER], pl.INT8],
    moe_shared_w_down_scale: pl.Tensor[[RANKS, MOE_STACK_LAYERS, D], pl.FP32],
    out_ffn_pre: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_ffn_post: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_ffn_res: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_ffn_input: pl.Out[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_ffn_partial: pl.Out[pl.Tensor[[RANKS, CAPACITY, D], pl.FP32]],
    out_ffn_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_x_hc_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16]],
    bb_ord: pl.Scalar[pl.INT32],
    moe_ord: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    ffn_epoch: pl.Scalar[pl.INT32],
):
    """Launch one MoE FFN half at the caller's per-program MoE and TP epochs."""
    lane_x_buffer = pld.alloc_window_buffer([RESIDENT_TOTAL_CAP, D], dtype=pl.INT8)
    lane_aux_buffer = pld.alloc_window_buffer([RESIDENT_TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
    lane_count_buffer = pld.alloc_window_buffer([N_RANKS, COUNT_PAD], dtype=pl.INT32)
    count_ready_buffer = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    x_ready_buffer = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    slot_out_buffer = pld.alloc_window_buffer([RESIDENT_ROUTES_PER_SRC, D], dtype=pl.FP32)
    return_ready_buffer = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    tp_partial_buffer = pld.alloc_window_buffer([CAPACITY, D], dtype=pl.FP32)
    tp_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.unroll(RANKS):
        lane_x = pld.window(lane_x_buffer, [RESIDENT_TOTAL_CAP, D], dtype=pl.INT8)
        lane_aux = pld.window(lane_aux_buffer, [RESIDENT_TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
        lane_count = pld.window(lane_count_buffer, [N_RANKS, COUNT_PAD], dtype=pl.INT32)
        count_ready = pld.window(count_ready_buffer, [N_RANKS, 1], dtype=pl.INT32)
        x_ready = pld.window(x_ready_buffer, [N_RANKS, 1], dtype=pl.INT32)
        slot_out = pld.window(slot_out_buffer, [RESIDENT_ROUTES_PER_SRC, D], dtype=pl.FP32)
        return_ready = pld.window(return_ready_buffer, [N_RANKS, 1], dtype=pl.INT32)
        tp_partial = pld.window(tp_partial_buffer, [CAPACITY, D], dtype=pl.FP32)
        tp_arrived = pld.window(tp_arrived_buffer, [EP_SIZE, 1], dtype=pl.INT32)
        prefill_moe_layer(
            x_hc_mid[rank], active_tokens[rank], mhc_ffn_fn[rank], mhc_ffn_scale[rank], mhc_ffn_base[rank],
            norm_post_norm[rank], router_gate_w[rank], router_gate_bias[rank], moe_w_gate_up[rank],
            moe_w_gate_up_scale[rank], moe_w_down[rank], moe_w_down_scale[rank], moe_shared_w_gate_up[rank],
            moe_shared_w_gate_up_scale[rank], moe_shared_w_down[rank], moe_shared_w_down_scale[rank], lane_x,
            lane_aux, lane_count, count_ready, x_ready, slot_out, return_ready, tp_partial, tp_arrived,
            out_ffn_pre[rank], out_ffn_post[rank], out_ffn_res[rank], out_ffn_input[rank],
            out_ffn_partial[rank], out_ffn_out[rank], out_x_hc_out[rank], bb_ord, moe_ord, rank, moe_epoch,
            ffn_epoch, device=rank,
        )


@pl.jit.host
def l3_prefill_sparse_attention(
    x_hc: pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16],
    request_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    mla_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    index_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    seg_start: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    pool_count_tok: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    tail_start: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    tail_count: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    kv_len_tok: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    pool_token_slots: pl.Tensor[[RANKS, POOL_AXIS, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[RANKS, POOL_AXIS], pl.INT32],
    pool_blocks: pl.Tensor[[RANKS, POOL_AXIS], pl.INT32],
    block_table: pl.Tensor[[RANKS, BATCH, TABLE_MAX], pl.INT32],
    active_tokens: pl.Tensor[[RANKS, 1], pl.INT32],
    mla_w_q_a: pl.Tensor[[RANKS, MLA_STACK_LAYERS, Q_LORA, D], pl.INT8],
    mla_w_q_a_scale: pl.Tensor[[RANKS, MLA_STACK_LAYERS, Q_LORA], pl.FP32],
    mla_q_a_norm: pl.Tensor[[RANKS, MLA_STACK_LAYERS, Q_LORA], pl.BF16],
    mla_w_q_b: pl.Tensor[[RANKS, MLA_STACK_LAYERS, LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    mla_w_q_b_scale: pl.Tensor[[RANKS, MLA_STACK_LAYERS, LOCAL_H * QK_DIM], pl.FP32],
    mla_w_kv_a: pl.Tensor[[RANKS, MLA_STACK_LAYERS, KV_LORA, D], pl.INT8],
    mla_w_kv_a_scale: pl.Tensor[[RANKS, MLA_STACK_LAYERS, KV_LORA], pl.FP32],
    mla_kv_a_norm: pl.Tensor[[RANKS, MLA_STACK_LAYERS, KV_LORA], pl.BF16],
    mla_w_k: pl.Tensor[[RANKS, MLA_STACK_LAYERS, LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    mla_w_v: pl.Tensor[[RANKS, MLA_STACK_LAYERS, LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    mla_w_o: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D, LOCAL_H * V_DIM], pl.INT8],
    mla_w_o_scale: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D], pl.FP32],
    idx_w_q_b: pl.Tensor[[RANKS, MLA_STACK_LAYERS, Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    idx_w_k: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_k_norm_w: pl.Tensor[[RANKS, MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_k_norm_b: pl.Tensor[[RANKS, MLA_STACK_LAYERS, INDEX_DIM], pl.BF16],
    idx_w_weights: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D, INDEX_H], pl.BF16],
    idx_w_compress_gate: pl.Tensor[[RANKS, MLA_STACK_LAYERS, D, INDEX_DIM], pl.BF16],
    idx_compress_ape: pl.Tensor[[RANKS, MLA_STACK_LAYERS, INDEX_KPOOL, INDEX_DIM], pl.BF16],
    mhc_attn_fn: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_attn_scale: pl.Tensor[[RANKS, NUM_LAYERS, 3], pl.FP32],
    mhc_attn_base: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC], pl.FP32],
    norm_input_norm: pl.Tensor[[RANKS, NORM_STACK_LAYERS, D], pl.BF16],
    hadamard: pl.Tensor[[RANKS, INDEX_DIM, INDEX_DIM], pl.BF16],
    mla_latent_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[
        pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]
    ],
    out_attn_pre: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_attn_post: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_attn_res: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_collapsed: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_normed: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_attn_partial: pl.Out[pl.Tensor[[RANKS, CAPACITY, D], pl.FP32]],
    out_sublayer_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_x_hc_mid: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16]],
    dsa_query: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_H, QK_DIM], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_H, V_DIM], pl.BF16]],
    index_q: pl.InOut[pl.Tensor[[RANKS, CAPACITY, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.InOut[pl.Tensor[[RANKS, CAPACITY, INDEX_DIM], pl.BF16]],
    index_scores: pl.InOut[pl.Tensor[[RANKS, CAPACITY, POOL_AXIS], pl.FP32]],
    selected_pools: pl.InOut[pl.Tensor[[RANKS, CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.InOut[pl.Tensor[[RANKS, CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    bb_ord: pl.Scalar[pl.INT32],
    dsa_ord: pl.Scalar[pl.INT32],
    attn_epoch: pl.Scalar[pl.INT32],
    skip_topk: pl.Scalar[pl.INT32],
):
    """Launch one DSA attention half at the caller's per-program TP epoch."""
    tp_partial_buffer = pld.alloc_window_buffer([CAPACITY, D], dtype=pl.FP32)
    tp_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.unroll(RANKS):
        tp_partial = pld.window(tp_partial_buffer, [CAPACITY, D], dtype=pl.FP32)
        tp_arrived = pld.window(tp_arrived_buffer, [EP_SIZE, 1], dtype=pl.INT32)
        prefill_sparse_attention(
            x_hc[rank], request_ids[rank], mla_slots[rank],
            index_slots[rank], seg_start[rank], pool_count_tok[rank], tail_start[rank], tail_count[rank],
            kv_len_tok[rank], pool_token_slots[rank], pool_slots[rank], pool_blocks[rank], block_table[rank],
            active_tokens[rank], mla_w_q_a[rank], mla_w_q_a_scale[rank], mla_q_a_norm[rank], mla_w_q_b[rank],
            mla_w_q_b_scale[rank], mla_w_kv_a[rank], mla_w_kv_a_scale[rank], mla_kv_a_norm[rank],
            mla_w_k[rank], mla_w_v[rank], mla_w_o[rank], mla_w_o_scale[rank], idx_w_q_b[rank], idx_w_k[rank],
            idx_k_norm_w[rank], idx_k_norm_b[rank], idx_w_weights[rank], idx_w_compress_gate[rank],
            idx_compress_ape[rank], mhc_attn_fn[rank], mhc_attn_scale[rank], mhc_attn_base[rank],
            norm_input_norm[rank], hadamard[rank], mla_latent_cache[rank],
            index_raw_cache[rank], index_pool_cache[rank], index_pool_scale[rank], tp_partial, tp_arrived,
            out_attn_pre[rank], out_attn_post[rank], out_attn_res[rank], out_collapsed[rank],
            out_normed[rank], out_attn_partial[rank], out_sublayer_out[rank], out_x_hc_mid[rank],
            dsa_query[rank], attn_out[rank], index_q[rank], index_k[rank], index_scores[rank],
            selected_pools[rank], selected_valid[rank], bb_ord, dsa_ord, rank, attn_epoch, skip_topk,
            device=rank,
            query_extent=CAPACITY,
        )


@pl.jit.host
def l3_prefill_head_rows(
    x_hc: pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16],
    logit_row_indices: pl.Tensor[[RANKS, LOGIT_ROWS], pl.INT32],
    head_hidden: pl.Out[pl.Tensor[[RANKS, LOGIT_ROWS, D], pl.BF16]],
):
    """Collapse the head and gather the logit rows (the forward's last dispatch)."""
    for rank in pl.unroll(RANKS):
        prefill_head_rows(x_hc[rank], logit_row_indices[rank], head_hidden[rank], device=rank)


@pl.jit
def prefill_lm_head_logits(
    head_hidden: pl.Tensor[[LOGIT_ROWS, D], pl.BF16],
    final_norm: pl.Tensor[[D], pl.BF16],
    weight: pl.Tensor[[FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[BATCH, FP32_HEAD_VOCAB_PAD], pl.FP32]],
):
    """Project the request prefix of the device's padded head-row buffer."""
    selected = head_hidden[0:BATCH, 0:D]
    lm_head_logits(selected, final_norm, weight, logits)
    return logits


@pl.jit.host
def l3_prefill_lm_head_logits(
    head_hidden: pl.Tensor[[RANKS, LOGIT_ROWS, D], pl.BF16],
    final_norm: pl.Tensor[[RANKS, D], pl.BF16],
    weight: pl.Tensor[[RANKS, FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    logits: pl.Out[pl.Tensor[[RANKS, BATCH, FP32_HEAD_VOCAB_PAD], pl.FP32]],
):
    """Normalize the gathered rows and project each rank's vocabulary shard."""
    for rank in pl.unroll(RANKS):
        prefill_lm_head_logits(head_hidden[rank], final_norm[rank], weight[rank], logits[rank], device=rank)


@pl.jit
def prefill_mtp_prepare(
    target_streams: pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16],
    embedding_rows: pl.Tensor[[CAPACITY, D], pl.BF16],
    feedback_norm: pl.Tensor[[D], pl.BF16],
    enorm: pl.Tensor[[D], pl.BF16],
    hnorm: pl.Tensor[[D], pl.BF16],
    eh_proj: pl.Tensor[[D, 2 * D], pl.BF16],
    request_ids: pl.Tensor[[CAPACITY], pl.INT32],
    cached_hidden: pl.Tensor[[BATCH, D], pl.BF16],
    cached_pools: pl.Tensor[[BATCH, KPOOL_SELECT_K], pl.INT32],
    cached_valid: pl.Tensor[[BATCH, KPOOL_SELECT_K], pl.INT32],
    selected_pools: pl.InOut[pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.InOut[pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    x_hc: pl.Out[pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16]],
    recycle: pl.Scalar[pl.INT32],
):
    """Project target or recycled MTP feedback and restore request-local proposal selections."""
    collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    if recycle == 0:
        mhc_head(target_streams, collapsed)
    else:
        for row in pl.spmd(CAPACITY, name_hint="mtp_feedback_gather"):
            request = pl.read(request_ids, [row])
            value = pl.full([1, D], dtype=pl.BF16, value=0.0)
            pools = pl.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=-1)
            valid = pl.full([1, KPOOL_SELECT_K], dtype=pl.INT32, value=0)
            if request >= 0:
                value = cached_hidden[request : request + 1, :]
                pools = cached_pools[request : request + 1, :]
                valid = cached_valid[request : request + 1, :]
            collapsed[row : row + 1, :] = value
            selected_pools[row : row + 1, :] = pools
            selected_valid[row : row + 1, :] = valid
    feedback = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    rmsnorm(collapsed, feedback_norm, feedback)
    projected = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    mtp_projection(embedding_rows, feedback, enorm, hnorm, eh_proj, projected)
    seed_hc_streams(projected, x_hc)
    return x_hc, selected_pools, selected_valid


@pl.jit.host
def l3_prefill_mtp_prepare(
    target_streams: pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16],
    embedding_rows: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    feedback_norm: pl.Tensor[[RANKS, D], pl.BF16],
    enorm: pl.Tensor[[RANKS, D], pl.BF16],
    hnorm: pl.Tensor[[RANKS, D], pl.BF16],
    eh_proj: pl.Tensor[[RANKS, D, 2 * D], pl.BF16],
    request_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    cached_hidden: pl.Tensor[[RANKS, BATCH, D], pl.BF16],
    cached_pools: pl.Tensor[[RANKS, BATCH, KPOOL_SELECT_K], pl.INT32],
    cached_valid: pl.Tensor[[RANKS, BATCH, KPOOL_SELECT_K], pl.INT32],
    selected_pools: pl.InOut[pl.Tensor[[RANKS, CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    selected_valid: pl.InOut[pl.Tensor[[RANKS, CAPACITY, KPOOL_SELECT_K], pl.INT32]],
    x_hc: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16]],
    recycle: pl.Scalar[pl.INT32],
):
    for rank in pl.unroll(RANKS):
        prefill_mtp_prepare(
            target_streams[rank], embedding_rows[rank], feedback_norm[rank], enorm[rank], hnorm[rank],
            eh_proj[rank], request_ids[rank], cached_hidden[rank], cached_pools[rank], cached_valid[rank],
            selected_pools[rank], selected_valid[rank], x_hc[rank], recycle, device=rank,
        )


@pl.jit
def prefill_mtp_head(
    head_hidden: pl.Tensor[[LOGIT_ROWS, D], pl.BF16],
    final_norm: pl.Tensor[[D], pl.BF16],
    weight: pl.Tensor[[FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    selected_pools: pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[CAPACITY, KPOOL_SELECT_K], pl.INT32],
    query_start_loc: pl.Tensor[[BATCH + 1], pl.INT32],
    cached_hidden: pl.InOut[pl.Tensor[[BATCH, D], pl.BF16]],
    cached_pools: pl.InOut[pl.Tensor[[BATCH, KPOOL_SELECT_K], pl.INT32]],
    cached_valid: pl.InOut[pl.Tensor[[BATCH, KPOOL_SELECT_K], pl.INT32]],
    logits: pl.Out[pl.Tensor[[BATCH, FP32_HEAD_VOCAB_PAD], pl.FP32]],
):
    for request in pl.spmd(BATCH, name_hint="mtp_feedback_commit"):
        begin = pl.read(query_start_loc, [request])
        end = pl.read(query_start_loc, [request + 1])
        if end > begin:
            cached_hidden[request : request + 1, :] = head_hidden[request : request + 1, :]
            cached_pools[request : request + 1, :] = selected_pools[end - 1 : end, :]
            cached_valid[request : request + 1, :] = selected_valid[end - 1 : end, :]
    selected = head_hidden[0:BATCH, 0:D]
    lm_head_mtp_logits(selected, final_norm, weight, logits)
    return cached_hidden, cached_pools, cached_valid, logits


@pl.jit.host
def l3_prefill_mtp_head(
    head_hidden: pl.Tensor[[RANKS, LOGIT_ROWS, D], pl.BF16],
    final_norm: pl.Tensor[[RANKS, D], pl.BF16],
    weight: pl.Tensor[[RANKS, FP32_HEAD_VOCAB_PAD, D], pl.FP32],
    selected_pools: pl.Tensor[[RANKS, CAPACITY, KPOOL_SELECT_K], pl.INT32],
    selected_valid: pl.Tensor[[RANKS, CAPACITY, KPOOL_SELECT_K], pl.INT32],
    query_start_loc: pl.Tensor[[RANKS, BATCH + 1], pl.INT32],
    cached_hidden: pl.InOut[pl.Tensor[[RANKS, BATCH, D], pl.BF16]],
    cached_pools: pl.InOut[pl.Tensor[[RANKS, BATCH, KPOOL_SELECT_K], pl.INT32]],
    cached_valid: pl.InOut[pl.Tensor[[RANKS, BATCH, KPOOL_SELECT_K], pl.INT32]],
    logits: pl.Out[pl.Tensor[[RANKS, BATCH, FP32_HEAD_VOCAB_PAD], pl.FP32]],
):
    for rank in pl.unroll(RANKS):
        prefill_mtp_head(
            head_hidden[rank], final_norm[rank], weight[rank], selected_pools[rank], selected_valid[rank],
            query_start_loc[rank], cached_hidden[rank], cached_pools[rank], cached_valid[rank], logits[rank],
            device=rank,
        )


def golden_prefill_layer(tensors):
    """Add rank-local shared outputs to routed results returned to their source ranks.

    Both branches reuse their standalone torch references. Each shared expert is
    complete, so different ranks' token outputs are never summed together.
    """

    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    # Block head, per rank, with the same torch references gate.py's test uses.
    route_indices = torch.zeros(N_RANKS, T, TOPK, dtype=torch.int32)
    route_weights = torch.zeros(N_RANKS, T, TOPK, dtype=torch.float32)
    x_int8 = torch.zeros(N_RANKS, T, D, dtype=torch.int8)
    x_scale = torch.zeros(N_RANKS, T, 1, dtype=torch.float32)
    for r in range(N_RANKS):
        x_norm, quantized, scale = golden_rmsnorm_quant(tensors["x_mixed"][r], tensors["norm_weight"][r])
        weights, indices = gate(x_norm, tensors["gate_weight"][r], tensors["correction_bias"][r])
        route_weights[r], route_indices[r] = weights, indices
        x_int8[r], x_scale[r] = quantized, scale
    if num_tokens < T:
        route_weights[:, num_tokens:] = 0
        route_indices[:, num_tokens:] = 0
        x_scale[:, num_tokens:] = 0

    routed_work = {
        "route_indices": route_indices,
        "route_weights": route_weights,
        "x_int8": x_int8,
        "x_scale": x_scale,
        "w_gate_up": tensors["w_gate_up"],
        "w_gate_up_scale": tensors["w_gate_up_scale"],
        "w_down": tensors["w_down"],
        "w_down_scale": tensors["w_down_scale"],
        "num_tokens": num_tokens,
        "routed": torch.zeros(N_RANKS, T, D, dtype=torch.float32),
    }
    golden_prefill_moe(routed_work)
    routed = routed_work["routed"]

    shared = torch.zeros(N_RANKS, T, D, dtype=torch.float32)
    for r in range(N_RANKS):
        work = {
            "x_int8": x_int8[r],
            "x_scale": x_scale[r],
            "w_gate_up": tensors["shared_w_gate_up"][r],
            "w_gate_up_scale": tensors["shared_w_gate_up_scale"][r],
            "w_down": tensors["shared_w_down"][r],
            "w_down_scale": tensors["shared_w_down_scale"][r],
            "output": torch.zeros(T, D, dtype=torch.float32),
        }
        golden_expert_shared_fn(work)
        shared[r] = work["output"]

    out = torch.zeros_like(tensors["y"])
    out[:, :num_tokens] = routed[:, :num_tokens] + shared[:, :num_tokens]
    tensors["y"][:] = out


def build_tensor_specs(num_tokens=T, *, num_ranks=EP_SIZE, checkpoint=None, layer_id=3, hidden=None):
    """Build generated fixtures or load a checkpoint's complete MoE FFN weights."""
    if checkpoint is not None:

        return build_checkpoint_moe_specs(
            checkpoint, layer_id, capacity=T, num_tokens=num_tokens, num_ranks=num_ranks, hidden=hidden,
        )
    if hidden is not None:
        raise ValueError("hidden activations require checkpoint weights")

    from golden import ScalarSpec, TensorSpec

    num_tokens = max(0, min(T, int(num_tokens)))

    def quantize_rows(weight, dequant_std):
        """Symmetric per-output-channel INT8 quant, rescaled to ``dequant_std``."""
        amax = weight.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        s = amax / INT8_SCALE_MAX
        w_i8 = torch.round(weight / s).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        s = (s.squeeze(-1) * (dequant_std / (w_i8.float() * s).std())).float()
        return w_i8, s

    def gen_shared_weight(shape, dequant_std, chan_cv):
        # Quantize one complete shared expert and replicate it to every EP rank.
        weight = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(shape[0], 1))
        w_i8, scale = quantize_rows(weight, dequant_std)
        return (
            w_i8.unsqueeze(0).expand(N_RANKS, *w_i8.shape).contiguous(),
            scale.unsqueeze(0).expand(N_RANKS, *scale.shape).contiguous(),
        )

    gen = torch.Generator().manual_seed(7)

    emb = torch.randn(N_RANKS, T, D, generator=gen).to(torch.bfloat16)
    hc_fn = torch.randn(MIX_HC, HC_DIM, generator=gen).to(torch.bfloat16)
    hc_scale = torch.rand(3, generator=gen)
    hc_base = torch.randn(MIX_HC, generator=gen)
    with torch.no_grad():
        batches = []
        for rank in range(N_RANKS):
            x_hc = hc_seed(emb[rank])
            pre_mix = hc_mixes(x_hc, hc_fn, hc_scale, hc_base)[0]
            batches.append(hc_pre(x_hc, pre_mix))
        x_mixed = torch.stack(batches)
    # Norm/router and full shared weights are replicated checkpoint tensors.
    gate_row = torch.randn(N_EXPERTS, D, generator=gen) / (D**0.5)
    gate_weight = gate_row.unsqueeze(0).expand(N_RANKS, *gate_row.shape).contiguous()

    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2}
    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2}
    (w_gu_i8, w_gu_s) = gen_routed_weight((N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
    w2_i8, w2_s = gen_routed_weight((N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])
    shared_gu_i8, shared_gu_s = gen_shared_weight((2 * MOE_INTER, D), SHARED_DEQUANT_STD["w1"], 0.50)
    shared_w2_i8, shared_w2_s = gen_shared_weight((D, MOE_INTER), SHARED_DEQUANT_STD["w2"], 0.33)

    return [
        TensorSpec("x_mixed", [N_RANKS, T, D], torch.bfloat16, init_value=lambda: x_mixed),
        TensorSpec("norm_weight", [N_RANKS, D], torch.bfloat16, init_value=lambda : torch.ones(N_RANKS, D, dtype=torch.bfloat16)),
        TensorSpec("gate_weight", [N_RANKS, N_EXPERTS, D], torch.bfloat16, init_value=lambda: gate_weight),
        # Fix a global top-8 expert set spanning the EP ranks for this fixture.
        TensorSpec("correction_bias", [N_RANKS, N_EXPERTS], torch.float32, init_value=lambda : make_correction_bias(N_RANKS, N_EXPERTS, TOPK)),
        TensorSpec("w_gate_up", [N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], torch.int8, init_value=lambda : w_gu_i8),
        TensorSpec("w_gate_up_scale", [N_RANKS, N_LOCAL_EXPERTS, 2 * MOE_INTER], torch.float32, init_value=lambda : w_gu_s),
        TensorSpec("w_down", [N_RANKS, N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("w_down_scale", [N_RANKS, N_LOCAL_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("shared_w_gate_up", [N_RANKS, 2 * MOE_INTER, D], torch.int8, init_value=lambda : pack_int8_nz(shared_gu_i8)),
        TensorSpec("shared_w_gate_up_scale", [N_RANKS, 2 * MOE_INTER], torch.float32, init_value=lambda : shared_gu_s),
        TensorSpec("shared_w_down", [N_RANKS, D, MOE_INTER], torch.int8, init_value=lambda : pack_int8_nz(shared_w2_i8)),
        TensorSpec("shared_w_down_scale", [N_RANKS, D], torch.float32, init_value=lambda: shared_w2_s),
        TensorSpec("y", [N_RANKS, T, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("num_ranks", torch.int32, num_ranks),
    ]


def prefill_layer_ring_heap():
    """Budget routed scratch, Gate buffers, shared tiles and FFN intermediates."""
    gate_rows = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
    # FP32 x*gamma, BF16 routing inputs, scale, scores, logits and sort buffers.
    gate_bytes = gate_rows * (6 * D + 4 + 12 * SCORE_PAD + 4 * N_EXPERTS + 8 * TOPK_PAD)
    shared_rows = ((T + SH_M_TILE - 1) // SH_M_TILE) * SH_M_TILE
    # Gate/up INT32, FP32/INT8 activations, padded scales and down INT32 outputs.
    shared_bytes = shared_rows * (13 * MOE_INTER + 4 * SH_ROW_PAD + 4 * D)
    # Routed/shared FP32 outputs, router results and the reused INT8 activation view.
    layer_bytes = T * (9 * D + 8 * TOPK + 4)
    return prefill_ring_heap(additional_bytes=gate_bytes + shared_bytes + layer_bytes)


@dataclass
class LayerRefWeights:
    """Everything one backbone layer's reference forward needs (FP32)."""

    layer_id: int
    config: Glm53LayerConfig
    input_norm: torch.Tensor  # [D]
    post_norm: torch.Tensor  # [D]
    # None exactly when config.has_hyper_connections is False (the MTP layer),
    # which carries no hc_* tensors and uses a plain residual wiring instead.
    attn_hc: MhcTensors | None
    ffn_hc: MhcTensors | None
    model: Glm53FlashConfig | None = None  # geometry the weights belong to (default FLASH)
    kda: KdaTensors | None = None
    mla: MlaTensors | None = None
    indexer: IndexerTensors | None = None
    dense: DenseTensors | None = None  # dense-MLP layers (0..first_k_dense)
    router: RouterTensors | None = None  # MoE layers
    experts: ExpertProvider | None = None  # MoE layers, streamed
    shared: DenseTensors | None = None  # MoE layers


def mla_forward(
    x: torch.Tensor,
    weights: MlaTensors,
    indexer: IndexerTensors,
    attention_mask: torch.Tensor,
    model: Glm53FlashConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One NoPE-MLA DSA sub-layer (port of ``Glm5NextTextAttention.forward``).

    Returns ``(output [B, S, D], internals)``.
    """
    batch_size, seq_len, _ = x.shape
    heads = model.num_attention_heads
    qk_dim, v_dim = model.qk_head_dim, model.v_head_dim

    q_resid = reference_rms_norm(x @ weights.w_q_a.t(), weights.q_a_norm, model.rms_norm_eps)
    query = (q_resid @ weights.w_q_b.t()).view(batch_size, seq_len, heads, qk_dim).transpose(1, 2)

    compressed = x @ weights.w_kv_a.t()  # NoPE: no rope columns
    k_latent = reference_rms_norm(compressed, weights.kv_a_norm, model.rms_norm_eps)

    # kv_b per head: [H, qk + v, kv_lora] split into key/value halves.
    kv_b = weights.kv_b.view(heads, qk_dim + v_dim, model.kv_lora_rank)
    k_nope = torch.einsum("bsl,hdl->bhsd", k_latent, kv_b[:, :qk_dim, :])
    value = torch.einsum("bsl,hdl->bhsd", k_latent, kv_b[:, qk_dim:, :])

    topk_indices, indexer_internals = reference_indexer_topk(x, q_resid, indexer, attention_mask, model)
    mask = topk_attention_mask(topk_indices, kv_len=seq_len)

    scores = query @ k_nope.transpose(-1, -2) * (qk_dim**-0.5)
    probs = torch.softmax(scores + mask, dim=-1, dtype=torch.float32)
    attn = probs @ value
    output = attn.transpose(1, 2).reshape(batch_size, seq_len, heads * v_dim) @ weights.w_o.t()

    internals = {
        "mla_q_resid": q_resid,
        "mla_k_latent": k_latent,
        "mla_topk_indices": topk_indices,
        **indexer_internals,
    }
    return output, internals


def reference_layer_forward(
    streams: torch.Tensor, weights: LayerRefWeights, attention_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One decoder layer on the mHC four-stream tensor (HF decoder-layer order).

    ``streams`` is ``[B, S, hc_mult, D]`` FP32; ``attention_mask`` is ``[B, S]``
    bool (defaults to all-visible). Returns the new streams and a dict of every
    sub-layer boundary (the default dump set plus internals). Geometry comes
    from ``weights.model`` so scaled mini-checkpoint models behave identically.
    """
    model_cfg = weights.model if weights.model is not None else FLASH
    boundaries: dict[str, torch.Tensor] = {"x_hc_in": streams}
    if attention_mask is None:
        attention_mask = torch.ones(streams.shape[:2], dtype=torch.bool)
    if weights.attn_hc is None or weights.ffn_hc is None:
        raise NotImplementedError(
            f"layer {weights.layer_id} has no hyper-connections (MTP/plain residual): its "
            "wiring (eh_proj/enorm/hnorm) is outside the prefill backbone. Load it with "
            "load_layer() and drive mla_forward()/moe_forward() on the sub-layer boundaries."
        )

    residual = streams
    pre, post, comb, collapsed = mhc_mapping(streams, weights.attn_hc, model_cfg)
    x = reference_rms_norm(collapsed, weights.input_norm, model_cfg.rms_norm_eps)
    boundaries.update(attn_hc_pre=pre, attn_hc_post=post, attn_hc_comb=comb, attn_norm_in=x)
    if weights.config.attention_kind is AttentionKind.LINEAR:
        attn_out, attn_internals = kda_forward(x, weights.kda, model_cfg, attention_mask)
    else:
        attn_out, attn_internals = mla_forward(x, weights.mla, weights.indexer, attention_mask, model_cfg)
    boundaries["attn_out"] = attn_out
    boundaries.update(attn_internals)
    streams = combine_streams(post, comb, attn_out, residual)
    boundaries["x_hc_mid"] = streams

    residual = streams
    pre, post, comb, collapsed = mhc_mapping(streams, weights.ffn_hc, model_cfg)
    x = reference_rms_norm(collapsed, weights.post_norm, model_cfg.rms_norm_eps)
    boundaries.update(ffn_hc_pre=pre, ffn_hc_post=post, ffn_hc_comb=comb, ffn_norm_in=x)
    if weights.config.mlp_kind is MlpKind.DENSE:
        ffn_out = dense_mlp_forward(x, weights.dense, model_cfg)
        ffn_internals: dict[str, torch.Tensor] = {}
    else:
        ffn_out, ffn_internals = moe_forward(x, weights.router, weights.experts, weights.shared, model_cfg)
    boundaries["ffn_out"] = ffn_out
    boundaries.update(ffn_internals)
    streams = combine_streams(post, comb, ffn_out, residual)
    boundaries["x_hc_out"] = streams
    return streams, boundaries


def synthetic_layer(model: Glm53FlashConfig, layer_id: int, seed: int) -> LayerRefWeights:
    """Synthetic layer weights at the true deployment geometry."""
    generator = torch.Generator().manual_seed(seed + 7919 * layer_id)
    config = model.layer_config(layer_id)
    layer = LayerRefWeights(
        layer_id=layer_id, config=config, model=model,
        input_norm=1.0 + torch.randn(model.hidden_size, generator=generator) * 0.05,
        post_norm=1.0 + torch.randn(model.hidden_size, generator=generator) * 0.05,
        attn_hc=synthetic_mhc(model, generator) if config.has_hyper_connections else None,
        ffn_hc=synthetic_mhc(model, generator) if config.has_hyper_connections else None,
    )
    if config.attention_kind is AttentionKind.LINEAR:
        layer.kda = synthetic_kda(model, generator)
    else:
        layer.mla = synthetic_mla(model, generator)
        layer.indexer = synthetic_indexer(model, generator)
    if config.mlp_kind is MlpKind.DENSE:
        layer.dense = synthetic_dense(model, generator, model.intermediate_size)
    else:
        layer.router = RouterTensors(
            weight=_randn((model.n_routed_experts, model.hidden_size), generator),
            bias=torch.randn(model.n_routed_experts, generator=generator) * 0.01,
        )
        layer.experts = SyntheticExpertProvider(model, seed + 31 * layer_id)
        layer.shared = synthetic_dense(model, generator, model.moe_intermediate_size)
    return layer


def _test_mla(model: Glm53FlashConfig) -> None:
    torch.manual_seed(19)
    b, s = 2, 18
    x = torch.randn(b, s, model.hidden_size) * 0.5
    weights = synthetic_mla(model, torch.Generator().manual_seed(3))
    indexer = synthetic_indexer(model, torch.Generator().manual_seed(4))
    mask = torch.ones(b, s, dtype=torch.bool)
    out, internals = mla_forward(x, weights, indexer, mask, model)
    assert out.shape == (b, s, model.hidden_size) and torch.isfinite(out).all()
    assert internals["mla_q_resid"].shape == (b, s, model.q_lora_rank)
    assert internals["mla_topk_indices"].shape[-1] == model.index_topk + model.index_kpool - 1
    print("  MLA: shapes finite, NoPE latent path, topk width")


def _test_router_and_moe(model: Glm53FlashConfig) -> None:
    torch.manual_seed(23)
    tokens = 5
    router = RouterTensors(
        weight=torch.randn(model.n_routed_experts, model.hidden_size) * 0.02,
        bias=torch.randn(model.n_routed_experts) * 0.01,
    )
    flat = torch.randn(tokens, model.hidden_size) * 0.5
    topk_weights, topk_indices = router_forward(flat, router, model)
    assert topk_indices.shape == (tokens, model.num_experts_per_tok)
    assert bool((topk_indices >= 0).all()) and bool((topk_indices < model.n_routed_experts).all())
    for row in range(tokens):
        assert len(set(topk_indices[row].tolist())) == model.num_experts_per_tok
        _assert_close(
            topk_weights[row].sum(), torch.tensor(model.routed_scaling_factor),
            "router weights must renormalize to routed_scaling_factor", atol=0.0001,
        )
    # Manual noaux-tc recompute: bias steers choice, weights stay bias-free.
    scores = (flat @ router.weight.t()).sigmoid()
    biased = scores + router.bias
    expected_idx = biased.topk(model.num_experts_per_tok, dim=-1, sorted=False).indices
    assert set(expected_idx[0].tolist()) == set(topk_indices[0].tolist())
    expected_w = scores[0].gather(0, topk_indices[0])
    expected_w = expected_w / (expected_w.sum() + 1e-20) * model.routed_scaling_factor
    _assert_close(topk_weights[0], expected_w, "router weight formula")

    # MoE output vs a manual per-token recompute (expert streaming correctness).
    shared = synthetic_dense(model, torch.Generator().manual_seed(9), model.moe_intermediate_size)
    provider = SyntheticExpertProvider(model, 77)
    x = torch.randn(2, 6, model.hidden_size) * 0.5
    out, internals = moe_forward(x, router, provider, shared, model)
    topk_weights, topk_indices = internals["router_topk_weights"], internals["router_topk_indices"]
    flat_x = x.reshape(-1, model.hidden_size)
    for token in (0, 7):
        manual = dense_mlp_forward(x.reshape(-1, model.hidden_size)[token], shared, model)
        for position, expert in enumerate(topk_indices[token].tolist()):
            gate_up = provider.gate_up(expert)
            gate, up = gate_up.chunk(2, dim=0)
            hidden_act = swiglu_limited(flat_x[token] @ gate.t(), flat_x[token] @ up.t(), model.swiglu_limit)
            manual = manual + (hidden_act @ provider.down(expert).t()) * topk_weights[token, position]
        _assert_close(
            out.reshape(-1, model.hidden_size)[token], manual, "MoE manual recompute", atol=0.0001,
            rtol=0.0001,
        )
    print("  router+MoE: noaux-tc formula, renormalized weights, expert streaming")


if __name__ == "__main__":
    import argparse

    from pypto.ir import DistributedConfig

    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("--ep", type=int, default=2, help="expert-parallel size / rank count")
    parser.add_argument(
        "-d", "--device", type=str, default=",".join((str(i) for i in range(N_RANKS))),
        help=f"comma-separated device ids (need {N_RANKS})",
    )
    parser.add_argument("--tokens", type=int, default=T, help="per-rank token capacity (module level)")
    parser.add_argument("--num-tokens", type=int, default=None, help="active rows; defaults to hidden rows or capacity")
    parser.add_argument("--checkpoint", type=Path, help="W8A8 checkpoint directory; default: generated weights")
    parser.add_argument("--layer", type=int, default=3, help="checkpoint MoE layer index (default: 3)")
    parser.add_argument("--hidden", type=Path, help="saved BF16 [EP, active tokens, D] x_mixed; requires --checkpoint")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.hidden is not None and args.checkpoint is None:
        parser.error("--hidden requires --checkpoint")
    hidden = torch.load(args.hidden, map_location="cpu", weights_only=True) if args.hidden is not None else None
    if hidden is not None and (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3):
        parser.error("--hidden must contain a BF16 tensor shaped [EP, active tokens, D]")
    num_tokens = args.num_tokens if args.num_tokens is not None else hidden.shape[1] if hidden is not None else T

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"
    num_ranks = len(device_ids)

    result = run(
        fn=l3_prefill_layer,
        specs=build_tensor_specs(num_tokens, num_ranks=num_ranks, checkpoint=args.checkpoint, layer_id=args.layer, hidden=hidden),
        golden_fn=golden_prefill_layer,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.backend_type, execution_mode=args.execution_mode,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
            ring_heap=prefill_layer_ring_heap(),
        ),
        rtol=0.001,
        atol=0.001,
        compare_fn={"y": ratio_reldiff(diff_thd=0.01, pct_thd=0.05)},
        compile_only=args.compile_only,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
