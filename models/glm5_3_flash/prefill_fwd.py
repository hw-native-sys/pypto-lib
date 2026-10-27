# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash resident forward execution and generation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import dataclasses
import inspect
import json
import math
import os
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from math import gcd
from typing import Mapping, Sequence

import pypto.language as pl
import torch
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig, StackedDeviceTensor

import models.glm5_3_flash.config as C
from models.glm5_3_flash.attention import MlaTensors
from models.glm5_3_flash.config import (
    BLOCK_SIZE, DENSE_INTER, DP_SIZE, EP_SIZE, FLASH, HC_MULT, IMAGE_TOKEN_ID, INDEX_DIM, INDEX_H,
    INDEX_KPOOL, INDEX_STATE_WIDTH, KDA_DIM, KPOOL_SELECT_K, KV_LORA, LOCAL_H, LOCAL_KDA_H,
    LOCAL_KDA_QKV_DIM, LOCAL_VOCAB, N_EXPERTS, QK_DIM, SUPPORTED_EP_SIZES, SUPPORTED_TP_SIZES, TP_SIZE,
    V_DIM, AttentionKind, D, Glm53FlashConfig, MlpKind,
)
from models.glm5_3_flash.dense_mlp import DenseTensors, _test_dense_mlp
from models.glm5_3_flash.expert_routed import CheckpointExpertProvider
from models.glm5_3_flash.gate import RouterTensors
from models.glm5_3_flash.indexer import POOL_SCALE_WIDTH, IndexerTensors, _test_indexer
from models.glm5_3_flash.kda import (
    BETA_PAD, CHUNK_TILE, CONV_DIM, STATE_LEN, KdaTensors, _test_kda_cores, kda_host_constants,
)
from models.glm5_3_flash.lm_head import FP32_HEAD_VOCAB_PAD, reference_lm_head_logits
from models.glm5_3_flash.mhc import MhcTensors, _test_mhc, combine_streams
from models.glm5_3_flash.prefill_layer import (
    BATCH, CACHE_LAYOUTS, CACHE_TOKENS, CAPACITY, DENSE_NUM_LAYERS, DSA_NUM_LAYERS, KDA_NUM_LAYERS,
    KDA_ROWS, LATENT_BLOCKS, LATENT_ROWS, LOGIT_ROWS, MAX_SEQ_LEN, MLA_STACK_LAYERS, MOE_STACK_LAYERS,
    NUM_LAYERS, POOL_AXIS, POOL_STRIDE, POOL_TABLE_MAX, POOL_TABLE_ROWS, PREFILL_MAX_TOKENS, RANKS, T_MOE,
    TABLE_MAX, LayerRefWeights, _shape_options, _test_mla, _test_router_and_moe, l3_prefill_dense_layer,
    l3_prefill_head_rows, l3_prefill_kda_attention, l3_prefill_lm_head_logits, l3_prefill_moe_layer,
    l3_prefill_mtp_head, l3_prefill_mtp_prepare, l3_prefill_seed, l3_prefill_sparse_attention,
    reference_layer_forward, synthetic_layer,
)
from models.glm5_3_flash.rmsnorm import reference_rms_norm
from models.glm5_3_flash.utils import (
    _FF_DOWN, _FF_FUSED, _FF_GATE, _FF_UP, ROLES, TEXT_LAYER_ROOTS, Checkpoint, QuantDescription,
    SharedCachePages, _as_batch_seqs, _assert_close, _find, _pad_to, _prefill_model, _prefixed,
    _read_quant, _resolve_name, _resolve_top_name, _scale_of, _try_find, _weight_shape,
    build_forward_metadata, build_indexer_chunk_metadata, check_ffn_scales, fuse_gate_up, gather_rows,
    golden_split_kv_b, layer_prefix, layer_prefixes, layer_router, load_shared_expert, load_tensors,
    pack_int8_nz, read_reference_float, scaled_model_from_config, shard_dense_mlp, shard_routed_experts,
    split_kv_b, sylvester_hadamard, unpack_int8_nz,
)
from models.glm5_3_flash.vision import VisionEncoder, VisionProcessor, VisualInput

if CACHE_TOKENS <= 0 or CACHE_TOKENS % C.BLOCK_SIZE:
    raise ValueError(f"--cache-tokens must be positive and aligned to {C.BLOCK_SIZE}")
if _shape_options.tokens is not None and _shape_options.tokens != T_MOE:
    raise ValueError(f"--tokens must match the resident owner capacity {T_MOE}")
_ci_rung_pinned = os.environ.get("GLM53_PREFILL_CI_PIN") == "2"
if (C.TP_SIZE, C.EP_SIZE, C.DP_SIZE) == (2, 2, 1) and _ci_rung_pinned:
    pass
elif (C.TP_SIZE, C.EP_SIZE, C.DP_SIZE) not in ((8, 8, 1), (16, 16, 1)):
    raise ValueError(
        "glm5_3_flash prefill_fwd requires TP8/EP8/DP1 or TP16/EP16/DP1; "
        f"got TP{C.TP_SIZE}/EP{C.EP_SIZE}/DP{C.DP_SIZE}. TP2/EP2 is accepted "
        "only for the marker-pinned CI subset (GLM53_PREFILL_CI_PIN)"
    )
if not 16 <= CAPACITY <= PREFILL_MAX_TOKENS or CAPACITY % 16:
    raise ValueError(f"--capacity must be 16..{PREFILL_MAX_TOKENS} and a multiple of 16, got {CAPACITY}")
if CAPACITY % 64:
    raise ValueError(f"--capacity must be a multiple of 64 for indexer_proj, got {CAPACITY}")
if not 1 <= MAX_SEQ_LEN <= FLASH.max_position_embeddings:
    raise ValueError(f"--max-seq-len must be 1..{FLASH.max_position_embeddings}, got {MAX_SEQ_LEN}")
_RESET_MLA_ELEMENTS = TABLE_MAX * BLOCK_SIZE * KV_LORA
_RESET_RAW_ELEMENTS = TABLE_MAX * BLOCK_SIZE * INDEX_STATE_WIDTH
_RESET_POOL_ELEMENTS = POOL_STRIDE * INDEX_DIM
_RESET_SCALE_ELEMENTS = POOL_STRIDE * POOL_SCALE_WIDTH
_RESET_CONV_ELEMENTS = STATE_LEN * CONV_DIM
_RESET_RECURRENT_ELEMENTS = LOCAL_KDA_H * KDA_DIM * KDA_DIM
_RESET_MLA_TILE = gcd(4096, _RESET_MLA_ELEMENTS)
_RESET_RAW_TILE = gcd(4096, _RESET_RAW_ELEMENTS)
_RESET_POOL_TILE = gcd(4096, _RESET_POOL_ELEMENTS)
_RESET_SCALE_TILE = gcd(BLOCK_SIZE // INDEX_KPOOL * POOL_SCALE_WIDTH, _RESET_SCALE_ELEMENTS)
_RESET_CONV_TILE = gcd(4096, _RESET_CONV_ELEMENTS)
_RESET_CONV_LANE_TILE = min(4, _RESET_CONV_ELEMENTS // _RESET_CONV_TILE)
_RESET_RECURRENT_TILE = gcd(4096, _RESET_RECURRENT_ELEMENTS)
_RESET_RECURRENT_LANE_TILE = min(4, _RESET_RECURRENT_ELEMENTS // _RESET_RECURRENT_TILE)
PREFIX_SNAPSHOT_ENTRIES = 2
_PREFIX_COPY_TILE = 1024
_PREFIX_COPY_LANE_TILE = 16
_PREFIX_SCALE_TILE = min(_PREFIX_COPY_TILE, BLOCK_SIZE // INDEX_KPOOL * POOL_SCALE_WIDTH)
CONSTANT_STEP_INPUTS = {"kda_tri_incl", "kda_tri_strict", "kda_head_onehot", "kda_eye_wide", "kda_state_rows"}

_FAMILY_PREFIX = {
    "kda": "self_attn.",
    "mla": "self_attn.",
    "indexer": "self_attn.indexer.",
    "mhc": "",
    "norm": "",
    "router": "mlp.",
}

RT_PROGRAMS = (
    "seed", "dense", "kda_attn", "dsa_attn", "moe_ffn",
    "head",
)
X_H_PINGPONG = ("ws_x_hc_a", "ws_x_hc_b")
GLM_TENSORS = (
    "embedding_rows", "query_start_loc", "request_ids",
    "mla_slots", "index_slots", "kda_state_rows", "seg_start",
    "pool_count_tok", "tail_start", "tail_count", "kv_len_tok",
    "pool_token_slots", "pool_slots", "pool_blocks",
    "block_table", "cache_page_owners", "logit_row_indices",
    "active_tokens", "kda_tri_incl", "kda_tri_strict", "kda_head_onehot", "kda_eye_wide", "kda_w_q",
    "kda_w_k", "kda_w_v", "kda_conv_q", "kda_conv_k", "kda_conv_v", "kda_w_f_a", "kda_w_f_b", "kda_w_g_a",
    "kda_w_g_b", "kda_w_b", "kda_dt_bias", "kda_a_log", "kda_o_norm", "kda_w_o",
    "mla_w_q_a", "mla_w_q_a_scale", "mla_q_a_norm", "mla_w_q_b", "mla_w_q_b_scale", "mla_w_kv_a",
    "mla_w_kv_a_scale", "mla_kv_a_norm", "mla_w_k", "mla_w_v", "mla_w_o", "mla_w_o_scale",
    "idx_w_q_b", "idx_w_k", "idx_k_norm_w", "idx_k_norm_b", "idx_w_weights", "idx_w_compress_gate",
    "idx_compress_ape", "mhc_attn_fn", "mhc_attn_scale", "mhc_attn_base", "mhc_ffn_fn", "mhc_ffn_scale",
    "mhc_ffn_base", "norm_input_norm", "norm_post_norm",
    "router_gate_w", "router_gate_bias", "moe_w_gate_up", "moe_w_gate_up_scale", "moe_w_down",
    "moe_w_down_scale", "moe_shared_w_gate_up", "moe_shared_w_gate_up_scale", "moe_shared_w_down",
    "moe_shared_w_down_scale", "dense_w_gate_up", "dense_w_gate_up_scale", "dense_w_down",
    "dense_w_down_scale", "hadamard", "mla_latent_cache",
    "index_raw_cache", "index_pool_cache", "index_pool_scale",
    "kda_conv_state", "kda_recurrent_state", "ws_attn_pre", "ws_attn_post", "ws_attn_res", "ws_collapsed",
    "ws_normed", "ws_attn_partial", "ws_sublayer_out", "ws_ffn_pre", "ws_ffn_post", "ws_ffn_res",
    "ws_ffn_input", "ws_ffn_partial", "ws_ffn_out", "ws_x_hc_a", "ws_x_hc_b",
    "kda_out", "dsa_query", "attn_out", "index_q", "index_k", "index_scores",
    "selected_pools", "selected_valid", "mixed_qkv",
    "decay", "beta", "kda_q", "kda_k", "kda_v", "dense_x_int8",
    "head_hidden",
)
GLM_WEIGHT_STACKS = (
    ("kda_w_q", "kda.w_q"),
    ("kda_w_k", "kda.w_k"),
    ("kda_w_v", "kda.w_v"),
    ("kda_conv_q", "kda.conv_q"),
    ("kda_conv_k", "kda.conv_k"),
    ("kda_conv_v", "kda.conv_v"),
    ("kda_w_f_a", "kda.w_f_a"),
    ("kda_w_f_b", "kda.w_f_b"),
    ("kda_w_g_a", "kda.w_g_a"),
    ("kda_w_g_b", "kda.w_g_b"),
    ("kda_w_b", "kda.w_b"),
    ("kda_dt_bias", "kda.dt_bias"),
    ("kda_a_log", "kda.a_log"),
    ("kda_o_norm", "kda.o_norm"),
    ("kda_w_o", "kda.w_o"),
    ("mla_w_q_a", "mla.w_q_a"),
    ("mla_w_q_a_scale", "mla.w_q_a_scale"),
    ("mla_q_a_norm", "mla.q_a_norm"),
    ("mla_w_q_b", "mla.w_q_b"),
    ("mla_w_q_b_scale", "mla.w_q_b_scale"),
    ("mla_w_kv_a", "mla.w_kv_a"),
    ("mla_w_kv_a_scale", "mla.w_kv_a_scale"),
    ("mla_kv_a_norm", "mla.kv_a_norm"),
    ("mla_w_k", "mla.w_k"),
    ("mla_w_v", "mla.w_v"),
    ("mla_w_o", "mla.w_o"),
    ("mla_w_o_scale", "mla.w_o_scale"),
    ("idx_w_q_b", "indexer.w_q_b"),
    ("idx_w_k", "indexer.w_k"),
    ("idx_k_norm_w", "indexer.k_norm_w"),
    ("idx_k_norm_b", "indexer.k_norm_b"),
    ("idx_w_weights", "indexer.w_weights"),
    ("idx_w_compress_gate", "indexer.w_compress_gate"),
    ("idx_compress_ape", "indexer.compress_ape"),
    ("mhc_attn_fn", "mhc.attn_fn"),
    ("mhc_attn_scale", "mhc.attn_scale"),
    ("mhc_attn_base", "mhc.attn_base"),
    ("mhc_ffn_fn", "mhc.ffn_fn"),
    ("mhc_ffn_scale", "mhc.ffn_scale"),
    ("mhc_ffn_base", "mhc.ffn_base"),
    ("norm_input_norm", "norm.input_norm"),
    ("norm_post_norm", "norm.post_norm"),
    ("router_gate_w", "router.gate_w"),
    ("router_gate_bias", "router.gate_bias"),
    ("moe_w_gate_up", "moe.w_gate_up"),
    ("moe_w_gate_up_scale", "moe.w_gate_up_scale"),
    ("moe_w_down", "moe.w_down"),
    ("moe_w_down_scale", "moe.w_down_scale"),
    ("moe_shared_w_gate_up", "moe.shared_w_gate_up"),
    ("moe_shared_w_gate_up_scale", "moe.shared_w_gate_up_scale"),
    ("moe_shared_w_down", "moe.shared_w_down"),
    ("moe_shared_w_down_scale", "moe.shared_w_down_scale"),
    ("dense_w_gate_up", "dense.w_gate_up"),
    ("dense_w_gate_up_scale", "dense.w_gate_up_scale"),
    ("dense_w_down", "dense.w_down"),
    ("dense_w_down_scale", "dense.w_down_scale"),
    ("hadamard", "hadamard"),
)
_PARAM_TO_STACK = dict(GLM_WEIGHT_STACKS)
_STACK_TO_PARAM = {stack: param for param, stack in GLM_WEIGHT_STACKS}
WORKSPACES = {
    "ws_attn_pre": ((CAPACITY, HC_MULT), torch.float32),
    "ws_attn_post": ((CAPACITY, HC_MULT), torch.float32),
    "ws_attn_res": ((CAPACITY, HC_MULT, HC_MULT), torch.float32),
    "ws_collapsed": ((CAPACITY, D), torch.bfloat16),
    "ws_normed": ((CAPACITY, D), torch.bfloat16),
    "ws_attn_partial": ((CAPACITY, D), torch.float32),
    "ws_sublayer_out": ((CAPACITY, D), torch.bfloat16),
    "ws_ffn_pre": ((CAPACITY, HC_MULT), torch.float32),
    "ws_ffn_post": ((CAPACITY, HC_MULT), torch.float32),
    "ws_ffn_res": ((CAPACITY, HC_MULT, HC_MULT), torch.float32),
    "ws_ffn_input": ((CAPACITY, D), torch.bfloat16),
    "ws_ffn_partial": ((CAPACITY, D), torch.float32),
    "ws_ffn_out": ((CAPACITY, D), torch.bfloat16),
    "ws_x_hc_a": ((CAPACITY, HC_MULT, D), torch.bfloat16),
    "ws_x_hc_b": ((CAPACITY, HC_MULT, D), torch.bfloat16),
    "kda_out": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    "dsa_query": ((CAPACITY, LOCAL_H, QK_DIM), torch.bfloat16),
    "attn_out": ((CAPACITY, LOCAL_H, V_DIM), torch.bfloat16),
    "index_q": ((CAPACITY, INDEX_H, INDEX_DIM), torch.bfloat16),
    "index_k": ((CAPACITY, INDEX_DIM), torch.bfloat16),
    "index_scores": ((CAPACITY, POOL_AXIS), torch.float32),
    "selected_pools": ((CAPACITY, KPOOL_SELECT_K), torch.int32),
    "selected_valid": ((CAPACITY, KPOOL_SELECT_K), torch.int32),
    "mixed_qkv": ((CAPACITY, 3 * LOCAL_KDA_QKV_DIM), torch.bfloat16),
    "decay": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.float32),
    "beta": ((CAPACITY, BETA_PAD), torch.float32),
    "kda_q": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    "kda_k": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    "kda_v": ((CAPACITY, LOCAL_KDA_H, KDA_DIM), torch.bfloat16),
    "dense_x_int8": ((CAPACITY, D), torch.int8),
}
OUTPUTS = {"head_hidden": ((LOGIT_ROWS, D), torch.bfloat16)}

_KDA_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("w_q", "self_attn.", ("q_proj.weight",)),
    ("w_k", "self_attn.", ("k_proj.weight",)),
    ("w_v", "self_attn.", ("v_proj.weight",)),
    ("conv_q", "self_attn.", ("q_conv1d.weight",)),
    ("conv_k", "self_attn.", ("k_conv1d.weight",)),
    ("conv_v", "self_attn.", ("v_conv1d.weight",)),
    ("w_f_a", "self_attn.", ("forget_gate.f_a_proj.weight", "f_a_proj.weight")),
    ("w_f_b", "self_attn.", ("forget_gate.f_b_proj.weight", "f_b_proj.weight")),
    ("dt_bias", "self_attn.", ("forget_gate.dt_bias", "dt_bias")),
    ("a_log", "self_attn.", ("forget_gate.A_log", "A_log")),
    ("w_g_a", "self_attn.", ("g_a_proj.weight",)),
    ("w_g_b", "self_attn.", ("g_b_proj.weight",)),
    ("w_b", "self_attn.", ("b_proj.weight",)),
    ("o_norm", "self_attn.", ("o_norm.weight",)),
    ("w_o", "self_attn.", ("o_proj.weight",)),
)
_MLA_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("w_q_a", "self_attn.", ("q_a_proj.weight",)),
    ("q_a_norm", "self_attn.", ("q_a_layernorm.weight",)),
    ("w_q_b", "self_attn.", ("q_b_proj.weight",)),
    ("w_kv_a", "self_attn.", ("kv_a_proj_with_mqa.weight",)),
    ("kv_a_norm", "self_attn.", ("kv_a_layernorm.weight",)),
    ("kv_b", "self_attn.", ("kv_b_proj.weight",)),
    ("w_o", "self_attn.", ("o_proj.weight",)),
)
_INDEXER_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("wq_b", "self_attn.indexer.", ("wq_b.weight",)),
    ("wk", "self_attn.indexer.", ("wk.weight",)),
    ("k_norm_w", "self_attn.indexer.", ("k_norm.weight",)),
    ("k_norm_b", "self_attn.indexer.", ("k_norm.bias",)),
    ("w_weights", "self_attn.indexer.", ("weights_proj.weight",)),
    ("w_compress_gate", "self_attn.indexer.", ("index_kpool_compress_gate",)),
    ("ape", "self_attn.indexer.", ("index_kpool_compress_ape",)),
)
_MHC_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("attn_fn", "", ("attn_hc.fn", "hc_attn_fn")),
    ("attn_scale", "", ("attn_hc.scale", "hc_attn_scale")),
    ("attn_base", "", ("attn_hc.base", "hc_attn_base")),
    ("ffn_fn", "", ("ffn_hc.fn", "hc_ffn_fn")),
    ("ffn_scale", "", ("ffn_hc.scale", "hc_ffn_scale")),
    ("ffn_base", "", ("ffn_hc.base", "hc_ffn_base")),
)
_NORM_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("input_norm", "", ("input_layernorm.weight",)),
    ("post_norm", "", ("post_attention_layernorm.weight",)),
)
DEFAULT_DUMP_KEYS = (
    "x_hc_in", "attn_norm_in", "attn_out", "x_hc_mid",
    "ffn_norm_in", "ffn_out", "x_hc_out",
)
INTERNAL_DUMP_KEYS = (
    "attn_hc_pre", "attn_hc_post", "attn_hc_comb",
    "ffn_hc_pre", "ffn_hc_post", "ffn_hc_comb", "kda_q", "kda_k", "kda_v", "kda_g", "kda_beta", "kda_core",
    "kda_gate", "kda_o_normed", "mla_q_resid", "mla_k_latent", "mla_topk_indices",
    "router_topk_weights", "router_topk_indices", "moe_routed", "moe_shared",
)

MTP_SPEC_TOKENS = 3
PREFIX_CACHE_FAMILIES = (
    "mla_latent_cache", "index_raw_cache", "index_pool_cache", "index_pool_scale",
    "kda_conv_state", "kda_recurrent_state",
)
PREFIX_SNAPSHOT_LAYOUTS = {
    "mla_latent_cache": ((PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_MLA_ELEMENTS), torch.bfloat16),
    "index_raw_cache": ((PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_RAW_ELEMENTS), torch.float32),
    "index_pool_cache": ((PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_POOL_ELEMENTS), torch.int8),
    "index_pool_scale": ((PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_SCALE_ELEMENTS), torch.float32),
    "kda_conv_state": ((PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_CONV_ELEMENTS), torch.bfloat16),
    "kda_recurrent_state": (
        (PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_RECURRENT_ELEMENTS),
        torch.float32,
    ),
}
PREFIX_SNAPSHOT_BYTES = sum(
    (shape[1] * shape[2] * torch.empty((), dtype=dtype).element_size() for (shape, dtype) in PREFIX_SNAPSHOT_LAYOUTS.values()),
)
VERIFY_X_H_C = os.environ.get("GLM53_PREFILL_VERIFY_XHC", "") == "1"
TRACE_DISPATCH = os.environ.get("GLM53_PREFILL_TRACE_DISPATCH", "") == "1"
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|user|>\n' + message['content'] + '<|assistant|>\n' }}{% endfor %}"
)
DEFAULT_PROMPTS = (
    "What is 2 + 3? Reply with just the number.",
    "What is the capital of France? Reply with just the city name.",
    "Translate 'hello' into Chinese. Reply with only the translation.",
    "Continue the sequence with three numbers: 1, 2, 3,",
    "Name the largest planet in the solar system. One word.",
    "What is 7 times 8? Reply with just the number.",
    "Which continent is Egypt in? One word.",
    "Spell the word 'computer'. Reply with only the spelling.",
    "What color do you get mixing blue and yellow? One word.",
    "How many days are in a leap year? Reply with just the number.",
    "Name the ocean between Europe and America. One word.",
    "What is the square root of 81? Reply with just the number.",
    "In what language is 'bonjour' a greeting? One word.",
    "Count from one to three in Spanish. Only the numbers.",
    "What is the chemical symbol for gold? Reply with only the symbol.",
    "Finish the proverb: 'An apple a day...'",
)


@pl.jit(auto_scope=False)
def copy_request_snapshot(
    mla_latent_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    kda_conv_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    bank_mla_latent_cache: pl.InOut[
        pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_MLA_ELEMENTS], pl.BF16]
    ],
    bank_index_raw_cache: pl.InOut[
        pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_RAW_ELEMENTS], pl.FP32]
    ],
    bank_index_pool_cache: pl.InOut[
        pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_POOL_ELEMENTS], pl.INT8]
    ],
    bank_index_pool_scale: pl.InOut[
        pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_SCALE_ELEMENTS], pl.FP32]
    ],
    bank_kda_conv_state: pl.InOut[
        pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_CONV_ELEMENTS], pl.BF16]
    ],
    bank_kda_recurrent_state: pl.InOut[
        pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_RECURRENT_ELEMENTS], pl.FP32]
    ],
    block_table: pl.Tensor[[BATCH, TABLE_MAX], pl.INT32],
    snapshot_length: pl.Scalar[pl.INT32],
    request_slot: pl.Scalar[pl.INT32],
    bank_id: pl.Scalar[pl.INT32],
    restore: pl.Scalar[pl.INT32],
):
    """Copy all private request-cache bytes to or from one immutable bank."""
    bank_latent_flat = pl.reshape(bank_mla_latent_cache, [PREFIX_SNAPSHOT_ENTRIES * DSA_NUM_LAYERS, _RESET_MLA_ELEMENTS])
    latent_flat = pl.reshape(mla_latent_cache, [DSA_NUM_LAYERS, LATENT_ROWS * KV_LORA])
    for block in pl.spmd(DSA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_mla_latent_cache"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(
            lane * _PREFIX_COPY_TILE, (snapshot_length) * KV_LORA, _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE
        ):
            valid = pl.min(_PREFIX_COPY_TILE, (snapshot_length) * KV_LORA - offset)
            logical_page = offset // ((BLOCK_SIZE) * KV_LORA)
            physical_page = pl.cast(pl.read(block_table, [request_slot, logical_page]), pl.INDEX)
            source_offset = physical_page * (BLOCK_SIZE) * KV_LORA + offset % ((BLOCK_SIZE) * KV_LORA)
            bank_row = pl.cast(bank_id, pl.INDEX) * DSA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_latent_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                latent_flat[layer : layer + 1, source_offset : source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(latent_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_latent_flat[bank_row : bank_row + 1, offset : offset + _PREFIX_COPY_TILE] = data

    bank_raw_flat = pl.reshape(bank_index_raw_cache, [PREFIX_SNAPSHOT_ENTRIES * DSA_NUM_LAYERS, _RESET_RAW_ELEMENTS])
    raw_flat = pl.reshape(index_raw_cache, [DSA_NUM_LAYERS, LATENT_ROWS * INDEX_STATE_WIDTH])
    for block in pl.spmd(DSA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_index_raw_cache"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(
            lane * _PREFIX_COPY_TILE,
            (snapshot_length) * INDEX_STATE_WIDTH,
            _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE,
        ):
            valid = pl.min(_PREFIX_COPY_TILE, (snapshot_length) * INDEX_STATE_WIDTH - offset)
            logical_page = offset // ((BLOCK_SIZE) * INDEX_STATE_WIDTH)
            physical_page = pl.cast(pl.read(block_table, [request_slot, logical_page]), pl.INDEX)
            source_offset = physical_page * BLOCK_SIZE * INDEX_STATE_WIDTH + offset % (BLOCK_SIZE * INDEX_STATE_WIDTH)
            bank_row = pl.cast(bank_id, pl.INDEX) * DSA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_raw_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                raw_flat[layer : layer + 1, source_offset : source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(raw_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_raw_flat[bank_row : bank_row + 1, offset : offset + _PREFIX_COPY_TILE] = data

    bank_pool_flat = pl.reshape(bank_index_pool_cache, [PREFIX_SNAPSHOT_ENTRIES * DSA_NUM_LAYERS, _RESET_POOL_ELEMENTS])
    pool_flat = pl.reshape(index_pool_cache, [DSA_NUM_LAYERS, POOL_TABLE_ROWS * INDEX_DIM])
    for block in pl.spmd(DSA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_index_pool_cache"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(
            lane * _PREFIX_COPY_TILE,
            (snapshot_length // INDEX_KPOOL) * INDEX_DIM,
            _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE,
        ):
            valid = pl.min(_PREFIX_COPY_TILE, (snapshot_length // INDEX_KPOOL) * INDEX_DIM - offset)
            logical_page = offset // ((BLOCK_SIZE // INDEX_KPOOL) * INDEX_DIM)
            physical_page = pl.cast(pl.read(block_table, [request_slot, logical_page]), pl.INDEX)
            source_offset = physical_page * (BLOCK_SIZE // INDEX_KPOOL) * INDEX_DIM + offset % (
                (BLOCK_SIZE // INDEX_KPOOL) * INDEX_DIM
            )
            bank_row = pl.cast(bank_id, pl.INDEX) * DSA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_pool_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                pool_flat[layer : layer + 1, source_offset : source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(pool_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_pool_flat[bank_row : bank_row + 1, offset : offset + _PREFIX_COPY_TILE] = data

    bank_scale_flat = pl.reshape(bank_index_pool_scale, [PREFIX_SNAPSHOT_ENTRIES * DSA_NUM_LAYERS, _RESET_SCALE_ELEMENTS])
    scale_flat = pl.reshape(index_pool_scale, [DSA_NUM_LAYERS, POOL_TABLE_ROWS * POOL_SCALE_WIDTH])
    for block in pl.spmd(DSA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_index_pool_scale"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(
            lane * _PREFIX_SCALE_TILE,
            (snapshot_length // INDEX_KPOOL) * POOL_SCALE_WIDTH,
            _PREFIX_COPY_LANE_TILE * _PREFIX_SCALE_TILE,
        ):
            valid = pl.min(_PREFIX_SCALE_TILE, (snapshot_length // INDEX_KPOOL) * POOL_SCALE_WIDTH - offset)
            logical_page = offset // ((BLOCK_SIZE // INDEX_KPOOL) * POOL_SCALE_WIDTH)
            physical_page = pl.cast(pl.read(block_table, [request_slot, logical_page]), pl.INDEX)
            source_offset = physical_page * (BLOCK_SIZE // INDEX_KPOOL) * POOL_SCALE_WIDTH + offset % (
                (BLOCK_SIZE // INDEX_KPOOL) * POOL_SCALE_WIDTH
            )
            bank_row = pl.cast(bank_id, pl.INDEX) * DSA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_scale_flat, [1, _PREFIX_SCALE_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                scale_flat[layer : layer + 1, source_offset : source_offset + _PREFIX_SCALE_TILE] = data
            else:
                data = pl.slice(scale_flat, [1, _PREFIX_SCALE_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_scale_flat[bank_row : bank_row + 1, offset : offset + _PREFIX_SCALE_TILE] = data

    bank_conv_flat = pl.reshape(bank_kda_conv_state, [PREFIX_SNAPSHOT_ENTRIES * KDA_NUM_LAYERS, _RESET_CONV_ELEMENTS])
    conv_flat = pl.reshape(kda_conv_state, [KDA_NUM_LAYERS, KDA_ROWS * STATE_LEN * CONV_DIM])
    for block in pl.spmd(KDA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_kda_conv_state"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(
            lane * _PREFIX_COPY_TILE, _RESET_CONV_ELEMENTS, _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE
        ):
            valid = pl.min(_PREFIX_COPY_TILE, _RESET_CONV_ELEMENTS - offset)
            source_offset = pl.cast(request_slot, pl.INDEX) * _RESET_CONV_ELEMENTS + offset
            bank_row = pl.cast(bank_id, pl.INDEX) * KDA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_conv_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                conv_flat[layer : layer + 1, source_offset : source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(conv_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_conv_flat[bank_row : bank_row + 1, offset : offset + _PREFIX_COPY_TILE] = data

    bank_recurrent_flat = pl.reshape(bank_kda_recurrent_state, [PREFIX_SNAPSHOT_ENTRIES * KDA_NUM_LAYERS, _RESET_RECURRENT_ELEMENTS])
    recurrent_flat = pl.reshape(kda_recurrent_state, [KDA_NUM_LAYERS, KDA_ROWS * LOCAL_KDA_H * KDA_DIM * KDA_DIM])
    for block in pl.spmd(KDA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_kda_recurrent_state"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(
            lane * _PREFIX_COPY_TILE, _RESET_RECURRENT_ELEMENTS, _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE
        ):
            valid = pl.min(_PREFIX_COPY_TILE, _RESET_RECURRENT_ELEMENTS - offset)
            source_offset = pl.cast(request_slot, pl.INDEX) * _RESET_RECURRENT_ELEMENTS + offset
            bank_row = pl.cast(bank_id, pl.INDEX) * KDA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_recurrent_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                recurrent_flat[layer : layer + 1, source_offset : source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(recurrent_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_recurrent_flat[bank_row : bank_row + 1, offset : offset + _PREFIX_COPY_TILE] = data
    return (
        mla_latent_cache,
        index_raw_cache,
        index_pool_cache,
        index_pool_scale,
        kda_conv_state,
        kda_recurrent_state,
        bank_mla_latent_cache,
        bank_index_raw_cache,
        bank_index_pool_cache,
        bank_index_pool_scale,
        bank_kda_conv_state,
        bank_kda_recurrent_state,
    )


@pl.jit.host
def l3_copy_request_snapshot(
    mla_latent_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[
        pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]
    ],
    kda_conv_state: pl.InOut[pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    bank_mla_latent_cache: pl.InOut[
        pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_MLA_ELEMENTS], pl.BF16]
    ],
    bank_index_raw_cache: pl.InOut[
        pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_RAW_ELEMENTS], pl.FP32]
    ],
    bank_index_pool_cache: pl.InOut[
        pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_POOL_ELEMENTS], pl.INT8]
    ],
    bank_index_pool_scale: pl.InOut[
        pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_SCALE_ELEMENTS], pl.FP32]
    ],
    bank_kda_conv_state: pl.InOut[
        pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_CONV_ELEMENTS], pl.BF16]
    ],
    bank_kda_recurrent_state: pl.InOut[
        pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_RECURRENT_ELEMENTS], pl.FP32]
    ],
    block_table: pl.Tensor[[RANKS, BATCH, TABLE_MAX], pl.INT32],
    snapshot_length: pl.Scalar[pl.INT32],
    request_slot: pl.Scalar[pl.INT32],
    bank_id: pl.Scalar[pl.INT32],
    restore: pl.Scalar[pl.INT32],
):
    """Copy the same request state on every resident TP/EP rank."""
    for rank in pl.unroll(RANKS):
        copy_request_snapshot(
            mla_latent_cache[rank],
            index_raw_cache[rank], index_pool_cache[rank], index_pool_scale[rank],
            kda_conv_state[rank], kda_recurrent_state[rank], bank_mla_latent_cache[rank],
            bank_index_raw_cache[rank], bank_index_pool_cache[rank], bank_index_pool_scale[rank],
            bank_kda_conv_state[rank], bank_kda_recurrent_state[rank], block_table[rank], snapshot_length,
            request_slot, bank_id, restore, device=rank,
        )


@pl.jit(auto_scope=False)
def reset_request_caches(
    mla_latent_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    kda_conv_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    cache_page_owners: pl.Tensor[[LATENT_BLOCKS], pl.INT32],
    slot_mask: pl.Scalar[pl.INT32],
):
    """Zero selected private segments, or all physical cache rows for a full reset."""
    mla_flat = pl.reshape(mla_latent_cache, [DSA_NUM_LAYERS, LATENT_ROWS * KV_LORA])
    for block in pl.spmd(DSA_NUM_LAYERS * 4, name_hint="reset_mla"):
        layer = block // 4
        lane = block % 4
        mla_zero = pl.tile.full([1, _RESET_MLA_TILE], dtype=pl.BF16, value=0)
        for page in pl.range(LATENT_BLOCKS):
            page_selected = pl.cast(0, pl.INT32)
            if slot_mask == (1 << BATCH) - 1:
                page_selected = pl.cast(1, pl.INT32)
            elif page < LATENT_BLOCKS:
                owner = pl.read(cache_page_owners, [page])
                if owner >= 0:
                    page_selected = pl.cast((slot_mask >> owner) & 1, pl.INT32)
            if page_selected != 0:
                for chunk in pl.range(lane, (BLOCK_SIZE) * KV_LORA // _RESET_MLA_TILE, 4):
                    offset = page * (BLOCK_SIZE) * KV_LORA + chunk * _RESET_MLA_TILE
                    pl.tile.store(mla_zero, [layer, offset], mla_flat)
    raw_flat = pl.reshape(index_raw_cache, [DSA_NUM_LAYERS, LATENT_ROWS * INDEX_STATE_WIDTH])
    for block in pl.spmd(DSA_NUM_LAYERS * 4, name_hint="reset_raw"):
        layer = block // 4
        lane = block % 4
        raw_zero = pl.tile.full([1, _RESET_RAW_TILE], dtype=pl.FP32, value=0)
        for page in pl.range(LATENT_BLOCKS):
            page_selected = pl.cast(0, pl.INT32)
            if slot_mask == (1 << BATCH) - 1:
                page_selected = pl.cast(1, pl.INT32)
            elif page < LATENT_BLOCKS:
                owner = pl.read(cache_page_owners, [page])
                if owner >= 0:
                    page_selected = pl.cast((slot_mask >> owner) & 1, pl.INT32)
            if page_selected != 0:
                for chunk in pl.range(lane, (BLOCK_SIZE) * INDEX_STATE_WIDTH // _RESET_RAW_TILE, 4):
                    offset = page * (BLOCK_SIZE) * INDEX_STATE_WIDTH + chunk * _RESET_RAW_TILE
                    pl.tile.store(raw_zero, [layer, offset], raw_flat)
    pool_flat = pl.reshape(index_pool_cache, [DSA_NUM_LAYERS, POOL_TABLE_ROWS * INDEX_DIM])
    for block in pl.spmd(DSA_NUM_LAYERS * 4, name_hint="reset_pool"):
        layer = block // 4
        lane = block % 4
        zero_i16 = pl.tile.full([1, _RESET_POOL_TILE], dtype=pl.INT16, value=0)
        pool_zero = pl.cast(zero_i16, pl.INT8)
        for page in pl.range(LATENT_BLOCKS + 1):
            page_selected = pl.cast(0, pl.INT32)
            if slot_mask == (1 << BATCH) - 1:
                page_selected = pl.cast(1, pl.INT32)
            elif page < LATENT_BLOCKS:
                owner = pl.read(cache_page_owners, [page])
                if owner >= 0:
                    page_selected = pl.cast((slot_mask >> owner) & 1, pl.INT32)
            if page_selected != 0:
                for chunk in pl.range(lane, (BLOCK_SIZE // INDEX_KPOOL) * INDEX_DIM // _RESET_POOL_TILE, 4):
                    offset = page * (BLOCK_SIZE // INDEX_KPOOL) * INDEX_DIM + chunk * _RESET_POOL_TILE
                    pl.tile.store(pool_zero, [layer, offset], pool_flat)
    scale_flat = pl.reshape(index_pool_scale, [DSA_NUM_LAYERS, POOL_TABLE_ROWS * POOL_SCALE_WIDTH])
    for block in pl.spmd(DSA_NUM_LAYERS * 4, name_hint="reset_scale"):
        layer = block // 4
        lane = block % 4
        scale_zero = pl.tile.full([1, _RESET_SCALE_TILE], dtype=pl.FP32, value=0)
        for page in pl.range(LATENT_BLOCKS + 1):
            page_selected = pl.cast(0, pl.INT32)
            if slot_mask == (1 << BATCH) - 1:
                page_selected = pl.cast(1, pl.INT32)
            elif page < LATENT_BLOCKS:
                owner = pl.read(cache_page_owners, [page])
                if owner >= 0:
                    page_selected = pl.cast((slot_mask >> owner) & 1, pl.INT32)
            if page_selected != 0:
                for chunk in pl.range(
                    lane, (BLOCK_SIZE // INDEX_KPOOL) * POOL_SCALE_WIDTH // _RESET_SCALE_TILE, 4
                ):
                    offset = page * (BLOCK_SIZE // INDEX_KPOOL) * POOL_SCALE_WIDTH + chunk * _RESET_SCALE_TILE
                    pl.tile.store(scale_zero, [layer, offset], scale_flat)
    conv_layer_tiles = KDA_ROWS * STATE_LEN * CONV_DIM // _RESET_CONV_TILE
    conv_slot_tiles = _RESET_CONV_ELEMENTS // _RESET_CONV_TILE
    conv_flat_rows = KDA_NUM_LAYERS * conv_layer_tiles
    conv_flat = pl.reshape(kda_conv_state, [conv_flat_rows, _RESET_CONV_TILE])
    for block in pl.spmd(KDA_NUM_LAYERS * _RESET_CONV_LANE_TILE, name_hint="reset_conv"):
        layer = block // _RESET_CONV_LANE_TILE
        lane = block % _RESET_CONV_LANE_TILE
        conv_zero = pl.tile.full([1, _RESET_CONV_TILE], dtype=pl.BF16, value=0)
        if slot_mask == (1 << BATCH) - 1:
            for chunk in pl.range(lane, conv_layer_tiles, _RESET_CONV_LANE_TILE):
                offset = layer * conv_layer_tiles + chunk
                pl.tile.store(conv_zero, [offset, 0], conv_flat)
        else:
            for slot in pl.range(BATCH):
                selected = (slot_mask >> pl.cast(slot, pl.INT32)) & 1
                if selected != 0:
                    for chunk in pl.range(lane, conv_slot_tiles, _RESET_CONV_LANE_TILE):
                        offset = layer * conv_layer_tiles + slot * conv_slot_tiles + chunk
                        pl.tile.store(conv_zero, [offset, 0], conv_flat)
    recurrent_layer_tiles = KDA_ROWS * LOCAL_KDA_H * KDA_DIM * KDA_DIM // _RESET_RECURRENT_TILE
    recurrent_slot_tiles = _RESET_RECURRENT_ELEMENTS // _RESET_RECURRENT_TILE
    recurrent_flat_rows = KDA_NUM_LAYERS * recurrent_layer_tiles
    recurrent_flat = pl.reshape(kda_recurrent_state, [recurrent_flat_rows, _RESET_RECURRENT_TILE])
    for block in pl.spmd(KDA_NUM_LAYERS * _RESET_RECURRENT_LANE_TILE, name_hint="reset_recurrent"):
        layer = block // _RESET_RECURRENT_LANE_TILE
        lane = block % _RESET_RECURRENT_LANE_TILE
        recurrent_zero = pl.tile.full([1, _RESET_RECURRENT_TILE], dtype=pl.FP32, value=0)
        if slot_mask == (1 << BATCH) - 1:
            for chunk in pl.range(lane, recurrent_layer_tiles, _RESET_RECURRENT_LANE_TILE):
                offset = layer * recurrent_layer_tiles + chunk
                pl.tile.store(recurrent_zero, [offset, 0], recurrent_flat)
        else:
            for slot in pl.range(BATCH):
                selected = (slot_mask >> pl.cast(slot, pl.INT32)) & 1
                if selected != 0:
                    for chunk in pl.range(lane, recurrent_slot_tiles, _RESET_RECURRENT_LANE_TILE):
                        offset = layer * recurrent_layer_tiles + slot * recurrent_slot_tiles + chunk
                        pl.tile.store(recurrent_zero, [offset, 0], recurrent_flat)
    return (mla_latent_cache, index_raw_cache, index_pool_cache, index_pool_scale, kda_conv_state, kda_recurrent_state)


@pl.jit.host
def l3_reset_request_caches(
    mla_latent_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[
        pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]
    ],
    kda_conv_state: pl.InOut[pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    cache_page_owners: pl.Tensor[[RANKS, LATENT_BLOCKS], pl.INT32],
    slot_mask: pl.Scalar[pl.INT32],
):
    """Clear the same selected request slots on every resident rank."""
    for rank in pl.unroll(RANKS):
        reset_request_caches(
            mla_latent_cache[rank],
            index_raw_cache[rank], index_pool_cache[rank], index_pool_scale[rank],
            kda_conv_state[rank], kda_recurrent_state[rank],
            cache_page_owners[rank], slot_mask, device=rank,
        )


def zeroed_caches() -> dict[str, torch.Tensor]:
    """Build bounded host fixtures; production clears its device allocations directly."""
    return {name: torch.zeros(shape, dtype=dtype) for name, (shape, dtype) in CACHE_LAYOUTS.items()}


@dataclass(frozen=True)
class _StackMember:
    """One kernel-side stack member: checkpoint spellings, view dtype and shard."""

    member: str
    suffixes: tuple[str, ...]
    dtype: torch.dtype
    transform: str = "identity"  # identity | transpose | split_k | split_v
    shard: str = "replicate"  # replicate | rows | columns


_KDA_MEMBERS = (
    _StackMember("w_q", ("q_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_k", ("k_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_v", ("v_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("conv_q", ("q_conv1d.weight",), torch.bfloat16, shard="rows"),
    _StackMember("conv_k", ("k_conv1d.weight",), torch.bfloat16, shard="rows"),
    _StackMember("conv_v", ("v_conv1d.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_f_a", ("forget_gate.f_a_proj.weight", "f_a_proj.weight"), torch.bfloat16),
    _StackMember("w_f_b", ("forget_gate.f_b_proj.weight", "f_b_proj.weight"), torch.bfloat16, shard="rows"),
    _StackMember("w_g_a", ("g_a_proj.weight",), torch.bfloat16),
    _StackMember("w_g_b", ("g_b_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_b", ("b_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("dt_bias", ("forget_gate.dt_bias", "dt_bias"), torch.float32, shard="rows"),
    _StackMember("a_log", ("forget_gate.A_log", "A_log"), torch.float32, shard="rows"),
    _StackMember("o_norm", ("o_norm.weight",), torch.bfloat16),
    _StackMember("w_o", ("o_proj.weight",), torch.bfloat16, shard="columns"),
)
_MLA_MEMBERS = (
    _StackMember("w_q_a", ("q_a_proj.weight",), torch.bfloat16),
    _StackMember("q_a_norm", ("q_a_layernorm.weight",), torch.bfloat16),
    _StackMember("w_q_b", ("q_b_proj.weight",), torch.bfloat16, shard="rows"),
    _StackMember("w_kv_a", ("kv_a_proj_with_mqa.weight",), torch.bfloat16),
    _StackMember("kv_a_norm", ("kv_a_layernorm.weight",), torch.bfloat16),
    _StackMember("w_k", ("kv_b_proj.weight",), torch.bfloat16, transform="split_k"),
    _StackMember("w_v", ("kv_b_proj.weight",), torch.bfloat16, transform="split_v"),
    _StackMember("w_o", ("o_proj.weight",), torch.bfloat16, shard="columns"),
)
_INDEXER_MEMBERS = (
    _StackMember("w_q_b", ("wq_b.weight",), torch.bfloat16, transform="transpose"),
    _StackMember("w_k", ("wk.weight",), torch.bfloat16, transform="transpose"),
    _StackMember("k_norm_w", ("k_norm.weight",), torch.bfloat16),
    _StackMember("k_norm_b", ("k_norm.bias",), torch.bfloat16),
    _StackMember("w_weights", ("weights_proj.weight",), torch.bfloat16, transform="transpose"),
    _StackMember("w_compress_gate", ("index_kpool_compress_gate",), torch.bfloat16, transform="transpose"),
    _StackMember("compress_ape", ("index_kpool_compress_ape",), torch.bfloat16),
)
_MHC_MEMBERS = (
    _StackMember("attn_fn", ("attn_hc.fn", "hc_attn_fn"), torch.bfloat16),
    _StackMember("attn_scale", ("attn_hc.scale", "hc_attn_scale"), torch.float32),
    _StackMember("attn_base", ("attn_hc.base", "hc_attn_base"), torch.float32),
    _StackMember("ffn_fn", ("ffn_hc.fn", "hc_ffn_fn"), torch.bfloat16),
    _StackMember("ffn_scale", ("ffn_hc.scale", "hc_ffn_scale"), torch.float32),
    _StackMember("ffn_base", ("ffn_hc.base", "hc_ffn_base"), torch.float32),
)
_NORM_MEMBERS = (
    _StackMember("input_norm", ("input_layernorm.weight",), torch.bfloat16),
    _StackMember("post_norm", ("post_attention_layernorm.weight",), torch.bfloat16),
)
_ROUTER_MEMBERS = (
    _StackMember("gate_w", ("gate.weight",), torch.bfloat16),
    _StackMember("gate_bias", ("gate.e_score_correction_bias", "e_score_correction_bias"), torch.float32),
)
_FAMILY_MEMBERS = {
    "kda": _KDA_MEMBERS,
    "mla": _MLA_MEMBERS,
    "indexer": _INDEXER_MEMBERS,
    "mhc": _MHC_MEMBERS,
    "norm": _NORM_MEMBERS,
    "router": _ROUTER_MEMBERS,
}
_TEXT_MODEL_ROOTS = ("model.language_model.", "model.")


def _source_shape(model: Glm53FlashConfig, family: str, member: str) -> tuple[int, ...]:
    """The checkpoint-side shape a member is read from (before any transform)."""
    hidden = model.hidden_size
    qkv = model.linear_num_heads * model.linear_head_dim
    if family == "kda":
        shapes = {
            "w_q": (qkv, hidden),
            "w_k": (qkv, hidden),
            "w_v": (qkv, hidden),
            "conv_q": (qkv, 1, model.linear_conv_kernel_dim),
            "conv_k": (qkv, 1, model.linear_conv_kernel_dim),
            "conv_v": (qkv, 1, model.linear_conv_kernel_dim),
            "w_f_a": (model.linear_head_dim, hidden),
            "w_f_b": (qkv, model.linear_head_dim),
            "w_g_a": (model.linear_head_dim, hidden),
            "w_g_b": (qkv, model.linear_head_dim),
            "w_b": (model.linear_num_heads, hidden),
            "dt_bias": (qkv,),
            "a_log": (model.linear_num_heads,),
            "o_norm": (model.linear_head_dim,),
            "w_o": (hidden, qkv),
        }
    elif family == "mla":
        shapes = {
            "w_q_a": (model.q_lora_rank, hidden),
            "q_a_norm": (model.q_lora_rank,),
            "w_q_b": (model.num_attention_heads * model.qk_head_dim, model.q_lora_rank),
            "w_kv_a": (model.kv_lora_rank, hidden),
            "kv_a_norm": (model.kv_lora_rank,),
            "w_k": (model.num_attention_heads * (model.qk_head_dim + model.v_head_dim), model.kv_lora_rank),
            "w_v": (model.num_attention_heads * (model.qk_head_dim + model.v_head_dim), model.kv_lora_rank),
            "w_o": (hidden, model.num_attention_heads * model.v_head_dim),
        }
    elif family == "indexer":
        shapes = {
            "w_q_b": (model.index_n_heads * model.index_head_dim, model.q_lora_rank),
            "w_k": (model.index_head_dim, hidden),
            "k_norm_w": (model.index_head_dim,),
            "k_norm_b": (model.index_head_dim,),
            "w_weights": (model.index_n_heads, hidden),
            "w_compress_gate": (model.index_head_dim, hidden),
            "compress_ape": (model.index_kpool, model.index_head_dim),
        }
    elif family == "mhc":
        shapes = {
            "attn_fn": (model.mix_hc, model.hc_dim),
            "attn_scale": (3,),
            "attn_base": (model.mix_hc,),
            "ffn_fn": (model.mix_hc, model.hc_dim),
            "ffn_scale": (3,),
            "ffn_base": (model.mix_hc,),
        }
    elif family == "norm":
        shapes = {"input_norm": (hidden,), "post_norm": (hidden,)}
    else:
        shapes = {"gate_w": (model.n_routed_experts, hidden), "gate_bias": (model.n_routed_experts,)}
    return shapes[member]


@dataclass(frozen=True)
class StackPlan:
    """One device-resident stacked tensor as the runner must allocate it."""

    name: str  # e.g. "kda.w_q"
    family: str  # kda | mla | indexer | mhc | norm | router | moe | dense | mtp | single
    layers: tuple[int, ...]  # source layer ids on the stack axis, ascending
    shape: tuple[int, ...]  # per-rank stacked shape (leading axis = family ordinal)
    dtype: torch.dtype

    @property
    def bytes(self) -> int:
        count = math.prod(self.shape)
        return count * torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class RankPlan:
    """Pure-metadata view of one rank's resident weights (no payload reads)."""

    rank: int
    tp_rank: int
    ep_rank: int
    local_expert_ids: tuple[int, ...]
    stacks: tuple[StackPlan, ...]
    device_bytes: int
    host_tables: tuple[str, ...]  # ("embed_tokens", "lm_head"), host-side aliases included


def _fused_ffn_at(
    tensors: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
    roles: tuple[str, str, str, str, str],
    *,
    expert: int | None,
    intermediate: int,
    hidden: int,
    channels: slice = slice(None),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``_fused_ffn`` carried over an explicit geometry: hidden and intermediate
    come from the staging model instead of the module constants, so scaled
    fixtures load through the same read/fuse pipeline as the deployment."""
    fmt = {"e": expert} if expert is not None else {}
    fused_role, fused_down_role, gate_role, up_role, down_role = roles
    fused_name = _try_find(tensors, _prefixed(prefixes, ROLES[fused_role], **fmt))
    down_name = _find(tensors, _prefixed(prefixes, ROLES[fused_down_role] + ROLES[down_role], **fmt))
    if fused_name is not None:
        if _weight_shape(tensors, fused_name) != (2 * intermediate, hidden):
            raise ValueError(f"{fused_name} must have shape {(2 * intermediate, hidden)}")
        if channels == slice(None):
            gate_up, gate_up_scale = _read_quant(tensors, fused_name)
        else:
            gate, gate_scale = _read_quant(tensors, fused_name, channels)
            up_rows = slice(intermediate + channels.start, intermediate + channels.stop)
            up, up_scale = _read_quant(tensors, fused_name, up_rows)
            gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    else:
        gate_name = _find(tensors, _prefixed(prefixes, ROLES[gate_role], **fmt))
        up_name = _find(tensors, _prefixed(prefixes, ROLES[up_role], **fmt))
        if _weight_shape(tensors, gate_name) != (intermediate, hidden) or _weight_shape(tensors, up_name) != (
            intermediate,
            hidden,
        ):
            raise ValueError(f"{gate_name} and {up_name} must have shape {(intermediate, hidden)}")
        gate, gate_scale = _read_quant(tensors, gate_name, channels)
        up, up_scale = _read_quant(tensors, up_name, channels)
        gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    if _weight_shape(tensors, down_name) != (hidden, intermediate):
        raise ValueError(f"{down_name} must have shape {(hidden, intermediate)}")
    down, down_scale = _read_quant(tensors, down_name, cols=channels)
    return gate_up, gate_up_scale, down, down_scale


def _ffn_roles(kind: str) -> tuple[str, str, str, str, str]:
    return {
        "routed": ("routed_fused", "routed_fused_down", "routed_gate", "routed_up", "routed_down"),
        "shared": ("shared_fused", "shared_fused_down", "shared_gate", "shared_up", "shared_down"),
        "dense": ("dense_fused", "dense_fused_down", "dense_gate", "dense_up", "dense_down"),
    }[kind]


class ResidentPrefillWeights:
    """Plan and stage the resident prefill weights for the TP/EP ranks.

    Construction goes through :meth:`load`, which reads headers only: it
    resolves every stack member's source name (both layer-root spellings, fused
    or separate gate/up, nested or flat mHC/KDA-gate spellings), validates the
    header shapes against the geometry, and builds the full stack plan. Payloads
    move only in :meth:`upload`, one staged layer-slice at a time, so host
    staging never exceeds a single layer of one family.

    Rank mapping: ranks are EP ranks; the TP rank is ``rank % tp_size``. In the
    deployment topology (TP16/EP16/DP1) the two coincide with the rank id, and
    TP1/EP16 collapses to tp_rank 0 for every rank.
    """

    def __init__(
        self,
        checkpoint,
        *,
        tp_size: int,
        ep_size: int,
        model: Glm53FlashConfig | None = None,
        mla_mode: str = "bf16",
    ):
        model = FLASH if model is None else model
        if isinstance(checkpoint, Checkpoint) and model is not FLASH:
            raise ValueError("Checkpoint validates the deployment geometry; pass model only for fixtures")
        if tp_size not in SUPPORTED_TP_SIZES:
            raise ValueError(f"tp_size must be one of {SUPPORTED_TP_SIZES}, got {tp_size}")
        if ep_size not in SUPPORTED_EP_SIZES:
            raise ValueError(f"ep_size must be one of {SUPPORTED_EP_SIZES}, got {ep_size}")
        if ep_size % tp_size:
            raise ValueError(f"EP{ep_size} must be divisible by TP{tp_size}")
        if model.num_attention_heads % tp_size:
            raise ValueError(f"{model.num_attention_heads} MLA heads do not divide TP{tp_size}")
        if model.linear_num_heads % tp_size:
            raise ValueError(f"{model.linear_num_heads} KDA heads do not divide TP{tp_size}")
        if model.intermediate_size % tp_size:
            raise ValueError(f"{model.intermediate_size} dense channels do not divide TP{tp_size}")
        if model.n_routed_experts % ep_size:
            raise ValueError(f"{model.n_routed_experts} routed experts do not divide EP{ep_size}")
        if mla_mode not in ("bf16", "native"):
            raise ValueError("mla_mode must be bf16 or native")
        self.mla_mode = mla_mode
        self._family_members = dict(_FAMILY_MEMBERS)
        if mla_mode == "native":
            native = []
            for spec in _MLA_MEMBERS:
                if spec.member in ("w_q_a", "w_q_b", "w_kv_a", "w_o"):
                    native.append(dataclasses.replace(spec, dtype=torch.int8))
                    native.append(dataclasses.replace(spec, member=spec.member + "_scale", dtype=torch.float32))
                else:
                    native.append(spec)
            self._family_members["mla"] = tuple(native)
        self.checkpoint = checkpoint
        self.model = model
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.ranks = ep_size
        self.local_h = model.num_attention_heads // tp_size
        self.local_kda_h = model.linear_num_heads // tp_size
        self.local_kda_qkv = model.linear_num_heads * model.linear_head_dim // tp_size
        self.n_local_experts = model.n_routed_experts // ep_size
        self.local_dense_inter = model.intermediate_size // tp_size
        mtp = model.mtp_layer_id
        self._family_layers = {
            "kda": tuple(model.kda_layer_ids),
            "mla": tuple(model.indexer_layer_ids),
            "indexer": tuple(model.indexer_layer_ids),
            "mhc": tuple(range(model.num_hidden_layers)),
            "norm": tuple(range(model.num_checkpoint_layers)),
            "router": tuple(range(model.first_k_dense_replace, model.num_checkpoint_layers)),
            "moe": tuple(range(model.first_k_dense_replace, model.num_checkpoint_layers)),
            "dense": tuple(range(model.first_k_dense_replace)),
        }
        self._mtp_layer = mtp
        self._lm_head_cache: torch.Tensor | None = None
        self._source: dict[tuple[str, int], str] = {}
        self.stacks: dict[str, StackPlan] = {}

    # -- construction ----------------------------------------------------------------

    @classmethod
    def load(
        cls,
        checkpoint,
        *,
        tp_size: int = 16,
        ep_size: int = 16,
        model: Glm53FlashConfig | None = None,
        mla_mode: str = "bf16",
    ) -> "ResidentPrefillWeights":
        """Build the full stack plan from checkpoint headers, reading no payloads."""
        self = cls(checkpoint, tp_size=tp_size, ep_size=ep_size, model=model, mla_mode=mla_mode)
        self._resolve_sources()
        self._build_stacks()
        return self

    def _resolve(self, candidates: Sequence[str], what: str) -> str:
        for name in candidates:
            if name in self.checkpoint:
                return name
        raise ValueError(f"checkpoint is missing {what} (tried {list(candidates)})")

    def _label(self, name: str) -> str:
        description = getattr(self.checkpoint, "description", None)
        label = None if description is None else description.get(name)
        if label not in ("FLOAT", "W8A8_DYNAMIC"):
            raise ValueError(f"{name} has unsupported quantization label {label!r}")
        return label

    def _resolve_sources(self) -> None:
        """Resolve and header-validate every stack member's source tensor."""
        model = self.model
        for family, members in self._family_members.items():
            prefix = _FAMILY_PREFIX[family]
            for layer in self._family_layers[family]:
                for spec in members:
                    candidates = [
                        root + f"{layer}." + prefix + suffix
                        for root in TEXT_LAYER_ROOTS
                        for suffix in spec.suffixes
                    ]
                    name = self._resolve(candidates, f"{family}.{spec.member} on layer {layer}")
                    self._source[(f"{family}.{spec.member}", layer)] = name
                    member = spec.member.removesuffix("_scale") if family == "mla" else spec.member
                    native = (
                        family == "mla"
                        and self.mla_mode == "native"
                        and member in ("w_q_a", "w_q_b", "w_kv_a", "w_o")
                    )
                    if native and self._label(name) != "W8A8_DYNAMIC":
                        raise ValueError(f"native MLA projection must be W8A8_DYNAMIC: {name}")
                    dtype = torch.bfloat16 if native else spec.dtype
                    self._validate_member_header(name, _source_shape(model, family, member), dtype)
        hidden = model.hidden_size
        for member, suffix, shape in (
            ("eh_proj", "eh_proj.weight", (hidden, 2 * hidden)),
            ("enorm", "enorm.weight", (hidden,)),
            ("hnorm", "hnorm.weight", (hidden,)),
            ("final_norm", "shared_head.norm.weight", (hidden,)),
        ):
            candidates = [root + f"{self._mtp_layer}." + suffix for root in TEXT_LAYER_ROOTS]
            name = self._resolve(candidates, f"mtp.{member}")
            self._source[(f"mtp.{member}", self._mtp_layer)] = name
            self._validate_member_header(name, shape, torch.bfloat16)
        self._source["norm.final_norm", -1] = self._resolve(
            [root + "norm.weight" for root in _TEXT_MODEL_ROOTS], "norm.final_norm",
        )
        self._validate_member_header(self._source[("norm.final_norm", -1)], (hidden,), torch.bfloat16)
        self._embed_name = self._resolve([root + "embed_tokens.weight" for root in _TEXT_MODEL_ROOTS], "embed_tokens")
        self._lm_head_name = self._resolve(("lm_head.weight",), "lm_head")

    def _validate_member_header(self, name: str, expected_shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Header-only shape and label gate; skipped for header-less fixture views.

        ``Checkpoint`` validates dtype at read time (``_float`` accepts only
        BF16/FP32 storage, ``quant`` only INT8); here the shape is checked from
        the shard header so a layout mismatch fails before any payload moves.
        """
        shape_of = getattr(self.checkpoint, "shape", None)
        if shape_of is None:
            return
        shape = shape_of(name)
        if tuple(shape) != expected_shape:
            raise ValueError(f"{name} has shape {tuple(shape)}, expected {expected_shape}")
        label = self._label(name)
        if label == "W8A8_DYNAMIC" and dtype != torch.bfloat16:
            # Only the MLA projections are INT8 in storage; every other member
            # must be FLOAT, and dequantizing one would silently change its ABI.
            raise ValueError(f"{name} is {label}; only BF16 members may dequantize, view is {dtype}")

    def _build_stacks(self) -> None:
        """Compute every stack's per-rank shape from the geometry alone."""
        model = self.model
        hidden = model.hidden_size
        stacks: dict[str, StackPlan] = {}
        for family, members in self._family_members.items():
            layers = self._family_layers[family]
            for spec in members:
                name = f"{family}.{spec.member}"
                shape = (len(layers), *self._member_shape(family, spec))
                stacks[name] = StackPlan(name, family, layers, shape, spec.dtype)
        moe_layers = self._family_layers["moe"]
        inter, local_experts = model.moe_intermediate_size, self.n_local_experts
        for member, shape, dtype in (
            ("w_gate_up", (local_experts, 2 * inter, hidden), torch.int8),
            ("w_gate_up_scale", (local_experts, 2 * inter), torch.float32),
            ("w_down", (local_experts, hidden, inter), torch.int8),
            ("w_down_scale", (local_experts, hidden), torch.float32),
            ("shared_w_gate_up", (2 * inter, hidden), torch.int8),
            ("shared_w_gate_up_scale", (2 * inter,), torch.float32),
            ("shared_w_down", (hidden, inter), torch.int8),
            ("shared_w_down_scale", (hidden,), torch.float32),
        ):
            name = f"moe.{member}"
            stacks[name] = StackPlan(name, "moe", moe_layers, (len(moe_layers), *shape), dtype)
        dense_layers = self._family_layers["dense"]
        local = self.local_dense_inter
        for member, shape, dtype in (
            ("w_gate_up", (2 * local, hidden), torch.int8),
            ("w_gate_up_scale", (2 * local,), torch.float32),
            ("w_down", (hidden, local), torch.int8),
            ("w_down_scale", (hidden,), torch.float32),
        ):
            name = f"dense.{member}"
            stacks[name] = StackPlan(name, "dense", dense_layers, (len(dense_layers), *shape), dtype)
        for name, family, shape in (
            ("norm.final_norm", "single", (hidden,)),
            ("mtp.eh_proj", "mtp", (hidden, 2 * hidden)),
            ("mtp.enorm", "mtp", (hidden,)),
            ("mtp.hnorm", "mtp", (hidden,)),
            ("mtp.final_norm", "mtp", (hidden,)),
            ("hadamard", "single", (model.index_head_dim, model.index_head_dim)),
        ):
            layers = (self._mtp_layer,) if family == "mtp" else ()
            stacks[name] = StackPlan(name, family, layers, shape, torch.bfloat16)
        self.stacks = stacks

    def _member_shape(self, family: str, spec: _StackMember) -> tuple[int, ...]:
        """One layer's rank-local shape for a stack member."""
        model = self.model
        hidden, qkv = model.hidden_size, self.local_kda_qkv
        if family == "kda":
            shapes = {
                "w_q": (qkv, hidden),
                "w_k": (qkv, hidden),
                "w_v": (qkv, hidden),
                "conv_q": (qkv, 1, model.linear_conv_kernel_dim),
                "conv_k": (qkv, 1, model.linear_conv_kernel_dim),
                "conv_v": (qkv, 1, model.linear_conv_kernel_dim),
                "w_f_a": (model.linear_head_dim, hidden),
                "w_f_b": (qkv, model.linear_head_dim),
                "w_g_a": (model.linear_head_dim, hidden),
                "w_g_b": (qkv, model.linear_head_dim),
                "w_b": (self.local_kda_h, hidden),
                "dt_bias": (qkv,),
                "a_log": (self.local_kda_h,),
                "o_norm": (model.linear_head_dim,),
                "w_o": (hidden, qkv),
            }
        elif family == "mla":
            shapes = {
                "w_q_a": (model.q_lora_rank, hidden),
                "q_a_norm": (model.q_lora_rank,),
                "w_q_b": (self.local_h * model.qk_head_dim, model.q_lora_rank),
                "w_kv_a": (model.kv_lora_rank, hidden),
                "kv_a_norm": (model.kv_lora_rank,),
                "w_k": (self.local_h, model.qk_head_dim, model.kv_lora_rank),
                "w_v": (self.local_h, model.v_head_dim, model.kv_lora_rank),
                "w_o": (hidden, self.local_h * model.v_head_dim),
            }
        elif family == "indexer":
            shapes = {
                "w_q_b": (model.q_lora_rank, model.index_n_heads * model.index_head_dim),
                "w_k": (hidden, model.index_head_dim),
                "k_norm_w": (model.index_head_dim,),
                "k_norm_b": (model.index_head_dim,),
                "w_weights": (hidden, model.index_n_heads),
                "w_compress_gate": (hidden, model.index_head_dim),
                "compress_ape": (model.index_kpool, model.index_head_dim),
            }
        elif family == "mhc":
            shapes = {
                "attn_fn": (model.mix_hc, model.hc_dim),
                "attn_scale": (3,),
                "attn_base": (model.mix_hc,),
                "ffn_fn": (model.mix_hc, model.hc_dim),
                "ffn_scale": (3,),
                "ffn_base": (model.mix_hc,),
            }
        elif family == "norm":
            shapes = {"input_norm": (hidden,), "post_norm": (hidden,)}
        else:
            shapes = {"gate_w": (model.n_routed_experts, hidden), "gate_bias": (model.n_routed_experts,)}
        if family == "mla" and spec.member.endswith("_scale"):
            return (shapes[spec.member.removesuffix("_scale")][0],)
        return shapes[spec.member]

    # -- public interface --------------------------------------------------------------

    @property
    def tensor_layouts(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        """Device stack name -> stacked shape/dtype, for ``runtime.alloc_tensor``."""
        return {plan.name: (plan.shape, plan.dtype) for plan in self.stacks.values()}

    @property
    def bytes_per_rank(self) -> int:
        """Device-resident bytes every rank allocates (identical across ranks)."""
        return sum(plan.bytes for plan in self.stacks.values())

    @property
    def host_bytes(self) -> int:
        """Host-side table bytes: the full LM head plus the embedding table.

        The backbone head is cached and embedding rows are gathered per prompt.
        MTP rows and head shards are read separately without retaining full tables.
        """
        return 2 * self.model.vocab_size * self.model.hidden_size * 2

    def plan(self, rank: int) -> RankPlan:
        """The rank's slice of every device stack plus byte totals; no payload reads."""
        if rank not in range(self.ranks):
            raise ValueError(f"rank must be in [0, {self.ranks}), got {rank}")
        first = rank * self.n_local_experts
        return RankPlan(
            rank=rank, tp_rank=rank % self.tp_size, ep_rank=rank,
            local_expert_ids=tuple(range(first, first + self.n_local_experts)),
            stacks=tuple(self.stacks.values()), device_bytes=self.bytes_per_rank,
            host_tables=("embed_tokens", "lm_head"),
        )

    def lookup_rows(self, token_ids: torch.Tensor, *, mtp: bool = False) -> torch.Tensor:
        """Gather one prompt's BF16 embedding rows from the host table."""
        ids = torch.as_tensor(token_ids, device="cpu")
        if ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("token ids must have an integer dtype")
        if bool(((ids < 0) | (ids >= self.model.vocab_size)).any()):
            raise ValueError("token id is outside the checkpoint vocabulary")
        name = self._embed_name
        if mtp:
            name = self._resolve(
                [root + f"{self._mtp_layer}.embed_tokens.weight" for root in TEXT_LAYER_ROOTS],
                "MTP embedding",
            )
        return gather_rows(self.checkpoint, name, ids).to(torch.bfloat16)

    def mtp_head_shard(self, rank: int) -> torch.Tensor:
        """Read the MTP-owned vocabulary shard without assuming tied tables."""
        if rank not in range(self.tp_size):
            raise ValueError("MTP head rank is outside the TP group")
        name = self._resolve(
            [root + f"{self._mtp_layer}.shared_head.head.weight" for root in TEXT_LAYER_ROOTS],
            "MTP shared head",
        )
        width = self.model.vocab_size // self.tp_size
        return self.checkpoint.read(name, slice(rank * width, (rank + 1) * width)).bfloat16()

    def lm_head(self) -> torch.Tensor:
        """The full ``[vocab, hidden]`` BF16 head matrix, read once and cached."""
        if self._lm_head_cache is None:
            head = self.checkpoint.read(self._lm_head_name)
            expected = (self.model.vocab_size, self.model.hidden_size)
            if tuple(head.shape) != expected:
                raise ValueError(f"lm_head has shape {tuple(head.shape)}, expected {expected}")
            self._lm_head_cache = head.to(torch.bfloat16).contiguous()
        return self._lm_head_cache

    # -- materialization ----------------------------------------------------------------

    def _read_source(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Authority-driven read: FLOAT casts, W8A8_DYNAMIC dequantizes to BF16.

        Dequant is one FP32 scale multiply and one BF16 rounding, so the INT8
        storage of the four MLA projections never crosses the host/device
        boundary; a non-zero ``weight_offset`` would invalidate the symmetric
        assumption and is rejected instead of silently dropped.
        """
        label = self._label(name)
        if label == "FLOAT":
            return self.checkpoint.read(name).to(dtype)
        if dtype != torch.bfloat16:
            raise ValueError(f"{name} is {label}; only BF16 consumers may dequantize, asked for {dtype}")
        weight, scale = _read_quant(self.checkpoint, name)
        product = weight.to(torch.float32) * scale.reshape(-1, *([1] * (weight.dim() - 1)))
        return product.to(torch.bfloat16)

    def _shard(self, tensor: torch.Tensor, shard: str, tp_rank: int) -> torch.Tensor:
        """Keep one TP rank's row/column block (heads sit contiguously)."""
        if shard == "replicate":
            return tensor
        axis = 0 if shard == "rows" else 1
        size = tensor.shape[axis]
        if size % self.tp_size:
            raise ValueError(f"{shard} axis {size} does not divide TP{self.tp_size}")
        width, start = size // self.tp_size, tp_rank * (size // self.tp_size)
        return tensor[start : start + width] if axis == 0 else tensor[:, start : start + width]

    def _member_tensor(self, family: str, spec: _StackMember, layer: int, tp_rank: int) -> torch.Tensor:
        name = self._source[(f"{family}.{spec.member}", layer)]
        value = self._read_source(name, spec.dtype)
        if spec.transform == "transpose":
            value = value.t()
        return self._shard(value, spec.shard, tp_rank).contiguous()

    def _mla_layer(self, layer: int, tp_rank: int) -> dict[str, torch.Tensor]:
        """One MLA layer's eight rank slices; ``kv_b_proj`` is read once."""
        out: dict[str, torch.Tensor] = {}
        kv_b = None
        for spec in self._family_members["mla"]:
            if spec.member.endswith("_scale"):
                continue
            if spec.dtype == torch.int8:
                name = self._source[(f"mla.{spec.member}", layer)]
                weight, scale = _read_quant(self.checkpoint, name)
                out[spec.member] = self._shard(weight, spec.shard, tp_rank).contiguous()
                if spec.shard == "rows":
                    scale = self._shard(scale, "rows", tp_rank)
                out[spec.member + "_scale"] = scale.contiguous()
            elif spec.transform in ("split_k", "split_v"):
                if kv_b is None:
                    kv_b = self._read_source(self._source[("mla.w_k", layer)], torch.bfloat16)
                (key, value) = split_kv_b(
                    kv_b, tp_rank, self.tp_size, qk_dim=self.model.qk_head_dim, v_dim=self.model.v_head_dim,
                    kv_lora=self.model.kv_lora_rank,
                )
                out[spec.member] = key if spec.transform == "split_k" else value
            else:
                out[spec.member] = self._member_tensor("mla", spec, layer, tp_rank)
        return out

    def _stack_family(self, family: str, tp_rank: int) -> dict[str, torch.Tensor]:
        """Materialize one non-FFN family as its stacked rank tensor per member."""
        collected: dict[str, list[torch.Tensor]] = {spec.member: [] for spec in self._family_members[family]}
        for layer in self._family_layers[family]:
            per_layer = (
                self._mla_layer(layer, tp_rank)
                if family == "mla"
                else {
                    spec.member: self._member_tensor(family, spec, layer, tp_rank)
                    for spec in self._family_members[family]
                }
            )
            for member, tensor in per_layer.items():
                collected[member].append(tensor)
        return {member: torch.stack(slices) for member, slices in collected.items()}

    def _stack_dense(self, tp_rank: int) -> dict[str, torch.Tensor]:
        """The dense layers' TP-sliced MLPs, stacked on the family axis."""
        model = self.model
        channels = slice(tp_rank * self.local_dense_inter, (tp_rank + 1) * self.local_dense_inter)
        shards = [
            _fused_ffn_at(
                self.checkpoint,
                layer_prefixes(layer),
                _ffn_roles("dense"),
                expert=None,
                intermediate=model.intermediate_size,
                hidden=model.hidden_size,
                channels=channels,
            )
            for layer in self._family_layers["dense"]
        ]
        return {
            "w_gate_up": torch.stack([shard[0] for shard in shards]),
            "w_gate_up_scale": torch.stack([shard[1] for shard in shards]),
            "w_down": torch.stack([shard[2] for shard in shards]),
            "w_down_scale": torch.stack([shard[3] for shard in shards]),
        }

    def _moe_layer(self, layer: int, ep_rank: int) -> dict[str, torch.Tensor]:
        """One MoE layer's local experts and replicated shared expert."""
        model = self.model
        prefixes = layer_prefixes(layer)
        first = ep_rank * self.n_local_experts
        stacked = None
        for expert in range(first, first + self.n_local_experts):
            values = _fused_ffn_at(
                self.checkpoint,
                prefixes, _ffn_roles("routed"), expert=expert, intermediate=model.moe_intermediate_size,
                hidden=model.hidden_size,
            )
            if stacked is None:
                stacked = [torch.empty((self.n_local_experts, *value.shape), dtype=value.dtype) for value in values]
            for target, value in zip(stacked, values):
                target[expert - first].copy_(value)
        shared = _fused_ffn_at(
            self.checkpoint,
            prefixes, _ffn_roles("shared"), expert=None, intermediate=model.moe_intermediate_size,
            hidden=model.hidden_size,
        )
        return {
            "w_gate_up": stacked[0],
            "w_gate_up_scale": stacked[1],
            "w_down": stacked[2],
            "w_down_scale": stacked[3],
            "shared_w_gate_up": shared[0],
            "shared_w_gate_up_scale": shared[1],
            "shared_w_down": shared[2],
            "shared_w_down_scale": shared[3],
        }

    def _single_tensors(self) -> dict[str, torch.Tensor]:
        """The unstacked residents: final norm, MTP extras, the Hadamard table."""

        model = self.model
        hadamard = sylvester_hadamard(model.index_head_dim)
        return {
            "norm.final_norm": self._read_source(self._source[("norm.final_norm", -1)], torch.bfloat16),
            "mtp.eh_proj": self._read_source(self._source[("mtp.eh_proj", self._mtp_layer)], torch.bfloat16),
            "mtp.enorm": self._read_source(self._source[("mtp.enorm", self._mtp_layer)], torch.bfloat16),
            "mtp.hnorm": self._read_source(self._source[("mtp.hnorm", self._mtp_layer)], torch.bfloat16),
            "mtp.final_norm": self._read_source(
                self._source[("mtp.final_norm", self._mtp_layer)], torch.bfloat16
            ),
            "hadamard": hadamard.contiguous(),
        }

    # -- staged upload --------------------------------------------------------------------

    def upload(self, runtime, rank: int, handles) -> None:
        """Stage one rank's resident stacks to its preallocated device tensors.

        Every stacked family stages one layer-slice at a time: the member
        slice is host-materialized, copied at its layer offset and dropped
        before the next layer, so peak host memory is one layer of one family
        (about 20 MB for a KDA layer at TP16, 280 MB at TP1) or one MoE
        layer-slice (the model's largest single buffer), never the whole
        stacked family doubled while ``torch.stack`` assembled it. Each
        per-layer ``put`` at ``layer_index`` preserves the logical layer slice
        ``torch.stack`` produced before. The two shared INT8
        matrices use NZ bytes; checkpoint-reader tensors and all other stacks
        retain logical row-major order.
        ``handles`` maps every layout name from :attr:`tensor_layouts` to a
        device handle carrying ``data_ptr``.
        """
        if rank not in range(self.ranks):
            raise ValueError(f"rank must be in [0, {self.ranks}), got {rank}")
        missing = [name for name in self.stacks if name not in handles]
        if missing:
            raise ValueError(f"handles are missing the stacks {missing}")
        tp_rank = rank % self.tp_size

        def put(name: str, data: torch.Tensor, layer_index: int | None = None) -> None:
            plan = self.stacks[name]
            if name in ("moe.shared_w_gate_up", "moe.shared_w_down"):
                data = pack_int8_nz(data)
            data = data.contiguous()
            expected = plan.shape if layer_index is None else plan.shape[1:]
            if tuple(data.shape) != expected or data.dtype != plan.dtype:
                raise ValueError(f"staged {name} is {tuple(data.shape)}/{data.dtype}, planned {expected}/{plan.dtype}")
            offset = 0 if layer_index is None else layer_index * data.numel() * data.element_size()
            runtime.copy_to(
                handles[name].data_ptr, data.data_ptr(), data.numel() * data.element_size(),
                dst_offset=offset, worker_id=rank,
            )

        for name, value in self._single_tensors().items():
            put(name, value)
        for family in ("norm", "mhc", "router", "kda", "mla", "indexer"):
            for index, layer in enumerate(self._family_layers[family]):
                per_layer = (
                    self._mla_layer(layer, tp_rank)
                    if family == "mla"
                    else {
                        spec.member: self._member_tensor(family, spec, layer, tp_rank)
                        for spec in self._family_members[family]
                    }
                )
                for member, value in per_layer.items():
                    put(f"{family}.{member}", value, layer_index=index)
        for member, stack in self._stack_dense(tp_rank).items():
            put(f"dense.{member}", stack)
        for index, layer in enumerate(self._family_layers["moe"]):
            for member, value in self._moe_layer(layer, rank).items():
                put(f"moe.{member}", value, layer_index=index)


def prepare_step_inputs(
    weights: ResidentPrefillWeights,
    sequences: list[list[int]],
    *,
    kv_seq_lens: torch.Tensor | None = None,
    visual_features: dict[int, torch.Tensor] | None = None,
    cache_pages: SharedCachePages | None = None,
    mtp: bool = False,
) -> dict:
    """Pack only new query tokens into stable request-owned cache slots."""
    if len(sequences) != BATCH:
        raise ValueError(f"the packed batch has {BATCH} request slots, got {len(sequences)}")
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)
    previous = torch.zeros(BATCH, dtype=torch.int32) if kv_seq_lens is None else kv_seq_lens.to(torch.int32)
    if previous.shape != (BATCH,) or bool((previous < 0).any()):
        raise ValueError("kv_seq_lens must contain one nonnegative prefix length per request slot")
    if bool((previous.to(torch.int64) + lengths > MAX_SEQ_LEN).any()):
        raise ValueError(f"request context exceeds the compiled maximum {MAX_SEQ_LEN}")
    total = int(lengths.sum().item())
    if total > CAPACITY:
        raise ValueError(f"{total} new query tokens exceed the compiled capacity {CAPACITY}")

    embedding_rows = torch.zeros(CAPACITY, D, dtype=torch.bfloat16)
    if total:
        ids = torch.tensor([token for seq in sequences for token in seq], dtype=torch.int64)
        embedding_rows[:total] = weights.lookup_rows(ids, mtp=mtp)
    if visual_features:

        offset = 0
        if any(type(slot) is not int or slot not in range(BATCH) for slot in visual_features):
            raise ValueError("visual feature keys must be request slots")
        for slot, sequence in enumerate(sequences):
            if slot in visual_features:
                features = visual_features[slot]
                indices = torch.tensor(
                    [row + offset for (row, token) in enumerate(sequence) if token == IMAGE_TOKEN_ID],
                )
                if features.dtype != torch.bfloat16 or tuple(features.shape) != (indices.numel(), D):
                    raise ValueError("visual features must be BF16 [image_placeholder_count, hidden_size]")
                if not torch.isfinite(features).all():
                    raise ValueError("visual features must be finite")
                embedding_rows[indices.long()] = features.cpu()
            offset += len(sequence)
    query_start_loc = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64), lengths]), 0).to(torch.int32)
    if cache_pages is None:
        if CACHE_TOKENS < BATCH * TABLE_MAX * BLOCK_SIZE:
            raise ValueError("a bounded shared cache requires caller-owned SharedCachePages")
        block_table = torch.arange(BATCH * TABLE_MAX, dtype=torch.int32).reshape(BATCH, TABLE_MAX)
        pool_table = torch.arange(BATCH * POOL_TABLE_MAX, dtype=torch.int32).reshape(BATCH, POOL_TABLE_MAX)
        pool_offsets = torch.arange(BATCH, dtype=torch.int32) * POOL_STRIDE
        owners = torch.arange(LATENT_BLOCKS, dtype=torch.int32) // TABLE_MAX
    else:
        block_table = cache_pages.table.clone()
        pool_table = cache_pages.pool_table()
        pool_offsets = cache_pages.pool_offsets()
        owners = cache_pages.owners.clone()
    metadata = build_forward_metadata(
        query_start_loc, kv_seq_lens=previous, mla_block_table=block_table,
        index_block_table=block_table,
        pool_block_table=pool_table,
        kda_state_rows=torch.arange(BATCH, dtype=torch.int32), tp_size=TP_SIZE,
    )
    step = build_indexer_chunk_metadata(
        metadata, block_table, pool_offsets, pools_cap=POOL_AXIS, pool_rows=POOL_TABLE_ROWS,
        pool_block_table=pool_table,
    )
    for name in ("seg_start", "pool_count_tok", "tail_start", "tail_count", "kv_len_tok"):
        step[name] = _pad_to(step[name], CAPACITY, 0)
    logit_rows = torch.where(lengths > 0, metadata.logit_row_indices, torch.zeros(BATCH, dtype=torch.int32))
    return {
        "embedding_rows": embedding_rows,
        "query_start_loc": query_start_loc,
        "request_ids": _pad_to(metadata.request_ids, CAPACITY, -1),
        "mla_slots": _pad_to(metadata.mla_slots, CAPACITY, -1),
        "index_slots": _pad_to(metadata.index_slots, CAPACITY, -1),
        "kda_state_rows": metadata.kda_state_rows,
        "block_table": block_table,
        "cache_page_owners": owners,
        "logit_row_indices": _pad_to(logit_rows, LOGIT_ROWS, 0),
        "active_tokens": torch.tensor([total], dtype=torch.int32),
        **step,
        **kda_host_constants(),
    }


def production_launch_plan() -> tuple[tuple, ...]:
    """The dispatch sequence of one full forward."""
    steps: list[tuple] = [("seed",)]
    for layer in range(NUM_LAYERS):
        if layer < DENSE_NUM_LAYERS:
            steps.append((layer, "layer"))
        else:
            steps.append((layer, "attn"))
            steps.append((layer, "ffn"))
    steps.append(("head",))
    return tuple(steps)


def production_program_entry(program: str):
    """The L3 entry one consolidated runtime-ordinal program compiles."""
    if program == "seed":
        return l3_prefill_seed
    if program == "dense":
        return l3_prefill_dense_layer
    if program == "kda_attn":
        return l3_prefill_kda_attention
    if program == "dsa_attn":
        return l3_prefill_sparse_attention
    if program == "moe_ffn":
        return l3_prefill_moe_layer
    if program == "head":
        return l3_prefill_head_rows
    raise ValueError(f"unknown consolidated program {program!r}")


def production_program_of_step(step: tuple) -> str:
    """Select the resident program for one dense layer or attention/FFN step."""
    if step in (("seed",), ("head",)):
        return step[0]
    layer, sublayer = step
    if sublayer == "layer":
        return "dense"
    if sublayer == "attn":
        return "dsa_attn" if layer % 4 == 3 else "kda_attn"
    if sublayer == "ffn":
        return "moe_ffn"
    raise ValueError(f"unknown sublayer {sublayer!r} in step {step}")


def production_step_ordinals(step: tuple, *, dispatch_index: int = 0) -> dict[str, int]:
    """Return layer ordinals and collective epochs for one program invocation.

    ``dispatch_index`` counts completed invocations of this prepared program,
    independent of other programs and request-cache slots. Zero selects a fresh
    window set for standalone validation. TP counters receive two notifications
    per epoch and must remain within signed INT32 range.
    """
    if type(dispatch_index) is not int or dispatch_index < 0:
        raise ValueError("dispatch_index must be a nonnegative integer")
    program = production_program_of_step(step)
    if program in ("seed", "head"):
        return {}
    layer = step[0]
    ordinals = {
        "bb_ord": layer,
        "skip_topk": 0,
        "kda_ord": layer - layer // 4,
        "dsa_ord": layer // 4,
        "dense_ord": layer % DENSE_NUM_LAYERS,
        "moe_ord": layer - DENSE_NUM_LAYERS,
    }
    epoch = dispatch_index + 1
    if program == "dense":
        ordinals.update(attn_epoch=2 * epoch - 1, ffn_epoch=2 * epoch)
        final_tp_epoch = 2 * epoch
    elif program == "moe_ffn":
        ordinals.update(moe_epoch=epoch, ffn_epoch=epoch)
        final_tp_epoch = epoch
    else:
        ordinals["attn_epoch"] = epoch
        final_tp_epoch = epoch
    if final_tp_epoch > (2**31 - 1) // 2:
        raise OverflowError("collective epochs exhausted; close and reopen the resident session")
    return ordinals


def _l3_arg_arity(entry) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """One entry's ``(tensor names, scalar names)``, both in signature order."""
    tensors, scalars = [], []
    for name, param in inspect.signature(entry._func).parameters.items():
        if param.annotation is pl.constexpr:
            continue
        if getattr(param.annotation, "shape", None) is None:
            scalars.append(name)
        else:
            tensors.append(name)
    return tuple(tensors), tuple(scalars)


def _step_key(step: tuple) -> str:
    """Manifest key of one launch-plan step (``seed`` / ``L4:attn`` / ``head``)."""
    if len(step) == 1:
        return step[0]
    layer, sublayer = step
    return f"L{layer}:{sublayer}"


def production_step_bindings(plan: tuple[tuple, ...] | None = None) -> dict[tuple, dict[str, str]]:
    """One launch-plan step's stream overrides: signature name -> workspace."""
    if plan is None:
        plan = production_launch_plan()
    bindings: dict[tuple, dict[str, str]] = {}
    side = 0  # X_H_PINGPONG index holding the stream after the previous step
    for step in plan:
        current, other = X_H_PINGPONG[side], X_H_PINGPONG[1 - side]
        if step in (("seed",), ("head",)):
            bindings[step] = {"x_hc": current}
        else:
            sublayer = step[1]
            if sublayer == "layer":  # dense whole-layer entry: mid parks, final returns
                bindings[step] = {"x_hc": current, "out_x_hc_mid": other, "out_x_hc_out": current}
            elif sublayer == "attn":  # moves the stream to the other side
                bindings[step] = {"x_hc": current, "out_x_hc_mid": other}
                side = 1 - side
            else:  # "ffn": returns the stream to the side the attention half read
                bindings[step] = {"x_hc_mid": current, "out_x_hc_out": other}
                side = 1 - side
    return bindings


def _alloc_name_of(param: str, overrides: dict[str, str]) -> str:
    """One signature parameter's allocation name in the runner's tensor table.

    Stream names answer the step's ping-pong override; every other ``out_*``
    boundary name answers its ``ws_*`` workspace; the rest (step inputs, cache
    pools, resident stacks, ``head_hidden``) allocate under their own names.
    """
    if param in overrides:
        return overrides[param]
    if param.startswith("out_"):
        workspace = "ws_" + param[len("out_") :]
        if workspace in WORKSPACES:
            return workspace
    return param


def _production_geometry() -> dict[str, int]:
    """Import-frozen query/cache dimensions, collective ABI and shared weight layout."""
    return {
        "version": 6,
        "mtp_cache": 1,
        "cache_tokens": CACHE_TOKENS,
        "native_mla_int8": 1,
        "shared_int8_nz": 1,
        "tp": TP_SIZE,
        "ep": EP_SIZE,
        "capacity": CAPACITY,
        "max_seq_len": MAX_SEQ_LEN,
        "batch": BATCH,
        "latent_rows": LATENT_ROWS,
        "pool_axis": POOL_AXIS,
    }


def compile_production(
    runtime_dir: Path,
    device_ids: list[int],
    *,
    platform: str = "a2a3",
    mode: str = "rt",
    programs: tuple[str, ...] | None = None,
    auxiliaries: bool = True,
) -> dict:
    """Compile the production programs and record the artifact manifest."""
    config = RunConfig(
        platform=platform, distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
    )
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"_mode": mode, "_geometry": _production_geometry()}
    if mode == "rt":
        for program in programs if programs is not None else RT_PROGRAMS:
            if program not in RT_PROGRAMS:
                raise ValueError(f"unknown consolidated program {program!r}")
            compiled = production_program_entry(program).compile(config=config)
            out_dir = getattr(compiled, "output_dir", None) or getattr(compiled, "work_dir")
            if out_dir is None:
                raise RuntimeError(f"no artifact directory for program {program}")
            manifest[program] = str(out_dir)
            print("GLM_PROD_COMPILED", program, out_dir, flush=True)
    elif mode != "rt":
        raise ValueError(f"mode must be 'rt', got {mode!r}")
    if auxiliaries:
        auxiliary = {}
        for name, entry in (
            ("reset", l3_reset_request_caches),
            ("head_logits", l3_prefill_lm_head_logits),
            ("snapshot", l3_copy_request_snapshot),
            ("mtp_prepare", l3_prefill_mtp_prepare),
            ("mtp_head", l3_prefill_mtp_head),
        ):
            compiled = entry.compile(config=config)
            auxiliary[name] = str(compiled.output_dir)
        manifest["_auxiliary"] = auxiliary
    (runtime_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return load_production(runtime_dir, device_ids, platform=platform)


def load_production(runtime_dir: Path, device_ids: list[int], *, platform: str = "a2a3") -> dict:
    """Load the production programs from a manifest runtime dir."""
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

    manifest = json.loads((Path(runtime_dir) / "manifest.json").read_text())
    if manifest.get("_mode") != "rt":
        raise ValueError(
            f"{runtime_dir}/manifest.json has unsupported mode {manifest.get('_mode')!r}; "
            "recompile with compile_production(mode='rt')"
        )
    if manifest.get("_geometry") != _production_geometry():
        raise ValueError(
            "runtime query/cache geometry, collective ABI or shared weight layout differs; recompile "
            "with matching --tp, --ep, --capacity, --max-seq-len and --cache-tokens"
        )
    programs_present = [key for key in manifest if not key.startswith("_")]
    unknown = [key for key in programs_present if key not in RT_PROGRAMS]
    if unknown:
        raise ValueError(f"rt manifest carries unknown programs {unknown}")
    result = {
        program: DistributedCompiledProgram.from_dir(
            manifest[program],
            platform=platform,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        )
        for program in programs_present
    }
    if "_auxiliary" in manifest:
        result["_auxiliary"] = {
            name: DistributedCompiledProgram.from_dir(
                path,
                platform=platform,
                distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            )
            for name, path in manifest["_auxiliary"].items()
        }
    return result


def _read_ffn(
    checkpoint: Checkpoint, layer: int, mlp_base: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One FFN as FP32 ``(gate, up, down)``, fused or separate gate/up spelling."""

    def names(suffixes: Sequence[str]) -> list[str]:
        return [f"{root}{layer}.{mlp_base}{s}" for root in TEXT_LAYER_ROOTS for s in suffixes]

    fused = next((n for n in names(_FF_FUSED) if n in checkpoint), None)
    if fused is not None:
        gate_up = read_reference_float(checkpoint, fused)
        gate, up = gate_up.chunk(2, dim=0)
    else:
        gate = read_reference_float(checkpoint, _resolve_name(checkpoint, layer, mlp_base, _FF_GATE))
        up = read_reference_float(checkpoint, _resolve_name(checkpoint, layer, mlp_base, _FF_UP))
    down = read_reference_float(checkpoint, _resolve_name(checkpoint, layer, mlp_base, _FF_DOWN))
    return gate.contiguous(), up.contiguous(), down.contiguous()


@dataclass
class RefRunResult:
    """What ``PrefillReferenceDriver.run`` returns."""

    final_hidden: torch.Tensor | None  # [B, S, D] post final-norm (full range only)
    streams_out: torch.Tensor  # [B, S, hc_mult, D] leaving the last layer
    layer_summaries: list[dict[str, float]] = field(default_factory=list)
    dump_paths: dict[int, Path] = field(default_factory=dict)
    final_dump_path: Path | None = None


class PrefillReferenceDriver:
    """Host-side reference prefill driver over real or synthetic weights.

    Real mode (``checkpoint_root`` given) wraps the deployment W8A8 checkpoint
    through :class:`Checkpoint` and streams one layer at a time; the
    routed experts of a MoE layer stream one expert at a time inside the layer.
    Synthetic mode (``checkpoint_root=None``) serves deterministic synthetic
    weights at the true ``config.py`` geometry for the no-checkpoint self-test.
    """

    def __init__(self, checkpoint_root: str | Path | None = None, model: Glm53FlashConfig = FLASH):
        if checkpoint_root is not None and model is not FLASH:
            raise ValueError("Checkpoint validates the deployment geometry; scaled models run checkpoint-less")
        self.model = model
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else None
        self.checkpoint = Checkpoint(self.checkpoint_root) if self.checkpoint_root is not None else None
        self._lm_head_cache: torch.Tensor | None = None

    # -- weight loading -----------------------------------------------------------------

    def load_layer(self, layer_id: int, *, seed: int = 0) -> LayerRefWeights:
        """Load (and dequantize) one layer, or synthesize it in checkpoint-less mode.

        Backbone layers are ``[0, num_hidden_layers)``; the MTP layer
        (``model.mtp_layer_id``) also loads when present — it has no mHC
        tensors (``has_hyper_connections=False``), so its ``attn_hc``/``ffn_hc``
        are ``None`` and its sub-layer weights feed :func:`mla_forward` /
        :func:`moe_forward` directly for the MTP diff harness.
        """
        model = self.model
        if not 0 <= layer_id <= model.mtp_layer_id:
            raise ValueError(
                f"layer_id must be in [0, {model.mtp_layer_id}] (backbone layers plus the "
                f"MTP layer), got {layer_id}"
            )
        if self.checkpoint is None:
            return synthetic_layer(model, layer_id, seed)
        checkpoint = self.checkpoint
        config = model.layer_config(layer_id)

        def read(sub_prefix: str, suffixes: Sequence[str]) -> torch.Tensor:
            return read_reference_float(checkpoint, _resolve_name(checkpoint, layer_id, sub_prefix, suffixes))

        norms = {member: read(prefix, suffixes) for member, prefix, suffixes in _NORM_TENSORS}

        def read_mhc(prefix_name: str) -> MhcTensors:
            picked = {
                member: read(prefix, suffixes)
                for member, prefix, suffixes in _MHC_TENSORS
                if member.startswith(prefix_name)
            }
            return MhcTensors(
                fn=picked[f"{prefix_name}fn"], base=picked[f"{prefix_name}base"],
                scale=picked[f"{prefix_name}scale"],
            )

        layer = LayerRefWeights(
            layer_id=layer_id,
            config=config,
            model=model,
            input_norm=norms["input_norm"],
            post_norm=norms["post_norm"],
            # The MTP layer carries no hc_* tensors: mHC reads are gated on the
            # layer's has_hyper_connections flag, never unconditional.
            attn_hc=read_mhc("attn_") if config.has_hyper_connections else None,
            ffn_hc=read_mhc("ffn_") if config.has_hyper_connections else None,
        )
        if config.attention_kind is AttentionKind.LINEAR:
            kda = {member: read(prefix, suffixes) for member, prefix, suffixes in _KDA_TENSORS}
            layer.kda = KdaTensors(**kda)
        else:
            mla = {member: read(prefix, suffixes) for member, prefix, suffixes in _MLA_TENSORS}
            layer.mla = MlaTensors(**mla)
            indexer = {member: read(prefix, suffixes) for member, prefix, suffixes in _INDEXER_TENSORS}
            layer.indexer = IndexerTensors(**indexer)
        if config.mlp_kind is MlpKind.DENSE:
            gate, up, down = _read_ffn(checkpoint, layer_id, "mlp.")
            layer.dense = DenseTensors(gate, up, down)
        else:
            router_weight = read("mlp.", ("gate.weight",))
            bias_name = _resolve_name(checkpoint, layer_id, "mlp.", ("gate.e_score_correction_bias",))
            layer.router = RouterTensors(router_weight, read_reference_float(checkpoint, bias_name))
            layer.experts = CheckpointExpertProvider(checkpoint, layer_id)
            gate, up, down = _read_ffn(checkpoint, layer_id, "mlp.shared_experts.")
            layer.shared = DenseTensors(gate, up, down)
        return layer

    def missing_layer_tensors(self, layer_id: int) -> list[str]:
        """Checkpoint tensors this layer still needs but no present shard carries.

        A non-empty answer means the download is incomplete and ``load_layer``
        will raise; empty means the layer can run against true weights today.
        Names are symbolic (canonical root spelling): a tensor counts as present
        when any accepted spelling resolves, mirroring :func:`_resolve_name`.
        mHC tensors are requested only when the layer has hyper-connections —
        the MTP layer carries none (``has_hyper_connections=False``).
        """
        if self.checkpoint is None:
            return []
        if not 0 <= layer_id <= self.model.mtp_layer_id:
            raise ValueError(f"layer_id must be in [0, {self.model.mtp_layer_id}], got {layer_id}")
        model = self.model
        config = model.layer_config(layer_id)
        checkpoint = self.checkpoint

        def present(sub_prefix: str, suffixes: Sequence[str]) -> bool:
            return any(
                (f"{root}{layer_id}.{sub_prefix}{suffix}" in checkpoint for root in TEXT_LAYER_ROOTS for suffix in suffixes),
            )

        def gap(sub_prefix: str, suffixes: Sequence[str]) -> list[str]:
            if present(sub_prefix, suffixes):
                return []
            return [f"{TEXT_LAYER_ROOTS[0]}{layer_id}.{sub_prefix}{suffixes[0]}"]

        def ffn_gaps(base: str) -> list[str]:
            gate_up_ok = present(base, _FF_FUSED) or (present(base, _FF_GATE) and present(base, _FF_UP))
            gaps = [] if gate_up_ok else [f"{TEXT_LAYER_ROOTS[0]}{layer_id}.{base}{_FF_FUSED[0]}"]
            if not present(base, _FF_DOWN):
                gaps.append(f"{TEXT_LAYER_ROOTS[0]}{layer_id}.{base}{_FF_DOWN[0]}")
            return gaps

        gaps: list[str] = []
        for _, prefix, suffixes in _NORM_TENSORS:
            gaps += gap(prefix, suffixes)
        if config.has_hyper_connections:
            for _, prefix, suffixes in _MHC_TENSORS:
                gaps += gap(prefix, suffixes)
        family = _KDA_TENSORS if config.attention_kind is AttentionKind.LINEAR else _MLA_TENSORS + _INDEXER_TENSORS
        for _, prefix, suffixes in family:
            gaps += gap(prefix, suffixes)
        if config.mlp_kind is MlpKind.DENSE:
            gaps += ffn_gaps("mlp.")
        else:
            gaps += gap("mlp.", ("gate.weight",))
            gaps += gap("mlp.", ("gate.e_score_correction_bias",))
            for expert in range(model.n_routed_experts):
                gaps += ffn_gaps(f"mlp.experts.{expert}.")
            gaps += ffn_gaps("mlp.shared_experts.")
        return gaps

    # -- host tables --------------------------------------------------------------------

    def embed_rows(
        self, token_ids: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Embedding rows ``[B, S, D]`` FP32: checkpoint row-gather or synthetic."""
        ids = _as_batch_seqs(token_ids)
        if self.checkpoint is not None:
            name = _resolve_top_name(self.checkpoint, ("embed_tokens.weight",))
            return gather_rows(self.checkpoint, name, ids).to(torch.float32)
        gen = generator if generator is not None else torch.Generator().manual_seed(1234)
        rows = torch.randn(ids.numel(), self.model.hidden_size, generator=gen) * 0.02
        return rows.view(*ids.shape, self.model.hidden_size)

    def final_norm_weight(self) -> torch.Tensor:
        """The final RMSNorm weight ``[D]`` (ones in synthetic mode)."""
        if self.checkpoint is not None:
            name = _resolve_top_name(self.checkpoint, ("norm.weight",))
            return read_reference_float(self.checkpoint, name)
        return torch.ones(self.model.hidden_size)

    def _lm_head(self) -> torch.Tensor:
        if self._lm_head_cache is None:
            if self.checkpoint is None:
                raise RuntimeError("lm_head top-1 needs a real checkpoint")
            name = _resolve_top_name(self.checkpoint, ("lm_head.weight",))
            self._lm_head_cache = self.checkpoint.read(name).to(torch.bfloat16).contiguous()
        return self._lm_head_cache

    def lm_head_top1(
        self, hidden: torch.Tensor, k: int = 1, *, chunk: int = 8192
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Host LM-head top-k: chunked FP32 matmul over the full ``[vocab, D]`` table.

        Returns ``(values, indices)``, each ``[N, k]``, rows of ``hidden`` ``[N, D]``.
        """
        head = self._lm_head()
        rows = hidden.reshape(-1, hidden.shape[-1]).to(torch.float32)
        best_values = torch.full((rows.shape[0], k), -torch.inf)
        best_indices = torch.zeros((rows.shape[0], k), dtype=torch.int64)
        for start in range(0, head.shape[0], chunk):
            logits = (head[start : start + chunk].to(torch.float32)) @ rows.t()  # [chunk, N]
            candidates = torch.cat([best_values, logits.t()], dim=1)
            indices = torch.cat(
                [best_indices, torch.arange(start, start + logits.shape[0])[None].expand(rows.shape[0], -1)],
                dim=1,
            )
            order = candidates.argsort(dim=1, descending=True)[:, :k]
            best_values = candidates.gather(1, order)
            best_indices = indices.gather(1, order)
        return best_values.contiguous(), best_indices.contiguous()

    # -- the streaming run ----------------------------------------------------------------

    def run(
        self,
        token_ids: torch.Tensor,
        *,
        layers: tuple[int, int] | None = None,
        attention_mask: torch.Tensor | None = None,
        initial_streams: torch.Tensor | None = None,
        dump_dir: Path | str | None = None,
        dump_layers: set[int] | None = None,
        dump_keys: Sequence[str] | None = None,
        dump_internals: bool = False,
        dump_dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> RefRunResult:
        """Stream the requested backbone layers on a prompt.

        ``layers`` is an inclusive-exclusive ``(start, stop)`` over
        ``[0, num_hidden_layers)``; default runs the whole backbone. Entering
        streams come from the embedding (all ``hc_mult`` streams equal) unless
        ``initial_streams [B, S, hc_mult, D]`` is given. When ``stop`` reaches
        the backbone depth the final collapse (mean over streams) and final
        RMSNorm run, and ``final_hidden`` is set. Layer ``L`` is dumped to
        ``dump_dir/layer_{L:02d}.pt`` when ``dump_layers`` selects it (default:
        every executed layer); ``dump_keys`` filters the payload (default:
        :data:`DEFAULT_DUMP_KEYS`, plus internals with ``dump_internals``).
        """
        model = self.model
        ids = _as_batch_seqs(token_ids)
        batch_size, seq_len = ids.shape
        start, stop = layers if layers is not None else (0, model.num_hidden_layers)
        if not 0 <= start < stop <= model.num_hidden_layers:
            raise ValueError(f"layers must satisfy 0 <= start < stop <= {model.num_hidden_layers}")

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask = attention_mask.to(torch.bool)

        if initial_streams is None:
            embedded = self.embed_rows(ids, generator=torch.Generator().manual_seed(seed))
            streams = embedded.unsqueeze(2).expand(-1, -1, model.hc_mult, -1).contiguous()
        else:
            streams = initial_streams.to(torch.float32).contiguous()

        keys = tuple(dump_keys) if dump_keys is not None else DEFAULT_DUMP_KEYS
        if dump_internals:
            keys = tuple(dict.fromkeys((*keys, *INTERNAL_DUMP_KEYS)))
        dump_dir_path = Path(dump_dir) if dump_dir is not None else None
        if dump_dir_path is not None:
            dump_dir_path.mkdir(parents=True, exist_ok=True)

        result = RefRunResult(final_hidden=None, streams_out=streams)
        for layer_id in range(start, stop):
            weights = self.load_layer(layer_id, seed=seed)
            streams, boundaries = reference_layer_forward(streams, weights, attention_mask)
            del weights
            result.layer_summaries.append(
                {"layer": layer_id, **{key: float(boundaries[key].abs().max()) for key in ("x_hc_in", "attn_out", "x_hc_mid", "ffn_out", "x_hc_out")}},
            )
            if dump_dir_path is not None and (dump_layers is None or layer_id in dump_layers):
                payload: dict[str, object] = {
                    "layer_id": layer_id,
                    "attention_kind": _attention_kind_name(layer_id, model),
                    "mlp_kind": _mlp_kind_name(layer_id, model),
                }
                payload.update({key: boundaries[key].to(dump_dtype) for key in keys if key in boundaries})
                path = dump_dir_path / f"layer_{layer_id:02d}.pt"
                torch.save(payload, path)
                result.dump_paths[layer_id] = path
            del boundaries

        result.streams_out = streams
        if stop == model.num_hidden_layers:
            collapsed = streams.mean(dim=2)
            result.final_hidden = reference_rms_norm(collapsed, self.final_norm_weight(), model.rms_norm_eps)
            if dump_dir_path is not None:
                result.final_dump_path = dump_dir_path / "final.pt"
                torch.save(
                    {"final_hidden": result.final_hidden.to(dump_dtype), "streams_mean_pre_norm": collapsed.to(dump_dtype), "token_ids": ids, "layer_range": (start, stop), "final_norm_eps": model.rms_norm_eps},
                    result.final_dump_path,
                )
        return result


def _attention_kind_name(layer_id: int, model: Glm53FlashConfig) -> str:
    return model.layer_config(layer_id).attention_kind.value


def _mlp_kind_name(layer_id: int, model: Glm53FlashConfig) -> str:
    return model.layer_config(layer_id).mlp_kind.value


def _test_layers_end_to_end(model: Glm53FlashConfig, tmp_root: Path) -> None:
    torch.manual_seed(31)
    b, s = 2, 17
    driver = PrefillReferenceDriver()  # synthetic mode
    mask = torch.ones(b, s, dtype=torch.bool)

    for layer_id, attention, mlp in (
        (0, "linear_attention", "dense"),
        (4, "linear_attention", "sparse"),
        (3, "deepseek_sparse_attention", "sparse"),
    ):
        weights = driver.load_layer(layer_id)
        assert weights.config.attention_kind.value == attention
        assert weights.config.mlp_kind.value == mlp
        embedded = torch.randn(b, s, model.hc_mult, model.hidden_size) * 0.5
        streams_a, bounds_a = reference_layer_forward(embedded, weights, mask)
        streams_b, _ = reference_layer_forward(embedded, weights, mask)
        assert streams_a.shape == (b, s, model.hc_mult, model.hidden_size)
        assert torch.equal(streams_a, streams_b), f"layer {layer_id} is nondeterministic"
        assert torch.isfinite(streams_a).all()
        for key in DEFAULT_DUMP_KEYS:
            assert key in bounds_a, f"layer {layer_id} dump key {key} missing"
        # Wiring: recompute the final combine from the dumped boundaries.
        recomputed = combine_streams(
            bounds_a["ffn_hc_post"], bounds_a["ffn_hc_comb"], bounds_a["ffn_out"], bounds_a["x_hc_mid"],
        )
        _assert_close(streams_a, recomputed, f"layer {layer_id} combine wiring", atol=1e-4, rtol=1e-4)
        print(f"  layer {layer_id:2d} ({attention}/{mlp}): deterministic, wired, finite")

    # Driver run over layers 0..4 with internals: dump-chain continuity and shapes.
    with tempfile.TemporaryDirectory(dir=str(tmp_root)) as tmp:
        token_ids = torch.randint(0, 1000, (b, s))
        result = driver.run(token_ids, layers=(0, 5), dump_dir=tmp, dump_internals=True)
        assert result.final_hidden is None, "mid-range stop must not produce final hidden"
        assert set(result.dump_paths) == {0, 1, 2, 3, 4}
        payloads = {L: torch.load(path, weights_only=False) for L, path in result.dump_paths.items()}
        for layer_id, payload in payloads.items():
            assert payload["layer_id"] == layer_id
            assert payload["x_hc_in"].shape == (b, s, model.hc_mult, model.hidden_size)
            assert payload["x_hc_out"].shape == (b, s, model.hc_mult, model.hidden_size)
        # Stream continuity: each layer's dumped input is the previous output.
        for layer_id in range(4):
            assert torch.equal(payloads[layer_id]["x_hc_out"], payloads[layer_id + 1]["x_hc_in"])
        assert payloads[3]["mla_topk_indices"].shape[-1] == model.index_topk + model.index_kpool - 1
        assert payloads[0]["kda_core"].shape == (
            b,
            s,
            model.linear_num_heads,
            model.linear_head_dim,
        )
        assert payloads[4]["router_topk_weights"].shape == (b * s, model.num_experts_per_tok)
        assert "router_topk_weights" not in payloads[1]  # layer 1 is a dense-MLP layer

    # Whole-backbone synthetic run (final-norm path + final.pt roundtrip).
    with tempfile.TemporaryDirectory(dir=str(tmp_root)) as tmp:
        result = driver.run(torch.randint(0, 1000, (1, 8)), dump_dir=tmp, dump_layers=set())
        assert result.final_hidden.shape == (1, 8, model.hidden_size)
        assert torch.isfinite(result.final_hidden).all()
        final_dump = torch.load(result.final_dump_path, weights_only=False)
        _assert_close(
            final_dump["final_hidden"].to(torch.float32), result.final_hidden, "final dump roundtrip",
            atol=0.0001,
        )
        manual = reference_rms_norm(result.streams_out.mean(dim=2), driver.final_norm_weight(), model.rms_norm_eps)
        _assert_close(result.final_hidden, manual, "final norm recompute", atol=1e-4)
    print("  driver.run: dump-chain continuity, internal keys, final norm + final.pt")


def _test_scaled_model(model: Glm53FlashConfig) -> None:
    """The mini-checkpoint consumption path: a scaled 4-layer schedule.

    ``layer_ckpt_l0l3`` (layers 0-3, num_hidden_layers=4, first_k_dense_replace=3)
    is the first real single-layer-diff weight source; this exercises the same
    model-override mechanics without needing its files on disk yet.
    """
    scaled = replace(model, num_hidden_layers=4, first_k_dense_replace=3)
    assert scaled.kda_layer_ids == (0, 1, 2) and scaled.dsa_layer_ids == (3,)
    assert scaled.layer_config(3).mlp_kind is MlpKind.SPARSE

    driver = PrefillReferenceDriver(model=scaled)  # synthetic weights, scaled schedule
    with tempfile.TemporaryDirectory() as tmp:
        result = driver.run(torch.randint(0, 1000, (1, 8)), dump_dir=tmp)
        assert len(result.layer_summaries) == 4
        assert result.final_hidden.shape == (1, 8, scaled.hidden_size)
        assert result.dump_paths[3].name == "layer_03.pt"
        payload = torch.load(result.dump_paths[3], weights_only=False)
        assert payload["attention_kind"] == "deepseek_sparse_attention"
        assert payload["mlp_kind"] == "sparse"
    # Layers past the MTP ordinal stay rejected under the scaled model.
    try:
        driver.load_layer(scaled.mtp_layer_id + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("scaled model must reject layers past the MTP ordinal")
    # ...and scaled_model_from_config round-trips a deployment-style config.json.
    with tempfile.TemporaryDirectory() as tmp:
        text = {
            "num_hidden_layers": 4,
            "first_k_dense_replace": 3,
            "hidden_size": model.hidden_size,
            "linear_attn_config": {"num_heads": model.linear_num_heads},
        }
        (Path(tmp) / "config.json").write_text(json.dumps({"text_config": text}))
        derived = scaled_model_from_config(tmp)
        assert derived.num_hidden_layers == 4 and derived.first_k_dense_replace == 3
        assert derived.hidden_size == model.hidden_size and derived.vocab_size == model.vocab_size

    # The MTP layer (layer 4 of the scaled schedule) loads without mHC tensors
    # and refuses the backbone wiring instead of demanding absent hc_* names.
    mtp = driver.load_layer(scaled.mtp_layer_id)
    assert mtp.attn_hc is None and mtp.ffn_hc is None
    assert mtp.config.has_hyper_connections is False
    assert driver.missing_layer_tensors(scaled.mtp_layer_id) == []
    try:
        reference_layer_forward(torch.randn(1, 4, scaled.hc_mult, scaled.hidden_size), mtp)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("MTP layer must reject the hyper-connection wiring")
    print("  scaled model: 4-layer schedule, DSA/MoE layer 3, config.json override, MTP gating")


def _test_lm_head_top1(model: Glm53FlashConfig) -> None:
    """Chunked top-k merge against one direct matmul (fake 20000-row head)."""
    torch.manual_seed(41)
    driver = PrefillReferenceDriver()
    driver._lm_head_cache = torch.randn(20000, model.hidden_size, dtype=torch.bfloat16) * 0.02
    hidden = torch.randn(7, model.hidden_size)
    values, indices = driver.lm_head_top1(hidden, k=3)  # 20000 rows -> 3 merge chunks
    reference = driver._lm_head().to(torch.float32) @ hidden.t()
    top = reference.topk(3, dim=0)
    assert indices.shape == (7, 3)
    assert torch.equal(indices, top.indices.t()), "top-k indices disagree with direct matmul"
    _assert_close(values, top.values.t(), "lm_head top-k merge values", atol=1e-3, rtol=1e-4)
    print("  lm_head top1: chunked merge == direct matmul (k=3, 3 chunks)")


def _real_checkpoint_smoke(root: str) -> None:
    """Against the real (possibly still downloading) W8A8 snapshot."""
    driver = PrefillReferenceDriver(root)
    print(f"  checkpoint opened: {len(driver.checkpoint)} indexed tensors")
    # Dequant one real W8A8 tensor through this module's reader and verify the
    # scale multiply against the raw INT8 payload directly.
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    if name in driver.checkpoint:
        dequant = read_reference_float(driver.checkpoint, name)
        raw = driver.checkpoint.read(name).to(torch.float32)
        scale = driver.checkpoint.read(name[: -len(".weight")] + ".weight_scale").float().reshape(-1)
        _assert_close(dequant, raw * scale[:, None], "real W8A8 dequant", atol=0, rtol=0)
        print(f"  real dequant exact: {name} {tuple(dequant.shape)}")
    for layer_id in (0, 3):
        missing = driver.missing_layer_tensors(layer_id)
        verdict = "loadable" if not missing else f"{len(missing)} tensors pending"
        print(f"  layer {layer_id} real-weight coverage: {verdict}")


def run_self_tests(model: Glm53FlashConfig = FLASH, checkpoint_root: str | None = None) -> None:
    """CPU self-test battery (synthetic weights at the true geometry)."""
    print(
        f"prefill reference self-test: hidden={model.hidden_size} layers={model.num_hidden_layers} heads={model.num_attention_heads}/{model.linear_num_heads} experts={model.n_routed_experts}",
    )
    _test_mhc(model)
    _test_kda_cores(model)
    _test_indexer(model)
    _test_mla(model)
    _test_router_and_moe(model)
    _test_dense_mlp(model)
    _test_lm_head_top1(model)
    _test_layers_end_to_end(model, Path("."))
    _test_scaled_model(model)
    if checkpoint_root is not None:
        print(f"real-checkpoint smoke: {checkpoint_root}")
        _real_checkpoint_smoke(checkpoint_root)
    print("prefill reference self-test: ALL PASS")


def _load_final_norm(checkpoint: Checkpoint) -> torch.Tensor:
    """The backbone final norm weight, resolving both checkpoint root spellings."""
    for name in ("model.language_model.norm.weight", "model.norm.weight"):
        if name in checkpoint:
            return checkpoint.read(name).to(torch.bfloat16).reshape(-1)
    raise ValueError("checkpoint is missing the final norm (model.language_model.norm.weight)")


def _resolve_eos_ids(checkpoint: Checkpoint, tokenizer, tokenizer_config: dict) -> set[int]:
    """Every generation-stop token id the checkpoint declares.

    ``generation_config.json``'s ``eos_token_id`` wins (the GLM-5.3-Flash snapshot
    ships the list ``[154820, 154827, 154829]``); a checkpoint without one falls
    back to the ``tokenizer_config.json`` ``eos_token`` spelling mapped through
    the tokenizer.
    """
    generation_path = checkpoint.root / "generation_config.json"
    value = None
    if generation_path.exists():
        value = json.loads(generation_path.read_text()).get("eos_token_id")
    ids = [int(token_id) for token_id in (value if isinstance(value, list) else [value] if value is not None else [])]
    if not ids:
        eos = tokenizer_config.get("eos_token")
        token_id = None if eos is None else tokenizer.token_to_id(eos["content"] if isinstance(eos, dict) else eos)
        if token_id is None:
            raise ValueError("checkpoint EOS token is missing from its tokenizer")
        ids = [int(token_id)]
    return set(ids)


def _render_prompt(prompt: str, chat_template: str | None) -> str:
    """Wrap one user prompt for encoding: the checkpoint template, else ChatML.

    Rendering follows the transformers convention: ``add_generation_prompt=True``
    lets the template append its assistant header (the GLM-5.3-Flash template
    gates ``<|assistant|>`` on that flag). Any template error falls back to the
    verbatim ChatML wrap.
    """
    template = chat_template if chat_template else DEFAULT_CHAT_TEMPLATE
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        # loopcontrols: the GLM-5.3-Flash template breaks out of loops; without
        # the extension jinja rejects the template and the verbatim fallback
        # would silently mask it.
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"],
        )
        return environment.from_string(template).render(
            messages=[{"role": "user", "content": prompt}], add_generation_prompt=True,
        )
    except Exception as error:  # noqa: BLE001 -- fall back verbatim on any template error
        print("GLM_CHAT_TEMPLATE_FALLBACK", repr(error), flush=True)
        return f"<|user|>\n{prompt}<|assistant|>\n"


class ResidentPrefillSession:
    """Resident weights and request-local caches for chunked GLM forwards."""

    def __init__(
        self,
        checkpoint: Checkpoint,
        programs: dict,
        device_ids: list[int],
        *,
        platform: str = "a2a3",
        run_config: RunConfig | None = None,
        host_num_threads: int | None = None,
        prefix_snapshot_entries: int = 0,
        prefix_snapshot_max_bytes: int = 128 * 1024**2,
        lm_head_backend: str | None = None,
        enable_mtp: bool = False,
    ):
        if host_num_threads is not None and (type(host_num_threads) is not int or host_num_threads <= 0):
            raise ValueError("host_num_threads must be a positive integer or None")
        if type(prefix_snapshot_entries) is not int or prefix_snapshot_entries not in (
            0,
            PREFIX_SNAPSHOT_ENTRIES,
        ):
            raise ValueError("prefix_snapshot_entries must be zero or two")
        required = prefix_snapshot_entries * PREFIX_SNAPSHOT_BYTES
        if type(prefix_snapshot_max_bytes) is not int or prefix_snapshot_max_bytes < required:
            raise ValueError(f"prefix snapshot allocation requires {required} bytes per rank")
        self._prefix_snapshot_entries = prefix_snapshot_entries
        self._snapshot_program = None
        self._snapshot_banks = {}
        self._snapshot_descriptors = [None] * prefix_snapshot_entries
        self._snapshot_generations = [0] * prefix_snapshot_entries
        self._host_num_threads = host_num_threads
        self._previous_host_num_threads = None
        self.checkpoint = checkpoint
        self.auxiliary_programs = programs.get("_auxiliary", {})
        self.programs = {name: value for name, value in programs.items() if not name.startswith("_")}
        programs = self.programs
        self.enable_mtp = bool(enable_mtp)
        self._mtp_lengths = torch.zeros(BATCH, dtype=torch.int32)
        self._mtp_ready = set()
        self._mtp_logits = torch.empty((BATCH, FLASH.vocab_size), dtype=torch.float32)
        self._last_chunk_sequences = None
        self._last_chunk_slots = None
        self.mtp_stats = {"proposed": 0, "accepted": 0, "rejected": 0, "rounds": 0}
        self.device_ids = list(device_ids)
        self.platform = platform
        self.run_config = run_config
        self.runtime = None
        self._lm_head_program = None
        self._device_lm_head = None
        self._device_mtp_head = None
        self._device_final_norm = None
        self._device_logits = None
        self._host_logits_shards = None
        self.lm_head = None
        self.lm_head_backend = (
            os.environ.get("GLM53_LM_HEAD_BACKEND", "npu") if lm_head_backend is None else lm_head_backend
        )
        if self.enable_mtp and self.lm_head_backend != "npu":
            raise ValueError("MTP requires the NPU LM head")
        self._mtp_prepare_program = None
        self._mtp_head_program = None
        self._context = None
        self.calls = 0
        self.guard_events: list[str] = []
        self._seq_lens = torch.zeros(BATCH, dtype=torch.int32)
        self.cache_pages = SharedCachePages(BATCH, MAX_SEQ_LEN, CACHE_TOKENS)
        self._invalid_slots: set[int] = set()
        self._reset_program = None
        self._dispatch_error: BaseException | None = None
        self._collective_dispatches = {program: 0 for program in RT_PROGRAMS if program not in ("seed", "head")}
        plan = production_launch_plan()
        if set(programs) != set(RT_PROGRAMS):
            raise ValueError(
                f"programs must cover exactly the consolidated set {list(RT_PROGRAMS)} "
                f"(got {sorted(programs)})"
            )
        # Signature names come from the entries, never from the compiled artifacts:
        # the persisted param metadata carries post-SSA spellings (x_hc__ssa_v0).
        # Every step dispatches its consolidated program by kind, with the layer
        # riding the dispatch as trailing ordinal scalars.
        arity = {program: _l3_arg_arity(production_program_entry(program)) for program in RT_PROGRAMS}
        step_program = {step: production_program_of_step(step) for step in plan}
        for program, (tensor_names, scalar_names) in arity.items():
            metadata = programs[program]._get_metadata()[0]
            expected = [False] * len(tensor_names) + [True] * len(scalar_names)
            if [info.shape is None for info in metadata] != expected:
                raise ValueError(f"compiled {program} has a stale collective ABI; recompile --runtime-dir")
        # Reject a different rank/capacity compile before loading any payload: the
        # seed entry leads with embedding_rows [RANKS, CAPACITY, D].
        seed_params = programs["seed"]._get_metadata()[0]
        if list(seed_params[0].shape) != [RANKS, CAPACITY, D]:
            raise ValueError(
                f"--runtime-dir must contain glm prefill compiled for "
                f"TP{TP_SIZE}/EP{EP_SIZE} with capacity {CAPACITY}"
            )
        if len(device_ids) != RANKS or len(set(device_ids)) != RANKS:
            raise ValueError(f"prefill needs {RANKS} distinct devices")
        dsa_params = programs["dsa_attn"]._get_metadata()[0]
        dsa_names = arity["dsa_attn"][0]
        expected_cache = [RANKS, DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA]
        if list(dsa_params[dsa_names.index("mla_latent_cache")].shape) != expected_cache:
            raise ValueError("compiled request-cache shape differs; recompile for --max-seq-len")

        self.plan, self.arity, self.step_program = plan, arity, step_program

    @property
    def lm_head_backend(self) -> str:
        """Select full FP32 logits on the NPU or the CPU reference."""
        return self._lm_head_backend

    @lm_head_backend.setter
    def lm_head_backend(self, value: str) -> None:
        if value not in ("npu", "cpu"):
            raise ValueError("lm_head_backend must be 'npu' or 'cpu'")
        if self.runtime is not None:
            if value == "npu" and self._lm_head_program is None:
                raise ValueError("open the session with lm_head_backend='npu' to prepare the device head")
            if value == "cpu" and self.lm_head is None:
                self.lm_head = self.weights.lm_head().to(torch.float32)
        self._lm_head_backend = value

    def __enter__(self):
        if self._context is not None:
            raise RuntimeError("the resident prefill session is already open")
        if self.run_config is not None and self.run_config.enable_chip_swimlane:
            raise ValueError("resident chunk sessions cannot replay two-pass swimlane dispatches")
        self.cache_pages = SharedCachePages(BATCH, MAX_SEQ_LEN, CACHE_TOKENS)
        self._mtp_lengths.zero_()
        self._mtp_ready.clear()
        self._last_chunk_sequences = None
        checkpoint_start = time.monotonic()
        checkpoint, programs, device_ids = self.checkpoint, self.programs, self.device_ids
        platform = self.platform
        plan, arity, step_program = self.plan, self.arity, self.step_program
        weights = ResidentPrefillWeights.load(
            checkpoint, tp_size=TP_SIZE, ep_size=EP_SIZE, model=_prefill_model(), mla_mode="native",
        )
        final_norm_w = _load_final_norm(checkpoint)
        lm_head = weights.lm_head().to(torch.float32) if self.lm_head_backend == "cpu" else None
        print("GLM_LM_HEAD_BACKEND", self.lm_head_backend, flush=True)
        config = self.run_config or RunConfig(
            platform=platform,
            ring_heap=2 * 1024**3,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        )

        self.runtime_started_at = time.monotonic()
        self.checkpoint_setup_seconds = self.runtime_started_at - checkpoint_start
        print("GLM_RESIDENT_BYTES_PER_RANK", weights.bytes_per_rank, flush=True)
        # Auxiliary cache maintenance shares the prepared worker, outside the forward manifest.
        reset_program = self.auxiliary_programs.get("reset") or l3_reset_request_caches.compile(config=config)
        primary, extras = programs["seed"], [programs[name] for name in RT_PROGRAMS[1:]]
        extras.append(reset_program)
        lm_head_program = None
        if self.lm_head_backend == "npu":
            lm_head_program = self.auxiliary_programs.get("head_logits") or l3_prefill_lm_head_logits.compile(
                config=config
            )
            extras.append(lm_head_program)
        snapshot_program = None
        if self._prefix_snapshot_entries:
            snapshot_program = self.auxiliary_programs.get("snapshot") or l3_copy_request_snapshot.compile(
                config=config
            )
            extras.append(snapshot_program)
        mtp_prepare_program = mtp_head_program = None
        if self.enable_mtp:
            mtp_prepare_program = self.auxiliary_programs.get(
                "mtp_prepare"
            ) or l3_prefill_mtp_prepare.compile(config=config)
            mtp_head_program = self.auxiliary_programs.get("mtp_head") or l3_prefill_mtp_head.compile(config=config)
            extras.extend((mtp_prepare_program, mtp_head_program))
        context = primary.prepare(config, extra_compiled=extras, persistent=True, reset_persistent_windows=False)
        runtime = context.__enter__()
        try:
            inputs = prepare_step_inputs(weights, [[] for _ in range(BATCH)], cache_pages=self.cache_pages)
            per_rank = [{} for _ in device_ids]
            for rank in range(RANKS):
                tensors = per_rank[rank]
                for name in GLM_TENSORS:
                    stack = _PARAM_TO_STACK.get(name)
                    if stack is not None:
                        shape, dtype = weights.tensor_layouts[stack]
                        tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                    elif name in CACHE_LAYOUTS:
                        shape, dtype = CACHE_LAYOUTS[name]
                        tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                    elif name in OUTPUTS:
                        shape, dtype = OUTPUTS[name]
                        tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                    elif name in WORKSPACES:
                        shape, dtype = WORKSPACES[name]
                        tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                    else:
                        value = inputs[name]
                        tensors[name] = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)
                # Stacks the graph ABI does not carry (final norm, MTP extras) still need
                # allocation because weights.upload stages every resident stack.
                for stack, (shape, dtype) in weights.tensor_layouts.items():
                    if stack not in _STACK_TO_PARAM and stack not in tensors:
                        tensors[stack] = runtime.alloc_tensor(shape, dtype, worker_id=rank)

            def upload(rank):
                handles = {stack: per_rank[rank][_STACK_TO_PARAM.get(stack, stack)] for stack in weights.tensor_layouts}
                weights.upload(runtime, rank, handles)
                print("GLM_RANK_WEIGHTS_READY", rank, flush=True)

            with ThreadPoolExecutor(max_workers=RANKS) as executor:
                list(executor.map(upload, range(RANKS)))
            stacked = {
                name: StackedDeviceTensor(
                    [rank[name] for rank in per_rank], (RANKS, *per_rank[0][name].shape), tuple(range(RANKS))
                )
                for name in GLM_TENSORS
            }
            runtime.run(
                reset_program, stacked["mla_latent_cache"], stacked["index_raw_cache"],
                stacked["index_pool_cache"], stacked["index_pool_scale"], stacked["kda_conv_state"],
                stacked["kda_recurrent_state"], stacked["cache_page_owners"], (1 << BATCH) - 1, config=config,
            )
            bindings = production_step_bindings(plan)
            for step in plan:
                tensor_names, _ = arity[step_program[step]]
                unbound = [name for name in tensor_names if _alloc_name_of(name, bindings[step]) not in stacked]
                if unbound:
                    raise ValueError(f"step {step} parameters have no allocated tensor: {unbound}")
            for name, (shape, dtype) in weights.tensor_layouts.items():
                if name not in _STACK_TO_PARAM:
                    stacked[name] = StackedDeviceTensor(
                        [rank[name] for rank in per_rank], (RANKS, *shape), tuple(range(RANKS)),
                    )
            if self.enable_mtp:
                mtp_layouts = {
                    "mtp_cached_hidden": ((BATCH, D), torch.bfloat16, 0),
                    "mtp_cached_pools": ((BATCH, KPOOL_SELECT_K), torch.int32, -1),
                    "mtp_cached_valid": ((BATCH, KPOOL_SELECT_K), torch.int32, 0),
                    "mtp_selected_pools": ((CAPACITY, KPOOL_SELECT_K), torch.int32, -1),
                    "mtp_selected_valid": ((CAPACITY, KPOOL_SELECT_K), torch.int32, 0),
                }
                for name, (shape, dtype, fill) in mtp_layouts.items():
                    value = torch.full(shape, fill, dtype=dtype)
                    for rank in range(RANKS):
                        per_rank[rank][name] = runtime.alloc_tensor(shape, dtype, init=value, worker_id=rank)
                    stacked[name] = StackedDeviceTensor(
                        [rank[name] for rank in per_rank], (RANKS, *shape), tuple(range(RANKS)),
                    )
            head_rows = torch.empty((RANKS, LOGIT_ROWS, D), dtype=torch.bfloat16)
            device_lm_head, device_final_norm, device_logits, host_logits_shards = None, None, None, None
            device_mtp_head = None
            if lm_head_program is not None:
                head_weights, mtp_head_weights, head_outputs = [], [], []
                head = weights.lm_head()
                for rank in range(RANKS):
                    shard = torch.zeros((FP32_HEAD_VOCAB_PAD, D), dtype=torch.float32)
                    shard[:LOCAL_VOCAB].copy_(head[rank * LOCAL_VOCAB : (rank + 1) * LOCAL_VOCAB])
                    head_weights.append(runtime.alloc_tensor(shard.shape, shard.dtype, init=shard, worker_id=rank))
                    if self.enable_mtp:
                        shard.zero_()
                        shard[:LOCAL_VOCAB].copy_(weights.mtp_head_shard(rank))
                        mtp_head_weights.append(
                            runtime.alloc_tensor(shard.shape, shard.dtype, init=shard, worker_id=rank),
                        )
                    output_shape = (BATCH, FP32_HEAD_VOCAB_PAD)
                    head_outputs.append(runtime.alloc_tensor(output_shape, torch.float32, worker_id=rank))
                workers = tuple(range(RANKS))
                device_lm_head = StackedDeviceTensor(head_weights, (RANKS, FP32_HEAD_VOCAB_PAD, D), workers)
                if self.enable_mtp:
                    device_mtp_head = StackedDeviceTensor(mtp_head_weights, (RANKS, FP32_HEAD_VOCAB_PAD, D), workers)
                device_final_norm = StackedDeviceTensor(
                    [rank["norm.final_norm"] for rank in per_rank], (RANKS, D), workers,
                )
                device_logits = StackedDeviceTensor(head_outputs, (RANKS, BATCH, FP32_HEAD_VOCAB_PAD), workers)
                host_logits_shards = torch.empty((RANKS, BATCH, FP32_HEAD_VOCAB_PAD), dtype=torch.float32)
                print("GLM_DEVICE_LM_HEAD_BYTES_PER_RANK", FP32_HEAD_VOCAB_PAD * D * 4, flush=True)
        except BaseException:
            context.__exit__(*sys.exc_info())
            raise
        self._context = context
        self.runtime, self.config = runtime, config
        self._lm_head_program = lm_head_program
        self._device_lm_head, self._device_final_norm = device_lm_head, device_final_norm
        self._device_mtp_head = device_mtp_head
        self._device_logits, self._host_logits_shards = device_logits, host_logits_shards
        self._reset_program = reset_program
        self._snapshot_program = snapshot_program
        self._mtp_prepare_program = mtp_prepare_program
        self._mtp_head_program = mtp_head_program
        self._snapshot_generations = [0] * self._prefix_snapshot_entries
        if self._prefix_snapshot_entries:
            try:
                for name, (shape, dtype) in PREFIX_SNAPSHOT_LAYOUTS.items():
                    shards = [runtime.alloc_tensor(shape, dtype, worker_id=rank) for rank in range(RANKS)]
                    self._snapshot_banks[name] = StackedDeviceTensor(shards, (RANKS, *shape), tuple(range(RANKS)))
            except BaseException:
                self.__exit__(*sys.exc_info())
                raise
        self.weights, self.final_norm_w, self.lm_head = weights, final_norm_w, lm_head
        self.per_rank, self.stacked, self.bindings = per_rank, stacked, bindings
        self.head_rows = head_rows
        self._seq_lens.zero_()
        self._invalid_slots.clear()
        self._dispatch_error = None
        for program in self._collective_dispatches:
            self._collective_dispatches[program] = 0
        try:
            if self._host_num_threads is not None:
                self._previous_host_num_threads = torch.get_num_threads()
                torch.set_num_threads(self._host_num_threads)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *exc_info):
        context, self._context = self._context, None
        self.runtime = None
        self._reset_program = None
        self._snapshot_program = None
        self._lm_head_program = None
        self._device_lm_head = None
        self._device_mtp_head = None
        self._device_final_norm = None
        self._device_logits = None
        self._host_logits_shards = None
        self._snapshot_banks.clear()
        self._snapshot_descriptors = [None] * self._prefix_snapshot_entries
        self.lm_head = None
        try:
            if context is not None:
                return context.__exit__(*exc_info)
            return False
        finally:
            self._restore_host_num_threads()

    def _restore_host_num_threads(self) -> None:
        previous, self._previous_host_num_threads = self._previous_host_num_threads, None
        if previous is not None:
            torch.set_num_threads(previous)

    def _require_healthy_runtime(self) -> None:
        """Require an open runtime with valid per-program communication state."""
        if self.runtime is None:
            raise RuntimeError("open the resident session before accessing request state")
        if self._dispatch_error is not None:
            raise RuntimeError("a dispatch failed; close and reopen the resident session") from self._dispatch_error

    def _snapshot_descriptor(self, descriptor):
        if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 3:
            raise ValueError("prefix snapshot descriptor must contain bank, generation and length")
        bank, generation, length = descriptor
        if any(type(value) is not int for value in descriptor):
            raise ValueError("prefix snapshot descriptor fields must be integers")
        if not 0 <= bank < self._prefix_snapshot_entries or generation <= 0:
            raise ValueError("prefix snapshot bank or generation is invalid")
        if not 0 < length <= MAX_SEQ_LEN or length % CHUNK_TILE:
            raise ValueError("prefix snapshot length must preserve the KDA recurrence boundary")
        return bank, generation, length

    def _copy_snapshot(self, slot: int, bank: int, length: int, *, restore: bool) -> None:
        self._require_healthy_runtime()
        if self._snapshot_program is None or type(slot) is not int or not 0 <= slot < BATCH:
            raise ValueError("prefix snapshots are disabled or the request slot is invalid")
        arguments = [self.stacked[name] for name in PREFIX_CACHE_FAMILIES]
        arguments.extend(self._snapshot_banks[name] for name in PREFIX_CACHE_FAMILIES)
        table = self.cache_pages.table.contiguous()
        for rank in range(RANKS):
            self.runtime.copy_to(
                self.per_rank[rank]["block_table"].data_ptr, table.data_ptr(),
                table.numel() * table.element_size(), worker_id=rank,
            )
        try:
            self.runtime.run(
                self._snapshot_program, *arguments, self.stacked["block_table"], length, slot, bank,
                int(restore), config=self.config,
            )
        except BaseException as error:
            self._dispatch_error = error
            raise

    def capture_prefix(self, slot: int, descriptor) -> None:
        """Capture a confirmed aligned prompt boundary into a new generation."""
        self._require_healthy_runtime()
        bank, generation, length = self._snapshot_descriptor(descriptor)
        if type(slot) is not int or not 0 <= slot < BATCH or slot in self._invalid_slots:
            raise ValueError("prefix capture requires a valid active request slot")
        if generation <= self._snapshot_generations[bank] or int(self._seq_lens[slot]) != length:
            raise ValueError("prefix capture generation or committed boundary is stale")
        if self.enable_mtp and int(self._mtp_lengths[slot]) != length:
            raise ValueError("MTP prefix capture requires refreshed target feedback")
        self._snapshot_descriptors[bank] = None
        self._snapshot_generations[bank] = generation
        try:
            self._copy_snapshot(slot, bank, length, restore=False)
        except BaseException:
            self._invalid_slots.add(slot)
            raise
        self._snapshot_descriptors[bank] = (bank, generation, length)

    def restore_prefix(self, slot: int, descriptor) -> None:
        """Restore immutable state into a reset request slot and commit its length."""
        self._require_healthy_runtime()
        bank, generation, length = self._snapshot_descriptor(descriptor)
        if self._snapshot_descriptors[bank] != (bank, generation, length):
            raise ValueError("prefix snapshot generation or length is stale")
        if type(slot) is not int or not 0 <= slot < BATCH or int(self._seq_lens[slot]) != 0:
            raise ValueError("prefix restore requires a reset request slot")
        lengths = self._seq_lens.clone()
        lengths[slot] = length
        self.cache_pages.reserve(lengths)
        self._invalid_slots.add(slot)
        self._copy_snapshot(slot, bank, length, restore=True)
        self._seq_lens[slot] = length
        self._mtp_lengths[slot] = length if self.enable_mtp else 0
        self._mtp_ready.discard(slot)
        self._invalid_slots.discard(slot)

    def reset_requests(self, request_slots: list[int]) -> None:
        """Clear request-owned state and cache rows before releasing or reusing slots."""
        self._require_healthy_runtime()
        slots = list(request_slots)
        if len(set(slots)) != len(slots) or any(slot < 0 or slot >= BATCH for slot in slots):
            raise ValueError(f"request slots must be unique integers in 0..{BATCH - 1}")
        if not slots:
            return
        # Mark requested slots invalid through the distributed clear.
        self._invalid_slots.update(slots)
        slot_mask = sum(1 << slot for slot in slots)
        owners = self.cache_pages.owners.contiguous()
        for rank in range(RANKS):
            self.runtime.copy_to(
                self.per_rank[rank]["cache_page_owners"].data_ptr, owners.data_ptr(),
                owners.numel() * owners.element_size(), worker_id=rank,
            )
        self.runtime.run(
            self._reset_program, self.stacked["mla_latent_cache"], self.stacked["index_raw_cache"],
            self.stacked["index_pool_cache"], self.stacked["index_pool_scale"],
            self.stacked["kda_conv_state"], self.stacked["kda_recurrent_state"],
            self.stacked["cache_page_owners"], slot_mask, config=self.config,
        )
        self._seq_lens[slots] = 0
        self.cache_pages.release(slots)
        self._mtp_lengths[slots] = 0
        self._mtp_ready.difference_update(slots)
        if self.enable_mtp:
            for name, width, dtype, fill in (
                ("mtp_cached_hidden", D, torch.bfloat16, 0),
                ("mtp_cached_pools", KPOOL_SELECT_K, torch.int32, -1),
                ("mtp_cached_valid", KPOOL_SELECT_K, torch.int32, 0),
            ):
                row = torch.full((1, width), fill, dtype=dtype)
                for rank in range(RANKS):
                    for slot in slots:
                        self.runtime.copy_to(
                            self.per_rank[rank][name].data_ptr, row.data_ptr(),
                            row.numel() * row.element_size(),
                            dst_offset=slot * row.numel() * row.element_size(), worker_id=rank,
                        )
        self._invalid_slots.difference_update(slots)

    def forward_chunks(
        self,
        token_sequences: list[list[int]],
        request_slots: list[int],
        kv_seq_lens: list[int] | None = None,
        *,
        visual_features: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Append only new query tokens to the selected persistent request slots."""
        self._require_healthy_runtime()
        slots = list(request_slots)
        if len(token_sequences) != len(slots) or not slots:
            raise ValueError("each nonempty query sequence needs one request slot")
        if len(set(slots)) != len(slots) or any(slot < 0 or slot >= BATCH for slot in slots):
            raise ValueError(f"request slots must be unique integers in 0..{BATCH - 1}")
        if any(not tokens for tokens in token_sequences):
            raise ValueError("active chunk sequences must contain at least one token")
        if self._invalid_slots.intersection(slots):
            raise RuntimeError("reset request slots after a failed dispatch before reusing their cache state")
        previous = self._seq_lens[slots].tolist()
        if kv_seq_lens is not None and list(kv_seq_lens) != previous:
            raise ValueError("chunk prefix lengths disagree with committed request cache state")
        sequences = [[] for _ in range(BATCH)]
        for slot, tokens in zip(slots, token_sequences, strict=True):
            sequences[slot] = list(tokens)
        if visual_features is not None and set(visual_features) - set(slots):
            raise ValueError("visual features must belong to active request slots")
        new_lengths = self._seq_lens + torch.tensor([len(tokens) for tokens in sequences], dtype=torch.int32)
        reservation = self.cache_pages.reserve(new_lengths)
        try:
            inputs = prepare_step_inputs(
                self.weights, sequences, kv_seq_lens=self._seq_lens, visual_features=visual_features,
                cache_pages=self.cache_pages,
            )
        except BaseException:
            self.cache_pages.cancel_reservation(reservation)
            raise
        try:
            logits = self._run_step(inputs)
        except BaseException:
            # Device caches may have advanced even when the host call did not commit its lengths.
            self._invalid_slots.update(slots)
            raise
        for slot, tokens in zip(slots, token_sequences, strict=True):
            self._seq_lens[slot] += len(tokens)
        self._last_chunk_sequences = [list(tokens) for tokens in token_sequences]
        self._last_chunk_slots = slots
        return logits[slots]

    def update_mtp(self, next_token_ids: list[int]) -> torch.Tensor:
        """Refresh the MTP prefix with true target feedback and one shifted token per query row."""
        if not self.enable_mtp or self._last_chunk_sequences is None:
            raise RuntimeError("MTP needs an enabled session and a completed target chunk")
        slots = self._last_chunk_slots
        sequences = self._last_chunk_sequences
        if len(next_token_ids) != len(slots):
            raise ValueError("MTP needs one next token per target request")
        expected = [int(self._seq_lens[slot]) - len(tokens) for slot, tokens in zip(slots, sequences)]
        if self._mtp_lengths[slots].tolist() != expected:
            raise RuntimeError("MTP cache must be refreshed after every target chunk")
        shifted = [tokens[1:] + [int(next_token)] for tokens, next_token in zip(sequences, next_token_ids)]
        logits = self._run_mtp(shifted, slots, recycle=False)
        self._mtp_ready.update(slots)
        self._last_chunk_sequences = None
        return logits

    def propose_mtp(self, request_slots: list[int], count: int) -> dict[int, list[int]]:
        """Draft with one MTP-owned TopK and roll speculative cache visibility back afterward."""
        slots = list(request_slots)
        if not self.enable_mtp or count not in range(1, MTP_SPEC_TOKENS + 1):
            raise ValueError(f"MTP proposal count must be in 1..{MTP_SPEC_TOKENS}")
        if len(set(slots)) != len(slots) or not slots or not set(slots) <= self._mtp_ready:
            raise ValueError("MTP proposal slots need fresh target feedback")
        if bool((self._mtp_lengths[slots] + count - 1 > MAX_SEQ_LEN).any()):
            raise ValueError("MTP proposal exceeds the context limit")
        base_lengths = self._mtp_lengths.clone()
        proposed = {slot: [] for slot in slots}
        for step in range(count):
            next_ids = self._mtp_logits[slots].argmax(dim=-1).tolist()
            for slot, token in zip(slots, next_ids):
                proposed[slot].append(int(token))
            if step + 1 < count:
                self._run_mtp([[int(token)] for token in next_ids], slots, recycle=True)
        # The target verifier consumes only confirmed input tokens. Its KDA states
        # never advance through a rejected draft. Uncommitted MTP payloads remain
        # owned but are invisible until a later true-feedback write replaces them.
        self._mtp_lengths.copy_(base_lengths)
        self._mtp_ready.difference_update(slots)
        self.mtp_stats["rounds"] += 1
        self.mtp_stats["proposed"] += count * len(slots)
        return proposed

    def _run_mtp(self, token_sequences, slots, *, recycle):
        self._require_healthy_runtime()
        sequences = [[] for _ in range(BATCH)]
        for slot, tokens in zip(slots, token_sequences):
            sequences[slot] = list(tokens)
        lengths = self._mtp_lengths + torch.tensor([len(tokens) for tokens in sequences], dtype=torch.int32)
        reservation = self.cache_pages.reserve(lengths)
        try:
            inputs = prepare_step_inputs(
                self.weights, sequences, kv_seq_lens=self._mtp_lengths, cache_pages=self.cache_pages,
                mtp=True,
            )
            for slot in slots:
                if int(self._mtp_lengths[slot]) == 0:
                    row = int(inputs["query_start_loc"][slot])
                    inputs["embedding_rows"][row].zero_()
        except BaseException:
            self.cache_pages.cancel_reservation(reservation)
            raise
        runtime, config, stacked = self.runtime, self.config, self.stacked
        try:
            for rank in range(RANKS):
                for name, value in inputs.items():
                    runtime.copy_to(
                        self.per_rank[rank][name].data_ptr, value.data_ptr(),
                        value.numel() * value.element_size(), worker_id=rank,
                    )
            feedback_norm = stacked["mtp.final_norm"] if recycle else self._device_final_norm
            runtime.run(
                self._mtp_prepare_program, stacked["ws_x_hc_a"], stacked["embedding_rows"], feedback_norm,
                stacked["mtp.enorm"], stacked["mtp.hnorm"], stacked["mtp.eh_proj"], stacked["request_ids"],
                stacked["mtp_cached_hidden"], stacked["mtp_cached_pools"], stacked["mtp_cached_valid"],
                stacked["mtp_selected_pools"], stacked["mtp_selected_valid"], stacked["ws_x_hc_b"],
                int(recycle), config=config,
            )
            for program, overrides in (
                ("dsa_attn", {"x_hc": "ws_x_hc_b", "out_x_hc_mid": "ws_x_hc_a"}),
                ("moe_ffn", {"x_hc_mid": "ws_x_hc_a", "out_x_hc_out": "ws_x_hc_b"}),
                ("head", {"x_hc": "ws_x_hc_b"}),
            ):
                tensor_names, scalar_names = self.arity[program]
                index = self._collective_dispatches.get(program, 0)
                epoch = index + 1
                if epoch > (2**31 - 1) // 2:
                    raise OverflowError("collective epochs exhausted; reopen the session")
                scalars = dict(
                    bb_ord=NUM_LAYERS, dsa_ord=MLA_STACK_LAYERS - 1, moe_ord=MOE_STACK_LAYERS - 1,
                    attn_epoch=epoch, moe_epoch=epoch, ffn_epoch=epoch, skip_topk=int(recycle),
                )
                arguments = []
                for name in tensor_names:
                    allocation = _alloc_name_of(name, overrides)
                    if name in ("selected_pools", "selected_valid"):
                        allocation = "mtp_" + name
                    arguments.append(stacked[allocation])
                arguments.extend(scalars[name] for name in scalar_names)
                runtime.run(self.programs[program], *arguments, config=config)
                if program in self._collective_dispatches:
                    self._collective_dispatches[program] = epoch
            runtime.run(
                self._mtp_head_program, stacked["head_hidden"], stacked["mtp.final_norm"],
                self._device_mtp_head, stacked["mtp_selected_pools"], stacked["mtp_selected_valid"],
                stacked["query_start_loc"], stacked["mtp_cached_hidden"], stacked["mtp_cached_pools"],
                stacked["mtp_cached_valid"], self._device_logits, config=config,
            )
            runtime.copy_stacked_from(self._device_logits, self._host_logits_shards)
            logits = torch.cat(tuple((shard[:, :LOCAL_VOCAB] for shard in self._host_logits_shards.unbind(0))), dim=1)
            if not torch.isfinite(logits[slots]).all():
                raise FloatingPointError("MTP produced nonfinite logits")
        except BaseException as error:
            self._dispatch_error = error
            self._invalid_slots.update(slots)
            raise
        self._mtp_lengths.copy_(lengths)
        self._mtp_logits[slots] = logits[slots]
        return logits[slots]

    def forward(self, sequences: list[list[int]]) -> torch.Tensor:
        """Run an independent full-prefix batch through the chunk-capable session."""
        if len(sequences) != BATCH or any(not tokens for tokens in sequences):
            raise ValueError(f"full-prefix forward requires {BATCH} nonempty sequences")
        self.reset_requests(list(range(BATCH)))
        return self.forward_chunks(sequences, list(range(BATCH)), [0] * BATCH)

    def _run_step(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Upload this query batch and dispatch the full backbone once."""
        self._require_healthy_runtime()
        runtime, config, programs = self.runtime, self.config, self.programs
        plan, arity, step_program = self.plan, self.arity, self.step_program
        per_rank, stacked, bindings = self.per_rank, self.stacked, self.bindings
        head_rows, guard_events = self.head_rows, self.guard_events
        final_norm_w, lm_head = self.final_norm_w, self.lm_head
        step = self.calls
        for rank in range(RANKS):
            for name, value in inputs.items():
                if name in CONSTANT_STEP_INPUTS or name in _PARAM_TO_STACK:
                    continue
                runtime.copy_to(
                    per_rank[rank][name].data_ptr, value.data_ptr(), value.numel() * value.element_size(),
                    worker_id=rank,
                )
        self.last_dispatch_started_at = time.monotonic()
        print("GLM_DEVICE_START", self.calls, flush=True)
        pending = deque()
        next_dispatches = dict(self._collective_dispatches)

        def complete_oldest():
            handle, key, dispatch_index = pending[0]
            handle.result()
            pending.popleft()
            if key in self._collective_dispatches:
                self._collective_dispatches[key] = dispatch_index + 1

        try:
            for launch in plan:
                if len(pending) == 2:
                    complete_oldest()
                overrides = bindings[launch]
                key = step_program[launch]
                tensor_names, scalar_names = arity[key]
                if TRACE_DISPATCH:
                    label = f"{_step_key(launch)} {key}"
                    print("GLM_DISPATCH", label, flush=True)
                dispatch_index = next_dispatches.get(key, 0)
                ords = production_step_ordinals(launch, dispatch_index=dispatch_index)
                arguments = [stacked[_alloc_name_of(name, overrides)] for name in tensor_names]
                arguments.extend(ords[name] for name in scalar_names)
                handle = runtime.submit(programs[key], *arguments, config=config)
                pending.append((handle, key, dispatch_index))
                if key in next_dispatches:
                    next_dispatches[key] = dispatch_index + 1
                if VERIFY_X_H_C and launch != ("head",):
                    while pending:
                        complete_oldest()
                    written = overrides.get("out_x_hc_out", overrides.get("out_x_hc_mid", overrides.get("x_hc")))
                    step_rows = torch.empty((RANKS, CAPACITY, HC_MULT, D), dtype=torch.bfloat16)
                    runtime.copy_stacked_from(stacked[written], step_rows)
                    finite_step = bool(torch.isfinite(step_rows.to(torch.float32)).all())
                    print(
                        "GLM_XHC_STEP", _step_key(launch), written, "finite=", finite_step, "absmax=",
                        float(step_rows.to(torch.float32).abs().max()), flush=True,
                    )
                    if not finite_step:
                        guard_events.append(f"nonfinite {written} after {_step_key(launch)}")
                        raise FloatingPointError(f"nonfinite {written} after {_step_key(launch)}")
            while pending:
                complete_oldest()
            if self.lm_head_backend == "npu":
                runtime.run(
                    self._lm_head_program, stacked["head_hidden"], self._device_final_norm,
                    self._device_lm_head, self._device_logits, config=config,
                )
                runtime.copy_stacked_from(self._device_logits, self._host_logits_shards)
            runtime.copy_stacked_from(stacked["head_hidden"], head_rows)
        except BaseException as error:
            self._dispatch_error = error
            for handle, _, _ in pending:
                try:
                    handle.result()
                except BaseException:
                    pass
            raise
        if VERIFY_X_H_C:
            print(
                "GLM_HEAD_HIDDEN finite=", bool(torch.isfinite(head_rows).all()), "absmax=",
                float(head_rows.to(torch.float32).abs().max()), flush=True,
            )
        if not bool(torch.isfinite(head_rows).all()):
            guard_events.append(f"nonfinite head_hidden at step {step}")
            raise FloatingPointError(f"nonfinite head_hidden at step {step}")
        if VERIFY_X_H_C:
            # The head's x_hc binding IS the last FFN launch's out_x_hc_out
            # (the ping-pong returns the stream to its entry side), so this
            # readback is the direct out_x_hc_out finiteness evidence.
            stream_name = bindings[("head",)]["x_hc"]
            stream_rows = torch.empty((RANKS, CAPACITY, HC_MULT, D), dtype=torch.bfloat16)
            runtime.copy_stacked_from(stacked[stream_name], stream_rows)
            finite_stream = bool(torch.isfinite(stream_rows.to(torch.float32)).all())
            print(
                "GLM_XHC_STREAM", stream_name, "finite=", finite_stream, "absmax=",
                float(stream_rows.to(torch.float32).abs().max()), flush=True,
            )
            if not finite_stream:
                guard_events.append(f"nonfinite {stream_name} at step {step}")
                raise FloatingPointError(f"nonfinite {stream_name} at step {step}")
        if self.lm_head_backend == "npu":
            shards = (shard[:, :LOCAL_VOCAB] for shard in self._host_logits_shards.unbind(dim=0))
            logits = torch.cat(tuple(shards), dim=1)
        else:
            # All ranks hold the same restored stream; rank 0's rows are the answer.
            logits = reference_lm_head_logits(head_rows[0, :BATCH], final_norm_w, lm_head)
        if not bool(torch.isfinite(logits[:BATCH]).all()):
            guard_events.append(f"nonfinite logits at step {step}")
            raise FloatingPointError(f"nonfinite last-token logits at step {step}")
        self.calls += 1
        return logits[:BATCH]


def generate(
    checkpoint: Checkpoint,
    programs: dict,
    device_ids: list[int],
    prompts: list[str],
    max_new_tokens: int,
    *,
    raw_prompts: bool = False,
    platform: str = "a2a3",
    expect_tokens: dict[int, list[int]] | None = None,
    dump_path: Path | None = None,
    visual_inputs: dict[int, list] | None = None,
    spec_tokens: int = 0,
):
    """Generate incrementally with resident caches and optional MTP verification.

    Report legacy ``seconds`` from runtime preparation and step ``seconds`` from
    dispatch, plus separate checkpoint/input setup and end-to-end wall times.
    """
    end_to_end_start = time.monotonic()
    from tokenizers import Tokenizer

    if type(spec_tokens) is not int or spec_tokens not in range(MTP_SPEC_TOKENS + 1):
        raise ValueError(f"spec_tokens must be in 0..{MTP_SPEC_TOKENS}")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if not 1 <= len(prompts) <= BATCH:
        raise ValueError(f"provide between 1 and {BATCH} prompts")

    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    tokenizer_config_path = checkpoint.root / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text()) if tokenizer_config_path.exists() else {}
    # The transformers convention: an inline tokenizer_config chat_template wins,
    # else the checkpoint's own chat_template.jinja beside the weights (the
    # GLM-5.3-Flash snapshot ships the file and no inline field).
    chat_template = tokenizer_config.get("chat_template")
    if not chat_template:
        template_path = checkpoint.root / "chat_template.jinja"
        if template_path.exists():
            chat_template = template_path.read_text()
    eos_ids = _resolve_eos_ids(checkpoint, tokenizer, tokenizer_config)

    visual_features = {}
    if visual_inputs:

        prompts = list(prompts)
        packed_visuals = []
        for request, visuals in visual_inputs.items():
            if type(request) is not int or request not in range(len(prompts)) or not visuals:
                raise ValueError("visual inputs need a nonempty list for a valid prompt index")
            for visual in visuals:
                if not isinstance(visual, VisualInput):
                    raise ValueError("visual inputs must be produced by VisionProcessor")
                packed_visuals.append((request, visual))
            remainder = prompts[request]
            expanded = []
            if "<|image|>" not in remainder and "<|video|>" not in remainder:
                prompts[request] = "\n".join(visual.prompt_fragment for visual in visuals) + "\n" + remainder
            else:
                for visual in visuals:
                    candidates = [
                        (remainder.find(marker), marker)
                        for marker in ("<|image|>", "<|video|>")
                        if marker in remainder
                    ]
                    if not candidates or min(candidates)[1] != visual.marker:
                        raise ValueError("prompt visual markers must match the supplied visual input order")
                    before, remainder = remainder.split(visual.marker, 1)
                    expanded.extend((before, visual.prompt_fragment))
                if "<|image|>" in remainder or "<|video|>" in remainder:
                    raise ValueError("each prompt visual marker needs one visual input")
                prompts[request] = "".join(expanded) + remainder
        pixels = torch.cat([visual.pixel_values for _, visual in packed_visuals])
        grid = torch.cat([visual.grid_thw for _, visual in packed_visuals])
        encoder = VisionEncoder(checkpoint, device_ids[0], platform=platform, compiled=programs.get("_vision"))
        encoded = encoder(pixels, grid)
        offset = 0
        for request, visual in packed_visuals:
            count = int(visual.grid_thw.prod(dim=-1).sum()) // 4
            part = encoded[offset : offset + count]
            visual_features[request] = (
                torch.cat((visual_features[request], part)) if request in visual_features else part
            )
            offset += count
        del encoder

    sequences = [
        tokenizer.encode(
            prompt if raw_prompts else _render_prompt(prompt, chat_template),
            add_special_tokens=False,
        ).ids
        for prompt in prompts
    ]
    if any(not sequence for sequence in sequences):
        raise ValueError("prompts must contain at least one token")
    if any(len(sequence) + max_new_tokens > MAX_SEQ_LEN for sequence in sequences):
        raise ValueError(f"prompt and generation exceed the compiled context limit {MAX_SEQ_LEN}")

    generated: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    step_log: list[dict] = []
    with ResidentPrefillSession(
        checkpoint, programs, device_ids, platform=platform, enable_mtp=bool(spec_tokens)
    ) as session:
        start = session.runtime_started_at
        logits = torch.empty(len(prompts), FLASH.vocab_size, dtype=torch.float32)
        remaining = [list(sequence) for sequence in sequences]
        feature_offsets = [0] * len(prompts)
        draft_queues = [[] for _ in prompts]
        for step in range(max_new_tokens):
            tick = time.monotonic()
            dispatch_seconds = input_setup_seconds = 0.0
            while any(remaining):
                budget, slots, chunks, features = CAPACITY, [], [], {}
                for request, tokens in enumerate(remaining):
                    if not tokens or budget == 0:
                        continue
                    count = min(len(tokens), budget)
                    if count < len(tokens):
                        count -= count % CHUNK_TILE
                    if count == 0:
                        continue
                    chunk, remaining[request] = tokens[:count], tokens[count:]
                    slots.append(request)
                    chunks.append(chunk)
                    if request in visual_features:

                        count_visual = chunk.count(IMAGE_TOKEN_ID)
                        begin = feature_offsets[request]
                        if count_visual:
                            features[request] = visual_features[request][begin : begin + count_visual]
                            feature_offsets[request] += count_visual
                    budget -= count
                dispatch_start = time.monotonic()
                partial = session.forward_chunks(chunks, slots, visual_features=features)
                dispatch_seconds += time.monotonic() - session.last_dispatch_started_at
                input_setup_seconds += session.last_dispatch_started_at - dispatch_start
                logits[slots] = partial
                if spec_tokens:
                    next_tokens = [
                        remaining[slot][0] if remaining[slot] else int(partial[index].argmax())
                        for index, slot in enumerate(slots)
                    ]
                    session.update_mtp(next_tokens)
            next_ids = logits.argmax(dim=-1).tolist()
            for request, token in enumerate(next_ids):
                if not finished[request]:
                    if draft_queues[request]:
                        candidate = draft_queues[request].pop(0)
                        if candidate == int(token):
                            session.mtp_stats["accepted"] += 1
                        else:
                            session.mtp_stats["rejected"] += 1
                            draft_queues[request].clear()
                    generated[request].append(int(token))
                    sequences[request].append(int(token))
                    finished[request] = int(token) in eos_ids
                    if not finished[request]:
                        remaining[request] = [int(token)]
            if spec_tokens and step + 1 < max_new_tokens:
                draft_slots = [
                    request
                    for request in range(len(prompts))
                    if not finished[request] and not draft_queues[request]
                ]
                if draft_slots:
                    count = min(spec_tokens, max_new_tokens - step - 1)
                    for slot, proposed in session.propose_mtp(draft_slots, count).items():
                        draft_queues[slot] = proposed
            step_end = time.monotonic()
            step_log.append(
                {
                    "step": step,
                    "seconds": dispatch_seconds,
                    "input_setup_seconds": input_setup_seconds,
                    "end_to_end_seconds": step_end - tick,
                    "token_ids": next_ids,
                    "text": [tokenizer.decode(tokens, skip_special_tokens=False) for tokens in generated],
                },
            )
            print("GLM_GENERATION_STEP", json.dumps(step_log[-1], ensure_ascii=False), flush=True)
            if all(finished):
                break
        resident_bytes_per_rank = session.weights.bytes_per_rank
        guard_events = session.guard_events

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(generated, indent=2) + "\n")
    if expect_tokens:
        for request, expected in expect_tokens.items():
            produced = generated[request][: len(expected)]
            if produced != list(expected):
                raise AssertionError(f"prompt {request}: expected top-1 prefix {list(expected)}, got {produced}")

    def _truncate_at_eos(tokens: list[int]) -> list[int]:
        stops = [index for index, token in enumerate(tokens) if token in eos_ids]
        return tokens[: stops[0]] if stops else tokens

    return {
        "platform": "a2a3",
        "tp": TP_SIZE,
        "ep": EP_SIZE,
        "dp": DP_SIZE,
        "precision": "official_bf16_w8a8",
        "checkpoint": str(checkpoint.root),
        "capacity": CAPACITY,
        "max_seq_len": MAX_SEQ_LEN,
        "cache_tokens": CACHE_TOKENS,
        "full_prefix_recomputation": False,
        "mtp": {"tokens": spec_tokens, "verification": "sequential_greedy", **session.mtp_stats},
        "layers_per_forward": NUM_LAYERS,
        "launch_mode": "rt",
        "device_launches_per_rank_per_forward": len(session.plan),
        "resident_bytes_per_rank": resident_bytes_per_rank,
        "seconds": time.monotonic() - start,
        "steps": step_log,
        "guard_events": guard_events,
        "checkpoint_setup_seconds": session.checkpoint_setup_seconds,
        "end_to_end_seconds": time.monotonic() - end_to_end_start,
        "cases": [
            {
                "prompt": prompt,
                "token_ids": _truncate_at_eos(tokens),
                "text": tokenizer.decode(_truncate_at_eos(tokens), skip_special_tokens=False),
                "eos_reached": done,
            }
            for prompt, tokens, done in zip(prompts, generated, finished)
        ],
    }


def run_weight_checks(argv=None):
    """CPU golden: layout, fusion and sharding against direct slicing.

    Host-side only, but the a2a3 sweep invokes every model file as
    ``python <file> -p a2a3 -d <ids>``, so the same switches are accepted and
    ignored. ``--tp`` / ``--ep`` are read by ``config.py`` at import time.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=str, default="0")
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--ep", type=int, default=16)
    parser.parse_args(argv)
    torch.manual_seed(23)

    def pieces(rows: int, width: int) -> tuple[torch.Tensor, ...]:
        """One FFN's canonical INT8 weights and per-channel scales."""
        return (
            torch.randint(-127, 128, (rows, width), dtype=torch.int8),
            torch.randint(-127, 128, (rows, width), dtype=torch.int8),
            torch.randint(-127, 128, (width, rows), dtype=torch.int8),
            torch.rand(rows, 1) * 1e-2,
            torch.rand(rows, 1) * 1e-2,
            torch.rand(width, 1) * 1e-2,
        )

    routed_global, routed_ep, routed_local = 32, 2, 16
    routed_pieces = {expert: pieces(2048, D) for expert in range(routed_global)}
    shared_pieces = pieces(2048, D)
    dense_pieces = pieces(DENSE_INTER, D)
    router_weight = torch.randn(N_EXPERTS, D, dtype=torch.bfloat16)
    router_bias = torch.randn(N_EXPERTS)

    def materialize(fused: bool, root: str = TEXT_LAYER_ROOTS[0]) -> dict[str, torch.Tensor]:
        """Write one canonical weight set out in either checkpoint spelling."""

        def prefix(layer: int) -> str:
            return f"{root}{layer}.mlp."

        out: dict[str, torch.Tensor] = {
            prefix(0) + "gate.weight": router_weight,
            prefix(0) + "gate.e_score_correction_bias": router_bias,
            prefix(3) + "gate.weight": router_weight,
            prefix(3) + "gate.e_score_correction_bias": router_bias,
        }

        def put(prefix: str, base: str, piece: tuple[torch.Tensor, ...]) -> None:
            gate, up, down, gate_scale, up_scale, down_scale = piece
            if fused:
                out[prefix + base + "w_gate_up"] = torch.cat([gate, up], dim=0)
                out[prefix + base + "w_gate_up.weight_scale"] = torch.cat([gate_scale, up_scale], dim=0)
                out[prefix + base + "w_down"] = down
                out[prefix + base + "w_down.weight_scale"] = down_scale
            else:
                out[prefix + base + "gate_proj.weight"] = gate
                out[prefix + base + "gate_proj.weight_scale"] = gate_scale
                out[prefix + base + "up_proj.weight"] = up
                out[prefix + base + "up_proj.weight_scale"] = up_scale
                out[prefix + base + "down_proj.weight"] = down
                out[prefix + base + "down_proj.weight_scale"] = down_scale

        for expert, piece in routed_pieces.items():
            put(prefix(3), f"experts.{expert}.", piece)
        put(prefix(3), "shared_experts.", shared_pieces)
        put(prefix(0), "", dense_pieces)
        return out

    sparse_fused = materialize(fused=True)
    sparse_separate = materialize(fused=False)

    def assert_close(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
        if a.shape != b.shape:
            raise AssertionError(f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}")
        if not torch.equal(a, b):
            raise AssertionError(f"{what}: values differ")

    # Fusing separate gate/up must reproduce the fused checkpoint bit-exactly,
    # and EP must pick whole experts in the kernel's [local, 2*inter, D] layout.
    for ep_rank in (0, 1):
        shard_args = dict(local_experts=routed_local, global_experts=routed_global)
        separate = shard_routed_experts(sparse_separate, 3, ep_rank, routed_ep, **shard_args)
        fused = shard_routed_experts(sparse_fused, 3, ep_rank, routed_ep, **shard_args)
        assert_close(separate.w_gate_up, fused.w_gate_up, f"ep{ep_rank} routed gate_up")
        assert_close(separate.w_gate_up_scale, fused.w_gate_up_scale, f"ep{ep_rank} routed gate_up_scale")
        assert_close(separate.w_down, fused.w_down, f"ep{ep_rank} routed down")
        assert_close(separate.w_down_scale, fused.w_down_scale, f"ep{ep_rank} routed down_scale")
        expected = torch.stack(
            [torch.cat([routed_pieces[e][0], routed_pieces[e][1]], dim=0) for e in range(ep_rank * routed_local, (ep_rank + 1) * routed_local)],
        )
        assert_close(separate.w_gate_up, expected, f"ep{ep_rank} routed layout")
        assert separate.w_gate_up.dtype == torch.int8
        assert separate.w_gate_up_scale.dtype == torch.float32
        assert separate.w_gate_up_scale.shape == (routed_local, 4096)

    # Shared expert: every EP rank keeps complete projections and scales.
    shared = load_shared_expert(sparse_fused, 3)
    shared_separate = load_shared_expert(sparse_separate, 3)
    inter = FLASH.moe_intermediate_size
    assert_close(shared.gate_up[:inter], shared_pieces[0], "shared gate")
    assert_close(shared.gate_up[inter:], shared_pieces[1], "shared up")
    assert_close(shared.down, shared_pieces[2], "shared down")
    assert_close(
        shared.gate_up_scale, torch.cat([shared_pieces[3].reshape(-1), shared_pieces[4].reshape(-1)]),
        "shared gate/up scale",
    )
    assert_close(shared.down_scale, shared_pieces[5].reshape(-1), "shared down scale")
    for field_name in ("gate_up", "gate_up_scale", "down", "down_scale"):
        assert_close(getattr(shared_separate, field_name), getattr(shared, field_name), f"shared fused {field_name}")

    # Dense layer: TP slices the intermediate width, and the router passes through
    # with the kernels' dtypes.
    dense = shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE)
    local_dense = DENSE_INTER // TP_SIZE
    assert dense.gate_up.shape == (2 * local_dense, D)
    assert dense.down.shape == (D, local_dense)
    assert dense.gate_up_scale.shape == (2 * local_dense,)
    assert dense.down_scale.shape == (D,)
    assert_close(dense.gate_up[:local_dense], dense_pieces[0][:local_dense], "dense gate slice")
    router = layer_router(sparse_fused, 0)
    assert router.weight.dtype == torch.bfloat16 and router.weight.shape == (N_EXPERTS, D)
    assert router.correction_bias.dtype == torch.float32 and router.correction_bias.shape == (N_EXPERTS,)
    assert_close(router.correction_bias, router_bias, "router bias")

    # The released checkpoint nests the text backbone under ``language_model``
    # (``model.language_model.layers.*``); both roots load the same tensors.
    nested = materialize(fused=True, root="model.language_model.layers.")
    ep_rank = 1
    nested_routed = shard_routed_experts(
        nested, 3, ep_rank, routed_ep, local_experts=routed_local, global_experts=routed_global,
    )
    canonical_routed = shard_routed_experts(
        sparse_fused, 3, ep_rank, routed_ep, local_experts=routed_local, global_experts=routed_global,
    )
    for field_name in ("w_gate_up", "w_gate_up_scale", "w_down", "w_down_scale"):
        assert_close(getattr(nested_routed, field_name), getattr(canonical_routed, field_name), f"nested routed {field_name}")
    nested_shared = load_shared_expert(nested, 3)
    assert_close(nested_shared.gate_up, shared.gate_up, "nested shared gate_up")
    assert_close(nested_shared.down, shared.down, "nested shared down")
    nested_dense = shard_dense_mlp(nested, 0, 0, TP_SIZE)
    assert_close(nested_dense.gate_up, dense.gate_up, "nested dense gate_up")
    nested_router = layer_router(nested, 3)
    canonical_router = layer_router(sparse_fused, 3)
    assert_close(nested_router.weight, canonical_router.weight, "nested router weight")
    assert_close(nested_router.correction_bias, canonical_router.correction_bias, "nested router bias")
    try:
        load_shared_expert(materialize(fused=True, root="model.text."), 3)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown layer root should have been rejected")

    # Boundary validation: the one shipped scheme, and a scale is mandatory.
    QuantDescription.parse({"model_quant_type": "W8A8_DYNAMIC", "group_size": 0, "is_rot_used": True}).require_dynamic_symmetric(
    )
    for bad in (
        {"model_quant_type": "FP8", "group_size": 0},
        {"model_quant_type": "W8A8_DYNAMIC", "group_size": 128},
    ):
        try:
            QuantDescription.parse(bad).require_dynamic_symmetric()
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should have been rejected")
    try:
        check_ffn_scales(sparse_fused, [layer_prefix(0) + "missing.weight"])
    except KeyError:
        pass
    else:
        raise AssertionError("a missing scale should have been rejected")

    # Shard geometry is validated rather than silently slicing short.
    for bad_call in (
        lambda: shard_dense_mlp(sparse_fused, 0, -1, TP_SIZE),
        lambda: shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE, intermediate=DENSE_INTER + 32),
        lambda: shard_dense_mlp(sparse_fused, 0, 0, TP_SIZE, local_inter=DENSE_INTER // TP_SIZE + 1),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid shard geometry should have been rejected")

    # The .safetensors branch of ``load_tensors`` is the one loader path with no
    # coverage when the optional package is absent; round-trip a small shard
    # through it whenever the package is present.
    try:
        from safetensors.torch import save_file  # noqa: PLC0415
    except ImportError:
        print("[GOLDEN] skip  weights safetensors reader (safetensors not installed)")
    else:
        import tempfile  # noqa: PLC0415

        probe = {
            "probe.weight": torch.arange(24, dtype=torch.int8).reshape(4, 6),
            "probe.weight_scale": torch.linspace(0.1, 1.0, 4).reshape(4, 1),
        }
        with tempfile.TemporaryDirectory() as tmp:
            save_file(probe, str(Path(tmp) / "shard-0.safetensors"))
            loaded = load_tensors(tmp)
            if not torch.equal(loaded["probe.weight"], probe["probe.weight"]):
                raise AssertionError("safetensors reader lost the weight payload")
            if not torch.equal(_scale_of(loaded, "probe.weight"), probe["probe.weight_scale"].reshape(-1)):
                raise AssertionError("safetensors reader did not flatten the [out, 1] scale")
        print("[GOLDEN] PASS weights (safetensors round-trip)")

    print(f"[GOLDEN] PASS weights (routed 32 experts over EP{routed_ep}, shared replicated, dense TP{TP_SIZE})")

    # ------------------- resident prefill staging (in-memory fixture) -------------------
    # Everything below runs against a scaled-geometry synthetic checkpoint held
    # in memory through the staging read protocol: never /data/models, never a
    # device, and never the optional safetensors package. ``Checkpoint`` parsing
    # and metadata validation are covered by the loader tests above; these cases
    # cover the staging itself - resolution, fusion, sharding, splitting,
    # stacking and the staged upload.
    import ctypes  # noqa: PLC0415

    TINY = Glm53FlashConfig(
        name="glm5_3_flash_tiny", vocab_size=128, hidden_size=64, num_hidden_layers=4,
        first_k_dense_replace=1, intermediate_size=32, rms_norm_eps=1e-05, max_position_embeddings=1024,
        swiglu_limit=10.0, tie_word_embeddings=False, num_attention_heads=4, q_lora_rank=12, kv_lora_rank=6,
        qk_nope_head_dim=8, qk_rope_head_dim=0, v_head_dim=8, index_n_heads=2, index_head_dim=8, index_topk=4,
        index_kpool=2, index_kpool_compress=True, index_kpool_always_select_tail=True,
        index_share_for_mtp_iteration=True, linear_num_heads=4, linear_head_dim=8, linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0, n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=2,
        moe_intermediate_size=32, scoring_func="sigmoid", topk_method="noaux_tc", norm_topk_prob=True,
        routed_scaling_factor=2.5, n_group=1, topk_group=1, mhc=True, hc_mult=2, hc_sinkhorn_iters=20,
        hc_eps=1e-06, num_nextn_predict_layers=1, dtype="bfloat16", quant_method="w8a8",
        activation_scheme="dynamic",
    )

    def tiny_root(layer: int) -> str:
        """The fixture's dual-spelling rule: layers 0-1 canonical, 2-4 nested."""
        return "model.layers." if layer <= 1 else "model.language_model.layers."

    class _FixtureWeights(Mapping[str, torch.Tensor]):
        """In-memory stand-in for :class:`Checkpoint` (the staging read protocol)."""

        def __init__(self, tensors: dict[str, torch.Tensor], labels: dict[str, str]):
            self.tensors, self.description = tensors, labels

        def __contains__(self, name: object) -> bool:
            return name in self.tensors

        def __iter__(self):
            return iter(self.tensors)

        def __len__(self) -> int:
            return len(self.tensors)

        def __getitem__(self, name: str) -> torch.Tensor:
            return self.tensors[name]

        def read(self, name: str, selection=None) -> torch.Tensor:
            return self.tensors[name] if selection is None else self.tensors[name][selection]

        def shape(self, name: str) -> tuple[int, ...]:
            return tuple(self.tensors[name].shape)

        def gather(self, name: str, rows: torch.Tensor) -> torch.Tensor:
            return self.tensors[name][rows]

    def build_tiny_fixture(mutate=None) -> _FixtureWeights:
        """Materialize the scaled-geometry deployment checkpoint the tests run against.

        Covers both layer-root spellings, nested and flat KDA-gate/mHC spellings,
        fused and separate gate/up layouts, the quantized MLA projections with
        scales and zero offsets, and the MTP duplicates of the host tables.
        ``mutate(tensors, labels)`` exists for the negative tests.
        """
        generator = torch.Generator().manual_seed(101)
        hidden = TINY.hidden_size
        qkv = TINY.linear_num_heads * TINY.linear_head_dim
        tensors: dict[str, torch.Tensor] = {}
        labels: dict[str, str] = {}

        def put(name: str, tensor: torch.Tensor, label: str = "FLOAT") -> None:
            tensors[name] = tensor
            labels[name] = label

        def bf16(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator).bfloat16()

        def int8_linear(stem: str, out: int, inn: int, *, bare: bool = False) -> None:
            name = stem if bare else stem + ".weight"
            weight = torch.randint(-127, 128, (out, inn), dtype=torch.int8, generator=generator)
            scale = torch.rand(out, 1, generator=generator).float() * 1e-2
            put(name, weight, "W8A8_DYNAMIC")
            put(stem + ".weight_scale", scale, "W8A8_DYNAMIC")
            put(stem + ".weight_offset", torch.zeros(out, 1), "W8A8_DYNAMIC")

        def ffn(base: str, inter: int, *, fused: bool, dotted: bool = False) -> None:
            # The canonical fused spelling is bare (``w_gate_up``); only the
            # routed expert entries also accept the ``.weight`` variant.
            if fused:
                int8_linear(base + "w_gate_up", 2 * inter, hidden, bare=not dotted)
                int8_linear(base + "w_down", hidden, inter, bare=not dotted)
            else:
                for part in ("gate", "up"):
                    int8_linear(base + part + "_proj", inter, hidden)
                int8_linear(base + "down_proj", hidden, inter)

        for layer in range(TINY.num_checkpoint_layers):
            base = tiny_root(layer) + f"{layer}."
            if layer in TINY.kda_layer_ids:
                attn = base + "self_attn."
                for part in ("q", "k", "v"):
                    put(attn + f"{part}_proj.weight", bf16(qkv, hidden))
                    put(attn + f"{part}_conv1d.weight", bf16(qkv, 1, TINY.linear_conv_kernel_dim))
                gates = attn if layer == 0 else attn + "forget_gate."  # flat vs nested spelling
                put(gates + "dt_bias", torch.randn(qkv, generator=generator).float())
                put(gates + "A_log", torch.randn(TINY.linear_num_heads, generator=generator).float())
                put(gates + "f_a_proj.weight", bf16(TINY.linear_head_dim, hidden))
                put(gates + "f_b_proj.weight", bf16(qkv, TINY.linear_head_dim))
                put(attn + "g_a_proj.weight", bf16(TINY.linear_head_dim, hidden))
                put(attn + "g_b_proj.weight", bf16(qkv, TINY.linear_head_dim))
                put(attn + "b_proj.weight", bf16(TINY.linear_num_heads, hidden))
                put(attn + "o_norm.weight", bf16(TINY.linear_head_dim))
                put(attn + "o_proj.weight", bf16(hidden, qkv))
            else:  # MLA + indexer (layers 3 and 4, including the MTP layer)
                attn = base + "self_attn."
                int8_linear(attn + "q_a_proj", TINY.q_lora_rank, hidden)
                put(attn + "q_a_layernorm.weight", bf16(TINY.q_lora_rank))
                int8_linear(attn + "q_b_proj", TINY.num_attention_heads * TINY.qk_head_dim, TINY.q_lora_rank)
                int8_linear(attn + "kv_a_proj_with_mqa", TINY.kv_lora_rank, hidden)
                put(attn + "kv_a_layernorm.weight", bf16(TINY.kv_lora_rank))
                put(
                    attn + "kv_b_proj.weight",
                    bf16(TINY.num_attention_heads * (TINY.qk_head_dim + TINY.v_head_dim), TINY.kv_lora_rank),
                )
                int8_linear(attn + "o_proj", hidden, TINY.num_attention_heads * TINY.v_head_dim)
                indexer = attn + "indexer."
                put(indexer + "wq_b.weight", bf16(TINY.index_n_heads * TINY.index_head_dim, TINY.q_lora_rank))
                put(indexer + "wk.weight", bf16(TINY.index_head_dim, hidden))
                put(indexer + "k_norm.weight", bf16(TINY.index_head_dim))
                put(indexer + "k_norm.bias", bf16(TINY.index_head_dim))
                put(indexer + "weights_proj.weight", bf16(TINY.index_n_heads, hidden))
                put(indexer + "index_kpool_compress_gate", bf16(TINY.index_head_dim, hidden))
                put(indexer + "index_kpool_compress_ape", bf16(TINY.index_kpool, TINY.index_head_dim))
            if layer < TINY.num_hidden_layers:  # mHC is backbone-only
                for kind in ("attn", "ffn"):
                    if layer == 0:  # flat spelling on one layer, canonical elsewhere
                        put(base + f"hc_{kind}_fn", bf16(TINY.mix_hc, TINY.hc_dim))
                        put(base + f"hc_{kind}_scale", torch.rand(3, generator=generator).float())
                        put(base + f"hc_{kind}_base", torch.rand(TINY.mix_hc, generator=generator).float())
                    else:
                        put(base + f"{kind}_hc.fn", bf16(TINY.mix_hc, TINY.hc_dim))
                        put(base + f"{kind}_hc.scale", torch.rand(3, generator=generator).float())
                        put(base + f"{kind}_hc.base", torch.rand(TINY.mix_hc, generator=generator).float())
            put(base + "input_layernorm.weight", bf16(hidden))
            put(base + "post_attention_layernorm.weight", bf16(hidden))
            if layer >= TINY.first_k_dense_replace:
                put(base + "mlp.gate.weight", bf16(TINY.n_routed_experts, hidden))
                flat_bias = layer == 1  # flat spelling on one layer, canonical elsewhere
                bias_stem = "mlp.e_score_correction_bias" if flat_bias else "mlp.gate.e_score_correction_bias"
                put(base + bias_stem, torch.randn(TINY.n_routed_experts, generator=generator).float())
                for expert in range(TINY.n_routed_experts):
                    dotted = layer == 4 and expert % 2 == 0  # the routed ``.weight`` variant
                    fused = layer == 2 or dotted
                    ffn(base + f"mlp.experts.{expert}.", TINY.moe_intermediate_size, fused=fused, dotted=dotted)
                ffn(base + "mlp.shared_experts.", TINY.moe_intermediate_size, fused=layer >= 3)
            else:
                ffn(base + "mlp.", TINY.intermediate_size, fused=False)
        embed = bf16(TINY.vocab_size, hidden)
        head = bf16(TINY.vocab_size, hidden)
        put("model.language_model.embed_tokens.weight", embed)
        put("lm_head.weight", head)
        put("model.language_model.norm.weight", bf16(hidden))
        mtp = tiny_root(TINY.mtp_layer_id) + f"{TINY.mtp_layer_id}."
        put(mtp + "eh_proj.weight", bf16(hidden, 2 * hidden))
        put(mtp + "enorm.weight", bf16(hidden))
        put(mtp + "hnorm.weight", bf16(hidden))
        put(mtp + "embed_tokens.weight", embed.clone())  # MTP duplicates alias the
        put(mtp + "shared_head.head.weight", head.clone())  # top-level host tables
        put(mtp + "shared_head.norm.weight", bf16(hidden))  # ignored, like rot.weight
        if mutate is not None:
            mutate(tensors, labels)
        return _FixtureWeights(tensors, labels)

    # Resident test 1: the fixture loads through the staging protocol and the
    # plan arithmetic holds.
    checkpoint = build_tiny_fixture()
    assert checkpoint.description["model.language_model.layers.3.self_attn.q_b_proj.weight"] == "W8A8_DYNAMIC"
    assert checkpoint.description["model.language_model.layers.3.self_attn.kv_b_proj.weight"] == "FLOAT"
    weights = ResidentPrefillWeights.load(checkpoint, tp_size=2, ep_size=2, model=TINY)
    assert weights.bytes_per_rank == sum(
        math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        for shape, dtype in weights.tensor_layouts.values()
    )
    assert weights.plan(1).device_bytes == weights.bytes_per_rank
    assert weights.plan(1).tp_rank == 1 and weights.plan(1).local_expert_ids == tuple(range(8, 16))
    try:
        weights.plan(2)
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-range rank should have been rejected")

    # Resident test 3: authority-driven dequant of the quantized MLA
    # projections equals int8 * scale with exactly one BF16 rounding, and
    # survives a matmul within that rounding.
    q_b_name = "model.language_model.layers.3.self_attn.q_b_proj.weight"
    q_b_stem = "model.language_model.layers.3.self_attn.q_b_proj"
    raw, scale = checkpoint[q_b_name], checkpoint[q_b_stem + ".weight_scale"].reshape(-1)
    dequant = weights._read_source(q_b_name, torch.bfloat16)
    assert dequant.dtype == torch.bfloat16
    assert_close(dequant, (raw.float() * scale.reshape(-1, 1)).to(torch.bfloat16), "mla q_b dequant")
    mla_layer = weights._mla_layer(3, tp_rank=1)
    assert_close(mla_layer["w_q_b"], dequant[16:], "mla q_b tp slice")  # 4 heads x 8 rows, head-major
    probe = torch.randn(6, TINY.q_lora_rank).bfloat16()
    got = (probe.float() @ mla_layer["w_q_b"].float().t()).to(torch.bfloat16)
    reference = (probe.float() @ (raw[16:].float() * scale[16:].reshape(-1, 1)).t()).to(torch.bfloat16)
    # One BF16 rounding of the weight matrix: the error scales with the
    # products' magnitude, so bound it by the reference's span, not an absolute.
    bound = reference.abs().max().item() / 64
    if not torch.allclose(got.float(), reference.float(), rtol=2.0 / 128, atol=bound):
        raise AssertionError("dequantized q_b disagrees with the reference within one BF16 rounding")
    kv_b = checkpoint["model.language_model.layers.3.self_attn.kv_b_proj.weight"]
    kv_b_passthrough = weights._read_source("model.language_model.layers.3.self_attn.kv_b_proj.weight", torch.bfloat16)
    assert_close(kv_b_passthrough, kv_b, "float passthrough")
    bad_offset = build_tiny_fixture(
        mutate=lambda tensors, labels: tensors.update({"model.language_model.layers.3.self_attn.q_b_proj.weight_offset": torch.ones(32, 1)}),
    )
    bad_offset_weights = ResidentPrefillWeights.load(bad_offset, tp_size=2, ep_size=2, model=TINY)
    try:
        bad_offset_weights._read_source(q_b_name, torch.bfloat16)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-zero MLA weight_offset should have been rejected")
    bad_ffn_offset = build_tiny_fixture(
        mutate=lambda tensors, labels: tensors.update({"model.layers.1.mlp.experts.0.gate_proj.weight_offset": torch.ones(16, 1)}),
    )
    bad_ffn_weights = ResidentPrefillWeights.load(bad_ffn_offset, tp_size=2, ep_size=2, model=TINY)
    try:
        bad_ffn_weights._moe_layer(1, ep_rank=0)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-zero FFN weight_offset should have been rejected")

    # Resident test 4: every family's rank slices are disjoint and cover the
    # source exactly once, across the bring-up geometries.
    for tp_size, ep_size in ((1, 1), (2, 2), (4, 16)):
        sharded = ResidentPrefillWeights.load(checkpoint, tp_size=tp_size, ep_size=ep_size, model=TINY)
        kda_stacks = [sharded._stack_family("kda", rank % tp_size) for rank in range(tp_size)]
        for stack in kda_stacks:
            if not all(tensor.is_contiguous() for tensor in stack.values()):
                raise AssertionError("every staged family tensor must be contiguous")
        for index, layer in enumerate(TINY.kda_layer_ids):
            source = checkpoint[tiny_root(layer) + f"{layer}.self_attn.q_proj.weight"]
            rebuilt = torch.cat([stack["w_q"][index] for stack in kda_stacks], dim=0)
            assert_close(rebuilt, source, f"kda w_q coverage tp{tp_size} layer {layer}")
        mla_stacks = [sharded._stack_family("mla", rank % tp_size) for rank in range(tp_size)]
        heads = TINY.num_attention_heads
        for index, layer in enumerate(TINY.indexer_layer_ids):
            prefix = tiny_root(layer) + f"{layer}.self_attn."
            kv_source = checkpoint[prefix + "kv_b_proj.weight"]
            reshaped = kv_source.reshape(heads, TINY.qk_head_dim + TINY.v_head_dim, TINY.kv_lora_rank)
            key, value = reshaped[:, : TINY.qk_head_dim], reshaped[:, TINY.qk_head_dim :]
            assert_close(torch.cat([s["w_k"][index] for s in mla_stacks], dim=0), key, f"kv_b key coverage tp{tp_size}")
            assert_close(
                torch.cat([s["w_v"][index] for s in mla_stacks], dim=0), value,
                f"kv_b value coverage tp{tp_size}",
            )
            # o_proj is W8A8 on the MLA layers: coverage is against the
            # dequantized source, which is what the ranks actually hold.
            o_source = sharded._read_source(prefix + "o_proj.weight", torch.bfloat16)
            assert_close(
                torch.cat([s["w_o"][index] for s in mla_stacks], dim=1), o_source,
                f"mla o_proj column coverage tp{tp_size}",
            )
        expert_ids = [expert for rank in range(ep_size) for expert in sharded.plan(rank).local_expert_ids]
        assert sorted(expert_ids) == list(range(TINY.n_routed_experts))
        layer_one = 1
        moe_per_rank = [sharded._moe_layer(layer_one, rank) for rank in range(ep_size)]
        for piece in moe_per_rank:
            if not all(tensor.is_contiguous() for tensor in piece.values()):
                raise AssertionError("every staged moe tensor must be contiguous")
        rebuilt_experts = torch.cat([piece["w_gate_up"] for piece in moe_per_rank], dim=0)

        def expert_tensor(expert: int, part: str) -> torch.Tensor:
            stem = tiny_root(layer_one) + f"{layer_one}.mlp.experts.{expert}.{part}_proj.weight"
            return checkpoint[stem]

        expected_experts = torch.stack(
            [torch.cat([expert_tensor(expert, "gate"), expert_tensor(expert, "up")], dim=0) for expert in range(TINY.n_routed_experts)],
        )
        assert_close(rebuilt_experts, expected_experts, f"routed expert coverage ep{ep_size}")
    for bad_call in (
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=8, ep_size=8, model=TINY),  # 4 MLA heads
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=3, ep_size=3, model=TINY),  # unsupported
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=2, ep_size=32, model=TINY),  # unsupported EP
        lambda: ResidentPrefillWeights.load(checkpoint, tp_size=4, ep_size=2, model=TINY),  # EP % TP
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid TP/EP geometry should have been rejected")

    # Resident test 5: the kv_b halves land reduction-first ([QK, KV_LORA]),
    # match the inline split sliced per head, and the full-geometry helper is
    # bit-equal to the shared projection golden with complete head coverage.
    kv_source = checkpoint["model.language_model.layers.3.self_attn.kv_b_proj.weight"]
    reshaped = kv_source.reshape(TINY.num_attention_heads, TINY.qk_head_dim + TINY.v_head_dim, TINY.kv_lora_rank)
    expected_key, expected_value = reshaped[:, : TINY.qk_head_dim], reshaped[:, TINY.qk_head_dim :]
    assert mla_layer["w_k"].shape == (2, TINY.qk_head_dim, TINY.kv_lora_rank)  # local_h = 4/2
    assert mla_layer["w_v"].shape == (2, TINY.v_head_dim, TINY.kv_lora_rank)
    assert_close(mla_layer["w_k"], expected_key[2:], "reduction-first w_k head slice")
    assert_close(mla_layer["w_v"], expected_value[2:], "reduction-first w_v head slice")
    generator = torch.Generator().manual_seed(202)
    full_kv_b = torch.randn(FLASH.num_attention_heads * (FLASH.qk_head_dim + FLASH.v_head_dim), FLASH.kv_lora_rank, generator=generator).bfloat16(
    )
    golden_key, golden_value = golden_split_kv_b(full_kv_b)
    rank_keys = [split_kv_b(full_kv_b, rank, 16)[0] for rank in range(16)]
    rank_values = [split_kv_b(full_kv_b, rank, 16)[1] for rank in range(16)]
    assert_close(torch.cat(rank_keys, dim=0), golden_key, "split_kv_b covers every head exactly once")
    assert_close(torch.cat(rank_values, dim=0), golden_value, "split_kv_b value coverage")
    assert rank_keys[3].shape == (FLASH.num_attention_heads // 16, FLASH.qk_head_dim, FLASH.kv_lora_rank)

    # Resident test 6: the Hadamard table is the scaled Sylvester form and
    # orthonormal at BF16 tolerance.
    singles = weights._single_tensors()
    hadamard = singles["hadamard"]
    matrix = torch.ones((1, 1))
    while matrix.shape[0] < TINY.index_head_dim:
        top = torch.cat([matrix, matrix], dim=1)
        bottom = torch.cat([matrix, -matrix], dim=1)
        matrix = torch.cat([top, bottom], dim=0)
    local_build = (matrix * TINY.index_head_dim**-0.5).to(torch.bfloat16)
    assert hadamard.dtype == torch.bfloat16 and torch.equal(hadamard, local_build)
    # The full-width table is the sylvester_hadamard the loader imports,
    # checked orthonormal at INDEX_DIM = 128.
    big = sylvester_hadamard(128)
    if not torch.allclose(big.float() @ big.float().t(), torch.eye(128), atol=2e-2):
        raise AssertionError("the Hadamard table is not orthonormal at BF16 tolerance")

    # Resident test 7: the indexer projections are transposed once into the
    # kernels' K-major [in, out] layout.
    indexer_stack = weights._stack_family("indexer", tp_rank=0)
    transposed = (
        ("w_q_b", "wq_b.weight"),
        ("w_k", "wk.weight"),
        ("w_weights", "weights_proj.weight"),
        ("w_compress_gate", "index_kpool_compress_gate"),
    )
    passthrough = (
        ("k_norm_w", "k_norm.weight"),
        ("k_norm_b", "k_norm.bias"),
        ("compress_ape", "index_kpool_compress_ape"),
    )
    for index, layer in enumerate(TINY.indexer_layer_ids):
        for member, stem in transposed:
            source = checkpoint[tiny_root(layer) + f"{layer}.self_attn.indexer.{stem}"]
            assert_close(indexer_stack[member][index], source.t(), f"indexer {member} transpose layer {layer}")
        for member, stem in passthrough:
            source = checkpoint[tiny_root(layer) + f"{layer}.self_attn.indexer.{stem}"]
            assert_close(indexer_stack[member][index], source, f"indexer {member} passthrough layer {layer}")

    # Resident test 8: backbone tables and independently addressed MTP tables.
    assert weights.host_bytes == 2 * TINY.vocab_size * TINY.hidden_size * 2
    assert not any("embed" in name or "head" in name for name in weights.tensor_layouts)

    def nudge_mtp_embed(tensors, labels):
        tensors["model.language_model.layers.4.embed_tokens.weight"][0].add_(1.0)
        tensors["model.language_model.layers.4.shared_head.head.weight"][0].add_(2.0)

    untied_weights = ResidentPrefillWeights.load(
        build_tiny_fixture(mutate=nudge_mtp_embed), tp_size=2, ep_size=2, model=TINY,
    )
    mtp_rows = untied_weights.lookup_rows(torch.tensor([0]), mtp=True)
    backbone_rows = untied_weights.lookup_rows(torch.tensor([0]))
    assert_close(mtp_rows, backbone_rows + 1, "MTP keeps its untied embedding values")
    mtp_head = untied_weights.mtp_head_shard(0)
    assert_close(mtp_head[0], untied_weights.lm_head()[0] + 2, "MTP keeps its untied head values")
    rows = weights.lookup_rows(torch.tensor([[3, 0], [127, 5]]))
    assert rows.shape == (2, 2, TINY.hidden_size) and rows.dtype == torch.bfloat16
    embed_table = checkpoint["model.language_model.embed_tokens.weight"]
    assert_close(rows[1, 0], embed_table[127], "lookup_rows gathers the host embed table")
    assert_close(rows[0, 1], embed_table[0], "lookup_rows keeps the request shape")
    assert weights.lm_head().shape == (TINY.vocab_size, TINY.hidden_size)
    try:
        weights.lookup_rows(torch.tensor([TINY.vocab_size]))
    except ValueError:
        pass
    else:
        raise AssertionError("an out-of-vocabulary token id should have been rejected")

    # Resident test 10: the staged upload - order, per-layer MoE staging, byte
    # coverage, and rank tagging - against a recording runtime.
    class _Handle:
        def __init__(self, pointer: int):
            self.data_ptr = pointer

    class _RecordingRuntime:
        def __init__(self):
            self.copies = []

        def copy_to(self, dst, src, nbytes, dst_offset=0, worker_id=None):
            self.copies.append((dst, dst_offset, nbytes, worker_id))

    runtime = _RecordingRuntime()
    handles = {name: _Handle(index) for index, name in enumerate(weights.tensor_layouts)}
    weights.upload(runtime, rank=1, handles=handles)
    names_by_pointer = {index: name for index, name in enumerate(weights.tensor_layouts)}
    per_name: dict[str, list[tuple[int, int]]] = {}
    for dst, offset, nbytes, worker in runtime.copies:
        assert worker == 1, "every copy must be tagged with the uploading rank"
        per_name.setdefault(names_by_pointer[dst], []).append((offset, nbytes))
    for name, plan in weights.stacks.items():
        copies = per_name.get(name, [])
        if not copies:
            raise AssertionError(f"stack {name} was never staged")
        if sum(nbytes for _, nbytes in copies) != plan.bytes:
            raise AssertionError(f"stack {name} staged {sum(n for _, n in copies)} of {plan.bytes} bytes")
        if plan.family in ("norm", "mhc", "router", "kda", "mla", "indexer", "moe"):
            layer_bytes = math.prod(plan.shape[1:]) * torch.empty((), dtype=plan.dtype).element_size()
            offsets = [offset for offset, _ in copies]
            if offsets != [index * layer_bytes for index in range(len(plan.layers))]:
                raise AssertionError(f"stack {name} was not staged one layer at a time in order")
            if any(nbytes != layer_bytes for _, nbytes in copies):
                raise AssertionError(f"stack {name} staged a partial layer slice")
        else:
            if copies != [(0, plan.bytes)]:
                raise AssertionError(f"unstacked resident {name} must be one whole copy")
    first_seen = list(
        dict.fromkeys((name.split(".")[0] for name in (names_by_pointer[dst] for (dst, _, _, _) in runtime.copies))),
    )
    expected_order = ["norm", "mtp", "hadamard", "mhc", "router", "kda", "mla", "indexer", "dense", "moe"]
    if first_seen != expected_order:
        raise AssertionError(f"staging order {first_seen} != {expected_order}")

    # Logical byte parity of per-layer staging versus whole-stack materialization:
    # decode the two shared NZ stacks before comparing to the logical
    # ``torch.stack`` reference. All other staged bytes stay unchanged.
    class _ByteRuntime(_RecordingRuntime):
        """Capture the staged source bytes at copy time (pointer valid there)."""

        def __init__(self):
            super().__init__()
            self.blobs: dict[str, bytearray] = {}

        def copy_to(self, dst, src, nbytes, dst_offset=0, worker_id=None):
            super().copy_to(dst, src, nbytes, dst_offset, worker_id)
            name = names_by_pointer[dst]
            blob = self.blobs.setdefault(name, bytearray(weights.stacks[name].bytes))
            blob[dst_offset : dst_offset + nbytes] = ctypes.string_at(src, nbytes)

    def raw_bytes(tensor: torch.Tensor) -> bytes:
        return bytes(tensor.contiguous().flatten().view(torch.uint8).numpy())

    byte_runtime = _ByteRuntime()
    weights.upload(byte_runtime, rank=1, handles=handles)
    reference: dict[str, torch.Tensor] = {}
    for family in ("norm", "mhc", "router", "kda", "mla", "indexer"):
        for member, stack in weights._stack_family(family, tp_rank=1).items():
            reference[f"{family}.{member}"] = stack
    for member, stack in weights._stack_dense(tp_rank=1).items():
        reference[f"dense.{member}"] = stack
    for member in (
        "w_gate_up",
        "w_gate_up_scale",
        "w_down",
        "w_down_scale",
        "shared_w_gate_up",
        "shared_w_gate_up_scale",
        "shared_w_down",
        "shared_w_down_scale",
    ):
        reference[f"moe.{member}"] = torch.stack(
            [weights._moe_layer(layer, 1)[member] for layer in weights._family_layers["moe"]],
        )
    for name, tensor in weights._single_tensors().items():
        reference[name] = tensor
    if set(reference) != set(byte_runtime.blobs):
        raise AssertionError("the parity reference and the staged stacks disagree on names")
    for name, tensor in reference.items():
        staged = bytes(byte_runtime.blobs[name])
        if name in ("moe.shared_w_gate_up", "moe.shared_w_down"):
            physical = torch.frombuffer(byte_runtime.blobs[name], dtype=tensor.dtype).reshape(tensor.shape)
            staged = raw_bytes(unpack_int8_nz(physical))
        if staged != raw_bytes(tensor):
            raise AssertionError(f"per-layer staging of {name} is not byte-identical to the stack")
    print(f"[GOLDEN] PASS weights upload byte parity ({len(reference)} stacks, rank 1 of TP2/EP2)")

    fresh = _RecordingRuntime()
    try:
        weights.upload(fresh, rank=0, handles={})
    except ValueError:
        pass
    else:
        raise AssertionError("upload with missing handles should have been rejected")
    try:
        weights.upload(fresh, rank=2, handles=handles)
    except ValueError:
        pass
    else:
        raise AssertionError("upload with an out-of-range rank should have been rejected")
    copies = len(runtime.copies)
    message = f"[GOLDEN] PASS weights resident loader (tiny checkpoint, TP1-4/EP1-16, {copies} copies)"
    print(message)

    class _FullGeometryCheckpoint(Mapping[str, torch.Tensor]):
        """Geometry-only checkpoint view: canonical names, no files, no payloads."""

        def __init__(self):
            model = FLASH
            names: dict[str, str] = {}
            for layer in range(model.num_checkpoint_layers):
                base = f"model.layers.{layer}."
                if layer in model.kda_layer_ids:
                    for part in ("q", "k", "v"):
                        names[base + f"self_attn.{part}_proj.weight"] = "FLOAT"
                        names[base + f"self_attn.{part}_conv1d.weight"] = "FLOAT"
                    for stem in ("dt_bias", "A_log", "f_a_proj.weight", "f_b_proj.weight"):
                        names[base + "self_attn.forget_gate." + stem] = "FLOAT"
                    kda_stems = (
                        "g_a_proj.weight",
                        "g_b_proj.weight",
                        "b_proj.weight",
                        "o_norm.weight",
                        "o_proj.weight",
                    )
                    for stem in kda_stems:
                        names[base + "self_attn." + stem] = "FLOAT"
                else:
                    quant_stems = ("q_a_proj.weight", "q_b_proj.weight", "kv_a_proj_with_mqa.weight", "o_proj.weight")
                    for stem in quant_stems:
                        names[base + "self_attn." + stem] = "W8A8_DYNAMIC"
                        names[base + "self_attn." + stem + ".weight_scale"] = "W8A8_DYNAMIC"
                        names[base + "self_attn." + stem + ".weight_offset"] = "W8A8_DYNAMIC"
                    for stem in ("q_a_layernorm.weight", "kv_a_layernorm.weight", "kv_b_proj.weight"):
                        names[base + "self_attn." + stem] = "FLOAT"
                    for stem in (
                        "wq_b.weight",
                        "wk.weight",
                        "k_norm.weight",
                        "k_norm.bias",
                        "weights_proj.weight",
                        "index_kpool_compress_gate",
                        "index_kpool_compress_ape",
                    ):
                        names[base + "self_attn.indexer." + stem] = "FLOAT"
                if layer < model.num_hidden_layers:
                    for kind in ("attn", "ffn"):
                        names[base + f"{kind}_hc.fn"] = "FLOAT"
                        names[base + f"{kind}_hc.scale"] = "FLOAT"
                        names[base + f"{kind}_hc.base"] = "FLOAT"
                for stem in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                    names[base + stem] = "FLOAT"
                if layer >= model.first_k_dense_replace:
                    names[base + "mlp.gate.weight"] = "FLOAT"
                    names[base + "mlp.gate.e_score_correction_bias"] = "FLOAT"
                    bases = [f"mlp.experts.{expert}." for expert in range(model.n_routed_experts)]
                    bases.append("mlp.shared_experts.")
                else:
                    bases = ["mlp."]
                for expert_base in bases:
                    for part in ("gate_proj", "up_proj", "down_proj"):
                        names[base + expert_base + part + ".weight"] = "W8A8_DYNAMIC"
                        names[base + expert_base + part + ".weight_scale"] = "W8A8_DYNAMIC"
                        names[base + expert_base + part + ".weight_offset"] = "W8A8_DYNAMIC"
            mtp = f"model.layers.{model.mtp_layer_id}."
            for stem in (
                "eh_proj.weight",
                "enorm.weight",
                "hnorm.weight",
                "embed_tokens.weight",
                "shared_head.head.weight",
                "shared_head.norm.weight",
            ):
                names[mtp + stem] = "FLOAT"
            names["model.language_model.norm.weight"] = "FLOAT"
            names["model.language_model.embed_tokens.weight"] = "FLOAT"
            names["lm_head.weight"] = "FLOAT"
            self._names = names

        def __contains__(self, name: object) -> bool:
            return name in self._names

        def __iter__(self):
            return iter(self._names)

        def __len__(self) -> int:
            return len(self._names)

        def __getitem__(self, name: str) -> torch.Tensor:
            raise AssertionError("the budget gate must not read payloads")

        @property
        def description(self) -> dict[str, str]:
            return self._names

        def read(self, name: str, selection=None) -> torch.Tensor:
            raise AssertionError("the budget gate must not read payloads")

    def family_bytes(resident, family: str, *, shared: bool | None = None) -> int:
        total = 0
        for plan in resident.stacks.values():
            if plan.family != family:
                continue
            if shared is True and not plan.name.startswith("moe.shared"):
                continue
            if shared is False and plan.name.startswith("moe.shared"):
                continue
            total += plan.bytes
        return total

    TP16_COMPONENTS = {
        "kda": 652_252_192,
        "mla": 364_953_600,
        "indexer": 179_324_928,
        "mhc": 70_788_600,
        "norm": 753_664,
        "router": 101_499_264,
        "routed": 19_503_710_208,
        "shared": 1_083_539_456,
        "dense": 28_379_136,
        "mtp": 67_133_440,
        "single": 40_960,
    }
    TP16_TOTAL = 22_052_375_448  # 22.05 GB per rank, the design's headline
    TP1_OVERRIDES = {"kda": 9_366_356_992, "mla": 2_818_621_440, "dense": 453_328_896}
    TP1_TOTAL = 33_645_097_848
    for tp_size, overrides, total in ((16, {}, TP16_TOTAL), (1, TP1_OVERRIDES, TP1_TOTAL)):
        resident = ResidentPrefillWeights.load(_FullGeometryCheckpoint(), tp_size=tp_size, ep_size=16)
        components = {**TP16_COMPONENTS, **overrides}
        for family in ("kda", "mla", "indexer", "mhc", "norm", "router", "dense", "mtp", "single"):
            got = family_bytes(resident, family)
            if got != components[family]:
                raise AssertionError(f"TP{tp_size} {family} budget {got} != design value {components[family]}")
        routed = family_bytes(resident, "moe", shared=False)
        shared = family_bytes(resident, "moe", shared=True)
        if (routed, shared) != (components["routed"], components["shared"]):
            raise AssertionError(f"TP{tp_size} moe budget {(routed, shared)} != design values")
        if resident.bytes_per_rank != total:
            raise AssertionError(f"TP{tp_size}/EP16 budget {resident.bytes_per_rank} != design total {total}")
        if resident.host_bytes != 2 * FLASH.vocab_size * FLASH.hidden_size * 2:
            raise AssertionError("host bytes must be the head plus the embedding table")
        assert resident.plan(0).device_bytes == resident.bytes_per_rank
    if abs(TP16_TOTAL / 1e9 - 22.05) > 0.01:
        raise AssertionError("the TP16 budget drifted off the design's 22.05 GB headline")
    print("[GOLDEN] PASS weights resident budget (TP16/EP16 22.05 GB, TP1/EP16 33.65 GB, headers only)")


def run_reference_checks_cli(argv=None) -> None:
    """Run the CPU self-test and optionally probe a real W8A8 checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="optional real W8A8 checkpoint root for a read/dequant/coverage smoke")
    args = parser.parse_args(argv)
    # Tiny-tensor CPU math parallelizes badly past a few dozen threads (this
    # host defaults to 320); cap the self-test only, library imports stay pure.
    torch.set_num_threads(min(32, torch.get_num_threads()))
    run_self_tests(checkpoint_root=args.checkpoint)


def main(argv=None):
    checks = argparse.ArgumentParser(add_help=False)
    checks.add_argument("--check", choices=["reference", "weights"])
    selected, argv = checks.parse_known_args(argv)
    if selected.check == "reference":
        return run_reference_checks_cli(argv)
    if selected.check == "weights":
        return run_weight_checks(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", action="store_true", help="accepted by the default CPU weight checks")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="resident W8A8 checkpoint root (required for generation)",
    )
    parser.add_argument("--runtime-dir", type=Path, default=None, help="manifest dir of the launch plan artifacts; compiled on first use")
    parser.add_argument(
        "-d", "--device", default=os.environ.get("TASK_DEVICE"),
        help="comma-separated device ids (accepted, ignored under --checkpoint)",
    )
    parser.add_argument("--tokens", type=int, default=T_MOE, help=argparse.SUPPRESS)
    parser.add_argument("--devices", default=os.environ.get("TASK_DEVICE"), help=argparse.SUPPRESS)
    parser.add_argument("--tp", type=int, default=C.TP_SIZE, choices=[16, 8, 2])
    parser.add_argument("--ep", type=int, default=RANKS, choices=[16, 8, 2])
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--cache-tokens", type=int, default=CACHE_TOKENS)
    parser.add_argument("--prompt", action="append", help=f"one prompt of the packed batch; provide between 1 and {BATCH}")
    parser.add_argument("--compile-only", action="store_true", help="build all programs without allocating NPU devices")
    parser.add_argument("--spec-tokens", type=int, choices=range(MTP_SPEC_TOKENS + 1), default=0)
    parser.add_argument("--image", action="append", type=Path, help="image for prompt zero; repeat for multiple images")
    parser.add_argument("--video", action="append", type=Path, help="video file for prompt zero; decoded with FFmpeg")
    parser.add_argument("--vision-runtime-dir", type=Path, help="precompiled single-rank vision artifact")
    parser.add_argument("--video-frame", action="append", type=Path, help="sampled video frame for prompt zero")
    parser.add_argument("--video-frame-fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--raw", action="store_true", help="encode prompts verbatim, no chat template")
    parser.add_argument("--expect-tokens", type=Path, help="JSON mapping prompt index to the expected top-1 token-id prefix")
    parser.add_argument("--dump-tokens", type=Path, help="write the produced token ids as JSON")
    options, unknown = parser.parse_known_args(argv)
    if options.checkpoint is None and options.runtime_dir is None and not options.compile_only:
        if options.prompt or options.image or options.video or options.video_frame or unknown:
            parser.error("generation requires --checkpoint and --runtime-dir")
        return run_weight_checks(["--tp", str(options.tp), "--ep", str(options.ep)])
    if options.sim:
        parser.error("resident generation requires a2a3 devices")
    if options.compile_only:
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        if options.runtime_dir is None:
            parser.error("--compile-only requires --runtime-dir")
        compile_production(options.runtime_dir, list(range(RANKS)))
        return
    if options.checkpoint is None:
        if options.prompt:
            raise ValueError("--prompt needs the generate() mode (--checkpoint)")
        parser.error("generation requires --checkpoint")
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if options.capacity != CAPACITY:
        raise ValueError(f"--capacity must match compilation ({CAPACITY}); recompile to change it")
    if options.cache_tokens != CACHE_TOKENS:
        raise ValueError(f"--cache-tokens must match compilation ({CACHE_TOKENS})")
    if options.max_seq_len != MAX_SEQ_LEN:
        raise ValueError(f"--max-seq-len must match compilation ({MAX_SEQ_LEN})")
    if (options.tp, options.ep) != (C.TP_SIZE, RANKS):
        raise ValueError(f"parallel sizes must match compilation: TP{C.TP_SIZE}/EP{RANKS}")
    if options.runtime_dir is None:
        raise ValueError("the generate() mode needs --runtime-dir")
    device_ids = [int(value) for value in (options.devices or "").split(",") if value]
    if len(device_ids) != RANKS or len(set(device_ids)) != RANKS:
        raise ValueError(f"provide {RANKS} distinct task-allocated a2a3 devices")
    prompts = list(options.prompt or DEFAULT_PROMPTS)
    if not 1 <= len(prompts) <= BATCH or options.max_new_tokens <= 0:
        raise ValueError(f"provide 1..{BATCH} prompts and a positive --max-new-tokens")

    expect_tokens = None
    if options.expect_tokens is not None:
        raw = json.loads(options.expect_tokens.read_text())
        expect_tokens = (
            {int(key): value for key, value in raw.items()}
            if isinstance(raw, dict)
            else {index: value for index, value in enumerate(raw)}
        )
    torch.set_num_threads(8)
    checkpoint = Checkpoint(options.checkpoint)
    if (options.runtime_dir / "manifest.json").exists():
        programs = load_production(options.runtime_dir, device_ids)
    else:
        programs = compile_production(options.runtime_dir, device_ids)
    visual_inputs = None
    if options.image or options.video or options.video_frame:

        processor = VisionProcessor(checkpoint)
        visuals = [processor.image(image) for image in options.image or []]
        visuals.extend(processor.video_file(video) for video in options.video or [])
        if options.video_frame:
            if not math.isfinite(options.video_frame_fps) or options.video_frame_fps <= 0:
                raise ValueError("--video-frame-fps must be positive and finite")
            times = [index / options.video_frame_fps for index in range(len(options.video_frame))]
            visuals.append(processor.video(options.video_frame, times))
        visual_inputs = {0: visuals}
    if options.vision_runtime_dir is not None:
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

        programs["_vision"] = DistributedCompiledProgram.from_dir(
            options.vision_runtime_dir, platform="a2a3",
            distributed_config=DistributedConfig(device_ids=[device_ids[0]], num_sub_workers=0),
        )
    report = generate(
        checkpoint, programs, device_ids, prompts, options.max_new_tokens, raw_prompts=options.raw,
        expect_tokens=expect_tokens, dump_path=options.dump_tokens, visual_inputs=visual_inputs,
        spec_tokens=options.spec_tokens,
    )
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("GLM_GENERATION_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
