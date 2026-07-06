# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
# ci: no-sim    # CI marker: full multi-layer / multi-card forward — device-only, skip on *sim
"""DeepSeek-V4 Flash packed-prefill forward experiment.

Mirrors ``decode_fwd.py``: a single rank-generic ``@pl.jit`` per-rank kernel
(``prefill_fwd``) is launched once per EP rank from an ``@pl.jit.host`` driver
(``l3_prefill_fwd``) via ``for r in pl.range(pld.world_size())``, so the same
program scales to EP 2 / 4 / 8.  The per-rank kernel hand-unrolls the model's
layer schedule and calls ``prefill_attention_{swa,hca,csa}`` + ``moe`` directly
(no ``prefill_layer`` wrapper).  Each attention / moe stage runs in its own
``pl.scope`` under ``auto_scope=False`` (matching ``decode_fwd``), and the final
hidden state passes ``hc_head`` -> final ``rms_norm`` -> TP-sharded ``lm_head``
to produce full-vocabulary logits.  This is a kernel-only smoke driver: it does
not run a golden comparison.  With ``lm_head`` composed in, ``--ep 2`` / TP=2
only (mirrors ``decode_fwd``).
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
from golden import run_jit
from pypto.ir.distributed_compiled_program import DistributedConfig

# prefill_fwd is self-contained: it imports kernels, constants, and per-kind
# spec builders directly from the leaf modules (no dependency on prefill_layer).
# The prefill path runs PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe, which freezes recv shapes and derives RECV_MAX = EP * MOE_TOKENS at import.
import config
config.MOE_TOKENS = config.PREFILL_TOKENS
# Import moe first: it applies the EP/FLASH override before the attention modules
# bake config-derived MoE shapes (matches prefill_layer's import order).
from moe import (
    AUX_PAD,
    D,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    T,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    moe,
)
from config import FLASH as MODEL_CONFIG
from prefill_attention_swa import (
    build_tensor_specs as build_swa_attention_tensor_specs,
    prefill_attention_swa,
)
from prefill_attention_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    build_tensor_specs as build_hca_attention_tensor_specs,
    prefill_attention_hca,
)
from prefill_attention_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    CSA_CMP_BLOCK_NUM,
    CSA_ORI_BLOCK_NUM,
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    H,
    HEAD_DIM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_SEQ_LEN,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    PREFILL_IDX_BLOCK_NUM,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    SPARSE_TOPK,
    START_POS,
    build_tensor_specs as build_csa_attention_tensor_specs,
    prefill_attention_csa,
)
from hc_head import hc_head
from rmsnorm import rms_norm

# ---------------------------------------------------------------------------
# Model layer schedule (DeepSeek-V4 Flash, 43 hidden layers):
#   layer 0, 1                     -> swa
#   layer 2, 4, ..., 40            -> csa   (20 layers, loop body)
#   layer 3, 5, ..., 41            -> hca   (20 layers, loop body)
#   layer 42 (FWD_LAST_LAYER)      -> csa   (final layer)
# CSA total = 20 (loop) + 1 (last) = 21 ; HCA total = 20.
# ---------------------------------------------------------------------------
MODEL_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
FWD_NUM_LAYERS = 43
CSA_NUM_LAYERS = 21
HCA_NUM_LAYERS = 20
FWD_LAST_LAYER = FWD_NUM_LAYERS - 1
CSA_LAST_ORDER = CSA_NUM_LAYERS - 1
LAST_MOE_EPOCH = 2 * HCA_NUM_LAYERS + 3
assert MODEL_NUM_LAYERS == 43, "DeepSeek-V4 Flash hidden layer count changed"

# Replicated head weights (per-rank, not layer-stacked): hc_head projection and
# the final RMSNorm gamma — mirrors decode_fwd.
HC_HEAD_NAMES = ["hc_head_fn", "hc_head_scale", "hc_head_base"]
FINAL_NORM_NAMES = ["final_norm_w"]

# Per-FWD-layer stacked weights (sliced by the FWD layer index 0..42).
FWD_LAYER_STACKED_NAMES = [
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "cmp_kv",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
]
# CSA-compact stacked weights (sliced by the CSA order index 0..20).
CSA_LAYER_STACKED_NAMES = [
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "csa_cmp_kv_state", "csa_cmp_score_state",
    "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_kv_state", "csa_inner_score_state", "idx_kv_cache",
]
# HCA-compact stacked weights (sliced by the HCA order index 0..19).
HCA_LAYER_STACKED_NAMES = [
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hca_cmp_kv_state", "hca_cmp_score_state",
]
# Replicated once and passed whole to every layer (block tables are smoke zeros;
# slot mappings depend only on token position + a fixed per-kind compress ratio,
# so a single copy per name is shared across all layers of that kind).
SHARED_NAMES = [
    "freqs_cos", "freqs_sin",
    "ori_block_table", "cmp_block_table", "idx_block_table",
    "hca_compress_state_block_table", "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "ori_slot_mapping", "position_ids", "input_ids",
    "hca_cmp_slot_mapping", "hca_state_slot_mapping",
    "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
    "csa_state_slot_mapping", "csa_inner_state_slot_mapping",
    "cmp_sparse_indices", "cmp_sparse_lens",
]

# KV / state caches: per-token persistent buffers, not weights — kept as host
# tensors (re-bound each dispatch) rather than device-resident.
CACHE_NAMES = {
    "kv_cache", "cmp_kv",
    "hca_cmp_kv_state", "hca_cmp_score_state",
    "csa_cmp_kv_state", "csa_cmp_score_state",
    "csa_inner_kv_state", "csa_inner_score_state", "idx_kv_cache",
}

# Static weight parameters to keep device-resident, sharded per rank. Every host
# param is a leading-dim-stacked ``[N_RANKS, *tail]`` tensor the orchestrator
# slices as ``weight[r]`` and dispatches to ``device=r``; marking these
# resident="stacked" makes the harness upload shard ``r`` to card ``r`` once (via
# ``alloc_stacked_tensor``) and reuse it across dispatches, skipping the
# per-dispatch H2D/D2H. Covers every stacked attention / MoE weight, the per-kind
# compressor weights, the replicated head weights, and the constant RoPE tables —
# but NOT the KV/state caches (``CACHE_NAMES``) nor the per-step metadata (slot
# mappings, block tables, ids, sparse indices), which change per token.
RESIDENT_WEIGHT_NAMES = frozenset(
    [
        n
        for n in (*FWD_LAYER_STACKED_NAMES, *CSA_LAYER_STACKED_NAMES, *HCA_LAYER_STACKED_NAMES)
        if n not in CACHE_NAMES
    ]
    + ["freqs_cos", "freqs_sin"]
    + HC_HEAD_NAMES
    + FINAL_NORM_NAMES
)

# KV / state caches to keep device-resident (child_memory) as well, skipping the
# per-dispatch H2D these otherwise pay every dispatch (they dominate the residual
# host-transfer cost). All of CACHE_NAMES becomes resident.
RESIDENT_CACHE_NAMES = frozenset(CACHE_NAMES)

# The subset the kernel updates in place (``pl.InOut`` at the l3_prefill_fwd host
# signature) is additionally read back once at the end for validation
# (``is_output=True`` -> ``copy_stacked_from``). The other caches are plain
# ``pl.Tensor`` inputs — read-only here, so resident but not read back. Only
# ``kv_cache`` is ``pl.InOut``; the rest (cmp_kv, *_kv_state, *_score_state,
# idx_kv_cache) are read-only inputs.
RESIDENT_CACHE_OUTPUT_NAMES = frozenset(["kv_cache"])


@pl.jit(auto_scope=False)
def prefill_fwd(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
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
    kv_cache: pl.InOut[pl.Tensor[[FWD_NUM_LAYERS * CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    cmp_kv: pl.Tensor[[FWD_NUM_LAYERS * CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_cmp_kv_state: pl.Tensor[[HCA_NUM_LAYERS * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_score_state: pl.Tensor[[HCA_NUM_LAYERS * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.Tensor[[CSA_NUM_LAYERS * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state: pl.Tensor[[CSA_NUM_LAYERS * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.Tensor[[CSA_NUM_LAYERS * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state: pl.Tensor[[CSA_NUM_LAYERS * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    idx_kv_cache: pl.Tensor[[CSA_NUM_LAYERS * PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    hca_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    x_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
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
    shared_w1: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    my_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, D], pl.BF16]:
    nt: pl.Scalar[pl.INT32] = num_tokens
    hidden: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)

    # ===================== layer 0 : swa =================================
    hc_attn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0])
    hc_attn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [0 * 3])
    hc_attn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [0 * MIX_HC])
    attn_norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [0 * D])
    wq_a_l0: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [0 * D, 0])
    wq_b_l0: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [0 * Q_LORA, 0])
    wq_b_scale_l0: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [0 * H * HEAD_DIM])
    wkv_l0: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [0 * D, 0])
    gamma_cq_l0: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [0 * Q_LORA])
    gamma_ckv_l0: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [0 * HEAD_DIM])
    kv_cache_l0: pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [0 * CSA_ORI_BLOCK_NUM, 0, 0, 0])
    attn_sink_l0: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [0 * H])
    wo_a_l0: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [0 * O_GROUPS, 0, 0])
    wo_b_l0: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [0 * D, 0])
    wo_b_scale_l0: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [0 * D])
    hc_ffn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0])
    hc_ffn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [0 * 3])
    hc_ffn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [0 * MIX_HC])
    norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [0 * D])
    gate_w_l0: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [0 * N_EXPERTS_GLOBAL, 0])
    gate_bias_l0: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [0 * N_EXPERTS_GLOBAL])
    tid2eid_l0: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [0 * VOCAB, 0])
    routed_w1_l0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [0 * N_LOCAL, 0, 0])
    routed_w1_scale_l0: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [0 * N_LOCAL, 0])
    routed_w3_l0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [0 * N_LOCAL, 0, 0])
    routed_w3_scale_l0: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [0 * N_LOCAL, 0])
    routed_w2_l0: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [0 * N_LOCAL, 0, 0])
    routed_w2_scale_l0: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [0 * N_LOCAL, 0])
    shared_w1_l0: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [0 * MOE_INTER, 0])
    shared_w1_scale_l0: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [0 * MOE_INTER])
    shared_w3_l0: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [0 * MOE_INTER, 0])
    shared_w3_scale_l0: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [0 * MOE_INTER])
    shared_w2_l0: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [0 * D, 0])
    shared_w2_scale_l0: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [0 * D])
    x_attn0: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    with pl.scope():
        prefill_attention_swa(
            x_hc,
            hc_attn_fn_l0, hc_attn_scale_l0, hc_attn_base_l0, attn_norm_w_l0,
            wq_a_l0, wq_b_l0, wq_b_scale_l0, wkv_l0, gamma_cq_l0, gamma_ckv_l0,
            freqs_cos, freqs_sin,
            kv_cache_l0, ori_block_table, ori_slot_mapping,
            cmp_sparse_indices, cmp_sparse_lens, position_ids,
            attn_sink_l0, wo_a_l0, wo_b_l0, wo_b_scale_l0,
            x_attn0, nt,
        )
    with pl.scope():
        moe(
            x_attn0,
            hc_ffn_fn_l0, hc_ffn_scale_l0, hc_ffn_base_l0,
            norm_w_l0, gate_w_l0, gate_bias_l0, tid2eid_l0, input_ids,
            routed_w1_l0, routed_w1_scale_l0, routed_w3_l0, routed_w3_scale_l0,
            routed_w2_l0, routed_w2_scale_l0,
            shared_w1_l0, shared_w1_scale_l0, shared_w3_l0, shared_w3_scale_l0,
            shared_w2_l0, shared_w2_scale_l0,
            hidden,
            recv_meta, recv_x, recv_aux, recv_route, arrived,
            routed_y_buf, combine_arrived,
            pl.cast(0, pl.INT32), nt, my_rank, pl.cast(1, pl.INT32),
        )

    # ===================== layer 1 : swa =================================
    hc_attn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0])
    hc_attn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [1 * 3])
    hc_attn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [1 * MIX_HC])
    attn_norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [1 * D])
    wq_a_l1: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [1 * D, 0])
    wq_b_l1: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [1 * Q_LORA, 0])
    wq_b_scale_l1: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [1 * H * HEAD_DIM])
    wkv_l1: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [1 * D, 0])
    gamma_cq_l1: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [1 * Q_LORA])
    gamma_ckv_l1: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [1 * HEAD_DIM])
    kv_cache_l1: pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [1 * CSA_ORI_BLOCK_NUM, 0, 0, 0])
    attn_sink_l1: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [1 * H])
    wo_a_l1: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [1 * O_GROUPS, 0, 0])
    wo_b_l1: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [1 * D, 0])
    wo_b_scale_l1: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [1 * D])
    hc_ffn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0])
    hc_ffn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [1 * 3])
    hc_ffn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [1 * MIX_HC])
    norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [1 * D])
    gate_w_l1: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [1 * N_EXPERTS_GLOBAL, 0])
    gate_bias_l1: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [1 * N_EXPERTS_GLOBAL])
    tid2eid_l1: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [1 * VOCAB, 0])
    routed_w1_l1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [1 * N_LOCAL, 0, 0])
    routed_w1_scale_l1: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [1 * N_LOCAL, 0])
    routed_w3_l1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [1 * N_LOCAL, 0, 0])
    routed_w3_scale_l1: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [1 * N_LOCAL, 0])
    routed_w2_l1: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [1 * N_LOCAL, 0, 0])
    routed_w2_scale_l1: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [1 * N_LOCAL, 0])
    shared_w1_l1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [1 * MOE_INTER, 0])
    shared_w1_scale_l1: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [1 * MOE_INTER])
    shared_w3_l1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [1 * MOE_INTER, 0])
    shared_w3_scale_l1: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [1 * MOE_INTER])
    shared_w2_l1: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [1 * D, 0])
    shared_w2_scale_l1: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [1 * D])
    x_attn1: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    with pl.scope():
        prefill_attention_swa(
            hidden,
            hc_attn_fn_l1, hc_attn_scale_l1, hc_attn_base_l1, attn_norm_w_l1,
            wq_a_l1, wq_b_l1, wq_b_scale_l1, wkv_l1, gamma_cq_l1, gamma_ckv_l1,
            freqs_cos, freqs_sin,
            kv_cache_l1, ori_block_table, ori_slot_mapping,
            cmp_sparse_indices, cmp_sparse_lens, position_ids,
            attn_sink_l1, wo_a_l1, wo_b_l1, wo_b_scale_l1,
            x_attn1, nt,
        )
    with pl.scope():
        moe(
            x_attn1,
            hc_ffn_fn_l1, hc_ffn_scale_l1, hc_ffn_base_l1,
            norm_w_l1, gate_w_l1, gate_bias_l1, tid2eid_l1, input_ids,
            routed_w1_l1, routed_w1_scale_l1, routed_w3_l1, routed_w3_scale_l1,
            routed_w2_l1, routed_w2_scale_l1,
            shared_w1_l1, shared_w1_scale_l1, shared_w3_l1, shared_w3_scale_l1,
            shared_w2_l1, shared_w2_scale_l1,
            hidden,
            recv_meta, recv_x, recv_aux, recv_route, arrived,
            routed_y_buf, combine_arrived,
            pl.cast(1, pl.INT32), nt, my_rank, pl.cast(2, pl.INT32),
        )

    # ============ loop : csa (even) + hca (odd) pairs, layers 2..41 ======
    for loop_i in pl.range(HCA_NUM_LAYERS):
        csa_layer: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 2, pl.INT32)
        hca_layer: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 3, pl.INT32)
        csa_moe_epoch: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 3, pl.INT32)
        hca_moe_epoch: pl.Scalar[pl.INT32] = pl.cast(loop_i * 2 + 4, pl.INT32)

        # ---- csa attention weights (per-FWD by csa_layer, compact by loop_i) ----
        hc_attn_fn_csa: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [csa_layer * MIX_HC, 0])
        hc_attn_scale_csa: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [csa_layer * 3])
        hc_attn_base_csa: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [csa_layer * MIX_HC])
        attn_norm_w_csa: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [csa_layer * D])
        wq_a_csa: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [csa_layer * D, 0])
        wq_b_csa: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [csa_layer * Q_LORA, 0])
        wq_b_scale_csa: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [csa_layer * H * HEAD_DIM])
        wkv_csa: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [csa_layer * D, 0])
        gamma_cq_csa: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [csa_layer * Q_LORA])
        gamma_ckv_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [csa_layer * HEAD_DIM])
        csa_cmp_wkv_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [loop_i * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_wgate_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [loop_i * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [loop_i * CSA_COMPRESS_RATIO, 0])
        csa_cmp_norm_w_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [loop_i * HEAD_DIM])
        csa_cmp_kv_state_csa: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_kv_state, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], [loop_i * CSA_STATE_BLOCK_NUM, 0, 0])
        csa_cmp_score_state_csa: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_score_state, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], [loop_i * CSA_STATE_BLOCK_NUM, 0, 0])
        csa_hadamard_idx_csa: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [loop_i * IDX_HEAD_DIM, 0])
        csa_idx_wq_b_csa: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8] = pl.slice(csa_idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [loop_i * Q_LORA, 0])
        csa_idx_wq_b_scale_csa: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32] = pl.slice(csa_idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [loop_i * IDX_N_HEADS * IDX_HEAD_DIM])
        csa_weights_proj_csa: pl.Tensor[[D, IDX_N_HEADS], pl.BF16] = pl.slice(csa_weights_proj, [D, IDX_N_HEADS], [loop_i * D, 0])
        csa_inner_wkv_csa: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wkv, [INNER_OUT_DIM, D], [loop_i * INNER_OUT_DIM, 0])
        csa_inner_wgate_csa: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wgate, [INNER_OUT_DIM, D], [loop_i * INNER_OUT_DIM, 0])
        csa_inner_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [loop_i * CSA_COMPRESS_RATIO, 0])
        csa_inner_norm_w_csa: pl.Tensor[[IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_inner_norm_w, [IDX_HEAD_DIM], [loop_i * IDX_HEAD_DIM])
        csa_inner_kv_state_csa: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_kv_state, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], [loop_i * INNER_STATE_BLOCK_NUM, 0, 0])
        csa_inner_score_state_csa: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_score_state, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], [loop_i * INNER_STATE_BLOCK_NUM, 0, 0])
        kv_cache_csa: pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer * CSA_ORI_BLOCK_NUM, 0, 0, 0])
        cmp_kv_csa: pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(cmp_kv, [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer * CSA_CMP_BLOCK_NUM, 0, 0, 0])
        idx_kv_cache_csa: pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16] = pl.slice(idx_kv_cache, [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], [loop_i * PREFILL_IDX_BLOCK_NUM, 0, 0, 0])
        attn_sink_csa: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [csa_layer * H])
        wo_a_csa: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [csa_layer * O_GROUPS, 0, 0])
        wo_b_csa: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [csa_layer * D, 0])
        wo_b_scale_csa: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [csa_layer * D])
        hc_ffn_fn_csa: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [csa_layer * MIX_HC, 0])
        hc_ffn_scale_csa: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [csa_layer * 3])
        hc_ffn_base_csa: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [csa_layer * MIX_HC])
        norm_w_csa: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [csa_layer * D])
        gate_w_csa: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [csa_layer * N_EXPERTS_GLOBAL, 0])
        gate_bias_csa: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [csa_layer * N_EXPERTS_GLOBAL])
        tid2eid_csa: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [csa_layer * VOCAB, 0])
        routed_w1_csa: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [csa_layer * N_LOCAL, 0, 0])
        routed_w1_scale_csa: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [csa_layer * N_LOCAL, 0])
        routed_w3_csa: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [csa_layer * N_LOCAL, 0, 0])
        routed_w3_scale_csa: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [csa_layer * N_LOCAL, 0])
        routed_w2_csa: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [csa_layer * N_LOCAL, 0, 0])
        routed_w2_scale_csa: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [csa_layer * N_LOCAL, 0])
        shared_w1_csa: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [csa_layer * MOE_INTER, 0])
        shared_w1_scale_csa: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [csa_layer * MOE_INTER])
        shared_w3_csa: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [csa_layer * MOE_INTER, 0])
        shared_w3_scale_csa: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [csa_layer * MOE_INTER])
        shared_w2_csa: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [csa_layer * D, 0])
        shared_w2_scale_csa: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [csa_layer * D])
        x_attn_csa: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
        hidden_mid: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
        with pl.scope():
            prefill_attention_csa(
                hidden,
                hc_attn_fn_csa, hc_attn_scale_csa, hc_attn_base_csa, attn_norm_w_csa,
                wq_a_csa, wq_b_csa, wq_b_scale_csa, wkv_csa, gamma_cq_csa, gamma_ckv_csa,
                freqs_cos, freqs_sin,
                csa_cmp_wkv_csa, csa_cmp_wgate_csa, csa_cmp_ape_csa, csa_cmp_norm_w_csa,
                csa_cmp_kv_state_csa, csa_cmp_score_state_csa, csa_compress_state_block_table,
                csa_hadamard_idx_csa,
                csa_idx_wq_b_csa, csa_idx_wq_b_scale_csa, csa_weights_proj_csa,
                csa_inner_wkv_csa, csa_inner_wgate_csa, csa_inner_ape_csa, csa_inner_norm_w_csa,
                csa_inner_kv_state_csa, csa_inner_score_state_csa, csa_inner_compress_state_block_table,
                kv_cache_csa, ori_block_table, ori_slot_mapping,
                cmp_kv_csa, cmp_block_table, idx_kv_cache_csa, idx_block_table,
                position_ids, csa_cmp_slot_mapping, csa_idx_slot_mapping,
                csa_state_slot_mapping, csa_inner_state_slot_mapping,
                attn_sink_csa, wo_a_csa, wo_b_csa, wo_b_scale_csa,
                x_attn_csa, nt,
            )
        with pl.scope():
            moe(
                x_attn_csa,
                hc_ffn_fn_csa, hc_ffn_scale_csa, hc_ffn_base_csa,
                norm_w_csa, gate_w_csa, gate_bias_csa, tid2eid_csa, input_ids,
                routed_w1_csa, routed_w1_scale_csa, routed_w3_csa, routed_w3_scale_csa,
                routed_w2_csa, routed_w2_scale_csa,
                shared_w1_csa, shared_w1_scale_csa, shared_w3_csa, shared_w3_scale_csa,
                shared_w2_csa, shared_w2_scale_csa,
                hidden_mid,
                recv_meta, recv_x, recv_aux, recv_route, arrived,
                routed_y_buf, combine_arrived,
                csa_layer, nt, my_rank, csa_moe_epoch,
            )

        # ---- hca attention weights (per-FWD by hca_layer, compact by loop_i) ----
        hc_attn_fn_hca: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [hca_layer * MIX_HC, 0])
        hc_attn_scale_hca: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [hca_layer * 3])
        hc_attn_base_hca: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [hca_layer * MIX_HC])
        attn_norm_w_hca: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [hca_layer * D])
        wq_a_hca: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [hca_layer * D, 0])
        wq_b_hca: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [hca_layer * Q_LORA, 0])
        wq_b_scale_hca: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [hca_layer * H * HEAD_DIM])
        wkv_hca: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [hca_layer * D, 0])
        gamma_cq_hca: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [hca_layer * Q_LORA])
        gamma_ckv_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [hca_layer * HEAD_DIM])
        hca_cmp_wkv_hca: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(hca_cmp_wkv, [HCA_MAIN_OUT_DIM, D], [loop_i * HCA_MAIN_OUT_DIM, 0])
        hca_cmp_wgate_hca: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(hca_cmp_wgate, [HCA_MAIN_OUT_DIM, D], [loop_i * HCA_MAIN_OUT_DIM, 0])
        hca_cmp_ape_hca: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(hca_cmp_ape, [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], [loop_i * HCA_COMPRESS_RATIO, 0])
        hca_cmp_norm_w_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(hca_cmp_norm_w, [HEAD_DIM], [loop_i * HEAD_DIM])
        hca_cmp_kv_state_hca: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(hca_cmp_kv_state, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], [loop_i * HCA_STATE_BLOCK_NUM, 0, 0])
        hca_cmp_score_state_hca: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(hca_cmp_score_state, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], [loop_i * HCA_STATE_BLOCK_NUM, 0, 0])
        kv_cache_hca: pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [hca_layer * CSA_ORI_BLOCK_NUM, 0, 0, 0])
        cmp_kv_hca: pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(cmp_kv, [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [hca_layer * CSA_CMP_BLOCK_NUM, 0, 0, 0])
        attn_sink_hca: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [hca_layer * H])
        wo_a_hca: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [hca_layer * O_GROUPS, 0, 0])
        wo_b_hca: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [hca_layer * D, 0])
        wo_b_scale_hca: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [hca_layer * D])
        hc_ffn_fn_hca: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [hca_layer * MIX_HC, 0])
        hc_ffn_scale_hca: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [hca_layer * 3])
        hc_ffn_base_hca: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [hca_layer * MIX_HC])
        norm_w_hca: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [hca_layer * D])
        gate_w_hca: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [hca_layer * N_EXPERTS_GLOBAL, 0])
        gate_bias_hca: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [hca_layer * N_EXPERTS_GLOBAL])
        tid2eid_hca: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [hca_layer * VOCAB, 0])
        routed_w1_hca: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [hca_layer * N_LOCAL, 0, 0])
        routed_w1_scale_hca: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [hca_layer * N_LOCAL, 0])
        routed_w3_hca: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [hca_layer * N_LOCAL, 0, 0])
        routed_w3_scale_hca: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [hca_layer * N_LOCAL, 0])
        routed_w2_hca: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [hca_layer * N_LOCAL, 0, 0])
        routed_w2_scale_hca: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [hca_layer * N_LOCAL, 0])
        shared_w1_hca: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [hca_layer * MOE_INTER, 0])
        shared_w1_scale_hca: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [hca_layer * MOE_INTER])
        shared_w3_hca: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [hca_layer * MOE_INTER, 0])
        shared_w3_scale_hca: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [hca_layer * MOE_INTER])
        shared_w2_hca: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [hca_layer * D, 0])
        shared_w2_scale_hca: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [hca_layer * D])
        x_attn_hca: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
        with pl.scope():
            prefill_attention_hca(
                hidden_mid,
                hc_attn_fn_hca, hc_attn_scale_hca, hc_attn_base_hca, attn_norm_w_hca,
                wq_a_hca, wq_b_hca, wq_b_scale_hca, wkv_hca, gamma_cq_hca, gamma_ckv_hca,
                freqs_cos, freqs_sin,
                hca_cmp_wkv_hca, hca_cmp_wgate_hca, hca_cmp_ape_hca, hca_cmp_norm_w_hca,
                hca_cmp_kv_state_hca, hca_cmp_score_state_hca, hca_compress_state_block_table,
                kv_cache_hca, ori_slot_mapping, ori_block_table,
                cmp_kv_hca, cmp_block_table, cmp_sparse_indices, cmp_sparse_lens,
                position_ids, hca_cmp_slot_mapping, hca_state_slot_mapping,
                attn_sink_hca, wo_a_hca, wo_b_hca, wo_b_scale_hca,
                x_attn_hca, nt,
            )
        with pl.scope():
            moe(
                x_attn_hca,
                hc_ffn_fn_hca, hc_ffn_scale_hca, hc_ffn_base_hca,
                norm_w_hca, gate_w_hca, gate_bias_hca, tid2eid_hca, input_ids,
                routed_w1_hca, routed_w1_scale_hca, routed_w3_hca, routed_w3_scale_hca,
                routed_w2_hca, routed_w2_scale_hca,
                shared_w1_hca, shared_w1_scale_hca, shared_w3_hca, shared_w3_scale_hca,
                shared_w2_hca, shared_w2_scale_hca,
                hidden,
                recv_meta, recv_x, recv_aux, recv_route, arrived,
                routed_y_buf, combine_arrived,
                hca_layer, nt, my_rank, hca_moe_epoch,
            )

    # ================ layer 42 (FWD_LAST_LAYER) : csa -> x_out ===========
    csa_layer_last: pl.Scalar[pl.INT32] = pl.cast(FWD_LAST_LAYER, pl.INT32)
    csa_order_last: pl.Scalar[pl.INT32] = pl.cast(CSA_LAST_ORDER, pl.INT32)
    last_moe_epoch: pl.Scalar[pl.INT32] = pl.cast(LAST_MOE_EPOCH, pl.INT32)
    hc_attn_fn_last: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [csa_layer_last * MIX_HC, 0])
    hc_attn_scale_last: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [csa_layer_last * 3])
    hc_attn_base_last: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [csa_layer_last * MIX_HC])
    attn_norm_w_last: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [csa_layer_last * D])
    wq_a_last: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [csa_layer_last * D, 0])
    wq_b_last: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [csa_layer_last * Q_LORA, 0])
    wq_b_scale_last: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [csa_layer_last * H * HEAD_DIM])
    wkv_last: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [csa_layer_last * D, 0])
    gamma_cq_last: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [csa_layer_last * Q_LORA])
    gamma_ckv_last: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [csa_layer_last * HEAD_DIM])
    csa_cmp_wkv_last: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [csa_order_last * CSA_MAIN_OUT_DIM, 0])
    csa_cmp_wgate_last: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [csa_order_last * CSA_MAIN_OUT_DIM, 0])
    csa_cmp_ape_last: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [csa_order_last * CSA_COMPRESS_RATIO, 0])
    csa_cmp_norm_w_last: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [csa_order_last * HEAD_DIM])
    csa_cmp_kv_state_last: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_kv_state, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], [csa_order_last * CSA_STATE_BLOCK_NUM, 0, 0])
    csa_cmp_score_state_last: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_score_state, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], [csa_order_last * CSA_STATE_BLOCK_NUM, 0, 0])
    csa_hadamard_idx_last: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [csa_order_last * IDX_HEAD_DIM, 0])
    csa_idx_wq_b_last: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8] = pl.slice(csa_idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [csa_order_last * Q_LORA, 0])
    csa_idx_wq_b_scale_last: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32] = pl.slice(csa_idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [csa_order_last * IDX_N_HEADS * IDX_HEAD_DIM])
    csa_weights_proj_last: pl.Tensor[[D, IDX_N_HEADS], pl.BF16] = pl.slice(csa_weights_proj, [D, IDX_N_HEADS], [csa_order_last * D, 0])
    csa_inner_wkv_last: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wkv, [INNER_OUT_DIM, D], [csa_order_last * INNER_OUT_DIM, 0])
    csa_inner_wgate_last: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wgate, [INNER_OUT_DIM, D], [csa_order_last * INNER_OUT_DIM, 0])
    csa_inner_ape_last: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [csa_order_last * CSA_COMPRESS_RATIO, 0])
    csa_inner_norm_w_last: pl.Tensor[[IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_inner_norm_w, [IDX_HEAD_DIM], [csa_order_last * IDX_HEAD_DIM])
    csa_inner_kv_state_last: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_kv_state, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], [csa_order_last * INNER_STATE_BLOCK_NUM, 0, 0])
    csa_inner_score_state_last: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_score_state, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], [csa_order_last * INNER_STATE_BLOCK_NUM, 0, 0])
    kv_cache_last: pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer_last * CSA_ORI_BLOCK_NUM, 0, 0, 0])
    cmp_kv_last: pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(cmp_kv, [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [csa_layer_last * CSA_CMP_BLOCK_NUM, 0, 0, 0])
    idx_kv_cache_last: pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16] = pl.slice(idx_kv_cache, [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], [csa_order_last * PREFILL_IDX_BLOCK_NUM, 0, 0, 0])
    attn_sink_last: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [csa_layer_last * H])
    wo_a_last: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [csa_layer_last * O_GROUPS, 0, 0])
    wo_b_last: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [csa_layer_last * D, 0])
    wo_b_scale_last: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [csa_layer_last * D])
    hc_ffn_fn_last: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [csa_layer_last * MIX_HC, 0])
    hc_ffn_scale_last: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [csa_layer_last * 3])
    hc_ffn_base_last: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [csa_layer_last * MIX_HC])
    norm_w_last: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [csa_layer_last * D])
    gate_w_last: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [csa_layer_last * N_EXPERTS_GLOBAL, 0])
    gate_bias_last: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [csa_layer_last * N_EXPERTS_GLOBAL])
    tid2eid_last: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [csa_layer_last * VOCAB, 0])
    routed_w1_last: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [csa_layer_last * N_LOCAL, 0, 0])
    routed_w1_scale_last: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [csa_layer_last * N_LOCAL, 0])
    routed_w3_last: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [csa_layer_last * N_LOCAL, 0, 0])
    routed_w3_scale_last: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [csa_layer_last * N_LOCAL, 0])
    routed_w2_last: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [csa_layer_last * N_LOCAL, 0, 0])
    routed_w2_scale_last: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [csa_layer_last * N_LOCAL, 0])
    shared_w1_last: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [csa_layer_last * MOE_INTER, 0])
    shared_w1_scale_last: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [csa_layer_last * MOE_INTER])
    shared_w3_last: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [csa_layer_last * MOE_INTER, 0])
    shared_w3_scale_last: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [csa_layer_last * MOE_INTER])
    shared_w2_last: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [csa_layer_last * D, 0])
    shared_w2_scale_last: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [csa_layer_last * D])
    x_attn_last: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    x_next_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16] = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    with pl.scope():
        prefill_attention_csa(
            hidden,
            hc_attn_fn_last, hc_attn_scale_last, hc_attn_base_last, attn_norm_w_last,
            wq_a_last, wq_b_last, wq_b_scale_last, wkv_last, gamma_cq_last, gamma_ckv_last,
            freqs_cos, freqs_sin,
            csa_cmp_wkv_last, csa_cmp_wgate_last, csa_cmp_ape_last, csa_cmp_norm_w_last,
            csa_cmp_kv_state_last, csa_cmp_score_state_last, csa_compress_state_block_table,
            csa_hadamard_idx_last,
            csa_idx_wq_b_last, csa_idx_wq_b_scale_last, csa_weights_proj_last,
            csa_inner_wkv_last, csa_inner_wgate_last, csa_inner_ape_last, csa_inner_norm_w_last,
            csa_inner_kv_state_last, csa_inner_score_state_last, csa_inner_compress_state_block_table,
            kv_cache_last, ori_block_table, ori_slot_mapping,
            cmp_kv_last, cmp_block_table, idx_kv_cache_last, idx_block_table,
            position_ids, csa_cmp_slot_mapping, csa_idx_slot_mapping,
            csa_state_slot_mapping, csa_inner_state_slot_mapping,
            attn_sink_last, wo_a_last, wo_b_last, wo_b_scale_last,
            x_attn_last, nt,
        )
    with pl.scope():
        moe(
            x_attn_last,
            hc_ffn_fn_last, hc_ffn_scale_last, hc_ffn_base_last,
            norm_w_last, gate_w_last, gate_bias_last, tid2eid_last, input_ids,
            routed_w1_last, routed_w1_scale_last, routed_w3_last, routed_w3_scale_last,
            routed_w2_last, routed_w2_scale_last,
            shared_w1_last, shared_w1_scale_last, shared_w3_last, shared_w3_scale_last,
            shared_w2_last, shared_w2_scale_last,
            x_next_hc,
            recv_meta, recv_x, recv_aux, recv_route, arrived,
            routed_y_buf, combine_arrived,
            csa_layer_last, nt, my_rank, last_moe_epoch,
        )
    x_head: pl.Tensor[[T, D], pl.BF16] = pl.create_tensor([T, D], dtype=pl.BF16)
    with pl.scope():
        hc_head(x_next_hc, hc_head_fn, hc_head_scale, hc_head_base, x_head)
        rms_norm(x_head, final_norm_w, x_out)
    return x_out


@pl.jit.host
def l3_prefill_fwd(
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16],
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
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    cmp_kv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_cmp_kv_state: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_score_state: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_score_state: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    csa_inner_score_state: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    idx_kv_cache: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[N_RANKS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[N_RANKS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, T], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, T], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[N_RANKS, T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[N_RANKS, T], pl.INT32],
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
    shared_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer(N_RANKS * N_LOCAL * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_aux_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * AUX_PAD * 4)
    recv_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)

    for r in pl.range(pld.world_size()):
        recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32] = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8] = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32] = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32] = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16] = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        prefill_fwd(
            x_hc[r],
            hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r], attn_norm_w[r],
            wq_a[r], wq_b[r], wq_b_scale[r], wkv[r], gamma_cq[r], gamma_ckv[r],
            kv_cache[r], attn_sink[r], wo_a[r], wo_b[r], wo_b_scale[r], cmp_kv[r],
            hca_cmp_wkv[r], hca_cmp_wgate[r], hca_cmp_ape[r], hca_cmp_norm_w[r],
            hca_cmp_kv_state[r], hca_cmp_score_state[r],
            csa_cmp_wkv[r], csa_cmp_wgate[r], csa_cmp_ape[r], csa_cmp_norm_w[r],
            csa_cmp_kv_state[r], csa_cmp_score_state[r],
            csa_hadamard_idx[r], csa_idx_wq_b[r], csa_idx_wq_b_scale[r], csa_weights_proj[r],
            csa_inner_wkv[r], csa_inner_wgate[r], csa_inner_ape[r], csa_inner_norm_w[r],
            csa_inner_kv_state[r], csa_inner_score_state[r], idx_kv_cache[r],
            hca_compress_state_block_table[r], csa_compress_state_block_table[r],
            csa_inner_compress_state_block_table[r],
            freqs_cos[r], freqs_sin[r],
            ori_block_table[r], cmp_block_table[r], idx_block_table[r],
            ori_slot_mapping[r], position_ids[r], input_ids[r],
            hca_cmp_slot_mapping[r], hca_state_slot_mapping[r],
            csa_cmp_slot_mapping[r], csa_idx_slot_mapping[r],
            csa_state_slot_mapping[r], csa_inner_state_slot_mapping[r],
            cmp_sparse_indices[r], cmp_sparse_lens[r],
            hc_head_fn[r], hc_head_scale[r], hc_head_base[r], final_norm_w[r],
            hidden_out[r],
            recv_meta, recv_x, recv_aux, recv_route, arrived,
            routed_y_buf, combine_arrived,
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r], norm_w[r],
            gate_w[r], gate_bias[r], tid2eid[r],
            routed_w1[r], routed_w1_scale[r], routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r], shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            r, num_tokens,
            device=r,
        )


# ---------------------------------------------------------------------------
# Fixtures (kernel-only smoke path: no golden).  Stacked weights reuse each
# layer's standalone attention/moe init; routing metadata, slot mappings and
# tid2eid carry meaningful values.
# ---------------------------------------------------------------------------
def _layer_count(name):
    if name in CSA_LAYER_STACKED_NAMES:
        return CSA_NUM_LAYERS
    if name in HCA_LAYER_STACKED_NAMES:
        return HCA_NUM_LAYERS
    if name in FWD_LAYER_STACKED_NAMES:
        return FWD_NUM_LAYERS
    return 1


def _make_stacked_spec(name, base_specs):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    count = _layer_count(name)
    packed_shape = [spec.shape[0], count * spec.shape[1], *spec.shape[2:]]

    def init_value():
        if name == "tid2eid":
            token_ids = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
            topk_ids = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
            rows = []
            for layer in range(count):
                rows.append((token_ids * TOPK + topk_ids + layer * TOPK) % N_EXPERTS_GLOBAL)
            packed = torch.cat(rows, dim=0)
            return packed.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
        base_init = spec.init_value
        return torch.cat([base_init() for _ in range(count)], dim=1)

    # Caches the kernel writes in place (kv_cache) are outputs read back for
    # validation; every other stacked tensor is a plain input (weight or read-only
    # cache).
    return TensorSpec(
        name, packed_shape, spec.dtype, init_value=init_value,
        is_output=name in RESIDENT_CACHE_OUTPUT_NAMES,
    )


def _make_shared_spec(name, base_specs, start_pos):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    pos = torch.arange(start_pos, start_pos + T, dtype=torch.int64)

    def ranked(single):
        return single.unsqueeze(0).expand(N_RANKS, *single.shape).contiguous()

    def init_value():
        if name == "position_ids":
            return ranked(pos.to(torch.int32))
        if name == "input_ids":
            return ranked((torch.arange(T, dtype=torch.int64) % VOCAB))
        if name == "ori_slot_mapping":
            return ranked((pos % BLOCK_SIZE).to(torch.int64))
        if name in ("hca_state_slot_mapping", "csa_state_slot_mapping", "csa_inner_state_slot_mapping"):
            return ranked(pos.to(torch.int64))
        if name == "hca_cmp_slot_mapping":
            out = torch.full((T,), -1, dtype=torch.int64)
            mask = ((pos + 1) % HCA_COMPRESS_RATIO) == 0
            out[mask] = ((pos[mask] + 1) // HCA_COMPRESS_RATIO) - 1
            return ranked(out)
        if name in ("csa_cmp_slot_mapping", "csa_idx_slot_mapping"):
            out = torch.full((T,), -1, dtype=torch.int64)
            mask = ((pos + 1) % CSA_COMPRESS_RATIO) == 0
            out[mask] = ((pos[mask] + 1) // CSA_COMPRESS_RATIO) - 1
            return ranked(out)
        if name == "cmp_sparse_lens":
            return ranked(torch.clamp(torch.arange(1, T + 1, dtype=torch.int32), max=SPARSE_TOPK))
        if name == "cmp_sparse_indices":
            out = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
            for t in range(T):
                valid = min(t + 1, SPARSE_TOPK)
                first = t + 1 - valid
                for k in range(valid):
                    out[t, k] = BLOCK_SIZE + first + k
            return ranked(out)
        # block tables and any remaining shared metadata: smoke zeros.
        return torch.zeros(list(spec.shape), dtype=spec.dtype)

    return TensorSpec(name, list(spec.shape), spec.dtype, init_value=init_value, is_output=False)


def _make_hc_head_spec(name):
    import torch
    from golden import TensorSpec

    if name == "hc_head_fn":
        return TensorSpec(
            name,
            [N_RANKS, HC_MULT, HC_DIM],
            torch.float32,
            init_value=lambda: torch.randn(N_RANKS, HC_MULT, HC_DIM) * 0.0519,
        )
    if name == "hc_head_scale":
        return TensorSpec(
            name,
            [N_RANKS, 1],
            torch.float32,
            init_value=lambda: torch.full((N_RANKS, 1), 0.076099, dtype=torch.float32),
        )
    if name == "hc_head_base":
        base = [5.9166, -3.6223, -2.9324, -3.3124]
        return TensorSpec(
            name,
            [N_RANKS, HC_MULT],
            torch.float32,
            init_value=lambda: torch.tensor(base, dtype=torch.float32).view(1, HC_MULT).expand(N_RANKS, -1).contiguous(),
        )
    raise ValueError(f"unclassified hc_head spec: {name}")


def _make_final_norm_spec(name):
    import torch
    from golden import TensorSpec

    if name == "final_norm_w":
        return TensorSpec(
            name,
            [N_RANKS, D],
            torch.bfloat16,
            init_value=lambda: (torch.randn(N_RANKS, D) * 0.1 + 1.0).to(torch.bfloat16),
        )
    raise ValueError(f"unclassified final norm spec: {name}")


# Canonical host-tensor order for a single unified prefill layer.
HOST_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "freqs_cos",
    "freqs_sin",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_cmp_kv_state",
    "hca_cmp_score_state",
    "hca_compress_state_block_table",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_cmp_kv_state",
    "csa_cmp_score_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_kv_state",
    "csa_inner_score_state",
    "csa_inner_compress_state_block_table",
    "kv_cache",
    "ori_block_table",
    "ori_slot_mapping",
    "cmp_kv",
    "cmp_block_table",
    "cmp_sparse_indices",
    "cmp_sparse_lens",
    "idx_kv_cache",
    "idx_block_table",
    "position_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "input_ids",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "x_next",
)


def _spec_value(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def _ranked_init(spec, n_ranks, torch):
    def init():
        values = [_spec_value(spec, torch) for _ in range(n_ranks)]
        return torch.stack(values, dim=0).contiguous()

    return init


def _ranked_x_hc_init(spec, n_ranks, active_tokens, torch):
    def init():
        values = [_spec_value(spec, torch) for _ in range(n_ranks)]
        stacked = torch.stack(values, dim=0).contiguous()
        active = min(active_tokens, stacked.shape[1])
        if active < stacked.shape[1]:
            inactive = torch.randn(stacked[:, active:].shape, dtype=torch.float32).to(stacked.dtype)
            stacked[:, active:] = inactive / 10.0
        return stacked

    return init


def _attention_kind_for_layer(layer_id):
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeek V4 attention compress ratio {ratio} at layer {layer_id}")


def build_single_layer_tensor_specs(start_pos=START_POS, num_tokens=T, layer_id=2):
    """Per-layer single-rank tensor specs: the base shapes/dtypes/inits that
    build_tensor_specs restacks across the forward layers."""
    import torch
    from golden import ScalarSpec, TensorSpec

    def kind_specs(build_fn):
        return {s.name: s for s in build_fn(start_pos=start_pos, num_tokens=num_tokens) if isinstance(s, TensorSpec)}

    swa = kind_specs(build_swa_attention_tensor_specs)
    hca = kind_specs(build_hca_attention_tensor_specs)
    csa = kind_specs(build_csa_attention_tensor_specs)
    active_kind = _attention_kind_for_layer(layer_id)
    active = {"swa": swa, "hca": hca, "csa": csa}[active_kind]
    active_tokens = num_tokens

    # (layer_name, source_spec). Shared state is taken from the active kind (its
    # init is what the active attention + its golden both consume). The hca_/csa_
    # compressor + indexer params are namespaced from their own kind; compressed
    # KV specs prefer the active attention kind and fall back to CSA for SWA.
    attention_specs = [
        ("x_hc", active["x_hc"]),
        ("hc_attn_fn", active["hc_attn_fn"]),
        ("hc_attn_scale", active["hc_attn_scale"]),
        ("hc_attn_base", active["hc_attn_base"]),
        ("attn_norm_w", active["attn_norm_w"]),
        ("wq_a", active["wq_a"]),
        ("wq_b", active["wq_b"]),
        ("wq_b_scale", active["wq_b_scale"]),
        ("wkv", active["wkv"]),
        ("gamma_cq", active["gamma_cq"]),
        ("gamma_ckv", active["gamma_ckv"]),
        ("freqs_cos", active["freqs_cos"]),
        ("freqs_sin", active["freqs_sin"]),
        ("hca_cmp_wkv", hca["cmp_wkv"]),
        ("hca_cmp_wgate", hca["cmp_wgate"]),
        ("hca_cmp_ape", hca["cmp_ape"]),
        ("hca_cmp_norm_w", hca["cmp_norm_w"]),
        ("hca_cmp_kv_state", hca["cmp_kv_state"]),
        ("hca_cmp_score_state", hca["cmp_score_state"]),
        ("hca_compress_state_block_table", hca["compress_state_block_table"]),
        ("csa_cmp_wkv", csa["cmp_wkv"]),
        ("csa_cmp_wgate", csa["cmp_wgate"]),
        ("csa_cmp_ape", csa["cmp_ape"]),
        ("csa_cmp_norm_w", csa["cmp_norm_w"]),
        ("csa_cmp_kv_state", csa["cmp_kv_state"]),
        ("csa_cmp_score_state", csa["cmp_score_state"]),
        ("csa_compress_state_block_table", csa["compress_state_block_table"]),
        ("csa_hadamard_idx", csa["hadamard_idx"]),
        ("csa_idx_wq_b", csa["idx_wq_b"]),
        ("csa_idx_wq_b_scale", csa["idx_wq_b_scale"]),
        ("csa_weights_proj", csa["idx_weights_proj"]),
        ("csa_inner_wkv", csa["inner_wkv"]),
        ("csa_inner_wgate", csa["inner_wgate"]),
        ("csa_inner_ape", csa["inner_ape"]),
        ("csa_inner_norm_w", csa["inner_norm_w"]),
        ("csa_inner_kv_state", csa["inner_kv_state"]),
        ("csa_inner_score_state", csa["inner_score_state"]),
        ("csa_inner_compress_state_block_table", csa["inner_compress_state_block_table"]),
        ("kv_cache", active["kv_cache"]),
        ("ori_block_table", active.get("ori_block_table", swa.get("block_table"))),
        ("ori_slot_mapping", active["ori_slot_mapping"]),
        ("cmp_kv", active.get("cmp_kv", csa["cmp_kv"])),
        ("cmp_block_table", active.get("cmp_block_table", csa["cmp_block_table"])),
        ("cmp_sparse_indices", active.get("cmp_sparse_indices", swa["cmp_sparse_indices"])),
        ("cmp_sparse_lens", active.get("cmp_sparse_lens", swa["cmp_sparse_lens"])),
        ("idx_kv_cache", csa["idx_kv_cache"]),
        ("idx_block_table", csa["idx_block_table"]),
        ("position_ids", active["position_ids"]),
        ("hca_cmp_slot_mapping", hca["cmp_slot_mapping"]),
        ("hca_state_slot_mapping", hca["state_slot_mapping"]),
        ("csa_cmp_slot_mapping", csa["cmp_slot_mapping"]),
        ("csa_idx_slot_mapping", csa["idx_slot_mapping"]),
        ("csa_state_slot_mapping", csa["state_slot_mapping"]),
        ("csa_inner_state_slot_mapping", csa["inner_state_slot_mapping"]),
        ("attn_sink", active["attn_sink"]),
        ("wo_a", active["wo_a"]),
        ("wo_b", active["wo_b"]),
        ("wo_b_scale", active["wo_b_scale"]),
    ]

    tensor_specs = [
        TensorSpec(
            name,
            [N_RANKS, *src.shape],
            src.dtype,
            init_value=(_ranked_x_hc_init(src, N_RANKS, active_tokens, torch) if name == "x_hc"
                        else _ranked_init(src, N_RANKS, torch)),
            is_output=src.is_output,
        )
        for name, src in attention_specs
    ]

    for spec in build_moe_tensor_specs(layer_id=layer_id):
        if not isinstance(spec, TensorSpec) or spec.name in {"x_hc", "x_next"}:
            continue
        if spec.name == "tid2eid":
            def init_tid2eid(spec=spec):
                _, vocab, topk = spec.shape
                ids = torch.arange(vocab, dtype=torch.int64).view(vocab, 1)
                ks = torch.arange(topk, dtype=torch.int64).view(1, topk)
                table = ((ids * topk + ks) % N_EXPERTS_GLOBAL).to(dtype=spec.dtype)
                return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

            tensor_specs.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_tid2eid))
        elif spec.name == "input_ids":
            def init_input_ids(spec=spec):
                _, tokens = spec.shape
                active = min(active_tokens, tokens)
                rows = []
                for rank in range(N_RANKS):
                    row = torch.roll(torch.arange(tokens, dtype=spec.dtype), shifts=rank)
                    if layer_id >= 3 and active < tokens:
                        row[active:] = -1
                    rows.append(row)
                return torch.stack(rows, dim=0).contiguous()

            tensor_specs.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_input_ids))
        else:
            tensor_specs.append(spec)

    tensor_specs.append(TensorSpec("x_next", [N_RANKS, T, HC_MULT, D], torch.bfloat16, is_output=True))
    tensor_by_name = {spec.name: spec for spec in tensor_specs}
    missing = [name for name in HOST_TENSOR_ORDER if name not in tensor_by_name]
    if missing:
        raise ValueError(f"missing unified prefill layer tensor specs: {missing}")
    return [tensor_by_name[name] for name in HOST_TENSOR_ORDER] + [
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("layer_id", torch.int32, layer_id),
    ]


def build_tensor_specs(start_pos=0, num_tokens=T):
    import torch
    from golden import TensorSpec

    base_specs = {
        spec.name: spec
        for spec in build_single_layer_tensor_specs(start_pos=start_pos, num_tokens=num_tokens, layer_id=0)
        if isinstance(spec, TensorSpec)
    }

    ordered_names = [
        "x_hc",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "cmp_kv",
        "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
        "hca_cmp_kv_state", "hca_cmp_score_state",
        "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
        "csa_cmp_kv_state", "csa_cmp_score_state",
        "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
        "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
        "csa_inner_kv_state", "csa_inner_score_state", "idx_kv_cache",
        "hca_compress_state_block_table", "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
        "freqs_cos", "freqs_sin",
        "ori_block_table", "cmp_block_table", "idx_block_table",
        "ori_slot_mapping", "position_ids", "input_ids",
        "hca_cmp_slot_mapping", "hca_state_slot_mapping",
        "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
        "csa_state_slot_mapping", "csa_inner_state_slot_mapping",
        "cmp_sparse_indices", "cmp_sparse_lens",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
        "hc_head_fn", "hc_head_scale", "hc_head_base",
        "final_norm_w",
    ]

    specs = []
    for name in ordered_names:
        if name == "x_hc":
            base = base_specs[name]

            def init_x_hc(b=base):
                return (torch.randn(list(b.shape)) * 0.05).to(b.dtype)

            specs.append(TensorSpec(name, list(base.shape), base.dtype, init_value=init_x_hc, is_output=False))
        elif name in SHARED_NAMES:
            specs.append(_make_shared_spec(name, base_specs, start_pos))
        elif name in HC_HEAD_NAMES:
            specs.append(_make_hc_head_spec(name))
        elif name in FINAL_NORM_NAMES:
            specs.append(_make_final_norm_spec(name))
        else:
            specs.append(_make_stacked_spec(name, base_specs))

    # Shard the static weight parameters per rank and keep them device-resident
    # (child_memory): each shard uploaded once to its card and reused across
    # dispatches, skipping per-dispatch H2D/D2H. RESIDENT_WEIGHT_NAMES are static
    # weights; RESIDENT_CACHE_NAMES are the KV/state caches (the written kv_cache
    # is also is_output=True and read back at the end via RESIDENT_CACHE_OUTPUT_NAMES).
    for spec in specs:
        if spec.name in RESIDENT_WEIGHT_NAMES or spec.name in RESIDENT_CACHE_NAMES:
            spec.resident = "stacked"

    specs.append(TensorSpec("hidden_out", [N_RANKS, T, D], torch.bfloat16, is_output=True))
    from golden import ScalarSpec
    specs.append(ScalarSpec("num_tokens", torch.int32, num_tokens))
    return specs


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Flash packed-prefill forward driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe).")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=T,
                        help=f"Active token rows for MoE routing/combine; default is T={T}.")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    specs = build_tensor_specs(start_pos=args.start_pos, num_tokens=args.num_tokens)

    result = run_jit(
        fn=l3_prefill_fwd,
        specs=specs,
        golden_fn=None,
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
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
