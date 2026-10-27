# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""GLM-5.3-Flash resident chunked generation, MTP verification and visual input composition."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import inspect
import json
import math
from math import gcd
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import models.glm5_3_flash.config as C

# prefill_moe.py freezes its owner-compacted row capacity from argv at import
# time. Each owner holds ceil(CAPACITY / TP_SIZE) rows, padded to 16 rows.
# The request batch size is independent of this device-owner partition.
_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--capacity", type=int, default=2048)
_shape_parser.add_argument("--max-seq-len", type=int)
_shape_parser.add_argument("--cache-tokens", type=int)
_shape_options = _shape_parser.parse_known_args()[0]
CAPACITY = _shape_options.capacity
MAX_SEQ_LEN = CAPACITY if _shape_options.max_seq_len is None else _shape_options.max_seq_len
_default_cache_tokens = min(16 * ((MAX_SEQ_LEN + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE) * C.BLOCK_SIZE, C.FLASH.max_position_embeddings)
CACHE_TOKENS = _default_cache_tokens if _shape_options.cache_tokens is None else _shape_options.cache_tokens
if CACHE_TOKENS <= 0 or CACHE_TOKENS % C.BLOCK_SIZE:
    raise ValueError(f"--cache-tokens must be positive and aligned to {C.BLOCK_SIZE}")
T_MOE = ((CAPACITY + C.TP_SIZE * 16 - 1) // (C.TP_SIZE * 16)) * 16
if not any(tok == "--tokens" or tok.startswith("--tokens=") for tok in sys.argv):
    sys.argv += ["--tokens", str(T_MOE)]

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig, StackedDeviceTensor
import torch
from models.glm5_3_flash.config import (
    BLOCK_SIZE,
    D,
    DENSE_INTER,
    DP_SIZE,
    EP_SIZE,
    FLASH,
    HC_DIM,
    HC_MULT,
    INDEX_DIM,
    INDEX_H,
    INDEX_KPOOL,
    INDEX_STATE_BLOCK_SIZE,
    INDEX_STATE_WIDTH,
    KDA_CONV_K,
    KDA_DIM,
    KV_LORA,
    LOCAL_H,
    LOCAL_KDA_H,
    LOCAL_KDA_QKV_DIM,
    LOCAL_VOCAB,
    MAX_BATCH_PER_DP,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS,
    N_LOCAL_EXPERTS,
    PREFILL_MAX_TOKENS,
    QK_DIM,
    Q_LORA,
    TOPK,
    TP_SIZE,
    V_DIM,
)

# Production attention and Dense use TP8 or TP16; routed experts use EP at
# the same width, and shared experts stay replicated. The TP2 CI rung covers
# only seed, MoE FFN and head, and requires the entry's explicit marker.
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
# indexer_proj tiles whole PROJ_T_TILE=64 token groups; the graph feeds the static
# capacity straight in, so the capacity itself must be 64-aligned (design said 16).
if CAPACITY % 64:
    raise ValueError(f"--capacity must be a multiple of 64 for indexer_proj, got {CAPACITY}")
if not 1 <= MAX_SEQ_LEN <= FLASH.max_position_embeddings:
    raise ValueError(f"--max-seq-len must be 1..{FLASH.max_position_embeddings}, got {MAX_SEQ_LEN}")

from models.glm5_3_flash.attention_tp import tp_all_reduce
from models.glm5_3_flash.dense_mlp import dense_mlp
from models.glm5_3_flash.expert_routed import expert_routed
from models.glm5_3_flash.expert_shared import expert_shared
from models.glm5_3_flash.gate import moe_gate
from models.glm5_3_flash.indexer_cache import (
    POOL_SCALE_WIDTH,
    POOL_VALID_WIDTH,
    indexer_cache_write,
    indexer_pool_write,
)
from models.glm5_3_flash.kda_conv import CONV_DIM, STATE_LEN, kda_conv_prefill
from models.glm5_3_flash.kda_output import kda_output
from models.glm5_3_flash.kda_projection import BETA_PAD, kda_projection
from models.glm5_3_flash.lm_head import reference_lm_head_logits, FP32_HEAD_VOCAB_PAD, lm_head_logits, lm_head_mtp_logits
from models.glm5_3_flash.metadata import SharedCachePages, build_forward_metadata, build_indexer_chunk_metadata
from models.glm5_3_flash.mhc import plain_residual_pre, plain_residual_post, mhc_head, mhc_mixes, mhc_post, mhc_pre
from models.glm5_3_flash.mla_cache import mla_cache_write
from models.glm5_3_flash.mla_epilog import mla_epilog_w8a8
from models.glm5_3_flash.mla_prolog import mla_prolog_w8a8
from models.glm5_3_flash.mtp_projection import mtp_projection
from models.glm5_3_flash.prefill_indexer import (
    KPOOL_SELECT_K,
    LEAF,
    TOPK_INDEX_WIDTH,
    indexer_expand,
    indexer_proj,
    indexer_score,
    indexer_topk,
)
from models.glm5_3_flash.prefill_kda import CHUNK, prefill_kda
from models.glm5_3_flash.prefill_layer import moe_layer_add
from models.glm5_3_flash.prefill_moe import (
    AUX_PAD,
    COUNT_PAD,
    GROUPED_CAP,
    N_RANKS,
    ROUTES_PER_SRC,
    T,
    TOTAL_CAP,
    prefill_moe_dispatch,
    prefill_moe_return,
)
# prefill_sparse_attn guards TP >= 4 at IMPORT time (its H_TILE tile budget).
# The import is conditional so this module stays importable at a CI fixture's
# TP2 rung, which never traces a DSA attention half: every attention program is
# width-locked and the narrow rung compiles only {seed, moe_ffn, head}. At
# TP >= 4 this is the plain unconditional import of old.
if TP_SIZE >= 4:
    from models.glm5_3_flash.prefill_sparse_attn import prefill_sparse_attn
else:  # pragma: no cover - the TP2 CI rung; tracing a DSA half here is a bug
    prefill_sparse_attn = None
from models.glm5_3_flash.rmsnorm import rmsnorm, rmsnorm_quant
from models.glm5_3_flash.weights import Checkpoint, ResidentPrefillWeights

__all__ = (
    "BATCH", "CAPACITY", "CONSTANT_STEP_INPUTS", "D", "DP_SIZE", "DSA_NUM_LAYERS",
    "DENSE_NUM_LAYERS", "EP_SIZE", "GLM_TENSORS", "GLM_WEIGHT_STACKS", "HC_MULT",
    "KDA_NUM_LAYERS", "LOGIT_ROWS", "MAX_SEQ_LEN", "MLA_STACK_LAYERS", "MOE_NUM_LAYERS",
    "MOE_STACK_LAYERS", "NORM_STACK_LAYERS", "NUM_LAYERS", "OUTPUTS", "RANKS",
    "ROUTER_STACK_LAYERS", "RT_PROGRAMS", "T_MOE", "TP_SIZE", "WORKSPACES",
    "X_H_PINGPONG", "Checkpoint", "MODEL", "ResidentPrefillWeights",
    "_PARAM_TO_STACK", "_STACK_TO_PARAM", "_alloc_name_of", "_l3_arg_arity",
    "_step_key", "compile_production", "l3_prefill_head_rows",
    "l3_prefill_layer_diff_dsa_attn_rt", "l3_prefill_layer_diff_kda_attn_rt",
    "l3_prefill_layer_diff_kda_dense_rt", "ResidentPrefillSession", "l3_prefill_layer_diff_moe_ffn_rt",
    "l3_prefill_seed", "load_production", "prepare_step_inputs", "prefill_head_rows",
    "prefill_layer_diff_dsa_attn_rt", "prefill_layer_diff_kda_attn_rt",
    "prefill_layer_diff_kda_dense_rt", "prefill_layer_diff_moe_ffn_rt",
    "prefill_seed", "production_launch_plan", "production_program_entry",
    "production_program_of_step", "production_step_bindings",
    "production_step_ordinals",
)


RANKS = EP_SIZE

# --- named layer constants (design S1), derived from the model -------------------
# The stacking depths follow the checkpoint the same way weights.py does (its
# _family_layers), so a fixture checkpoint implementing the weights.py read protocol feeds a
# matching ABI: override the model before import via GLM53_PREFILL_MODEL, a JSON
# object of Glm53FlashConfig field overrides (num_hidden_layers,
# first_k_dense_replace, ...). The attention schedule itself (KDA unless
# layer % 4 == 3) is architectural and lives in the config class, so the branch
# arithmetic below is model-faithful at every depth.

def _prefill_model():
    overrides = json.loads(os.environ.get("GLM53_PREFILL_MODEL", "{}"))
    return dataclasses.replace(C.FLASH, **overrides) if overrides else C.FLASH


MODEL = _prefill_model()
NUM_LAYERS = MODEL.num_hidden_layers      # backbone layers; the loop bound and nothing else
KDA_NUM_LAYERS = len(MODEL.kda_layer_ids)  # linear-attention stack depth
DSA_NUM_LAYERS = len(MODEL.dsa_layer_ids) + 1  # sparse-attention cache stack depth
# Includes the MTP layer's private latent and indexer state.
DENSE_NUM_LAYERS = MODEL.first_k_dense_replace  # dense-MLP stack depth
MOE_NUM_LAYERS = NUM_LAYERS - DENSE_NUM_LAYERS  # MoE layers on the backbone
# Loader stack depths: weights.py carries the MTP layer one entry past the backbone
# families; the backbone ordinals never reach the tails.
MLA_STACK_LAYERS = len(MODEL.indexer_layer_ids)
MOE_STACK_LAYERS = MODEL.num_checkpoint_layers - DENSE_NUM_LAYERS
NORM_STACK_LAYERS = MODEL.num_checkpoint_layers
ROUTER_STACK_LAYERS = MOE_STACK_LAYERS

BATCH = 16                           # first-light packed batch: 16 prompts, one batch for all ranks
LOGIT_ROWS = MAX_BATCH_PER_DP        # 32; covers BATCH with headroom (design S3.4)
LOCAL_DENSE_INTER = DENSE_INTER // TP_SIZE

# --- derived cache geometry (design S3.3), all static capacity-bound -------------
TABLE_MAX = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
LATENT_BLOCKS = CACHE_TOKENS // BLOCK_SIZE
LATENT_ROWS = LATENT_BLOCKS * BLOCK_SIZE
# Each request owns a page-aligned pooled-key segment. The score window includes
# one full LEAF of slack for the scorer and top-k tile stores.
POOL_STRIDE = (
    ((MAX_SEQ_LEN + INDEX_KPOOL - 1) // INDEX_KPOOL + INDEX_STATE_BLOCK_SIZE - 1)
    // INDEX_STATE_BLOCK_SIZE * INDEX_STATE_BLOCK_SIZE
)
POOL_AXIS = ((max(LATENT_ROWS // INDEX_KPOOL, CAPACITY // INDEX_KPOOL + BATCH) + 2 * LEAF - 2) // LEAF) * LEAF
POOL_TABLE_ROWS = LATENT_ROWS // INDEX_KPOOL + BLOCK_SIZE // INDEX_KPOOL  # final raw-page span is scratch
KDA_ROWS = MAX_BATCH_PER_DP                      # per-request state pool depth (32)
# max logical pool pages per request: pools of INDEX_KPOOL tokens, paged at
# INDEX_STATE_BLOCK_SIZE pool rows (the field is only used by build_forward_metadata)
POOL_TABLE_MAX = (
    MAX_SEQ_LEN + INDEX_KPOOL * INDEX_STATE_BLOCK_SIZE - 1
) // (INDEX_KPOOL * INDEX_STATE_BLOCK_SIZE)

# Cache reset geometry.
_RESET_MLA_ELEMENTS = TABLE_MAX * BLOCK_SIZE * KV_LORA
_RESET_RAW_ELEMENTS = TABLE_MAX * BLOCK_SIZE * INDEX_STATE_WIDTH
_RESET_POOL_ELEMENTS = POOL_STRIDE * INDEX_DIM
_RESET_SCALE_ELEMENTS = POOL_STRIDE * POOL_SCALE_WIDTH
_RESET_CONV_ELEMENTS = STATE_LEN * CONV_DIM
_RESET_RECURRENT_ELEMENTS = LOCAL_KDA_H * KDA_DIM * KDA_DIM

# Cache reset tiling.
_RESET_MLA_TILE = gcd(4096, _RESET_MLA_ELEMENTS)
_RESET_RAW_TILE = gcd(4096, _RESET_RAW_ELEMENTS)
_RESET_POOL_TILE = gcd(4096, _RESET_POOL_ELEMENTS)
_RESET_SCALE_TILE = gcd(BLOCK_SIZE // INDEX_KPOOL * POOL_SCALE_WIDTH, _RESET_SCALE_ELEMENTS)
_RESET_CONV_TILE = gcd(4096, _RESET_CONV_ELEMENTS)
_RESET_CONV_LANE_TILE = min(4, _RESET_CONV_ELEMENTS // _RESET_CONV_TILE)
_RESET_RECURRENT_TILE = gcd(4096, _RESET_RECURRENT_ELEMENTS)
_RESET_RECURRENT_LANE_TILE = min(4, _RESET_RECURRENT_ELEMENTS // _RESET_RECURRENT_TILE)

# Static packed extents (the donor's settled answer, ``deepseek_v4_1_flash/
# prefill_layer.py`` ``make_layer_program``): every L2/L3 signature below carries
# the token, request, table and state extents as the Python integers derived
# here instead of the config's dynamic symbols. The kernels' own symbolic
# parameter annotations bind to these values at launch, and a kernel-written
# buffer keeps its literal extents across the sublayer handoffs -- with symbolic
# extents the writer's dim expression no longer unifies with the callee's type
# variable, which is exactly the Error-104 "ForStmt dim equality" class the
# conv -> prefill_kda and mla_prolog -> sparse-attn handoffs hit before this
# switch (donor docstring: "a static extent unifies with the T_DYN parameters
# of every sublayer operator, including across the boundary where one
# sublayer's result feeds the next").

assert T == T_MOE and ROUTES_PER_SRC == T_MOE * TOPK
assert LOGIT_ROWS >= BATCH

# =====================================================================================
# Small composition glue (new code; no kernel file is modified anywhere in this file)
# =====================================================================================


@pl.jit.inline
def seed_hc_streams(
    embedding_rows: pl.Tensor[[CAPACITY, D], pl.BF16],
    x_hc: pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16],
):
    """Seed the stream: broadcast the BF16 embedding rows into all four streams
    (``hc_seed``, golden.py). Each site collapses with its own pre-mix (the
    ``attn_hc``/``ffn_hc`` order of modeling_glm5_next.py), so there is no delayed
    pre to seed and no one-hot initial mix.

    Line discipline: each block owns 16 consecutive token rows.
    """
    rows = pl.tensor.dim(embedding_rows, 0)
    x_flat = pl.reshape(x_hc, [rows, HC_DIM])
    with pl.spmd((rows + 15) // 16, name_hint="seed_hc_streams"):
        t0 = pl.tile.get_block_idx() * 16
        for row in pl.unroll(16):
            t = t0 + row
            embedded = pl.tile.load(embedding_rows, [t, 0], [1, D])
            for stream in pl.unroll(HC_MULT):
                pl.tile.store(embedded, [t, stream * D], x_flat)


@pl.jit.inline
def stack_conv_weight(
    conv_q: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    conv_k: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    conv_v: pl.Tensor[[LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
):
    """Assemble this layer's ``[KDA_CONV_K, CONV_DIM]`` depthwise conv weight from
    the three per-family checkpoint stacks (q|k|v channel blocks of 512 each).

    The loader keeps the checkpoint spelling (three ``[channels, 1, kernel]``
    tensors); the kernel wants tap-major rows with the channel axis concatenated
    q-then-k-then-v (kda_conv.py module docstring). Each block owns one 32-column
    BF16 line of every tap row, so the narrow scalar stores never share a 64-byte
    line across blocks; a 32-column group never crosses a family boundary because
    the family stride (512) is itself 32-aligned.
    """
    with pl.spmd(CONV_DIM // 32, name_hint="stack_conv_weight"):
        c0 = pl.tile.get_block_idx() * 32
        for tap in pl.unroll(KDA_CONV_K):
            line = pl.tile.full([1, 32], dtype=pl.BF16, value=0.0)
            for col in pl.unroll(32):
                c = c0 + col
                if c < LOCAL_KDA_QKV_DIM:
                    value = pl.read(conv_q, [c, 0, tap])
                elif c < 2 * LOCAL_KDA_QKV_DIM:
                    value = pl.read(conv_k, [c - LOCAL_KDA_QKV_DIM, 0, tap])
                else:
                    value = pl.read(conv_v, [c - 2 * LOCAL_KDA_QKV_DIM, 0, tap])
                pl.tile.write(line, [0, col], value)
            pl.tile.store(line, [tap, c0], weight)


@pl.jit.inline
def zero_bf16_padding_rows(
    x: pl.Tensor[[CAPACITY, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Zero BF16 rows at or beyond the active prefix (donor ``zero_bf16_padding``).

    tp_all_reduce only publishes rows below ``num_tokens``, so its output's padding
    rows hold stale window bytes; left alone they would flow through mhc_post into
    the stream and could be non-finite. A full-capacity batch floors to one task
    whose body guards itself.
    """
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
):
    """MLA NoPE projection -> latent cache publish -> kpool indexer chain -> sparse
    attention -> head projection, leaving the row-parallel FP32 partial.

    Ordering is the standalone step's proven order (prefill_indexer.py step test):
    the cache and pool writes must precede the sparse attention because the
    always-appended tail reads this dispatch's own rows.
    """
    mla_prolog_w8a8(normed, w_q_a, w_q_a_scale, q_a_norm, w_q_b, w_q_b_scale, w_kv_a, w_kv_a_scale, kv_a_norm, q_resid, query, kv_latent)
    mla_cache_write(kv_latent, mla_slots, latent_cache)
    indexer_proj(
        normed, q_resid, idx_w_q_b, idx_w_k, idx_k_norm_w, idx_k_norm_b,
        idx_w_weights, idx_w_compress_gate, index_q, index_k, head_weights, gate_scores, skip_topk,
    )
    indexer_cache_write(index_k, gate_scores, index_slots, raw_cache)
    indexer_pool_write(
        raw_cache, compress_ape, hadamard, pool_token_slots, pool_slots,
        pool_cache, pool_scale, pool_valid,
    )
    if skip_topk == 0:
        indexer_score(
            index_q, hadamard, pool_cache, pool_scale, pool_blocks, head_weights,
            seg_start, pool_count_tok, index_scores,
        )
        indexer_topk(index_scores, seg_start, pool_count_tok, selected_pools, selected_valid)
    indexer_expand(selected_pools, selected_valid, tail_start, tail_count, kv_len_tok, topk_indices)
    prefill_sparse_attn(query, latent_cache, block_table, request_ids, w_k, w_v, topk_indices, attn_out)
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
    kda_state_rows: pl.Tensor[[BATCH], pl.INT32],
    tri_incl: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    tri_strict: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    eye_wide: pl.Tensor[[CHUNK, KDA_DIM], pl.FP32],
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
):
    """KDA projection -> causal conv -> chunked delta attention -> gated output."""
    stack_conv_weight(conv_q, conv_k, conv_v, conv_w)
    kda_projection(
        normed, w_q, w_k, w_v, w_f_a, w_f_b, dt_bias, a_log, w_b, w_g_a, w_g_b,
        mixed_qkv, decay, beta, out_gate,
    )
    kda_conv_prefill(
        mixed_qkv, conv_w, conv_state, query_start_loc, kda_state_rows, query, key, value
    )
    prefill_kda(
        query, key, value, decay, beta, recurrent_state, query_start_loc, kda_state_rows,
        tri_incl, tri_strict, head_onehot, eye_wide, kda_out,
    )
    # kda_output runs at the L2 level, not here, so the diff entries expose the
    # recurrence-output boundary (the buffer kda_output consumes) that the CPU
    # reference dumps for the per-layer diff.


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
    lane_x: pld.DistributedTensor[[TOTAL_CAP, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[TOTAL_CAP, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    count_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[ROUTES_PER_SRC, D], pl.FP32],
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
    slab_x: pl.Tensor[[GROUPED_CAP, D], pl.INT8],
    slab_scale: pl.Tensor[[GROUPED_CAP, 1], pl.FP32],
    slab_weight: pl.Tensor[[GROUPED_CAP, 1], pl.FP32],
    slab_route: pl.Tensor[[GROUPED_CAP, 1], pl.INT32],
    slab_src: pl.Tensor[[GROUPED_CAP, 1], pl.INT32],
    expert_y: pl.Tensor[[GROUPED_CAP, D], pl.FP32],
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
    moe_gate(
        local_ffn, post_norm_w, gate_w, gate_bias,
        route_weights, route_indices, x_int8, x_scale, local_tokens,
    )
    dispatch_tid = prefill_moe_dispatch(
        route_indices, route_weights, x_int8, x_scale,
        send_counts, recv_expert_counts, expert_counts, expert_offsets,
        lane_x, lane_aux, lane_count, count_ready, x_ready,
        slab_x, slab_scale, slab_weight, slab_route, slab_src,
        local_tokens, my_rank, epoch,
    )
    expert_routed(
        slab_x, slab_scale, w_gate_up, w_gate_up_scale, w_down, w_down_scale,
        expert_offsets, slab_weight, expert_y,
    )
    prefill_moe_return(
        expert_y, slab_route, slab_src, expert_offsets, expert_counts,
        routed, slot_out, return_ready, local_tokens, my_rank, epoch, dispatch_tid,
    )
    if local_tokens > 0:
        expert_shared(
            x_int8, x_scale, shared_w_gate_up, shared_w_gate_up_scale,
            shared_w_down, shared_w_down_scale, shared_out,
        )
    moe_layer_add(routed, shared_out, y_local, local_tokens)
    scatter_local_tokens_fp32(y_local, ffn_partial, num_tokens, my_rank)
    return ffn_partial


# =====================================================================================
# L2: production step entries (donor make_layer_program pattern). The first revision
# composed all 45 layers into ONE traced L2 function; pypto's dim-fold/proof path
# degrades past one sublayer pair per function (logs_t16/README.md), so the file now
# launches small per-sublayer entries instead -- see l3_prefill_fwd below.
# =====================================================================================


# Prefix snapshot bank geometry.
PREFIX_SNAPSHOT_ENTRIES = 2
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
    "kda_recurrent_state": ((PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_RECURRENT_ELEMENTS), torch.float32),
}
PREFIX_SNAPSHOT_BYTES = sum(
    shape[1] * shape[2] * torch.empty((), dtype=dtype).element_size()
    for shape, dtype in PREFIX_SNAPSHOT_LAYOUTS.values()
)


# Prefix snapshot copy tiling.
_PREFIX_COPY_TILE = 1024
_PREFIX_COPY_LANE_TILE = 16
_PREFIX_SCALE_TILE = min(_PREFIX_COPY_TILE, BLOCK_SIZE // INDEX_KPOOL * POOL_SCALE_WIDTH)


@pl.jit(auto_scope=False)
def copy_request_snapshot(
    mla_latent_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    kda_conv_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    bank_mla_latent_cache: pl.InOut[pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_MLA_ELEMENTS], pl.BF16]],
    bank_index_raw_cache: pl.InOut[pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_RAW_ELEMENTS], pl.FP32]],
    bank_index_pool_cache: pl.InOut[pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_POOL_ELEMENTS], pl.INT8]],
    bank_index_pool_scale: pl.InOut[pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_SCALE_ELEMENTS], pl.FP32]],
    bank_kda_conv_state: pl.InOut[pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_CONV_ELEMENTS], pl.BF16]],
    bank_kda_recurrent_state: pl.InOut[pl.Tensor[[PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_RECURRENT_ELEMENTS], pl.FP32]],
    block_table: pl.Tensor[[BATCH, TABLE_MAX], pl.INT32],
    snapshot_length: pl.Scalar[pl.INT32],
    request_slot: pl.Scalar[pl.INT32],
    bank_id: pl.Scalar[pl.INT32],
    restore: pl.Scalar[pl.INT32],
):
    """Copy all private request-cache bytes to or from one immutable bank."""
    latent_flat = pl.reshape(mla_latent_cache, [DSA_NUM_LAYERS, LATENT_ROWS * KV_LORA])
    bank_latent_flat = pl.reshape(bank_mla_latent_cache, [PREFIX_SNAPSHOT_ENTRIES * DSA_NUM_LAYERS, _RESET_MLA_ELEMENTS])
    for block in pl.spmd(DSA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_mla_latent_cache"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(lane * _PREFIX_COPY_TILE, (snapshot_length) * KV_LORA, _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE):
            valid = pl.min(_PREFIX_COPY_TILE, (snapshot_length) * KV_LORA - offset)
            logical_page = offset // ((BLOCK_SIZE) * KV_LORA)
            physical_page = pl.cast(pl.read(block_table, [request_slot, logical_page]), pl.INDEX)
            source_offset = physical_page * (BLOCK_SIZE) * KV_LORA + offset % ((BLOCK_SIZE) * KV_LORA)
            bank_row = pl.cast(bank_id, pl.INDEX) * DSA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_latent_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                latent_flat[layer:layer + 1, source_offset:source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(latent_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_latent_flat[bank_row:bank_row + 1, offset:offset + _PREFIX_COPY_TILE] = data

    raw_flat = pl.reshape(index_raw_cache, [DSA_NUM_LAYERS, LATENT_ROWS * INDEX_STATE_WIDTH])
    bank_raw_flat = pl.reshape(bank_index_raw_cache, [PREFIX_SNAPSHOT_ENTRIES * DSA_NUM_LAYERS, _RESET_RAW_ELEMENTS])
    for block in pl.spmd(DSA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_index_raw_cache"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(lane * _PREFIX_COPY_TILE, (snapshot_length) * INDEX_STATE_WIDTH, _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE):
            valid = pl.min(_PREFIX_COPY_TILE, (snapshot_length) * INDEX_STATE_WIDTH - offset)
            logical_page = offset // ((BLOCK_SIZE) * INDEX_STATE_WIDTH)
            physical_page = pl.cast(pl.read(block_table, [request_slot, logical_page]), pl.INDEX)
            source_offset = physical_page * (BLOCK_SIZE) * INDEX_STATE_WIDTH + offset % ((BLOCK_SIZE) * INDEX_STATE_WIDTH)
            bank_row = pl.cast(bank_id, pl.INDEX) * DSA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_raw_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                raw_flat[layer:layer + 1, source_offset:source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(raw_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_raw_flat[bank_row:bank_row + 1, offset:offset + _PREFIX_COPY_TILE] = data

    pool_flat = pl.reshape(index_pool_cache, [DSA_NUM_LAYERS, POOL_TABLE_ROWS * INDEX_DIM])
    bank_pool_flat = pl.reshape(bank_index_pool_cache, [PREFIX_SNAPSHOT_ENTRIES * DSA_NUM_LAYERS, _RESET_POOL_ELEMENTS])
    for block in pl.spmd(DSA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_index_pool_cache"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(lane * _PREFIX_COPY_TILE, (snapshot_length // INDEX_KPOOL) * INDEX_DIM, _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE):
            valid = pl.min(_PREFIX_COPY_TILE, (snapshot_length // INDEX_KPOOL) * INDEX_DIM - offset)
            logical_page = offset // ((BLOCK_SIZE // INDEX_KPOOL) * INDEX_DIM)
            physical_page = pl.cast(pl.read(block_table, [request_slot, logical_page]), pl.INDEX)
            source_offset = physical_page * (BLOCK_SIZE // INDEX_KPOOL) * INDEX_DIM + offset % ((BLOCK_SIZE // INDEX_KPOOL) * INDEX_DIM)
            bank_row = pl.cast(bank_id, pl.INDEX) * DSA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_pool_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                pool_flat[layer:layer + 1, source_offset:source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(pool_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_pool_flat[bank_row:bank_row + 1, offset:offset + _PREFIX_COPY_TILE] = data

    scale_flat = pl.reshape(index_pool_scale, [DSA_NUM_LAYERS, POOL_TABLE_ROWS * POOL_SCALE_WIDTH])
    bank_scale_flat = pl.reshape(bank_index_pool_scale, [PREFIX_SNAPSHOT_ENTRIES * DSA_NUM_LAYERS, _RESET_SCALE_ELEMENTS])
    for block in pl.spmd(DSA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_index_pool_scale"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(lane * _PREFIX_SCALE_TILE, (snapshot_length // INDEX_KPOOL) * POOL_SCALE_WIDTH, _PREFIX_COPY_LANE_TILE * _PREFIX_SCALE_TILE):
            valid = pl.min(_PREFIX_SCALE_TILE, (snapshot_length // INDEX_KPOOL) * POOL_SCALE_WIDTH - offset)
            logical_page = offset // ((BLOCK_SIZE // INDEX_KPOOL) * POOL_SCALE_WIDTH)
            physical_page = pl.cast(pl.read(block_table, [request_slot, logical_page]), pl.INDEX)
            source_offset = physical_page * (BLOCK_SIZE // INDEX_KPOOL) * POOL_SCALE_WIDTH + offset % ((BLOCK_SIZE // INDEX_KPOOL) * POOL_SCALE_WIDTH)
            bank_row = pl.cast(bank_id, pl.INDEX) * DSA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_scale_flat, [1, _PREFIX_SCALE_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                scale_flat[layer:layer + 1, source_offset:source_offset + _PREFIX_SCALE_TILE] = data
            else:
                data = pl.slice(scale_flat, [1, _PREFIX_SCALE_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_scale_flat[bank_row:bank_row + 1, offset:offset + _PREFIX_SCALE_TILE] = data

    conv_flat = pl.reshape(kda_conv_state, [KDA_NUM_LAYERS, KDA_ROWS * STATE_LEN * CONV_DIM])
    bank_conv_flat = pl.reshape(bank_kda_conv_state, [PREFIX_SNAPSHOT_ENTRIES * KDA_NUM_LAYERS, _RESET_CONV_ELEMENTS])
    for block in pl.spmd(KDA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_kda_conv_state"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(lane * _PREFIX_COPY_TILE, _RESET_CONV_ELEMENTS, _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE):
            valid = pl.min(_PREFIX_COPY_TILE, _RESET_CONV_ELEMENTS - offset)
            source_offset = pl.cast(request_slot, pl.INDEX) * _RESET_CONV_ELEMENTS + offset
            bank_row = pl.cast(bank_id, pl.INDEX) * KDA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_conv_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                conv_flat[layer:layer + 1, source_offset:source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(conv_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_conv_flat[bank_row:bank_row + 1, offset:offset + _PREFIX_COPY_TILE] = data

    recurrent_flat = pl.reshape(kda_recurrent_state, [KDA_NUM_LAYERS, KDA_ROWS * LOCAL_KDA_H * KDA_DIM * KDA_DIM])
    bank_recurrent_flat = pl.reshape(bank_kda_recurrent_state, [PREFIX_SNAPSHOT_ENTRIES * KDA_NUM_LAYERS, _RESET_RECURRENT_ELEMENTS])
    for block in pl.spmd(KDA_NUM_LAYERS * _PREFIX_COPY_LANE_TILE, name_hint="snapshot_kda_recurrent_state"):
        layer = block // _PREFIX_COPY_LANE_TILE
        lane = block % _PREFIX_COPY_LANE_TILE
        for offset in pl.range(lane * _PREFIX_COPY_TILE, _RESET_RECURRENT_ELEMENTS, _PREFIX_COPY_LANE_TILE * _PREFIX_COPY_TILE):
            valid = pl.min(_PREFIX_COPY_TILE, _RESET_RECURRENT_ELEMENTS - offset)
            source_offset = pl.cast(request_slot, pl.INDEX) * _RESET_RECURRENT_ELEMENTS + offset
            bank_row = pl.cast(bank_id, pl.INDEX) * KDA_NUM_LAYERS + layer
            if restore != 0:
                data = pl.slice(bank_recurrent_flat, [1, _PREFIX_COPY_TILE], [bank_row, offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                recurrent_flat[layer:layer + 1, source_offset:source_offset + _PREFIX_COPY_TILE] = data
            else:
                data = pl.slice(recurrent_flat, [1, _PREFIX_COPY_TILE], [layer, source_offset], valid_shape=[1, valid])
                data = pl.set_validshape(data, 1, valid)
                bank_recurrent_flat[bank_row:bank_row + 1, offset:offset + _PREFIX_COPY_TILE] = data
    return (
        mla_latent_cache, index_raw_cache, index_pool_cache, index_pool_scale,
        kda_conv_state, kda_recurrent_state,
        bank_mla_latent_cache, bank_index_raw_cache, bank_index_pool_cache, bank_index_pool_scale,
        bank_kda_conv_state, bank_kda_recurrent_state,
    )


@pl.jit.host
def l3_copy_request_snapshot(
    mla_latent_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    kda_conv_state: pl.InOut[pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    bank_mla_latent_cache: pl.InOut[pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_MLA_ELEMENTS], pl.BF16]],
    bank_index_raw_cache: pl.InOut[pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_RAW_ELEMENTS], pl.FP32]],
    bank_index_pool_cache: pl.InOut[pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_POOL_ELEMENTS], pl.INT8]],
    bank_index_pool_scale: pl.InOut[pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, DSA_NUM_LAYERS, _RESET_SCALE_ELEMENTS], pl.FP32]],
    bank_kda_conv_state: pl.InOut[pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_CONV_ELEMENTS], pl.BF16]],
    bank_kda_recurrent_state: pl.InOut[pl.Tensor[[RANKS, PREFIX_SNAPSHOT_ENTRIES, KDA_NUM_LAYERS, _RESET_RECURRENT_ELEMENTS], pl.FP32]],
    block_table: pl.Tensor[[RANKS, BATCH, TABLE_MAX], pl.INT32],
    snapshot_length: pl.Scalar[pl.INT32],
    request_slot: pl.Scalar[pl.INT32],
    bank_id: pl.Scalar[pl.INT32],
    restore: pl.Scalar[pl.INT32],
):
    """Copy the same request state on every resident TP/EP rank."""
    for rank in pl.unroll(RANKS):
        copy_request_snapshot(
            mla_latent_cache[rank], index_raw_cache[rank], index_pool_cache[rank],
            index_pool_scale[rank], kda_conv_state[rank], kda_recurrent_state[rank],
            bank_mla_latent_cache[rank], bank_index_raw_cache[rank], bank_index_pool_cache[rank],
            bank_index_pool_scale[rank], bank_kda_conv_state[rank], bank_kda_recurrent_state[rank],
            block_table[rank], snapshot_length, request_slot, bank_id, restore, device=rank,
        )


@pl.jit(auto_scope=False)
def reset_request_caches(
    mla_latent_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    kda_conv_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
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
                for chunk in pl.range(lane, (BLOCK_SIZE // INDEX_KPOOL) * POOL_SCALE_WIDTH // _RESET_SCALE_TILE, 4):
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
    return (
        mla_latent_cache, index_raw_cache, index_pool_cache, index_pool_scale,
        kda_conv_state, kda_recurrent_state,
    )


@pl.jit.host
def l3_reset_request_caches(
    mla_latent_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA], pl.BF16]],
    index_raw_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    index_pool_cache: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM], pl.INT8]],
    index_pool_scale: pl.InOut[pl.Tensor[[RANKS, DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    kda_conv_state: pl.InOut[pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    kda_recurrent_state: pl.InOut[pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    cache_page_owners: pl.Tensor[[RANKS, LATENT_BLOCKS], pl.INT32],
    slot_mask: pl.Scalar[pl.INT32],
):
    """Clear the same selected request slots on every resident rank."""
    for rank in pl.unroll(RANKS):
        reset_request_caches(
            mla_latent_cache[rank], index_raw_cache[rank],
            index_pool_cache[rank], index_pool_scale[rank],
            kda_conv_state[rank], kda_recurrent_state[rank], cache_page_owners[rank], slot_mask, device=rank,
        )


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
    head_all = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    mhc_head(x_hc, head_all)
    gather_head_rows(head_all, logit_row_indices, head_hidden)
    return head_hidden



# =====================================================================================
# L2: the single-layer diff entry (verifier handoff, design S2.1)
# =====================================================================================


# Per-kind diff entries: a traced If around the kernel chains degrades every
# dim to a runtime var, so the attention/FFN kind is selected HERE, in
# Python, before tracing: the dense/kda pairing covers layers below
# first_k_dense_replace; every other backbone layer is kda_moe or dsa_moe by
# the % 4 schedule.
@pl.jit(auto_scope=False)
def prefill_layer_diff_kda_attn_rt(
    # -- sublayer input: the stream entering the attention half ----------------------
    x_hc: pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16],
    # -- metadata and per-step inputs (the KDA chain's own subset) -------------------
    query_start_loc: pl.Tensor[[BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[BATCH], pl.INT32],
    active_tokens: pl.Tensor[[1], pl.INT32],
    kda_tri_incl: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    kda_tri_strict: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    kda_head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    kda_eye_wide: pl.Tensor[[CHUNK, KDA_DIM], pl.FP32],
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
    kda_conv_state: pl.InOut[
        pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]
    ],
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
):
    """Runtime-ordinal KDA-attention entry (t29 probe origin; the retired
    constexpr twin is recoverable at git tag ``glm53-pretrim-baseline``).

    Identical dataflow and kernel chain; the only difference is how the layer's
    weight views are taken: ``bb_ord`` (backbone ordinal -- the mHC attention
    site and the input norm) and ``kda_ord`` (KDA family ordinal -- the kda
    weight stacks and the conv/recurrent cache pools) arrive as dispatch-time
    ``pl.Scalar[pl.INT32]`` arguments and index the resident stacks as dynamic
    subscripts, the ``deepseek_v4_flash_mtp`` donor's production form
    (``wq_a[layer_index]``: a runtime-offset ``tensor.slice`` whose extents stay
    static, so no dim ever degrades to a runtime variable). One artifact
    therefore serves every KDA-attention sublayer of the schedule; the host
    computes ``bb_ord = layer`` and ``kda_ord = layer - layer // 4`` and passes
    them per dispatch. This twin began as the t29 consolidation probe and is
    now the ONLY KDA-attention entry: t29/t30 proved its artifact compiles
    without the dim-fold poison and matches the retired per-layer constexpr
    artifacts' outputs bit-for-bit (``logs_t29/``, ``logs_t30/README.md``).
    """
    num_tokens: pl.Scalar[pl.INT32] = pl.read(active_tokens, [0])
    t_tokens = pl.tensor.dim(x_hc, 0)

    conv_w = pl.create_tensor([KDA_CONV_K, CONV_DIM], dtype=pl.BF16)
    out_gate = pl.create_tensor([t_tokens, LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)

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

    with pl.scope():
        mhc_mixes(x_hc, attn_fn_l, attn_scale_l, attn_base_l,
                  out_attn_pre, out_attn_post, out_attn_res)
        mhc_pre(x_hc, out_attn_pre, out_collapsed)
        rmsnorm(out_collapsed, input_norm_l, out_normed)
        kda_attention_sublayer(
            out_normed,
            kda_w_q_l, kda_w_k_l, kda_w_v_l,
            kda_w_f_a_l, kda_w_f_b_l,
            kda_dt_bias_l, kda_a_log_l, kda_w_b_l,
            kda_w_g_a_l, kda_w_g_b_l,
            kda_conv_q_l, kda_conv_k_l, kda_conv_v_l,
            kda_o_norm_l, kda_w_o_l,
            conv_state_l, recurrent_state_l,
            query_start_loc, kda_state_rows,
            kda_tri_incl, kda_tri_strict, kda_head_onehot, kda_eye_wide,
            conv_w, mixed_qkv, decay, beta, out_gate,
            kda_q, kda_k, kda_v, kda_out,
            out_attn_partial,
        )
        kda_output(kda_out, kda_o_norm_l, out_gate, kda_w_o_l, out_attn_partial)
        tp_all_reduce(out_attn_partial, tp_partial, tp_arrived, out_sublayer_out,
                      num_tokens, my_rank, 0, attn_epoch)
        zero_bf16_padding_rows(out_sublayer_out, num_tokens)
        mhc_post(out_sublayer_out, x_hc, out_attn_post, out_attn_res, out_x_hc_mid)

    return (
        out_attn_pre, out_attn_post, out_attn_res, out_collapsed, out_normed,
        out_attn_partial, out_sublayer_out, out_x_hc_mid,
    )



@pl.jit(auto_scope=False)
def prefill_layer_diff_moe_ffn_rt(
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
    lane_x: pld.DistributedTensor[[TOTAL_CAP, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[TOTAL_CAP, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[N_RANKS, COUNT_PAD], pl.INT32],
    count_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_ready: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[ROUTES_PER_SRC, D], pl.FP32],
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
    """Runtime-ordinal MoE-FFN entry (t29 probe origin; the retired constexpr
    twin is recoverable at git tag ``glm53-pretrim-baseline``).

    Identical dataflow and kernel chain; the layer's weight views come from
    two dispatch-time ``pl.Scalar[pl.INT32]`` ordinals: ``bb_ord`` (backbone
    ordinal -- the mHC FFN site and the post norm) and ``moe_ord`` (the MoE
    family ordinal -- router and the routed/shared expert stacks), used as
    dynamic subscripts in the ``deepseek_v4_flash_mtp`` donor's production
    form (runtime-offset ``tensor.slice``, static extents). The host computes
    ``bb_ord = layer`` and ``moe_ord = layer - DENSE_NUM_LAYERS``; the graph
    carries no layer arithmetic. This twin began as the t29 probe deciding
    whether the launch plan's 42 FFN steps can share one artifact and is now
    the ONLY MoE-FFN entry (bit-equal to the retired constexpr artifacts:
    ``logs_t29/``, ``logs_t30/README.md``).
    """
    num_tokens: pl.Scalar[pl.INT32] = pl.read(active_tokens, [0])
    t_tokens = pl.tensor.dim(x_hc_mid, 0)

    local_ffn = pl.create_tensor([T_MOE, D], dtype=pl.BF16)
    moe_route_weights = pl.create_tensor([T_MOE, TOPK], dtype=pl.FP32)
    moe_route_indices = pl.create_tensor([T_MOE, TOPK], dtype=pl.INT32)
    moe_x_int8 = pl.create_tensor([T_MOE, D], dtype=pl.INT8)
    moe_x_scale = pl.create_tensor([T_MOE, 1], dtype=pl.FP32)
    send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32)
    recv_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL_EXPERTS], dtype=pl.INT32)
    expert_counts = pl.create_tensor([N_LOCAL_EXPERTS, 1], dtype=pl.INT32)
    expert_offsets = pl.create_tensor([N_LOCAL_EXPERTS + 1], dtype=pl.INT32)
    slab_x = pl.create_tensor([GROUPED_CAP, D], dtype=pl.INT8)
    slab_scale = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.FP32)
    slab_weight = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.FP32)
    slab_route = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.INT32)
    slab_src = pl.create_tensor([GROUPED_CAP, 1], dtype=pl.INT32)
    expert_y = pl.create_tensor([GROUPED_CAP, D], dtype=pl.FP32)
    routed = pl.create_tensor([T_MOE, D], dtype=pl.FP32)
    shared_out = pl.create_tensor([T_MOE, D], dtype=pl.FP32)
    y_local = pl.create_tensor([T_MOE, D], dtype=pl.FP32)

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

    with pl.scope():
        if bb_ord == NUM_LAYERS:
            plain_residual_pre(x_hc_mid, out_ffn_input, out_ffn_pre, out_ffn_post, out_ffn_res)
        else:
            ffn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16] = mhc_ffn_fn[bb_ord]
            ffn_scale_l: pl.Tensor[[3], pl.FP32] = mhc_ffn_scale[bb_ord]
            ffn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = mhc_ffn_base[bb_ord]
            mhc_mixes(x_hc_mid, ffn_fn_l, ffn_scale_l, ffn_base_l,
                      out_ffn_pre, out_ffn_post, out_ffn_res)
            mhc_pre(x_hc_mid, out_ffn_pre, out_ffn_input)
        moe_ffn_sublayer(
            out_ffn_input, post_norm_l,
            router_gate_w_l, router_gate_bias_l,
            moe_w_gate_up_l, moe_w_gate_up_scale_l,
            moe_w_down_l, moe_w_down_scale_l,
            moe_shared_w_gate_up_l, moe_shared_w_gate_up_scale_l,
            moe_shared_w_down_l, moe_shared_w_down_scale_l,
            lane_x, lane_aux, lane_count, count_ready, x_ready,
            slot_out, return_ready,
            local_ffn, moe_route_weights, moe_route_indices,
            moe_x_int8, moe_x_scale,
            send_counts, recv_expert_counts, expert_counts, expert_offsets,
            slab_x, slab_scale, slab_weight, slab_route, slab_src,
            expert_y, routed, shared_out, y_local, out_ffn_partial,
            num_tokens, my_rank, moe_epoch,
        )
        tp_all_reduce(out_ffn_partial, tp_partial, tp_arrived, out_ffn_out,
                      num_tokens, my_rank, 0, ffn_epoch)
        zero_bf16_padding_rows(out_ffn_out, num_tokens)
        if bb_ord == NUM_LAYERS:
            plain_residual_post(out_ffn_out, x_hc_mid, out_x_hc_out)
        else:
            mhc_post(out_ffn_out, x_hc_mid, out_ffn_post, out_ffn_res, out_x_hc_out)

    return (
        out_ffn_pre, out_ffn_post, out_ffn_res, out_ffn_input, out_ffn_partial,
        out_ffn_out, out_x_hc_out,
    )



@pl.jit(auto_scope=False)
def prefill_layer_diff_kda_dense_rt(
    # -- layer input: the stream entering the layer --------------------------------
    x_hc: pl.Tensor[[CAPACITY, HC_MULT, D], pl.BF16],
    # -- metadata and per-step inputs (the KDA chain's own subset; the dense
    #    layer runs no DSA/indexer/MoE chain, so those tensors left the ABI) --------
    query_start_loc: pl.Tensor[[BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[BATCH], pl.INT32],
    active_tokens: pl.Tensor[[1], pl.INT32],
    kda_tri_incl: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    kda_tri_strict: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    kda_head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    kda_eye_wide: pl.Tensor[[CHUNK, KDA_DIM], pl.FP32],
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
    kda_conv_state: pl.InOut[
        pl.Tensor[[KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]
    ],
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
):
    """Runtime-ordinal dense whole-layer entry (t30; the retired constexpr
    twin is recoverable at git tag ``glm53-pretrim-baseline``).

    Identical dataflow and kernel chain -- the KDA attention half and the dense
    MLP half in one entry, the only entry whose two ``tp_all_reduce`` calls
    share one fresh window set (epochs 1 then 2). The layer enters as three
    dispatch-time ``pl.Scalar[pl.INT32]`` ordinals: ``bb_ord`` (backbone
    ordinal -- both mHC sites and both norms), ``kda_ord`` (the KDA family:
    weight stacks and the conv/recurrent cache pools) and ``dense_ord`` (the
    dense MLP family), each a dynamic ``stack[ordinal]`` view with static
    extents (the t29-proven ``deepseek_v4_flash_mtp`` production form). The
    host computes ``bb_ord = layer``, ``kda_ord = layer - layer // 4`` and
    ``dense_ord = layer % DENSE_NUM_LAYERS``; the graph carries no layer
    arithmetic.

    One deliberate trim versus the retired constexpr whole-layer entry (git
    tag ``glm53-pretrim-baseline``): that entry descends from the pre-split
    single graph and carried the other kinds' dead surface (MLA/indexer/
    router/MoE stacks and their dead views, the DSA boundary buffers, the MoE
    lane windows). A runtime-ordinal program cannot keep dead views -- their
    ordinals would also have to ride the dispatch for nothing -- so this
    entry carries exactly the dense layer's live dataflow. Every boundary
    name and mutability the retired entry actually writes is unchanged (t30:
    12/12 dispatches bit-equal, ``logs_t30/README.md``).
    """
    num_tokens: pl.Scalar[pl.INT32] = pl.read(active_tokens, [0])
    t_tokens = pl.tensor.dim(x_hc, 0)

    conv_w = pl.create_tensor([KDA_CONV_K, CONV_DIM], dtype=pl.BF16)
    out_gate = pl.create_tensor([t_tokens, LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)
    dense_normed = pl.create_tensor([t_tokens, D], dtype=pl.BF16)
    dense_x_scale = pl.create_tensor([t_tokens, 1], dtype=pl.FP32)

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

    with pl.scope():
        mhc_mixes(x_hc, attn_fn_l, attn_scale_l, attn_base_l,
                  out_attn_pre, out_attn_post, out_attn_res)
        mhc_pre(x_hc, out_attn_pre, out_collapsed)
        rmsnorm(out_collapsed, input_norm_l, out_normed)
        kda_attention_sublayer(
            out_normed,
            kda_w_q_l, kda_w_k_l, kda_w_v_l,
            kda_w_f_a_l, kda_w_f_b_l,
            kda_dt_bias_l, kda_a_log_l, kda_w_b_l,
            kda_w_g_a_l, kda_w_g_b_l,
            kda_conv_q_l, kda_conv_k_l, kda_conv_v_l,
            kda_o_norm_l, kda_w_o_l,
            conv_state_l, recurrent_state_l,
            query_start_loc, kda_state_rows,
            kda_tri_incl, kda_tri_strict, kda_head_onehot, kda_eye_wide,
            conv_w, mixed_qkv, decay, beta, out_gate,
            kda_q, kda_k, kda_v, kda_out,
            out_attn_partial,
        )
        kda_output(kda_out, kda_o_norm_l, out_gate, kda_w_o_l, out_attn_partial)
        tp_all_reduce(out_attn_partial, tp_partial, tp_arrived, out_sublayer_out,
                      num_tokens, my_rank, 0, attn_epoch)
        zero_bf16_padding_rows(out_sublayer_out, num_tokens)
        mhc_post(out_sublayer_out, x_hc, out_attn_post, out_attn_res, out_x_hc_mid)

    with pl.scope():
        mhc_mixes(out_x_hc_mid, ffn_fn_l, ffn_scale_l, ffn_base_l,
                  out_ffn_pre, out_ffn_post, out_ffn_res)
        mhc_pre(out_x_hc_mid, out_ffn_pre, out_ffn_input)
        dense_ffn_sublayer(
            out_ffn_input, post_norm_l,
            dense_w_gate_up_l, dense_w_gate_up_scale_l,
            dense_w_down_l, dense_w_down_scale_l,
            num_tokens, dense_normed, dense_x_int8, dense_x_scale, out_ffn_partial,
        )
        tp_all_reduce(out_ffn_partial, tp_partial, tp_arrived, out_ffn_out,
                      num_tokens, my_rank, 0, ffn_epoch)
        zero_bf16_padding_rows(out_ffn_out, num_tokens)
        mhc_post(out_ffn_out, out_x_hc_mid, out_ffn_post, out_ffn_res, out_x_hc_out)

    return (
        out_attn_pre, out_attn_post, out_attn_res, out_collapsed, out_normed,
        out_attn_partial, out_sublayer_out, out_x_hc_mid, out_ffn_pre, out_ffn_post,
        out_ffn_res, out_ffn_input, out_ffn_partial, out_ffn_out, out_x_hc_out,
    )


@pl.jit(auto_scope=False)
def prefill_layer_diff_dsa_attn_rt(
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
):
    """Runtime-ordinal DSA-attention entry (t30; the retired constexpr twin
    is recoverable at git tag ``glm53-pretrim-baseline``).

    Identical dataflow and kernel chain; the layer's weight views come from two
    dispatch-time ``pl.Scalar[pl.INT32]`` ordinals: ``bb_ord`` (backbone
    ordinal -- the mHC attention site and the input norm) and ``dsa_ord`` (the
    DSA family ordinal -- the MLA/indexer stacks and the latent/raw/pool cache
    pools), each a dynamic ``stack[ordinal]`` view with static extents (the
    t29-proven production form). The host computes ``bb_ord = layer`` and
    ``dsa_ord = layer // 4``; the graph carries no layer arithmetic. One
    artifact therefore serves every DSA-attention sublayer of the schedule.
    """
    num_tokens: pl.Scalar[pl.INT32] = pl.read(active_tokens, [0])
    t_tokens = pl.tensor.dim(x_hc, 0)
    pools_axis = pl.tensor.dim(pool_token_slots, 0)

    q_resid = pl.create_tensor([t_tokens, Q_LORA], dtype=pl.BF16)
    kv_latent = pl.create_tensor([t_tokens, KV_LORA], dtype=pl.BF16)
    head_weights = pl.create_tensor([t_tokens, INDEX_H], dtype=pl.FP32)
    gate_scores = pl.create_tensor([t_tokens, INDEX_DIM], dtype=pl.FP32)
    topk_indices = pl.create_tensor([t_tokens, TOPK_INDEX_WIDTH], dtype=pl.INT32)
    pool_valid = pl.create_tensor([pools_axis, POOL_VALID_WIDTH], dtype=pl.INT32)

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

    with pl.scope():
        if bb_ord == NUM_LAYERS:
            plain_residual_pre(x_hc, out_collapsed, out_attn_pre, out_attn_post, out_attn_res)
        else:
            attn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16] = mhc_attn_fn[bb_ord]
            attn_scale_l: pl.Tensor[[3], pl.FP32] = mhc_attn_scale[bb_ord]
            attn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = mhc_attn_base[bb_ord]
            mhc_mixes(x_hc, attn_fn_l, attn_scale_l, attn_base_l,
                      out_attn_pre, out_attn_post, out_attn_res)
            mhc_pre(x_hc, out_attn_pre, out_collapsed)
        rmsnorm(out_collapsed, input_norm_l, out_normed)
        dsa_attention_sublayer(
            out_normed,
            mla_w_q_a_l, mla_w_q_a_scale_l, mla_q_a_norm_l, mla_w_q_b_l, mla_w_q_b_scale_l,
            mla_w_kv_a_l, mla_w_kv_a_scale_l, mla_kv_a_norm_l,
            mla_w_k_l, mla_w_v_l, mla_w_o_l, mla_w_o_scale_l,
            idx_w_q_b_l, idx_w_k_l, idx_k_norm_w_l,
            idx_k_norm_b_l, idx_w_weights_l,
            idx_w_compress_gate_l, idx_compress_ape_l, hadamard,
            latent_cache_l, raw_cache_l,
            pool_cache_l, pool_scale_l,
            block_table, request_ids, mla_slots, index_slots,
            pool_token_slots, pool_slots, pool_blocks,
            seg_start, pool_count_tok, tail_start, tail_count, kv_len_tok,
            q_resid, dsa_query, kv_latent, attn_out,
            index_q, index_k, head_weights, gate_scores, index_scores,
            selected_pools, selected_valid, topk_indices, pool_valid,
            out_attn_partial, skip_topk,
        )
        tp_all_reduce(out_attn_partial, tp_partial, tp_arrived, out_sublayer_out,
                      num_tokens, my_rank, 0, attn_epoch)
        zero_bf16_padding_rows(out_sublayer_out, num_tokens)
        if bb_ord == NUM_LAYERS:
            plain_residual_post(out_sublayer_out, x_hc, out_x_hc_mid)
        else:
            mhc_post(out_sublayer_out, x_hc, out_attn_post, out_attn_res, out_x_hc_mid)

    return (
        out_attn_pre, out_attn_post, out_attn_res, out_collapsed, out_normed,
        out_attn_partial, out_sublayer_out, out_x_hc_mid,
    )


# =====================================================================================
# L3: rank axis and per-program communication windows



# Per-kind L3 wrappers (see the L2 split note: no traced If may select the
# kind).
@pl.jit.host
def l3_prefill_layer_diff_kda_attn_rt(
    x_hc: pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16],
    query_start_loc: pl.Tensor[[RANKS, BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[RANKS, BATCH], pl.INT32],
    active_tokens: pl.Tensor[[RANKS, 1], pl.INT32],
    kda_tri_incl: pl.Tensor[[RANKS, CHUNK, CHUNK], pl.FP32],
    kda_tri_strict: pl.Tensor[[RANKS, CHUNK, CHUNK], pl.FP32],
    kda_head_onehot: pl.Tensor[[RANKS, LOCAL_KDA_H, BETA_PAD], pl.FP32],
    kda_eye_wide: pl.Tensor[[RANKS, CHUNK, KDA_DIM], pl.FP32],
    kda_w_q: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_k: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_v: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_conv_q: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_k: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_v: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_w_f_a: pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_f_b: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_g_a: pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_g_b: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_b: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_H, D], pl.BF16],
    kda_dt_bias: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM], pl.FP32],
    kda_a_log: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_H], pl.FP32],
    kda_o_norm: pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_DIM], pl.BF16],
    kda_w_o: pl.Tensor[[RANKS, KDA_NUM_LAYERS, D, LOCAL_KDA_QKV_DIM], pl.BF16],
    mhc_attn_fn: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_attn_scale: pl.Tensor[[RANKS, NUM_LAYERS, 3], pl.FP32],
    mhc_attn_base: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC], pl.FP32],
    norm_input_norm: pl.Tensor[[RANKS, NORM_STACK_LAYERS, D], pl.BF16],
    kda_conv_state: pl.InOut[
        pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]
    ],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    out_attn_pre: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_attn_post: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_attn_res: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_collapsed: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_normed: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_attn_partial: pl.Out[pl.Tensor[[RANKS, CAPACITY, D], pl.FP32]],
    out_sublayer_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_x_hc_mid: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16]],
    kda_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    mixed_qkv: pl.InOut[pl.Tensor[[RANKS, CAPACITY, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.InOut[pl.Tensor[[RANKS, CAPACITY, BETA_PAD], pl.FP32]],
    kda_q: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_k: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_v: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    bb_ord: pl.Scalar[pl.INT32],
    kda_ord: pl.Scalar[pl.INT32],
    attn_epoch: pl.Scalar[pl.INT32],
):
    """Launch one KDA attention half at the caller's per-program TP epoch."""
    tp_partial_buffer = pld.alloc_window_buffer([CAPACITY, D], dtype=pl.FP32)
    tp_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.unroll(RANKS):
        tp_partial = pld.window(tp_partial_buffer, [CAPACITY, D], dtype=pl.FP32)
        tp_arrived = pld.window(tp_arrived_buffer, [EP_SIZE, 1], dtype=pl.INT32)
        prefill_layer_diff_kda_attn_rt(
            x_hc[rank],
            query_start_loc[rank], kda_state_rows[rank], active_tokens[rank],
            kda_tri_incl[rank], kda_tri_strict[rank], kda_head_onehot[rank], kda_eye_wide[rank],
            kda_w_q[rank], kda_w_k[rank], kda_w_v[rank],
            kda_conv_q[rank], kda_conv_k[rank], kda_conv_v[rank],
            kda_w_f_a[rank], kda_w_f_b[rank], kda_w_g_a[rank], kda_w_g_b[rank],
            kda_w_b[rank], kda_dt_bias[rank], kda_a_log[rank], kda_o_norm[rank], kda_w_o[rank],
            mhc_attn_fn[rank], mhc_attn_scale[rank], mhc_attn_base[rank],
            norm_input_norm[rank],
            kda_conv_state[rank], kda_recurrent_state[rank],
            tp_partial, tp_arrived,
            out_attn_pre[rank], out_attn_post[rank], out_attn_res[rank],
            out_collapsed[rank], out_normed[rank],
            out_attn_partial[rank], out_sublayer_out[rank], out_x_hc_mid[rank],
            kda_out[rank], mixed_qkv[rank], decay[rank], beta[rank],
            kda_q[rank], kda_k[rank], kda_v[rank],
            bb_ord, kda_ord, rank, attn_epoch, device=rank,
        )



@pl.jit.host
def l3_prefill_layer_diff_moe_ffn_rt(
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
    lane_x_buffer = pld.alloc_window_buffer([TOTAL_CAP, D], dtype=pl.INT8)
    lane_aux_buffer = pld.alloc_window_buffer([TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
    lane_count_buffer = pld.alloc_window_buffer([N_RANKS, COUNT_PAD], dtype=pl.INT32)
    count_ready_buffer = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    x_ready_buffer = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    slot_out_buffer = pld.alloc_window_buffer([ROUTES_PER_SRC, D], dtype=pl.FP32)
    return_ready_buffer = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    tp_partial_buffer = pld.alloc_window_buffer([CAPACITY, D], dtype=pl.FP32)
    tp_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.unroll(RANKS):
        lane_x = pld.window(lane_x_buffer, [TOTAL_CAP, D], dtype=pl.INT8)
        lane_aux = pld.window(lane_aux_buffer, [TOTAL_CAP, AUX_PAD], dtype=pl.FP32)
        lane_count = pld.window(lane_count_buffer, [N_RANKS, COUNT_PAD], dtype=pl.INT32)
        count_ready = pld.window(count_ready_buffer, [N_RANKS, 1], dtype=pl.INT32)
        x_ready = pld.window(x_ready_buffer, [N_RANKS, 1], dtype=pl.INT32)
        slot_out = pld.window(slot_out_buffer, [ROUTES_PER_SRC, D], dtype=pl.FP32)
        return_ready = pld.window(return_ready_buffer, [N_RANKS, 1], dtype=pl.INT32)
        tp_partial = pld.window(tp_partial_buffer, [CAPACITY, D], dtype=pl.FP32)
        tp_arrived = pld.window(tp_arrived_buffer, [EP_SIZE, 1], dtype=pl.INT32)
        prefill_layer_diff_moe_ffn_rt(
            x_hc_mid[rank],
            active_tokens[rank],
            mhc_ffn_fn[rank], mhc_ffn_scale[rank], mhc_ffn_base[rank],
            norm_post_norm[rank],
            router_gate_w[rank], router_gate_bias[rank],
            moe_w_gate_up[rank], moe_w_gate_up_scale[rank],
            moe_w_down[rank], moe_w_down_scale[rank],
            moe_shared_w_gate_up[rank], moe_shared_w_gate_up_scale[rank],
            moe_shared_w_down[rank], moe_shared_w_down_scale[rank],
            lane_x, lane_aux, lane_count, count_ready, x_ready,
            slot_out, return_ready,
            tp_partial, tp_arrived,
            out_ffn_pre[rank], out_ffn_post[rank], out_ffn_res[rank],
            out_ffn_input[rank], out_ffn_partial[rank], out_ffn_out[rank], out_x_hc_out[rank],
            bb_ord, moe_ord, rank, moe_epoch, ffn_epoch, device=rank,
        )


@pl.jit.host
def l3_prefill_layer_diff_kda_dense_rt(
    x_hc: pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16],
    query_start_loc: pl.Tensor[[RANKS, BATCH + 1], pl.INT32],
    kda_state_rows: pl.Tensor[[RANKS, BATCH], pl.INT32],
    active_tokens: pl.Tensor[[RANKS, 1], pl.INT32],
    kda_tri_incl: pl.Tensor[[RANKS, CHUNK, CHUNK], pl.FP32],
    kda_tri_strict: pl.Tensor[[RANKS, CHUNK, CHUNK], pl.FP32],
    kda_head_onehot: pl.Tensor[[RANKS, LOCAL_KDA_H, BETA_PAD], pl.FP32],
    kda_eye_wide: pl.Tensor[[RANKS, CHUNK, KDA_DIM], pl.FP32],
    kda_w_q: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_k: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_w_v: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    kda_conv_q: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_k: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_conv_v: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, 1, KDA_CONV_K], pl.BF16],
    kda_w_f_a: pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_f_b: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_g_a: pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_DIM, D], pl.BF16],
    kda_w_g_b: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    kda_w_b: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_H, D], pl.BF16],
    kda_dt_bias: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_QKV_DIM], pl.FP32],
    kda_a_log: pl.Tensor[[RANKS, KDA_NUM_LAYERS, LOCAL_KDA_H], pl.FP32],
    kda_o_norm: pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_DIM], pl.BF16],
    kda_w_o: pl.Tensor[[RANKS, KDA_NUM_LAYERS, D, LOCAL_KDA_QKV_DIM], pl.BF16],
    mhc_attn_fn: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_attn_scale: pl.Tensor[[RANKS, NUM_LAYERS, 3], pl.FP32],
    mhc_attn_base: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC], pl.FP32],
    mhc_ffn_fn: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC, HC_DIM], pl.BF16],
    mhc_ffn_scale: pl.Tensor[[RANKS, NUM_LAYERS, 3], pl.FP32],
    mhc_ffn_base: pl.Tensor[[RANKS, NUM_LAYERS, MIX_HC], pl.FP32],
    norm_input_norm: pl.Tensor[[RANKS, NORM_STACK_LAYERS, D], pl.BF16],
    norm_post_norm: pl.Tensor[[RANKS, NORM_STACK_LAYERS, D], pl.BF16],
    dense_w_gate_up: pl.Tensor[[RANKS, DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER, D], pl.INT8],
    dense_w_gate_up_scale: pl.Tensor[[RANKS, DENSE_NUM_LAYERS, 2 * LOCAL_DENSE_INTER], pl.FP32],
    dense_w_down: pl.Tensor[[RANKS, DENSE_NUM_LAYERS, D, LOCAL_DENSE_INTER], pl.INT8],
    dense_w_down_scale: pl.Tensor[[RANKS, DENSE_NUM_LAYERS, D], pl.FP32],
    kda_conv_state: pl.InOut[
        pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM], pl.BF16]
    ],
    kda_recurrent_state: pl.InOut[
        pl.Tensor[[RANKS, KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]
    ],
    out_attn_pre: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_attn_post: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_attn_res: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_collapsed: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_normed: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_attn_partial: pl.Out[pl.Tensor[[RANKS, CAPACITY, D], pl.FP32]],
    out_sublayer_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_x_hc_mid: pl.InOut[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16]],
    out_ffn_pre: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_ffn_post: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32]],
    out_ffn_res: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, HC_MULT], pl.FP32]],
    out_ffn_input: pl.Out[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_ffn_partial: pl.Out[pl.Tensor[[RANKS, CAPACITY, D], pl.FP32]],
    out_ffn_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.BF16]],
    out_x_hc_out: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16]],
    kda_out: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    mixed_qkv: pl.InOut[pl.Tensor[[RANKS, CAPACITY, 3 * LOCAL_KDA_QKV_DIM], pl.BF16]],
    decay: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.FP32]],
    beta: pl.InOut[pl.Tensor[[RANKS, CAPACITY, BETA_PAD], pl.FP32]],
    kda_q: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_k: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    kda_v: pl.InOut[pl.Tensor[[RANKS, CAPACITY, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    dense_x_int8: pl.InOut[pl.Tensor[[RANKS, CAPACITY, D], pl.INT8]],
    bb_ord: pl.Scalar[pl.INT32],
    kda_ord: pl.Scalar[pl.INT32],
    dense_ord: pl.Scalar[pl.INT32],
    attn_epoch: pl.Scalar[pl.INT32],
    ffn_epoch: pl.Scalar[pl.INT32],
):
    """Launch one dense KDA layer at two consecutive caller-owned TP epochs."""
    tp_partial_buffer = pld.alloc_window_buffer([CAPACITY, D], dtype=pl.FP32)
    tp_arrived_buffer = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.unroll(RANKS):
        tp_partial = pld.window(tp_partial_buffer, [CAPACITY, D], dtype=pl.FP32)
        tp_arrived = pld.window(tp_arrived_buffer, [EP_SIZE, 1], dtype=pl.INT32)
        prefill_layer_diff_kda_dense_rt(
            x_hc[rank],
            query_start_loc[rank], kda_state_rows[rank], active_tokens[rank],
            kda_tri_incl[rank], kda_tri_strict[rank], kda_head_onehot[rank], kda_eye_wide[rank],
            kda_w_q[rank], kda_w_k[rank], kda_w_v[rank],
            kda_conv_q[rank], kda_conv_k[rank], kda_conv_v[rank],
            kda_w_f_a[rank], kda_w_f_b[rank], kda_w_g_a[rank], kda_w_g_b[rank],
            kda_w_b[rank], kda_dt_bias[rank], kda_a_log[rank], kda_o_norm[rank], kda_w_o[rank],
            mhc_attn_fn[rank], mhc_attn_scale[rank], mhc_attn_base[rank],
            mhc_ffn_fn[rank], mhc_ffn_scale[rank], mhc_ffn_base[rank],
            norm_input_norm[rank], norm_post_norm[rank],
            dense_w_gate_up[rank], dense_w_gate_up_scale[rank],
            dense_w_down[rank], dense_w_down_scale[rank],
            kda_conv_state[rank], kda_recurrent_state[rank],
            tp_partial, tp_arrived,
            out_attn_pre[rank], out_attn_post[rank], out_attn_res[rank],
            out_collapsed[rank], out_normed[rank],
            out_attn_partial[rank], out_sublayer_out[rank], out_x_hc_mid[rank],
            out_ffn_pre[rank], out_ffn_post[rank], out_ffn_res[rank],
            out_ffn_input[rank], out_ffn_partial[rank], out_ffn_out[rank], out_x_hc_out[rank],
            kda_out[rank], mixed_qkv[rank], decay[rank], beta[rank],
            kda_q[rank], kda_k[rank], kda_v[rank], dense_x_int8[rank],
            bb_ord, kda_ord, dense_ord, rank,
            attn_epoch, ffn_epoch, device=rank,
        )


@pl.jit.host
def l3_prefill_layer_diff_dsa_attn_rt(
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
    index_raw_cache: pl.InOut[
        pl.Tensor[[RANKS, DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH], pl.FP32]
    ],
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
        prefill_layer_diff_dsa_attn_rt(
            x_hc[rank],
            request_ids[rank], mla_slots[rank], index_slots[rank],
            seg_start[rank], pool_count_tok[rank],
            tail_start[rank], tail_count[rank], kv_len_tok[rank],
            pool_token_slots[rank], pool_slots[rank], pool_blocks[rank],
            block_table[rank], active_tokens[rank],
            mla_w_q_a[rank], mla_w_q_a_scale[rank], mla_q_a_norm[rank], mla_w_q_b[rank], mla_w_q_b_scale[rank],
            mla_w_kv_a[rank], mla_w_kv_a_scale[rank], mla_kv_a_norm[rank],
            mla_w_k[rank], mla_w_v[rank], mla_w_o[rank], mla_w_o_scale[rank],
            idx_w_q_b[rank], idx_w_k[rank], idx_k_norm_w[rank], idx_k_norm_b[rank],
            idx_w_weights[rank], idx_w_compress_gate[rank], idx_compress_ape[rank],
            mhc_attn_fn[rank], mhc_attn_scale[rank], mhc_attn_base[rank],
            norm_input_norm[rank], hadamard[rank],
            mla_latent_cache[rank], index_raw_cache[rank],
            index_pool_cache[rank], index_pool_scale[rank],
            tp_partial, tp_arrived,
            out_attn_pre[rank], out_attn_post[rank], out_attn_res[rank],
            out_collapsed[rank], out_normed[rank],
            out_attn_partial[rank], out_sublayer_out[rank], out_x_hc_mid[rank],
            dsa_query[rank], attn_out[rank], index_q[rank], index_k[rank],
            index_scores[rank], selected_pools[rank], selected_valid[rank],
            bb_ord, dsa_ord, rank, attn_epoch, skip_topk, device=rank,
        )


@pl.jit.host
def l3_prefill_seed(
    embedding_rows: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[RANKS, CAPACITY, HC_MULT, D], pl.BF16]],
):
    """Seed the carried stream on every rank (the forward's first dispatch)."""
    for rank in pl.unroll(RANKS):
        prefill_seed(embedding_rows[rank], x_hc[rank], device=rank)


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
                value = cached_hidden[request:request + 1, :]
                pools = cached_pools[request:request + 1, :]
                valid = cached_valid[request:request + 1, :]
            collapsed[row:row + 1, :] = value
            selected_pools[row:row + 1, :] = pools
            selected_valid[row:row + 1, :] = valid
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
        prefill_mtp_prepare(target_streams[rank], embedding_rows[rank], feedback_norm[rank],
                            enorm[rank], hnorm[rank], eh_proj[rank], request_ids[rank],
                            cached_hidden[rank], cached_pools[rank], cached_valid[rank],
                            selected_pools[rank], selected_valid[rank], x_hc[rank], recycle, device=rank)


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
            cached_hidden[request:request + 1, :] = head_hidden[request:request + 1, :]
            cached_pools[request:request + 1, :] = selected_pools[end - 1:end, :]
            cached_valid[request:request + 1, :] = selected_valid[end - 1:end, :]
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
        prefill_mtp_head(head_hidden[rank], final_norm[rank], weight[rank],
                         selected_pools[rank], selected_valid[rank], query_start_loc[rank],
                         cached_hidden[rank], cached_pools[rank], cached_valid[rank], logits[rank], device=rank)


def production_launch_plan() -> tuple[tuple, ...]:
    """The dispatch sequence of one full forward (design S1 schedule).

    The donor ``make_layer_program`` shape at the RUNNER level: one L3
    dispatch per step -- the seed, one launch per dense KDA layer, two per
    MoE layer (attention half, then FFN half), then the head. A single
    dispatch cannot host the schedule (Error-104 dim-fold degradation; 2,784
    diagnostics in ``logs_t16/attempt3/prod_cap128_try3.log``), so the host
    drives the loop over per-sublayer programs prepared on ONE shared L3
    worker (``prepare(extra_compiled=...)``); since t30 the 89 steps dispatch
    SIX consolidated runtime-ordinal programs
    (:func:`production_program_of_step`) with the layer as trailing ordinals.
    """
    steps: list[tuple] = [("seed",)]
    for layer in range(NUM_LAYERS):
        if layer < DENSE_NUM_LAYERS:
            steps.append((layer, "layer"))
        else:
            steps.append((layer, "attn"))
            steps.append((layer, "ffn"))
    steps.append(("head",))
    return tuple(steps)


# =====================================================================================
# Consolidated runtime-ordinal programs (t30): 89 per-layer artifacts -> 6 programs
# =====================================================================================
# The t29 probe proved the mechanism (zero Error-104, 18/12 dispatches
# bitwise-equal to the retired per-layer artifacts); t30 completed the
# family: ONE consolidated program per sublayer kind, the layer riding the
# dispatch as trailing host-computed INT32 ordinals. This dissolves the
# MAX_REGISTERED_CALLABLE_IDS=64 wall (89 programs cannot register on one
# worker; 6 can) and the 89-artifact compile cost; the constexpr twin path
# was retired after t30 (logs_t29/, logs_t30/README.md; git tag
# glm53-pretrim-baseline).
RT_PROGRAMS = ("seed", "dense", "kda_attn", "dsa_attn", "moe_ffn", "head")


def production_program_entry(program: str):
    """The L3 entry one consolidated runtime-ordinal program compiles."""
    if program == "seed":
        return l3_prefill_seed
    if program == "dense":
        return l3_prefill_layer_diff_kda_dense_rt
    if program == "kda_attn":
        return l3_prefill_layer_diff_kda_attn_rt
    if program == "dsa_attn":
        return l3_prefill_layer_diff_dsa_attn_rt
    if program == "moe_ffn":
        return l3_prefill_layer_diff_moe_ffn_rt
    if program == "head":
        return l3_prefill_head_rows
    raise ValueError(f"unknown consolidated program {program!r}")


def production_program_of_step(step: tuple) -> str:
    """The ONE consolidated program that serves a launch-plan step.

    Pure Python dispatch on the step's kind: dense KDA layers run whole; every
    MoE layer is the split attention/FFN pair with the attention half's kind
    picked by the ``layer % 4`` schedule.
    """
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
    """One entry's ``(tensor names, scalar names)``, both in signature order.

    Splits the entry's signature by the annotation: a runtime-ordinal entry's
    ``pl.Scalar`` parameters have no shape and dispatch as one Python int each
    (``distributed_runner`` forwards scalars as-is), so a dispatch is the
    tensor arguments in signature order followed by the scalar values in
    signature order -- the t29-verified ABI.
    """
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


# The carried BF16 four-stream ping-pongs between two runner-allocated workspaces.
X_H_PINGPONG = ("ws_x_hc_a", "ws_x_hc_b")


def production_step_bindings(plan: tuple[tuple, ...] | None = None) -> dict[tuple, dict[str, str]]:
    """One launch-plan step's stream overrides: signature name -> workspace.

    The documented ping-pong (design S2.1): the seed writes side ``a``; a dense
    whole-layer launch reads the current side (``x_hc``), parks its mid boundary
    on the other side (``out_x_hc_mid``) and writes the final stream back to the
    current side (``out_x_hc_out``); an attention launch reads the current side
    (``x_hc``) and moves the stream to the other (``out_x_hc_mid``); the FFN
    launch reads that side (``x_hc_mid``) and writes the stream back
    (``out_x_hc_out``); the head collapses whichever side holds the final stream
    (``x_hc``). After every layer the stream is back on the side it entered the
    layer on, so the first launch of every layer always reads ``ws_x_hc_a``.
    """
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
        workspace = "ws_" + param[len("out_"):]
        if workspace in WORKSPACES:
            return workspace
    return param

def _production_geometry() -> dict[str, int]:
    """Import-frozen query/cache dimensions, collective ABI and shared weight layout."""
    return {
        "version": 6, "mtp_cache": 1, "cache_tokens": CACHE_TOKENS, "native_mla_int8": 1, "shared_int8_nz": 1, "tp": TP_SIZE, "ep": EP_SIZE, "capacity": CAPACITY,
        "max_seq_len": MAX_SEQ_LEN, "batch": BATCH, "latent_rows": LATENT_ROWS,
        "pool_axis": POOL_AXIS,
    }


def compile_production(
    runtime_dir: Path, device_ids: list[int], *, platform: str = "a2a3", mode: str = "rt",
    programs: tuple[str, ...] | None = None,
    auxiliaries: bool = True,
) -> dict:
    """Compile the production programs and record the artifact manifest.

    ``mode="rt"`` (the only production path, t30): the SIX consolidated
    runtime-ordinal programs of :data:`RT_PROGRAMS`, the layer riding each
    dispatch as trailing INT32 ordinals (:func:`production_step_ordinals`)
    -- 6 compiles instead of 89. The retired per-layer constexpr twin is
    recoverable at git tag ``glm53-pretrim-baseline`` and rejected
    explicitly here. The manifest records the mode under ``_mode`` for
    :func:`load_production`; ``programs`` narrows the compile to a subset of
    :data:`RT_PROGRAMS` (the CI fixture's width-safe subset).
    """
    config = RunConfig(
        platform=platform,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
    )
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"_mode": mode, "_geometry": _production_geometry()}
    if mode == "rt":
        for program in (programs if programs is not None else RT_PROGRAMS):
            if program not in RT_PROGRAMS:
                raise ValueError(f"unknown consolidated program {program!r}")
            compiled = production_program_entry(program).compile(config=config)
            out_dir = getattr(compiled, "output_dir", None) or getattr(compiled, "work_dir")
            if out_dir is None:
                raise RuntimeError(f"no artifact directory for program {program}")
            manifest[program] = str(out_dir)
            print("GLM_PROD_COMPILED", program, out_dir, flush=True)
    elif mode != "rt":
        raise ValueError(
            f"mode must be 'rt', got {mode!r}: the per-layer constexpr path was "
            "retired after t30 proved the six consolidated programs bit-equal "
            "(logs_t29/, logs_t30/README.md); recover it from git tag "
            "glm53-pretrim-baseline and recompile there"
        )
    if auxiliaries:
        auxiliary = {}
        for name, entry in (("reset", l3_reset_request_caches), ("head_logits", l3_prefill_lm_head_logits),
                            ("snapshot", l3_copy_request_snapshot), ("mtp_prepare", l3_prefill_mtp_prepare),
                            ("mtp_head", l3_prefill_mtp_head)):
            compiled = entry.compile(config=config)
            auxiliary[name] = str(compiled.output_dir)
        manifest["_auxiliary"] = auxiliary
    (runtime_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return load_production(runtime_dir, device_ids, platform=platform)


def load_production(runtime_dir: Path, device_ids: list[int], *, platform: str = "a2a3") -> dict:
    """Load the production programs from a manifest runtime dir.

    The manifest must be an ``rt`` manifest (``_mode: "rt"``, the t30
    production path): the consolidated program names it records -- all six,
    or the CI fixture's subset -- map to their artifacts. Pre-t30 per-layer
    manifests are rejected explicitly (retired path; git tag
    ``glm53-pretrim-baseline``).
    """
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

    manifest = json.loads((Path(runtime_dir) / "manifest.json").read_text())
    if manifest.get("_mode") != "rt":
        raise ValueError(
            f"{runtime_dir}/manifest.json is not an rt manifest (_mode="
            f"{manifest.get('_mode')!r}): the per-layer constexpr path was "
            "retired after t30 (logs_t29/, logs_t30/README.md); recompile "
            "with compile_production(mode='rt'), or recover the retired path "
            "from git tag glm53-pretrim-baseline"
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
            manifest[program], platform=platform,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        )
        for program in programs_present
    }
    if "_auxiliary" in manifest:
        result["_auxiliary"] = {
            name: DistributedCompiledProgram.from_dir(path, platform=platform,
                  distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0))
            for name, path in manifest["_auxiliary"].items()
        }
    return result


GLM_TENSORS = (
    "embedding_rows",
    "query_start_loc",
    "request_ids",
    "mla_slots",
    "index_slots",
    "kda_state_rows",
    "seg_start",
    "pool_count_tok",
    "tail_start",
    "tail_count",
    "kv_len_tok",
    "pool_token_slots",
    "pool_slots",
    "pool_blocks",
    "block_table",
    "cache_page_owners",
    "logit_row_indices",
    "active_tokens",
    "kda_tri_incl",
    "kda_tri_strict",
    "kda_head_onehot",
    "kda_eye_wide",
    "kda_w_q",
    "kda_w_k",
    "kda_w_v",
    "kda_conv_q",
    "kda_conv_k",
    "kda_conv_v",
    "kda_w_f_a",
    "kda_w_f_b",
    "kda_w_g_a",
    "kda_w_g_b",
    "kda_w_b",
    "kda_dt_bias",
    "kda_a_log",
    "kda_o_norm",
    "kda_w_o",
    "mla_w_q_a",
    "mla_w_q_a_scale",
    "mla_q_a_norm",
    "mla_w_q_b",
    "mla_w_q_b_scale",
    "mla_w_kv_a",
    "mla_w_kv_a_scale",
    "mla_kv_a_norm",
    "mla_w_k",
    "mla_w_v",
    "mla_w_o",
    "mla_w_o_scale",
    "idx_w_q_b",
    "idx_w_k",
    "idx_k_norm_w",
    "idx_k_norm_b",
    "idx_w_weights",
    "idx_w_compress_gate",
    "idx_compress_ape",
    "mhc_attn_fn",
    "mhc_attn_scale",
    "mhc_attn_base",
    "mhc_ffn_fn",
    "mhc_ffn_scale",
    "mhc_ffn_base",
    "norm_input_norm",
    "norm_post_norm",
    "router_gate_w",
    "router_gate_bias",
    "moe_w_gate_up",
    "moe_w_gate_up_scale",
    "moe_w_down",
    "moe_w_down_scale",
    "moe_shared_w_gate_up",
    "moe_shared_w_gate_up_scale",
    "moe_shared_w_down",
    "moe_shared_w_down_scale",
    "dense_w_gate_up",
    "dense_w_gate_up_scale",
    "dense_w_down",
    "dense_w_down_scale",
    "hadamard",
    "mla_latent_cache",
    "index_raw_cache",
    "index_pool_cache",
    "index_pool_scale",
    "kda_conv_state",
    "kda_recurrent_state",
    "ws_attn_pre",
    "ws_attn_post",
    "ws_attn_res",
    "ws_collapsed",
    "ws_normed",
    "ws_attn_partial",
    "ws_sublayer_out",
    "ws_ffn_pre",
    "ws_ffn_post",
    "ws_ffn_res",
    "ws_ffn_input",
    "ws_ffn_partial",
    "ws_ffn_out",
    "ws_x_hc_a",
    "ws_x_hc_b",
    "kda_out",
    "dsa_query",
    "attn_out",
    "index_q",
    "index_k",
    "index_scores",
    "selected_pools",
    "selected_valid",
    "mixed_qkv",
    "decay",
    "beta",
    "kda_q",
    "kda_k",
    "kda_v",
    "dense_x_int8",
    "head_hidden",
)

# Device-resident weight stacks the graph consumes, as (graph parameter, loader
# stack name). The loader also stages norm.final_norm and the mtp.* extras; the
# runner allocates and uploads them alongside but the graph ABI does not carry them.
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

# Graph parameter -> loader stack, including the stacks the graph does not carry so
# allocation/upload stay keyed by the loader's own names.
_PARAM_TO_STACK = dict(GLM_WEIGHT_STACKS)
_STACK_TO_PARAM = {stack: param for param, stack in GLM_WEIGHT_STACKS}

# Per-rank shapes of the production orchestrator's boundary workspaces: every
# launch writes what it reads, so the runner allocates them once per session and
# never re-uploads them between generate steps (they are absent from the step
# input dict on purpose).
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

# Resident stacks, KDA tables and stable KDA request rows survive every dispatch.
# Request caches are initialized once and cleared only by reset_requests().
CONSTANT_STEP_INPUTS = {
    "kda_tri_incl", "kda_tri_strict", "kda_head_onehot", "kda_eye_wide",
    "kda_state_rows",
}

# =====================================================================================
# Host-side metadata (design S5.2): the indexer step builder shared by the runner and
# the future CI fixture, so the two can never drift.
# =====================================================================================


def _pad_to(tensor: torch.Tensor, rows: int, value: int) -> torch.Tensor:
    """Right-pad a 1-D int tensor to ``rows`` entries."""
    if tensor.numel() > rows:
        raise ValueError(f"{tensor.numel()} rows exceed the static capacity {rows}")
    if tensor.numel() == rows:
        return tensor
    pad = torch.full((rows - tensor.numel(),), value, dtype=tensor.dtype)
    return torch.cat([tensor, pad])


def kda_host_constants() -> dict[str, torch.Tensor]:
    """The shared KDA tables every rank uploads once (prefill_kda.py fixtures)."""
    return {
        "kda_tri_incl": torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.float32)),
        "kda_tri_strict": torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.float32), diagonal=-1),
        "kda_head_onehot": torch.eye(BETA_PAD, dtype=torch.float32)[:LOCAL_KDA_H].contiguous(),
        "kda_eye_wide": torch.nn.functional.pad(
            torch.eye(CHUNK, dtype=torch.float32), (0, KDA_DIM - CHUNK)
        ).contiguous(),
    }


CACHE_LAYOUTS = {
    "mla_latent_cache": ((DSA_NUM_LAYERS, LATENT_ROWS, KV_LORA), torch.bfloat16),
    "index_raw_cache": ((DSA_NUM_LAYERS, LATENT_ROWS, INDEX_STATE_WIDTH), torch.float32),
    "index_pool_cache": ((DSA_NUM_LAYERS, POOL_TABLE_ROWS, INDEX_DIM), torch.int8),
    "index_pool_scale": ((DSA_NUM_LAYERS, POOL_TABLE_ROWS, POOL_SCALE_WIDTH), torch.float32),
    "kda_conv_state": ((KDA_NUM_LAYERS, KDA_ROWS, STATE_LEN, CONV_DIM), torch.bfloat16),
    "kda_recurrent_state": ((KDA_NUM_LAYERS, KDA_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM), torch.float32),
}


def zeroed_caches() -> dict[str, torch.Tensor]:
    """Build bounded host fixtures; production clears its device allocations directly."""
    return {name: torch.zeros(shape, dtype=dtype) for name, (shape, dtype) in CACHE_LAYOUTS.items()}


def prepare_step_inputs(
    weights: ResidentPrefillWeights, sequences: list[list[int]],
    *, kv_seq_lens: torch.Tensor | None = None,
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
        from models.glm5_3_flash.vision.config import IMAGE_TOKEN_ID

        offset = 0
        if any(type(slot) is not int or slot not in range(BATCH) for slot in visual_features):
            raise ValueError("visual feature keys must be request slots")
        for slot, sequence in enumerate(sequences):
            if slot in visual_features:
                features = visual_features[slot]
                indices = torch.tensor([row + offset for row, token in enumerate(sequence) if token == IMAGE_TOKEN_ID])
                if features.dtype != torch.bfloat16 or tuple(features.shape) != (indices.numel(), D):
                    raise ValueError("visual features must be BF16 [image_placeholder_count, hidden_size]")
                if not torch.isfinite(features).all():
                    raise ValueError("visual features must be finite")
                embedding_rows[indices.long()] = features.cpu()
            offset += len(sequence)
    query_start_loc = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.int64), lengths]), 0
    ).to(torch.int32)
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
        query_start_loc, kv_seq_lens=previous,
        mla_block_table=block_table, index_block_table=block_table,
        pool_block_table=pool_table, kda_state_rows=torch.arange(BATCH, dtype=torch.int32),
        tp_size=TP_SIZE,
    )
    step = build_indexer_chunk_metadata(
        metadata, block_table, pool_offsets,
        pools_cap=POOL_AXIS, pool_rows=POOL_TABLE_ROWS, pool_block_table=pool_table,
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

# GLM53_PREFILL_VERIFY_XHC=1 additionally reads the final stream workspace (the
# last FFN launch's out_x_hc_out) back after every forward and refuses non-finite
# values -- the plan-driven proof's stream evidence. Off by default because the
# readback is a full [RANKS, CAPACITY, HC_MULT, D] copy per generate step.
VERIFY_X_H_C = os.environ.get("GLM53_PREFILL_VERIFY_XHC", "") == "1"
# GLM53_PREFILL_TRACE_DISPATCH=1 prints one line per launch-plan dispatch
# (the dispatch-order evidence the sim proof's log greps for).
TRACE_DISPATCH = os.environ.get("GLM53_PREFILL_TRACE_DISPATCH", "") == "1"

# Built-in chat-template fallback used when the checkpoint ships neither an
# inline tokenizer_config.json chat_template nor a chat_template.jinja file.
# GLM-5.3 chat turns are ChatML-style; the checkpoint's own template always wins
# when present (see generate's resolution order).
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|user|>\n' + message['content'] + '<|assistant|>\n' }}"
    "{% endfor %}"
)

def host_lm_head_logits(
    head_rows: torch.Tensor, final_norm_w: torch.Tensor, lm_head: torch.Tensor
) -> torch.Tensor:
    """CPU reference for final FP32 RMSNorm and the full-vocabulary projection.

    ``head_rows`` is ``[rows, D]`` BF16; the result is FP32 ``[rows, VOCAB]`` with
    no BF16 rounding between normalization and projection.
    """
    return reference_lm_head_logits(head_rows, final_norm_w, lm_head)


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
    ids = [
        int(token_id)
        for token_id in (value if isinstance(value, list) else [value] if value is not None else [])
    ]
    if not ids:
        eos = tokenizer_config.get("eos_token")
        token_id = (
            None if eos is None else tokenizer.token_to_id(
                eos["content"] if isinstance(eos, dict) else eos
            )
        )
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
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        return environment.from_string(template).render(
            messages=[{"role": "user", "content": prompt}], add_generation_prompt=True
        )
    except Exception as error:  # noqa: BLE001 -- fall back verbatim on any template error
        print("GLM_CHAT_TEMPLATE_FALLBACK", repr(error), flush=True)
        return f"<|user|>\n{prompt}<|assistant|>\n"

class ResidentPrefillSession:
    """Resident weights and request-local caches for chunked GLM forwards."""

    def __init__(
        self, checkpoint: Checkpoint, programs: dict, device_ids: list[int],
        *, platform: str = "a2a3", run_config: RunConfig | None = None,
        host_num_threads: int | None = None,
        prefix_snapshot_entries: int = 0, prefix_snapshot_max_bytes: int = 128 * 1024**2,
        lm_head_backend: str | None = None,
        enable_mtp: bool = False,
    ):
        if host_num_threads is not None and (type(host_num_threads) is not int or host_num_threads <= 0):
            raise ValueError("host_num_threads must be a positive integer or None")
        if type(prefix_snapshot_entries) is not int or prefix_snapshot_entries not in (0, PREFIX_SNAPSHOT_ENTRIES):
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
        self._collective_dispatches = {
            program: 0 for program in RT_PROGRAMS if program not in ("seed", "head")
        }
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
        arity = {program: _l3_arg_arity(production_program_entry(program))
                 for program in RT_PROGRAMS}
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
            checkpoint, tp_size=TP_SIZE, ep_size=EP_SIZE, model=_prefill_model(), mla_mode="native"
        )
        final_norm_w = _load_final_norm(checkpoint)
        lm_head = weights.lm_head().to(torch.float32) if self.lm_head_backend == "cpu" else None
        print("GLM_LM_HEAD_BACKEND", self.lm_head_backend, flush=True)
        config = self.run_config or RunConfig(
            platform=platform, ring_heap=2 * 1024**3,
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
            lm_head_program = self.auxiliary_programs.get("head_logits") or l3_prefill_lm_head_logits.compile(config=config)
            extras.append(lm_head_program)
        snapshot_program = None
        if self._prefix_snapshot_entries:
            snapshot_program = self.auxiliary_programs.get("snapshot") or l3_copy_request_snapshot.compile(config=config)
            extras.append(snapshot_program)
        mtp_prepare_program = mtp_head_program = None
        if self.enable_mtp:
            mtp_prepare_program = self.auxiliary_programs.get("mtp_prepare") or l3_prefill_mtp_prepare.compile(config=config)
            mtp_head_program = self.auxiliary_programs.get("mtp_head") or l3_prefill_mtp_head.compile(config=config)
            extras.extend((mtp_prepare_program, mtp_head_program))
        context = primary.prepare(
            config, extra_compiled=extras, persistent=True, reset_persistent_windows=False,
        )
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
                        tensors[name] = runtime.alloc_tensor(
                            value.shape, value.dtype, init=value, worker_id=rank
                        )
                # Stacks the graph ABI does not carry (final norm, MTP extras) still need
                # allocation because weights.upload stages every resident stack.
                for stack, (shape, dtype) in weights.tensor_layouts.items():
                    if stack not in _STACK_TO_PARAM and stack not in tensors:
                        tensors[stack] = runtime.alloc_tensor(shape, dtype, worker_id=rank)

            def upload(rank):
                handles = {
                    stack: per_rank[rank][_STACK_TO_PARAM.get(stack, stack)]
                    for stack in weights.tensor_layouts
                }
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
            runtime.run(reset_program, stacked["mla_latent_cache"], stacked["index_raw_cache"],
                        stacked["index_pool_cache"], stacked["index_pool_scale"], stacked["kda_conv_state"],
                        stacked["kda_recurrent_state"], stacked["cache_page_owners"], (1 << BATCH) - 1,
                        config=config)
            # Per-step parameter -> buffer binding (design S2.1 ping-pong): every
            # launch's signature names must resolve to exactly one allocated tensor
            # before any dispatch happens, so a plan/ABI drift fails loudly here
            # instead of as an opaque dispatch-time shape error.
            bindings = production_step_bindings(plan)
            for step in plan:
                tensor_names, _ = arity[step_program[step]]
                unbound = [
                    name for name in tensor_names
                    if _alloc_name_of(name, bindings[step]) not in stacked
                ]
                if unbound:
                    raise ValueError(f"step {step} parameters have no allocated tensor: {unbound}")
            for name, (shape, dtype) in weights.tensor_layouts.items():
                if name not in _STACK_TO_PARAM:
                    stacked[name] = StackedDeviceTensor([rank[name] for rank in per_rank],
                                   (RANKS, *shape), tuple(range(RANKS)))
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
                    stacked[name] = StackedDeviceTensor([rank[name] for rank in per_rank],
                                   (RANKS, *shape), tuple(range(RANKS)))
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
                        mtp_head_weights.append(runtime.alloc_tensor(shard.shape, shard.dtype, init=shard, worker_id=rank))
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
        if not 0 < length <= MAX_SEQ_LEN or length % CHUNK:
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
            self.runtime.copy_to(self.per_rank[rank]["block_table"].data_ptr, table.data_ptr(),
                                 table.numel() * table.element_size(), worker_id=rank)
        try:
            self.runtime.run(self._snapshot_program, *arguments, self.stacked["block_table"],
                             length, slot, bank, int(restore), config=self.config)
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
            self.runtime.copy_to(self.per_rank[rank]["cache_page_owners"].data_ptr, owners.data_ptr(),
                                 owners.numel() * owners.element_size(), worker_id=rank)
        self.runtime.run(
            self._reset_program,
            self.stacked["mla_latent_cache"], self.stacked["index_raw_cache"],
            self.stacked["index_pool_cache"], self.stacked["index_pool_scale"],
            self.stacked["kda_conv_state"], self.stacked["kda_recurrent_state"],
            self.stacked["cache_page_owners"], slot_mask, config=self.config,
        )
        self._seq_lens[slots] = 0
        self.cache_pages.release(slots)
        self._mtp_lengths[slots] = 0
        self._mtp_ready.difference_update(slots)
        if self.enable_mtp:
            for name, width, dtype, fill in (("mtp_cached_hidden", D, torch.bfloat16, 0),
                                            ("mtp_cached_pools", KPOOL_SELECT_K, torch.int32, -1),
                                            ("mtp_cached_valid", KPOOL_SELECT_K, torch.int32, 0)):
                row = torch.full((1, width), fill, dtype=dtype)
                for rank in range(RANKS):
                    for slot in slots:
                        self.runtime.copy_to(self.per_rank[rank][name].data_ptr, row.data_ptr(),
                                             row.numel() * row.element_size(), dst_offset=slot * row.numel() * row.element_size(),
                                             worker_id=rank)
        self._invalid_slots.difference_update(slots)

    def forward_chunks(
        self, token_sequences: list[list[int]], request_slots: list[int],
        kv_seq_lens: list[int] | None = None,
        *, visual_features: dict[int, torch.Tensor] | None = None,
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
            inputs = prepare_step_inputs(self.weights, sequences, kv_seq_lens=self._seq_lens,
                                         visual_features=visual_features, cache_pages=self.cache_pages)
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
        if not self.enable_mtp or count not in range(1, C.MTP_SPEC_TOKENS + 1):
            raise ValueError(f"MTP proposal count must be in 1..{C.MTP_SPEC_TOKENS}")
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
            inputs = prepare_step_inputs(self.weights, sequences, kv_seq_lens=self._mtp_lengths,
                                         cache_pages=self.cache_pages, mtp=True)
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
                    runtime.copy_to(self.per_rank[rank][name].data_ptr, value.data_ptr(),
                                    value.numel() * value.element_size(), worker_id=rank)
            feedback_norm = stacked["mtp.final_norm"] if recycle else self._device_final_norm
            runtime.run(self._mtp_prepare_program, stacked["ws_x_hc_a"], stacked["embedding_rows"],
                        feedback_norm, stacked["mtp.enorm"], stacked["mtp.hnorm"], stacked["mtp.eh_proj"],
                        stacked["request_ids"], stacked["mtp_cached_hidden"], stacked["mtp_cached_pools"],
                        stacked["mtp_cached_valid"], stacked["mtp_selected_pools"], stacked["mtp_selected_valid"],
                        stacked["ws_x_hc_b"], int(recycle), config=config)
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
                scalars = dict(bb_ord=NUM_LAYERS, dsa_ord=MLA_STACK_LAYERS - 1,
                               moe_ord=MOE_STACK_LAYERS - 1, attn_epoch=epoch,
                               moe_epoch=epoch, ffn_epoch=epoch, skip_topk=int(recycle))
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
            runtime.run(self._mtp_head_program, stacked["head_hidden"], stacked["mtp.final_norm"],
                        self._device_mtp_head, stacked["mtp_selected_pools"], stacked["mtp_selected_valid"],
                        stacked["query_start_loc"], stacked["mtp_cached_hidden"], stacked["mtp_cached_pools"],
                        stacked["mtp_cached_valid"], self._device_logits, config=config)
            runtime.copy_stacked_from(self._device_logits, self._host_logits_shards)
            logits = torch.cat(tuple(shard[:, :LOCAL_VOCAB] for shard in self._host_logits_shards.unbind(0)), dim=1)
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
                    per_rank[rank][name].data_ptr, value.data_ptr(),
                    value.numel() * value.element_size(), worker_id=rank,
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
                    written = overrides.get(
                        "out_x_hc_out", overrides.get("out_x_hc_mid", overrides.get("x_hc"))
                    )
                    step_rows = torch.empty((RANKS, CAPACITY, HC_MULT, D), dtype=torch.bfloat16)
                    runtime.copy_stacked_from(stacked[written], step_rows)
                    finite_step = bool(torch.isfinite(step_rows.to(torch.float32)).all())
                    print(
                        "GLM_XHC_STEP", _step_key(launch), written, "finite=", finite_step,
                        "absmax=", float(step_rows.to(torch.float32).abs().max()), flush=True,
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
                "GLM_HEAD_HIDDEN finite=", bool(torch.isfinite(head_rows).all()),
                "absmax=", float(head_rows.to(torch.float32).abs().max()), flush=True,
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
                "GLM_XHC_STREAM", stream_name, "finite=", finite_stream,
                "absmax=", float(stream_rows.to(torch.float32).abs().max()), flush=True,
            )
            if not finite_stream:
                guard_events.append(f"nonfinite {stream_name} at step {step}")
                raise FloatingPointError(f"nonfinite {stream_name} at step {step}")
        if self.lm_head_backend == "npu":
            shards = (shard[:, :LOCAL_VOCAB] for shard in self._host_logits_shards.unbind(dim=0))
            logits = torch.cat(tuple(shards), dim=1)
        else:
            # All ranks hold the same restored stream; rank 0's rows are the answer.
            logits = host_lm_head_logits(head_rows[0, :BATCH], final_norm_w, lm_head)
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

    if type(spec_tokens) is not int or spec_tokens not in range(C.MTP_SPEC_TOKENS + 1):
        raise ValueError(f"spec_tokens must be in 0..{C.MTP_SPEC_TOKENS}")
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
        from models.glm5_3_flash.vision.forward import VisionEncoder
        from models.glm5_3_flash.vision.processor import VisualInput

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
                    candidates = [(remainder.find(marker), marker) for marker in ("<|image|>", "<|video|>")
                                  if marker in remainder]
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
            part = encoded[offset:offset + count]
            visual_features[request] = torch.cat((visual_features[request], part)) if request in visual_features else part
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
    with ResidentPrefillSession(checkpoint, programs, device_ids, platform=platform, enable_mtp=bool(spec_tokens)) as session:
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
                        count -= count % CHUNK
                    if count == 0:
                        continue
                    chunk, remaining[request] = tokens[:count], tokens[count:]
                    slots.append(request)
                    chunks.append(chunk)
                    if request in visual_features:
                        from models.glm5_3_flash.vision.config import IMAGE_TOKEN_ID

                        count_visual = chunk.count(IMAGE_TOKEN_ID)
                        begin = feature_offsets[request]
                        if count_visual:
                            features[request] = visual_features[request][begin:begin + count_visual]
                            feature_offsets[request] += count_visual
                    budget -= count
                dispatch_start = time.monotonic()
                partial = session.forward_chunks(chunks, slots, visual_features=features)
                dispatch_seconds += time.monotonic() - session.last_dispatch_started_at
                input_setup_seconds += session.last_dispatch_started_at - dispatch_start
                logits[slots] = partial
                if spec_tokens:
                    next_tokens = [remaining[slot][0] if remaining[slot] else int(partial[index].argmax())
                                   for index, slot in enumerate(slots)]
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
                draft_slots = [request for request in range(len(prompts))
                               if not finished[request] and not draft_queues[request]]
                if draft_slots:
                    count = min(spec_tokens, max_new_tokens - step - 1)
                    for slot, proposed in session.propose_mtp(draft_slots, count).items():
                        draft_queues[slot] = proposed
            step_end = time.monotonic()
            step_log.append({
                "step": step, "seconds": dispatch_seconds,
                "input_setup_seconds": input_setup_seconds,
                "end_to_end_seconds": step_end - tick, "token_ids": next_ids,
                "text": [tokenizer.decode(tokens, skip_special_tokens=False) for tokens in generated],
            })
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
                raise AssertionError(
                    f"prompt {request}: expected top-1 prefix {list(expected)}, got {produced}"
                )

    def _truncate_at_eos(tokens: list[int]) -> list[int]:
        stops = [index for index, token in enumerate(tokens) if token in eos_ids]
        return tokens[: stops[0]] if stops else tokens

    return {
        "platform": "a2a3", "tp": TP_SIZE, "ep": EP_SIZE, "dp": DP_SIZE,
        "precision": "official_bf16_w8a8", "checkpoint": str(checkpoint.root),
        "capacity": CAPACITY, "max_seq_len": MAX_SEQ_LEN, "cache_tokens": CACHE_TOKENS,
        "full_prefix_recomputation": False,
        "mtp": {"tokens": spec_tokens, "verification": "sequential_greedy", **session.mtp_stats},
        "layers_per_forward": NUM_LAYERS,
        "launch_mode": "rt",
        "device_launches_per_rank_per_forward": len(session.plan),
        "resident_bytes_per_rank": resident_bytes_per_rank,
        "seconds": time.monotonic() - start, "steps": step_log, "guard_events": guard_events,
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

def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="production generate() mode: the resident W8A8 checkpoint root (required; "
             "the CI fixture case ships in the follow-up CI PR)",
    )
    parser.add_argument(
        "--runtime-dir", type=Path, default=None,
        help="manifest dir of the launch plan artifacts; compiled on first use",
    )
    # The follow-up CI entry carries the -p/--platform flag this file used to
    # forward: without it detect_changes.py's has_platform_cli probe no longer
    # selects this production CLI as a CI case (the fixture ships separately).
    # -d stays accepted-and-ignored under --checkpoint (pre-t37 behavior).
    parser.add_argument(
        "-d", "--device", default=os.environ.get("TASK_DEVICE"),
        help="comma-separated device ids (accepted, ignored under --checkpoint)",
    )
    parser.add_argument(
        "--tokens", type=int, default=T, help=argparse.SUPPRESS  # consumed at import time
    )
    parser.add_argument(
        "--devices", default=os.environ.get("TASK_DEVICE"),
        help=argparse.SUPPRESS,  # generate() device list, kept for the production CLI
    )
    parser.add_argument("--tp", type=int, default=C.TP_SIZE, choices=[16, 8, 2])
    parser.add_argument("--ep", type=int, default=RANKS, choices=[16, 8, 2])
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--cache-tokens", type=int, default=CACHE_TOKENS)
    parser.add_argument("--prompt", action="append", help=f"one prompt of the packed batch; provide between 1 and {BATCH}")
    parser.add_argument("--compile-only", action="store_true", help="build all programs without allocating NPU devices")
    parser.add_argument("--spec-tokens", type=int, choices=range(C.MTP_SPEC_TOKENS + 1), default=0)
    parser.add_argument("--image", action="append", type=Path, help="image for prompt zero; repeat for multiple images")
    parser.add_argument("--video", action="append", type=Path, help="video file for prompt zero; decoded with FFmpeg")
    parser.add_argument("--vision-runtime-dir", type=Path, help="precompiled single-rank vision artifact")
    parser.add_argument("--video-frame", action="append", type=Path, help="sampled video frame for prompt zero")
    parser.add_argument("--video-frame-fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--raw", action="store_true", help="encode prompts verbatim, no chat template")
    parser.add_argument(
        "--expect-tokens", type=Path,
        help="JSON mapping prompt index to the expected top-1 token-id prefix",
    )
    parser.add_argument("--dump-tokens", type=Path, help="write the produced token ids as JSON")
    options, unknown = parser.parse_known_args(argv)
    if options.compile_only:
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        if options.runtime_dir is None:
            parser.error("--compile-only requires --runtime-dir")
        compile_production(options.runtime_dir, list(range(RANKS)))
        return
    if options.checkpoint is None:
        # Bare runs are the CI fixture case, which ships in the follow-up CI
        # PR (the prefill_fwd_ci entry); production mode needs --checkpoint.
        if options.prompt:
            raise ValueError("--prompt needs the generate() mode (--checkpoint)")
        parser.error(
            "the CI fixture case ships in the follow-up CI PR; production needs --checkpoint"
        )
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if options.capacity != CAPACITY:
        raise ValueError(
            f"--capacity must match compilation ({CAPACITY}); recompile to change it"
        )
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
            {int(key): value for key, value in raw.items()} if isinstance(raw, dict)
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
        from models.glm5_3_flash.vision.processor import VisionProcessor

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

        programs["_vision"] = DistributedCompiledProgram.from_dir(options.vision_runtime_dir, platform="a2a3",
                              distributed_config=DistributedConfig(device_ids=[device_ids[0]], num_sub_workers=0))
    report = generate(
        checkpoint, programs, device_ids, prompts, options.max_new_tokens,
        raw_prompts=options.raw,
        expect_tokens=expect_tokens,
        dump_path=options.dump_tokens,
        visual_inputs=visual_inputs,
        spec_tokens=options.spec_tokens,
    )
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("GLM_GENERATION_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)
    return report

if __name__ == "__main__":
    main()
