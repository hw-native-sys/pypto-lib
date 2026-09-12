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
"""DeepSeek-V4 Flash prefill entry with lib-owned context parallelism.

The serving-facing ``l3_prefill_fwd`` ABI owns request distribution and CP
workspace. Every supported request uses CP, including short prompts. The
current contract is CP=EP>1, one active request owner, position zero, and
1..CP_SIZE*1024 input tokens. CP1, prefix reuse, chunk continuation and
multiple active owners are not supported by this entry. Invalid runtime
metadata produces NaN hidden outputs; callers must enforce this contract.

One layer schedule follows model order through attention and MoE,
then HC head and final RMSNorm. The HOST projects selected hidden rows through
the established LM head. The colocated standalone runner is an execution
smoke check, not a numerical oracle.
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
from golden import run
from pypto.ir import DistributedConfig

# Leaf kernels own attention math; this module owns the full layer schedule.
import config
# Import moe first: it applies the EP/FLASH override before the attention modules
# bake config-derived MoE shapes.
from moe import (
    PrefillMoELayout,
    make_prefill_moe,
    D,
    HC_DIM,
    HC_MULT,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    clear_prefill_moe_signals,
    check_prefill_moe_slab,
    PREFILL_MOE_EXPERT_SCALE_PAD,
    PREFILL_MOE_SCALE_PAD,
)

T = config.PREFILL_TOKENS
from config import FLASH as MODEL_CONFIG
from prefill_swa import (
    build_tensor_specs as build_swa_attention_tensor_specs,
)
from prefill_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
)
from prefill_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    CSA_CMP_BLOCK_NUM,
    CSA_ORI_BLOCK_NUM,
    H,
    HEAD_DIM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
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
)
from prefill_compressor_ratio4 import CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_STATE_MAX_BLOCKS
from prefill_indexer_compressor import INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_MAX_BLOCKS
from prefill_compressor_ratio128 import HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_STATE_MAX_BLOCKS
from hc_head import hc_head
from lm_head import (
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
    lm_head_test,
)
from rmsnorm import rms_norm

assert config.PREFILL_TOKENS == T


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
HCA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // HCA_COMPRESS_RATIO
CSA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // CSA_COMPRESS_RATIO
HCA_COMPRESS_STATE_DIM = 2 * HCA_MAIN_OUT_DIM
CSA_COMPRESS_STATE_DIM = 2 * CSA_MAIN_OUT_DIM
CSA_INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
FWD_LAST_LAYER = FWD_NUM_LAYERS - 1
CSA_LAST_ORDER = CSA_NUM_LAYERS - 1

FWD_TOKENS_DYN = pl.dynamic("PREFILL_FWD_TOKENS_DYN")
PRE_HC_COPY_TOKEN_TILE = 4
assert T % PRE_HC_COPY_TOKEN_TILE == 0

# The LM head owns its barrier counters, so its epoch restarts at 1 rather
# than continuing the MoE numbering.
LM_HEAD_COMM_EPOCH = 1
assert MODEL_NUM_LAYERS == 43, "DeepSeek-V4 Flash hidden layer count changed"

# Physical cache pools are runtime-sized.  The first dimension of each
# stacked cache is the per-layer pool size multiplied by its layer count.
FWD_ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
FWD_HCA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_CMP_BLOCK_NUM_DYN")
FWD_CSA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_CMP_BLOCK_NUM_DYN")
FWD_IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
FWD_HCA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")
FWD_CSA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
FWD_INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")

# Replicated head weights (per-rank, not layer-stacked): hc_head projection and
# the final RMSNorm gamma — mirrors decode_fwd.
HC_HEAD_NAMES = ["hc_head_fn", "hc_head_scale", "hc_head_base"]
FINAL_NORM_NAMES = ["final_norm_w"]

# Per-FWD-layer stacked weights (sliced by the FWD layer index 0..42).
FWD_LAYER_STACKED_NAMES = [
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "hca_cmp_kv", "csa_cmp_kv",
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
    "csa_compress_state",
    "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_compress_state", "csa_cmp_kv", "idx_kv_cache", "idx_kv_scale",
]
# HCA-compact stacked weights (sliced by the HCA order index 0..19).
HCA_LAYER_STACKED_NAMES = [
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hca_compress_state", "hca_cmp_kv",
]
# Replicated once and passed whole to every layer (block tables are smoke zeros;
# slot mappings depend only on token position + a fixed per-kind compress ratio,
# so a single copy per name is shared across all layers of that kind).
SHARED_NAMES = [
    "freqs_cos", "freqs_sin",
    "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
    "hca_compress_state_block_table", "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "ori_slot_mapping", "position_ids", "input_ids",
    "hca_cmp_slot_mapping", "hca_state_slot_mapping",
    "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
    "csa_state_slot_mapping", "csa_inner_state_slot_mapping",
]

# KV / state caches: per-token persistent buffers, not weights — kept as host
# tensors (re-bound each dispatch) rather than device-resident.
CACHE_NAMES = {
    "kv_cache", "hca_cmp_kv", "csa_cmp_kv",
    "hca_compress_state", "csa_compress_state", "csa_inner_compress_state",
    "idx_kv_cache", "idx_kv_scale",
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

# Every cache in this set is mutated by one of the packed attention/compressor
# kernels and must remain visible to the following decode invocation.
RESIDENT_CACHE_OUTPUT_NAMES = RESIDENT_CACHE_NAMES


# CP scheduling and communication capacities.
from prefill_swa import (
    CP_CHOICES, CP_SIZE, CP_TAIL_WINDOW_ROWS, LOCAL_PARTS, NUM_SEGMENTS,
    ORI_MAX_BLOCKS, OVERLAY_BASE, OVERLAY_ROWS, OVERLAY_SOURCES, TAIL_ROWS, WIN,
    prefill_attention_swa,
)
from prefill_cp_zigzag import MAX_SEGMENT_TILES, CP_PREFILL_CMP_BLOCK_NUM as PREFILL_CMP_BLOCK_NUM
from prefill_hca import prefill_attention_hca
from prefill_sparse_attn import HCA_MAX_COMPRESSED_ROWS
from prefill_csa import (
    LOCAL_LEAVES as CSA_LOCAL_LEAVES, MAX_COMPRESS_LEAVES as CSA_MAX_COMPRESS_LEAVES,
    prefill_attention_csa,
)
from config import PREFILL_CMP_MAX_BLOCKS
from prefill_cp_exchange import (
    CMP_META_DIM, CMP_WINDOW_ROWS, META_DIM, RECORDS_PER_WINDOW, SCALE_TILE_COLS,
    STATE_META_DIM, STATE_RECORDS_PER_WINDOW, STATE_WINDOW_ROWS,
    _clear_prefill_cp_exchange_signals,
)

ATTN_TILE_ROWS = TAIL_ROWS
NUM_ATTN_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
CP_LOCAL_ROWS = NUM_ATTN_TILES * ATTN_TILE_ROWS
CP_MOE_LAYOUT = PrefillMoELayout(CP_LOCAL_ROWS)
cp_prefill_moe = make_prefill_moe(CP_MOE_LAYOUT)
CP_MOE_ROUTES_PER_SRC = CP_MOE_LAYOUT.routes_per_source
CP_MOE_TOTAL_CAP = CP_MOE_LAYOUT.total_capacity
MOE_ID_COPY_TILE = 8

check_prefill_moe_slab(CP_LOCAL_ROWS)
assert CP_SIZE in CP_CHOICES, f"--cp must be one of {CP_CHOICES} (got {CP_SIZE})"
assert CP_SIZE in (1, N_RANKS), f"CP must be 1 or EP (got CP={CP_SIZE}, EP={N_RANKS})"
assert LOCAL_PARTS == 2 and ATTN_TILE_ROWS == 128 and CP_LOCAL_ROWS == 1024, "CP leaf/slab capacity changed"
assert ATTN_TILE_ROWS % PRE_HC_COPY_TOKEN_TILE == 0
assert CP_LOCAL_ROWS % MOE_ID_COPY_TILE == 0


@pl.jit.inline
def _fwd_attention_stage_barrier_from_completion(
    completion_token: pl.Tensor[[NUM_ATTN_TILES, 1, 8], pl.FP32],
) -> pl.Scalar[pl.TASK_ID]:
    """Complete the attention stage before dispatching MoE."""
    stage_token = pl.create_tensor([1, 1, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fwd_attn_stage_token", allow_early_resolve=False) as stage_tid:
        stage_token[0:1, 0:1, 0:8] = pl.slice(completion_token, [1, 1, 8], [0, 0, 0])
    # Complete attention before constructing the dependent MoE graph.
    _completed = pl.read(stage_token, [0, 0, 0])
    return stage_tid


@pl.jit.inline
def _fwd_attention_stage_barrier_from_x_attn(
    x_attn: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32],
) -> pl.Scalar[pl.TASK_ID]:
    """HCA counterpart: one task samples every attention output tile."""
    x_attn_flat = pl.reshape(x_attn, [CP_LOCAL_ROWS, HC_MULT, D])
    stage_tokens = pl.create_tensor([NUM_ATTN_TILES, 1, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fwd_attn_stage_token", allow_early_resolve=False) as stage_tid:
        for tile in pl.range(NUM_ATTN_TILES):
            row0 = tile * ATTN_TILE_ROWS
            stage_tokens[tile : tile + 1, 0:1, 0:8] = pl.slice(x_attn_flat, [1, 1, 8], [row0, 0, 0])
    # Complete attention before constructing the dependent MoE graph.
    _completed = pl.read(stage_tokens, [0, 0, 0])
    return stage_tid


@pl.jit.inline
def _fwd_pack_moe_inputs(
    x_attn: pl.Tensor[[CP_LOCAL_ROWS, HC_MULT, D], pl.FP32],
    input_ids: pl.Tensor[[CP_LOCAL_ROWS], pl.INT64],
    active_flat: pl.Tensor[[NUM_ATTN_TILES, OVERLAY_SOURCES], pl.INT32],
    packed_x: pl.Tensor[[CP_LOCAL_ROWS, HC_MULT, D], pl.FP32],
    packed_ids: pl.Tensor[[CP_LOCAL_ROWS], pl.INT64],
    attention_done_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.INT32]:
    """Pack the two real segment prefixes and zero the unused MoE capacity."""
    first_tokens = pl.cast(0, pl.INDEX)
    second_tokens = pl.cast(0, pl.INDEX)
    for tile in pl.range(MAX_SEGMENT_TILES):
        first_tokens = first_tokens + pl.cast(pl.read(active_flat, [tile, 1]), pl.INDEX)
        second_tokens = second_tokens + pl.cast(pl.read(active_flat, [MAX_SEGMENT_TILES + tile, 1]), pl.INDEX)
    active_tokens = first_tokens + second_tokens
    for block in pl.spmd(
        (CP_LOCAL_ROWS // PRE_HC_COPY_TOKEN_TILE) * HC_MULT,
        name_hint="pack_moe_hidden", deps=[attention_done_tid],
    ):
        row0 = (block // HC_MULT) * PRE_HC_COPY_TOKEN_TILE
        hc_lane = block % HC_MULT
        for lane in pl.range(PRE_HC_COPY_TOKEN_TILE):
            row = row0 + lane
            if row < active_tokens:
                source = row
                if row >= first_tokens:
                    source = MAX_SEGMENT_TILES * ATTN_TILE_ROWS + row - first_tokens
                packed_x[row : row + 1, hc_lane : hc_lane + 1, :] = pl.slice(x_attn, [1, 1, D], [source, hc_lane, 0])
            else:
                packed_x[row : row + 1, hc_lane : hc_lane + 1, :] = pl.full([1, 1, D], dtype=pl.FP32, value=0.0)
    for block in pl.spmd(CP_LOCAL_ROWS // MOE_ID_COPY_TILE, name_hint="pack_moe_ids"):
        for lane in pl.range(MOE_ID_COPY_TILE):
            row = block * MOE_ID_COPY_TILE + lane
            value = pl.cast(0, pl.INT64)
            if row < active_tokens:
                source = row
                if row >= first_tokens:
                    source = MAX_SEGMENT_TILES * ATTN_TILE_ROWS + row - first_tokens
                value = pl.read(input_ids, [source])
            pl.write(packed_ids, [row], value)
    return pl.cast(active_tokens, pl.INT32)


@pl.jit.inline
def _fwd_restore_hidden_layout(
    x_next_work: pl.Tensor[[CP_LOCAL_ROWS, HC_MULT, D], pl.FP32],
    active_flat: pl.Tensor[[NUM_ATTN_TILES, OVERLAY_SOURCES], pl.INT32],
    hidden_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32]],
    moe_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32]:
    """Restore packed MoE rows to the two attention slots and zero padding."""
    hidden_flat = pl.reshape(hidden_out, [CP_LOCAL_ROWS, HC_MULT, D])
    first_tokens = pl.cast(0, pl.INDEX)
    for first_tile in pl.range(MAX_SEGMENT_TILES):
        first_tokens = first_tokens + pl.cast(pl.read(active_flat, [first_tile, 1]), pl.INDEX)
    tile_blocks = (ATTN_TILE_ROWS // PRE_HC_COPY_TOKEN_TILE) * HC_MULT
    with pl.spmd(NUM_ATTN_TILES * tile_blocks, name_hint="restore_hidden_layout", deps=[moe_tid]) as _restore_tid:
        block = pl.tile.get_block_idx()
        tile = block // tile_blocks
        tile_block = block % tile_blocks
        token_block = tile_block // HC_MULT
        hc_lane = tile_block % HC_MULT
        token0 = token_block * PRE_HC_COPY_TOKEN_TILE
        active = pl.read(active_flat, [tile, 1])
        for dt in pl.range(PRE_HC_COPY_TOKEN_TILE):
            token = token0 + dt
            row = tile * ATTN_TILE_ROWS + token
            if token < active:
                packed_row = row
                if tile >= MAX_SEGMENT_TILES:
                    packed_row = first_tokens + row - MAX_SEGMENT_TILES * ATTN_TILE_ROWS
                hidden_flat[
                    row : row + 1,
                    hc_lane : hc_lane + 1,
                    0:D,
                ] = pl.slice(x_next_work, [1, 1, D], [packed_row, hc_lane, 0])
            else:
                hidden_flat[row : row + 1, hc_lane : hc_lane + 1, 0:D] = pl.full([1, 1, D], dtype=pl.FP32, value=0.0)
    return pl.reshape(hidden_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D])


@pl.jit.inline
def _fwd_wait_previous_moe(
    attention_tid: pl.Scalar[pl.TASK_ID],
    previous_completion: pl.Tensor[[1, 1, 8], pl.FP32],
) -> pl.Scalar[pl.TASK_ID]:
    # Empty CP ranks may not read the preceding hidden at all. Keep their
    # source-side communication epochs ordered by the explicit completion.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="fwd_previous_moe_complete",
        deps=[attention_tid],
        allow_early_resolve=False,
    ) as completed_tid:
        _previous = pl.read(previous_completion, [0, 0, 0])
    return completed_tid


@pl.jit.inline
def _fwd_moe_tail(
    x_attn: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32],
    overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32],
    input_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS], pl.INT64],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    moe_x_mixed: pl.InOut[pl.Tensor[[CP_LOCAL_ROWS, D], pl.BF16]],
    moe_post_ffn: pl.InOut[pl.Tensor[[CP_LOCAL_ROWS, HC_MULT], pl.FP32]],
    moe_comb_ffn: pl.InOut[pl.Tensor[[CP_LOCAL_ROWS, HC_MULT * HC_MULT], pl.FP32]],
    moe_ffn_out: pl.InOut[pl.Tensor[[CP_LOCAL_ROWS, D], pl.BF16]],
    moe_dense_scale: pl.InOut[pl.Tensor[[CP_MOE_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32]],
    moe_returned_y: pl.InOut[pl.Tensor[[CP_MOE_ROUTES_PER_SRC, D], pl.BF16]],
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prefill_moe_x_target: pld.DistributedTensor[[CP_MOE_TOTAL_CAP, D], pl.INT8],
    prefill_moe_x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prefill_moe_scale_target: pld.DistributedTensor[[CP_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    prefill_moe_reverse_target: pld.DistributedTensor[[CP_MOE_TOTAL_CAP, D], pl.BF16],
    prefill_moe_reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    hidden_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32]],
    completion_anchor: pl.Out[pl.Tensor[[1, 1, 8], pl.FP32]],
    attention_done_tid: pl.Scalar[pl.TASK_ID],
    layer_id: pl.Scalar[pl.INT32],
) -> pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32]:
    """Pack real CP rows, run one MoE slab, and restore the attention slots."""
    x_attn_flat = pl.reshape(x_attn, [CP_LOCAL_ROWS, HC_MULT, D])
    x_next_work = pl.create_tensor([CP_LOCAL_ROWS, HC_MULT, D], dtype=pl.FP32)
    active_flat = pl.reshape(overlay_active_lengths, [NUM_ATTN_TILES, OVERLAY_SOURCES])
    input_ids_flat = pl.reshape(input_ids, [CP_LOCAL_ROWS])
    packed_x = pl.create_tensor([CP_LOCAL_ROWS, HC_MULT, D], dtype=pl.FP32)
    packed_ids = pl.create_tensor([CP_LOCAL_ROWS], dtype=pl.INT64)
    active_tokens = _fwd_pack_moe_inputs(
        x_attn_flat, input_ids_flat, active_flat, packed_x, packed_ids, attention_done_tid
    )

    moe_dense_x = pl.create_tensor([CP_MOE_TOTAL_CAP, D], dtype=pl.INT8)
    moe_tid = cp_prefill_moe(
        packed_x,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, packed_ids,
        routed_w1, routed_w1_scale,
        routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next_work,
        moe_x_mixed, moe_post_ffn, moe_comb_ffn, moe_ffn_out,
        moe_dense_x, moe_dense_scale,
        moe_returned_y,
        count_target, count_signal,
        prefill_moe_x_target, prefill_moe_x_signal,
        prefill_moe_scale_target,
        prefill_moe_reverse_target, prefill_moe_reverse_signal,
        attention_done_tid, layer_id,
        pl.cast(layer_id + 1, pl.INT32),
        active_tokens,
    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="local1024_moe_completion_anchor",
        deps=[moe_tid],
        allow_early_resolve=False,
    ):
        final_element = pl.slice(x_next_work, [1, 1, 8], [CP_LOCAL_ROWS - 1, 0, 0])
        completion_anchor[0:1, 0:1, 0:8] = final_element

    hidden_out = _fwd_restore_hidden_layout(x_next_work, active_flat, hidden_out, moe_tid)
    return hidden_out


# ---------------------------------------------------------------------------
# CP request metadata and layer schedule
# ---------------------------------------------------------------------------
@pl.jit.inline
def _prefill_cp_metadata(
    control: pl.Tensor[[1, 16], pl.INT32],
    ori_block_table: pl.Tensor[[ORI_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_position_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    query_token_to_request: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    overlay_position_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_token_to_request: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32],
    swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    cache_owner_rank_t: pl.Tensor[[1], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    leaf_positions_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT32],
    leaf_main_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_idx_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_main_state_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_inner_state_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_num_tokens_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES], pl.INT32],
    hca_history_slots: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    hca_history_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    csa_history_slots: pl.Tensor[[TAIL_ROWS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Build CP coordinates and physical slots for one chunk on the owning device."""
    query_position_ids_flat = pl.reshape(query_position_ids, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), TAIL_ROWS])
    query_token_to_request_flat = pl.reshape(query_token_to_request, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), TAIL_ROWS])
    overlay_position_ids_flat = pl.reshape(overlay_position_ids, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), OVERLAY_ROWS])
    overlay_token_to_request_flat = pl.reshape(
        overlay_token_to_request, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), OVERLAY_ROWS]
    )
    overlay_active_lengths_flat = pl.reshape(
        overlay_active_lengths, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), OVERLAY_SOURCES]
    )
    swa_indices_flat = pl.reshape(swa_indices, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES) * (TAIL_ROWS), WIN])
    leaf_positions_input_flat = pl.reshape(
        leaf_positions_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )
    leaf_main_slots_input_flat = pl.reshape(
        leaf_main_slots_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )
    leaf_idx_slots_input_flat = pl.reshape(
        leaf_idx_slots_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )
    leaf_main_state_slots_input_flat = pl.reshape(
        leaf_main_state_slots_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )
    leaf_inner_state_slots_input_flat = pl.reshape(
        leaf_inner_state_slots_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )

    for cp_request_segments_block in pl.spmd(1, name_hint="cp_request_segments"):
        length = pl.cast(pl.read(control, [0, 2]), pl.INDEX)
        span = pl.cast(pl.read(control, [0, 3]), pl.INDEX)
        base = pl.cast(pl.read(control, [0, 4]), pl.INDEX)
        pl.write(cache_owner_rank_t, [0], pl.read(control, [0, 1]))
        pl.write(final_segment_t, [0], pl.cast((length - 1) // span, pl.INT32))
        for segment in pl.range(NUM_SEGMENTS):
            start = base + segment * span
            active = pl.max(0, pl.min(span, base + length - start))
            if segment < CP_SIZE:
                owner = segment
                part = pl.cast(0, pl.INDEX)
            else:
                owner = NUM_SEGMENTS - 1 - segment
                part = pl.cast(1, pl.INDEX)
            pl.write(segment_starts_t, [segment], pl.cast(start, pl.INT32))
            pl.write(segment_lengths_t, [segment], pl.cast(active, pl.INT32))
            pl.write(owner_rank_table, [segment], pl.cast(owner, pl.INT32))
            pl.write(owner_part_table, [segment], pl.cast(part, pl.INT32))
            pl.write(reverse_index, [segment], pl.cast(2 * owner + part, pl.INT32))
            for row in pl.range(TAIL_ROWS):
                position = pl.cast(-1, pl.INDEX)
                if row < pl.min(TAIL_ROWS, active):
                    position = pl.cast(start + pl.max(0, active - TAIL_ROWS) + row, pl.INDEX)
                pl.write(segment_tail_positions, [segment, row], pl.cast(position, pl.INT32))
        for part in pl.range(LOCAL_PARTS):
            if part == 0:
                segment = pl.cast(my_rank, pl.INDEX)
            else:
                segment = pl.cast(NUM_SEGMENTS - 1 - my_rank, pl.INDEX)
            start = base + segment * span
            active = pl.max(0, pl.min(span, base + length - start))
            end = start + active
            valid = pl.cast(0, pl.INDEX)
            if active > 0:
                valid = pl.min(TAIL_ROWS, end)
            pl.write(owner_segments_t, [part], pl.cast(segment, pl.INT32))
            pl.write(predecessor_segments, [part], pl.cast(segment - 1, pl.INT32))
            pl.write(segment_active_lengths, [part], pl.cast(active, pl.INT32))
            pl.write(snapshot_valid, [part], pl.cast(valid, pl.INT32))
            for row in pl.range(TAIL_ROWS):
                position = pl.cast(-1, pl.INDEX)
                if row < valid:
                    position = pl.cast(end - valid + row, pl.INDEX)
                pl.write(snapshot_positions, [part, row], pl.cast(position, pl.INT32))
        for row in pl.range(TAIL_ROWS):
            position = base + length - TAIL_ROWS + row
            source = pl.cast(-1, pl.INT32)
            source_row = pl.cast(-1, pl.INT32)
            slot = pl.cast(-1, pl.INT32)
            if position >= base:
                segment = (position - base) // span
                start = base + segment * span
                active = pl.max(0, pl.min(span, base + length - start))
                source = pl.cast(segment, pl.INT32)
                source_row = pl.cast(position - start - pl.max(0, active - TAIL_ROWS), pl.INT32)
                page = pl.read(ori_block_table, [position // BLOCK_SIZE])
                if page >= 0:
                    slot = pl.cast(page * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT32)
            pl.write(final_win_seg_src, [row], source)
            pl.write(final_win_row_src, [row], source_row)
            pl.write(final_slot_mapping, [row], slot)
        # Window an earlier chunk of this request already committed. HCA and CSA
        # read it out of the paged cache instead of re-projecting hidden states
        # they no longer hold; -1 means "no history", i.e. a fresh request.
        for part in pl.range(LOCAL_PARTS):
            for row in pl.range(TAIL_ROWS):
                history_slot = pl.cast(-1, pl.INT32)
                history_position = pl.cast(-1, pl.INT32)
                if part == 0:
                    absolute = base - TAIL_ROWS + row
                    if absolute >= 0:
                        history_position = pl.cast(absolute, pl.INT32)
                        history_page = pl.read(ori_block_table, [absolute // BLOCK_SIZE])
                        if history_page >= 0:
                            history_slot = pl.cast(
                                history_page * BLOCK_SIZE + absolute % BLOCK_SIZE, pl.INT32
                            )
                pl.write(hca_history_slots, [part, row], history_slot)
                pl.write(hca_history_positions, [part, row], history_position)
        for row in pl.range(TAIL_ROWS):
            csa_slot = pl.cast(-1, pl.INT32)
            csa_absolute = base - TAIL_ROWS + row
            if csa_absolute >= 0:
                csa_page = pl.read(ori_block_table, [csa_absolute // BLOCK_SIZE])
                if csa_page >= 0:
                    csa_slot = pl.cast(csa_page * BLOCK_SIZE + csa_absolute % BLOCK_SIZE, pl.INT32)
            pl.write(csa_history_slots, [row], csa_slot)

    # One block owns all small active-length fields to avoid cache-line sharing.
    for cp_query_coordinates_block in pl.spmd(1, name_hint="cp_query_coordinates"):
        length = pl.cast(pl.read(control, [0, 2]), pl.INDEX)
        span = pl.cast(pl.read(control, [0, 3]), pl.INDEX)
        base = pl.cast(pl.read(control, [0, 4]), pl.INDEX)
        for part in pl.range(LOCAL_PARTS):
            if part == 0:
                query_segment = pl.cast(my_rank, pl.INDEX)
            else:
                query_segment = pl.cast(NUM_SEGMENTS - 1 - my_rank, pl.INDEX)
            start = base + query_segment * span
            seg_length = pl.max(0, pl.min(span, base + length - start))
            for tile in pl.range(MAX_SEGMENT_TILES):
                active = pl.max(0, pl.min(ATTN_TILE_ROWS, seg_length - tile * ATTN_TILE_ROWS))
                tile_start = start + tile * ATTN_TILE_ROWS
                if tile > 0:
                    pred_start = start + (tile - 1) * ATTN_TILE_ROWS
                    pred_length = pl.max(0, pl.min(ATTN_TILE_ROWS, seg_length - (tile - 1) * ATTN_TILE_ROWS))
                else:
                    if query_segment > 0:
                        pred_seg_start = base + (query_segment - 1) * span
                        pred_seg_length = pl.max(0, pl.min(span, base + length - pred_seg_start))
                        pred_length = pl.min(TAIL_ROWS, pred_seg_length)
                        pred_start = pred_seg_start + pl.max(0, pred_seg_length - TAIL_ROWS)
                    else:
                        pred_start = pl.cast(0, pl.INDEX)
                        pred_length = pl.cast(0, pl.INDEX)
                pl.write(overlay_active_lengths_flat, [part * MAX_SEGMENT_TILES + tile, 0], pl.cast(pred_length, pl.INT32))
                pl.write(overlay_active_lengths_flat, [part * MAX_SEGMENT_TILES + tile, 1], pl.cast(active, pl.INT32))
                for row in pl.range(ATTN_TILE_ROWS):
                    query = pl.cast(0, pl.INT32)
                    request = pl.cast(-1, pl.INT32)
                    current = pl.cast(-1, pl.INT32)
                    previous = pl.cast(-1, pl.INT32)
                    previous_request = pl.cast(-1, pl.INT32)
                    if row < active:
                        query = pl.cast(tile_start + row, pl.INT32)
                        request = pl.cast(0, pl.INT32)
                        current = query
                    if row < pred_length:
                        previous = pl.cast(pred_start + row, pl.INT32)
                        previous_request = pl.cast(0, pl.INT32)
                    pl.write(query_position_ids_flat, [part * MAX_SEGMENT_TILES + tile, row], query)
                    pl.write(query_token_to_request_flat, [part * MAX_SEGMENT_TILES + tile, row], request)
                    pl.write(overlay_position_ids_flat, [part * MAX_SEGMENT_TILES + tile, row], previous)
                    pl.write(overlay_token_to_request_flat, [part * MAX_SEGMENT_TILES + tile, row], previous_request)
                    pl.write(overlay_position_ids_flat, [part * MAX_SEGMENT_TILES + tile, ATTN_TILE_ROWS + row], current)
                    pl.write(overlay_token_to_request_flat, [part * MAX_SEGMENT_TILES + tile, ATTN_TILE_ROWS + row], request)


    for block in pl.spmd(LOCAL_PARTS * MAX_SEGMENT_TILES):
        part = block // MAX_SEGMENT_TILES
        tile = block % MAX_SEGMENT_TILES
        swa_base = pl.cast(pl.read(control, [0, 4]), pl.INDEX)
        for row in pl.range(ATTN_TILE_ROWS):
            query = pl.read(query_position_ids_flat, [block, row])
            request = pl.read(query_token_to_request_flat, [block, row])
            active = pl.cast(pl.read(overlay_active_lengths_flat, [block, 1]), pl.INDEX)
            pred_length = pl.cast(pl.read(overlay_active_lengths_flat, [block, 0]), pl.INDEX)
            tile_start = pl.cast(pl.read(overlay_position_ids_flat, [block, ATTN_TILE_ROWS]), pl.INDEX)
            pred_start = pl.cast(pl.read(overlay_position_ids_flat, [block, 0]), pl.INDEX)
            for col in pl.range(WIN):
                key = query - WIN + 1 + col
                index = pl.cast(-1, pl.INT32)
                if request >= 0:
                    # Lower pre-chunk keys to owner-cache rows; overlays use current-chunk keys.
                    if key >= 0:
                        if key < swa_base:
                            history_key = pl.cast(key, pl.INDEX)
                            history_page = pl.read(ori_block_table, [history_key // BLOCK_SIZE])
                            if history_page >= 0:
                                index = pl.cast(
                                    history_page * BLOCK_SIZE + history_key % BLOCK_SIZE, pl.INT32
                                )
                    if key >= tile_start:
                        if key < tile_start + active:
                            index = pl.cast(OVERLAY_BASE + ATTN_TILE_ROWS + key - tile_start, pl.INT32)
                    else:
                        if key >= pred_start:
                            if key < pred_start + pred_length:
                                index = pl.cast(OVERLAY_BASE + key - pred_start, pl.INT32)
                pl.write(swa_indices_flat, [(part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS + row, col], index)


    for cp_csa_leaf_coordinates_block in pl.spmd(1, name_hint="cp_csa_leaf_coordinates"):
        length = pl.cast(pl.read(control, [0, 2]), pl.INDEX)
        span = pl.cast(pl.read(control, [0, 3]), pl.INDEX)
        base = pl.cast(pl.read(control, [0, 4]), pl.INDEX)
        for part in pl.range(LOCAL_PARTS):
            if part == 0:
                leaf_segment = pl.cast(my_rank, pl.INDEX)
            else:
                leaf_segment = pl.cast(NUM_SEGMENTS - 1 - my_rank, pl.INDEX)
            start = base + leaf_segment * span
            seg_length = pl.max(0, pl.min(span, base + length - start))
            seed = pl.cast(0, pl.INDEX)
            if leaf_segment > 0:
                if seg_length > 0:
                    predecessor_length = pl.min(TAIL_ROWS, pl.max(0, pl.min(span, base + length - (base + (leaf_segment - 1) * span))))
                    seed = pl.min(start % CSA_COMPRESS_RATIO + CSA_COMPRESS_RATIO, predecessor_length)
            for leaf in pl.range(CSA_MAX_COMPRESS_LEAVES):
                if leaf == 0:
                    active = seed
                else:
                    active = pl.max(0, pl.min(ATTN_TILE_ROWS, seg_length - (leaf - 1) * ATTN_TILE_ROWS))
                pl.write(leaf_num_tokens_input, [part, leaf], pl.cast(active, pl.INT32))
                for row in pl.range(ATTN_TILE_ROWS):
                    position = pl.cast(0, pl.INDEX)
                    main_slot = pl.cast(-1, pl.INT64)
                    inner_slot = pl.cast(-1, pl.INT64)
                    cmp_slot = pl.cast(-1, pl.INT64)
                    idx_slot = pl.cast(-1, pl.INT64)
                    if row < active:
                        if leaf == 0:
                            position = start - seed + row
                        else:
                            position = start + (leaf - 1) * ATTN_TILE_ROWS + row
                        if position // CSA_STATE_BLOCK_SIZE < CSA_STATE_MAX_BLOCKS:
                            page = pl.read(csa_compress_state_block_table, [position // CSA_STATE_BLOCK_SIZE])
                            if page >= 0:
                                main_slot = pl.cast(page * CSA_STATE_BLOCK_SIZE + position % CSA_STATE_BLOCK_SIZE, pl.INT64)
                        if position // INNER_STATE_BLOCK_SIZE < INNER_STATE_MAX_BLOCKS:
                            page = pl.read(csa_inner_compress_state_block_table, [position // INNER_STATE_BLOCK_SIZE])
                            if page >= 0:
                                inner_slot = pl.cast(page * INNER_STATE_BLOCK_SIZE + position % INNER_STATE_BLOCK_SIZE, pl.INT64)
                        if leaf > 0:
                            if (position + 1) % CSA_COMPRESS_RATIO == 0:
                                logical = (position + 1) // CSA_COMPRESS_RATIO - 1
                                if logical // CSA_CMP_STORAGE_BLOCK_SIZE < PREFILL_CMP_MAX_BLOCKS:
                                    # Local compressor output uses the compact cyclic scratch pool.
                                    scratch_page = logical // CSA_CMP_STORAGE_BLOCK_SIZE % PREFILL_CMP_BLOCK_NUM
                                    cmp_slot = pl.cast(scratch_page * CSA_CMP_STORAGE_BLOCK_SIZE + logical % CSA_CMP_STORAGE_BLOCK_SIZE, pl.INT64)
                                if logical // CSA_CMP_STORAGE_BLOCK_SIZE < IDX_CACHE_MAX_BLOCKS:
                                    page = pl.read(idx_block_table, [logical // CSA_CMP_STORAGE_BLOCK_SIZE])
                                    if page >= 0:
                                        idx_slot = pl.cast(page * CSA_CMP_STORAGE_BLOCK_SIZE + logical % CSA_CMP_STORAGE_BLOCK_SIZE, pl.INT64)
                    pl.write(leaf_positions_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], pl.cast(position, pl.INT32))
                    pl.write(leaf_main_state_slots_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], main_slot)
                    pl.write(leaf_inner_state_slots_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], inner_slot)
                    pl.write(leaf_main_slots_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], cmp_slot)
                    pl.write(leaf_idx_slots_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], idx_slot)
    return (
        segment_starts_t,
        predecessor_segments,
        query_position_ids,
        query_token_to_request,
        overlay_position_ids,
        overlay_token_to_request,
        overlay_active_lengths,
        swa_indices,
        reverse_index,
        owner_rank_table,
        final_win_seg_src,
        final_win_row_src,
        final_slot_mapping,
        segment_active_lengths,
        cache_owner_rank_t,
        owner_segments_t,
        final_segment_t,
        segment_tail_positions,
        snapshot_positions,
        snapshot_valid,
        owner_part_table,
        segment_lengths_t,
        leaf_positions_input,
        leaf_main_slots_input,
        leaf_idx_slots_input,
        leaf_main_state_slots_input,
        leaf_inner_state_slots_input,
        leaf_num_tokens_input,
        hca_history_slots,
        hca_history_positions,
        csa_history_slots,
    )


@pl.jit.inline(auto_scope=False)
def prefill_fwd(
    x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32],
    # SWA attention weights (layer-stacked: FWD_NUM_LAYERS * <unit>).
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
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # Raw KV pools use one caller-sized physical block span per layer.
    kv_cache: pl.InOut[pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    # Compressed KV pools are sliced by each attention type's layer ordinal.
    # One compressed-KV pool per flavour: a cache block holds
    # BLOCK_SIZE / COMPRESS_RATIO rows, which differs between HCA and CSA.
    hca_cmp_kv: pl.InOut[pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_position_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    query_token_to_request: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    overlay_position_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_token_to_request: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32],
    swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    # --- Shared CP metadata needed by CSA/HCA cores (beyond SWA) ----------
    # Request ownership and segment metadata are shared across model layers.
    # CSA and HCA consume the relevant fields through their stage contracts.
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    cache_owner_rank_t: pl.Tensor[[1], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    # --- HCA type-specific (layers 3, 5, ..., 41) -----------------------------
    # HCA compact compressor weights (ratio-128: OUT_DIM == HEAD_DIM, so
    # cmp_wkv/cmp_wgate are [HEAD_DIM, D] and cmp_ape is [RATIO, HEAD_DIM]).
    # Stacked by HCA_NUM_LAYERS on axis 0 -> [HCA_NUM_LAYERS * unit, ...];
    # the child slices its type ordinal (0 for L3, 1 for L5) per layer.
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    # HCA persistent compressor state (rank-local InOut root; stacked by
    # HCA_NUM_LAYERS on axis 0 -> [HCA_NUM_LAYERS * unit, ...]).
    hca_compress_state: pl.InOut[pl.Tensor[[FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    # HCA-specific metadata.
    segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    # --- CSA type-specific (layers 2, 4, ..., 42) -----------------------------
    # CSA main compressor weights (ratio-4: MAIN_OUT_DIM = 2*HEAD_DIM).
    # Stacked by CSA_NUM_LAYERS on axis 0 -> [CSA_NUM_LAYERS * unit, ...];
    # the child slices its type ordinal (0 for L2, 1 for L4) per layer.
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    # CSA indexer weights (stacked by CSA_NUM_LAYERS on axis 0).
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    # CSA inner compressor weights (ratio-4: INNER_OUT_DIM = 2*IDX_HEAD_DIM).
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    # CSA persistent state/caches (rank-local InOut roots; stacked by
    # CSA_NUM_LAYERS on axis 0 -> [CSA_NUM_LAYERS * unit, ...]).
    csa_compress_state: pl.InOut[pl.Tensor[[FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    # CSA-specific metadata.
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    leaf_positions_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT32],
    leaf_main_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_idx_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_main_state_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_inner_state_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_num_tokens_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES], pl.INT32],
    hca_history_slots: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    hca_history_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    csa_history_slots: pl.Tensor[[TAIL_ROWS], pl.INT32],
    # Indexed by logical compressed page, whose row count is per-flavour.
    hca_cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    # --- Communication windows ------------------------------------------
    # Domain 1: shared tail exchange (SWA + CSA + HCA reuse one bank under
    # monotonic tail_comm_epoch). The dual-tail exchange also needs a
    # The legacy KV-tail window remains for CSA/HCA. Recipes-aligned SWA and
    # the compressor paths exchange normalized hidden tails through the
    # hidden-tail window, then project KV on the receiving rank.
    hidden_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Domain 2: HCA compact (one bank reused by HCA layers; compact_comm_epoch).
    cmp_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    cmp_meta_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32],
    state_window: pld.DistributedTensor[[STATE_WINDOW_ROWS, HCA_COMPRESS_STATE_DIM], pl.FP32],
    state_meta_window: pld.DistributedTensor[[CP_SIZE, STATE_META_DIM], pl.INT32],
    hca_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hca_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Domain 3: CSA compact/index/state (one bank reused by CSA layers).
    main_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, CSA_MAIN_OUT_DIM], pl.BF16],
    idx_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, IDX_HEAD_DIM], pl.INT8],
    scale_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP16],
    record_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, META_DIM], pl.INT32],
    main_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, CSA_COMPRESS_STATE_DIM], pl.FP32],
    main_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32],
    inner_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32],
    inner_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32],
    csa_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    csa_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # MoE weights (layer-stacked).
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS], pl.INT64],
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
    # Rank-local resident MoE workspaces, reused by every serialized layer.
    # Compact count/x/scale/reverse windows. all_to_all_v owns reusable,
    # self-clearing collective signals, so no per-wave epoch ABI remains.
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prefill_moe_x_target: pld.DistributedTensor[[CP_MOE_TOTAL_CAP, D], pl.INT8],
    prefill_moe_x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prefill_moe_scale_target: pld.DistributedTensor[[CP_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    prefill_moe_reverse_target: pld.DistributedTensor[[CP_MOE_TOTAL_CAP, D], pl.BF16],
    prefill_moe_reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # Final normalization weights (HC head + final RMSNorm). The HC head
    # projects the [HC_MULT, D] hyper-connection mix to a single [D] row; the
    # final RMSNorm normalizes it into hidden_out for the LM head.
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    # Final outputs. pre_hc_hidden_out is the FP32 pre-HC MoE result (one
    # [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D] slab);
    # hidden_out is the BF16 post-RMSNorm final hidden ([LOCAL_ROWS, D]). The
    # host's next stage broadcasts its unique global-final row before LM head.
    # Both are pl.Out so the host ties them to host-level output slots.
    pre_hc_hidden_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32]],
    hidden_out: pl.Out[pl.Tensor[[CP_LOCAL_ROWS, D], pl.BF16]],
    # Scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[CP_LOCAL_ROWS, D], pl.BF16]:
    """Run the chronological attention/MoE schedule over caller-owned caches."""

    # Invocation-local MoE scratch is reused by the chronological layer loop.
    moe_x_mixed = pl.create_tensor([CP_LOCAL_ROWS, D], dtype=pl.BF16)
    moe_post_ffn = pl.create_tensor([CP_LOCAL_ROWS, HC_MULT], dtype=pl.FP32)
    moe_comb_ffn = pl.create_tensor([CP_LOCAL_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
    moe_ffn_out = pl.create_tensor([CP_LOCAL_ROWS, D], dtype=pl.BF16)
    moe_dense_scale = pl.create_tensor([CP_MOE_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32)
    moe_returned_y = pl.create_tensor([CP_MOE_ROUTES_PER_SRC, D], dtype=pl.BF16)

    swa_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    swa_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    compressed_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = (
        pl.slice(freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [1, 0, 0])
    )
    compressed_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = (
        pl.slice(freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [1, 0, 0])
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

    raw_blocks = pl.tensor.dim(kv_cache, 0) // FWD_NUM_LAYERS
    hca_cmp_blocks = pl.tensor.dim(hca_cmp_kv, 0) // HCA_NUM_LAYERS
    hca_state_blocks = pl.tensor.dim(hca_compress_state, 0) // HCA_NUM_LAYERS
    csa_cmp_blocks = pl.tensor.dim(csa_cmp_kv, 0) // CSA_NUM_LAYERS
    csa_idx_blocks = pl.tensor.dim(idx_kv_cache, 0) // CSA_NUM_LAYERS
    csa_state_blocks = pl.tensor.dim(csa_compress_state, 0) // CSA_NUM_LAYERS
    csa_inner_state_blocks = (pl.tensor.dim(csa_inner_compress_state, 0) // CSA_NUM_LAYERS)
    # CSA scratch shares each sliced state root's physical page capacity.
    main_state_workspace0 = pl.create_tensor(
        [csa_state_blocks, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
        dtype=pl.FP32,
    )
    main_state_workspace1 = pl.create_tensor(
        [csa_state_blocks, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
        dtype=pl.FP32,
    )
    inner_state_workspace0 = pl.create_tensor(
        [csa_inner_state_blocks, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM],
        dtype=pl.FP32,
    )
    inner_state_workspace1 = pl.create_tensor(
        [csa_inner_state_blocks, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM],
        dtype=pl.FP32,
    )
    effective_x_workspace = pl.create_tensor([CSA_LOCAL_LEAVES * ATTN_TILE_ROWS, D], dtype=pl.BF16)

    # Every layer uses the same local storage and monotonically increasing
    # protocol epochs. Empty ranks retain the explicit previous-MoE fence.
    hca_history_cmp_rows = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fwd_hca_history_cmp_rows"):
        fwd_base = pl.read(segment_starts_t, [0])
        pl.write(
            hca_history_cmp_rows, [0],
            pl.cast(pl.min(fwd_base // HCA_COMPRESS_RATIO, HCA_MAX_COMPRESSED_ROWS), pl.INT32),
        )
    layer_output = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], dtype=pl.FP32)
    moe_completion = pl.create_tensor([1, 1, 8], dtype=pl.FP32)
    x_attn = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], dtype=pl.FP32)
    # Carry the stage dependency beyond the attention scope, as in DSpark HCA.
    attention_stage_deps = pl.array.create(1, pl.TASK_ID)
    layer_hidden = x_hc
    for layer_id in pl.range(FWD_NUM_LAYERS):
        layer_index: pl.Scalar[pl.INT32] = pl.cast(layer_id, pl.INT32)
        kv_cache_layer = pl.slice(kv_cache, [raw_blocks, BLOCK_SIZE, 1, HEAD_DIM], [layer_index * raw_blocks, 0, 0, 0])
        hc_attn_fn_layer: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [layer_index * MIX_HC, 0]
        )
        hc_attn_scale_layer: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [layer_index * 3])
        hc_attn_base_layer: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [layer_index * MIX_HC])
        attn_norm_w_layer: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [layer_index * D])
        wq_a_layer: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [layer_index * D, 0])
        wq_b_layer: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [layer_index * Q_LORA, 0]
        )
        wq_b_scale_layer: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [layer_index * H * HEAD_DIM]
        )
        wkv_layer: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [layer_index * D, 0])
        gamma_cq_layer: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [layer_index * Q_LORA])
        gamma_ckv_layer: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [layer_index * HEAD_DIM])
        attn_sink_layer: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [layer_index * H])
        wo_a_layer: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
            wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [layer_index * O_GROUPS, 0, 0]
        )
        wo_b_layer: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(
            wo_b, [D, O_GROUPS * O_LORA], [layer_index * D, 0]
        )
        wo_b_scale_layer: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [layer_index * D])
        hc_ffn_fn_layer: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [layer_index * MIX_HC, 0]
        )
        hc_ffn_scale_layer: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [layer_index * 3])
        hc_ffn_base_layer: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [layer_index * MIX_HC])
        norm_w_layer: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [layer_index * D])
        gate_w_layer: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(
            gate_w, [N_EXPERTS_GLOBAL, D], [layer_index * N_EXPERTS_GLOBAL, 0]
        )
        gate_bias_layer: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(
            gate_bias, [N_EXPERTS_GLOBAL], [layer_index * N_EXPERTS_GLOBAL]
        )
        tid2eid_layer: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [layer_index * VOCAB, 0])
        routed_w1_layer: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w1, [N_LOCAL, MOE_INTER, D], [layer_index * N_LOCAL, 0, 0]
        )
        routed_w1_scale_layer: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
            routed_w1_scale, [N_LOCAL, MOE_INTER], [layer_index * N_LOCAL, 0]
        )
        routed_w3_layer: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w3, [N_LOCAL, MOE_INTER, D], [layer_index * N_LOCAL, 0, 0]
        )
        routed_w3_scale_layer: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
            routed_w3_scale, [N_LOCAL, MOE_INTER], [layer_index * N_LOCAL, 0]
        )
        routed_w2_layer: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(
            routed_w2, [N_LOCAL, D, MOE_INTER], [layer_index * N_LOCAL, 0, 0]
        )
        routed_w2_scale_layer: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(
            routed_w2_scale, [N_LOCAL, D], [layer_index * N_LOCAL, 0]
        )
        shared_w1_layer: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w1, [MOE_INTER, D], [layer_index * MOE_INTER, 0]
        )
        shared_w1_scale_layer: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w1_scale, [MOE_INTER], [layer_index * MOE_INTER]
        )
        shared_w3_layer: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w3, [MOE_INTER, D], [layer_index * MOE_INTER, 0]
        )
        shared_w3_scale_layer: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w3_scale, [MOE_INTER], [layer_index * MOE_INTER]
        )
        shared_w2_layer: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [layer_index * D, 0])
        shared_w2_scale_layer: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [layer_index * D])
        tail_comm_epoch = pl.cast(layer_index, pl.INT32)
        if pl.read(segment_starts_t, [0]) > 0:
            tail_comm_epoch = pl.cast(layer_index + pl.min(layer_index, 2), pl.INT32)
        with pl.scope():
            attention_completion = pl.create_tensor([NUM_ATTN_TILES, 1, 8], dtype=pl.FP32)
            if layer_index < 2:
                prefill_attention_swa(
                    layer_hidden,
                    hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
                    attn_norm_w_layer,
                    wq_a_layer, wq_b_layer, wq_b_scale_layer,
                    wkv_layer,
                    gamma_cq_layer, gamma_ckv_layer,
                    swa_freqs_cos,
                    swa_freqs_sin,
                    kv_cache_layer, csa_history_slots,
                    attn_sink_layer,
                    wo_a_layer, wo_b_layer, wo_b_scale_layer,
                    segment_starts_t, segment_tail_positions,
                    predecessor_segments,
                    query_position_ids, query_token_to_request,
                    overlay_position_ids, overlay_token_to_request, overlay_active_lengths,
                    swa_indices,
                    reverse_index,
                    owner_rank_table,
                    final_win_seg_src, final_win_row_src,
                    final_slot_mapping,
                    hidden_tail_window,
                    tail_ready, tail_consumed,
                    x_attn,
                    attention_completion,
                    pl.read(cache_owner_rank_t, [0]),
                    my_rank,
                    tail_comm_epoch,
                )
            elif layer_index % 2 == 0:
                type_index = (layer_index - 2) // 2
                cmp_kv_csa = pl.slice(
                    csa_cmp_kv,
                    [csa_cmp_blocks, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                    [type_index * csa_cmp_blocks, 0, 0, 0],
                )
                csa_cmp_wkv_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
                    csa_cmp_wkv,
                    [CSA_MAIN_OUT_DIM, D],
                    [type_index * CSA_MAIN_OUT_DIM, 0],
                )
                csa_cmp_wgate_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
                    csa_cmp_wgate,
                    [CSA_MAIN_OUT_DIM, D],
                    [type_index * CSA_MAIN_OUT_DIM, 0],
                )
                csa_cmp_ape_csa: pl.Tensor[
                    [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32
                ] = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [type_index * CSA_COMPRESS_RATIO, 0])
                csa_cmp_norm_w_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
                    csa_cmp_norm_w, [HEAD_DIM], [type_index * HEAD_DIM]
                )
                hadamard_idx_csa: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16] = (
                    pl.slice(csa_hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [type_index * IDX_HEAD_DIM, 0])
                )
                idx_wq_b_csa: pl.Tensor[
                    [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8
                ] = pl.slice(csa_idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [type_index * Q_LORA, 0])
                idx_wq_b_scale_csa: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32] = (
                    pl.slice(csa_idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [type_index * IDX_N_HEADS * IDX_HEAD_DIM])
                )
                idx_weights_proj_csa: pl.Tensor[[D, IDX_N_HEADS], pl.BF16] = pl.slice(
                    csa_weights_proj, [D, IDX_N_HEADS], [type_index * D, 0]
                )
                csa_inner_wkv_csa: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = (
                    pl.slice(csa_inner_wkv, [INNER_OUT_DIM, D], [type_index * INNER_OUT_DIM, 0])
                )
                csa_inner_wgate_csa: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = (
                    pl.slice(csa_inner_wgate, [INNER_OUT_DIM, D], [type_index * INNER_OUT_DIM, 0])
                )
                csa_inner_ape_csa: pl.Tensor[
                    [CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32
                ] = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [type_index * CSA_COMPRESS_RATIO, 0])
                csa_inner_norm_w_csa: pl.Tensor[[IDX_HEAD_DIM], pl.BF16] = pl.slice(
                    csa_inner_norm_w, [IDX_HEAD_DIM], [type_index * IDX_HEAD_DIM]
                )
                csa_compress_state_csa = pl.slice(
                    csa_compress_state,
                    [csa_state_blocks, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
                    [type_index * csa_state_blocks, 0, 0],
                )
                csa_inner_compress_state_csa = pl.slice(
                    csa_inner_compress_state,
                    [
                        csa_inner_state_blocks,
                        INNER_STATE_BLOCK_SIZE,
                        CSA_INNER_COMPRESS_STATE_DIM,
                    ],
                    [type_index * csa_inner_state_blocks, 0, 0],
                )
                idx_kv_cache_csa = pl.slice(
                    idx_kv_cache,
                    [csa_idx_blocks, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
                    [type_index * csa_idx_blocks, 0, 0, 0],
                )
                idx_kv_scale_csa = pl.slice(
                    idx_kv_scale,
                    [csa_idx_blocks, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1],
                    [type_index * csa_idx_blocks, 0, 0, 0],
                )
                prefill_attention_csa(
                    layer_hidden,
                    hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
                    attn_norm_w_layer,
                    wq_a_layer, wq_b_layer, wq_b_scale_layer,
                    wkv_layer,
                    gamma_cq_layer, gamma_ckv_layer,
                    compressed_freqs_cos,
                    compressed_freqs_sin,
                    csa_cmp_wkv_csa, csa_cmp_wgate_csa, csa_cmp_ape_csa, csa_cmp_norm_w_csa,
                    hadamard_idx_csa,
                    idx_wq_b_csa, idx_wq_b_scale_csa, idx_weights_proj_csa,
                    csa_inner_wkv_csa, csa_inner_wgate_csa, csa_inner_ape_csa, csa_inner_norm_w_csa,
                    main_state_workspace0,
                    inner_state_workspace0,
                    main_state_workspace1,
                    inner_state_workspace1,
                    csa_compress_state_csa,
                    csa_compress_state_block_table,
                    csa_inner_compress_state_csa, csa_inner_compress_state_block_table,
                    kv_cache_layer,
                    cmp_kv_csa,
                    csa_cmp_block_table,
                    idx_kv_cache_csa, idx_kv_scale_csa, idx_block_table,
                    segment_starts_t, segment_lengths_t, segment_active_lengths,
                    owner_segments_t,
                    predecessor_segments,
                    query_position_ids, query_token_to_request,
                    overlay_active_lengths,
                    swa_indices,
                    csa_history_slots,
                    final_segment_t,
                    reverse_index,
                    owner_rank_table,
                    final_win_seg_src, final_win_row_src,
                    final_slot_mapping,
                    leaf_positions_input,
                    leaf_main_slots_input,
                    leaf_idx_slots_input,
                    leaf_main_state_slots_input,
                    leaf_inner_state_slots_input,
                    leaf_num_tokens_input,
                    effective_x_workspace,
                    hidden_tail_window,
                    tail_ready, tail_consumed,
                    main_window,
                    idx_window,
                    scale_window,
                    record_window,
                    main_state_window, main_state_meta_window,
                    inner_state_window, inner_state_meta_window,
                    csa_compact_ready,
                    csa_compact_consumed,
                    attn_sink_layer,
                    wo_a_layer, wo_b_layer, wo_b_scale_layer,
                    x_attn,
                    attention_completion,
                    pl.read(cache_owner_rank_t, [0]),
                    my_rank,
                    tail_comm_epoch,
                    type_index,
                )
            else:
                type_index = (layer_index - 3) // 2
                cmp_kv_hca = pl.slice(
                    hca_cmp_kv,
                    [hca_cmp_blocks, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                    [type_index * hca_cmp_blocks, 0, 0, 0],
                )
                hca_cmp_wkv_hca: pl.Tensor[[HEAD_DIM, D], pl.BF16] = pl.slice(
                    hca_cmp_wkv, [HEAD_DIM, D], [type_index * HEAD_DIM, 0]
                )
                hca_cmp_wgate_hca: pl.Tensor[[HEAD_DIM, D], pl.BF16] = pl.slice(
                    hca_cmp_wgate, [HEAD_DIM, D], [type_index * HEAD_DIM, 0]
                )
                hca_cmp_ape_hca: pl.Tensor[[HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32] = (
                    pl.slice(hca_cmp_ape, [HCA_COMPRESS_RATIO, HEAD_DIM], [type_index * HCA_COMPRESS_RATIO, 0])
                )
                hca_cmp_norm_w_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
                    hca_cmp_norm_w, [HEAD_DIM], [type_index * HEAD_DIM]
                )
                hca_compress_state_hca = pl.slice(
                    hca_compress_state,
                    [hca_state_blocks, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
                    [type_index * hca_state_blocks, 0, 0],
                )
                prefill_attention_hca(
                    layer_hidden,
                    hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
                    attn_norm_w_layer,
                    wq_a_layer, wq_b_layer, wq_b_scale_layer,
                    wkv_layer,
                    gamma_cq_layer, gamma_ckv_layer,
                    compressed_freqs_cos,
                    compressed_freqs_sin,
                    hca_cmp_wkv_hca, hca_cmp_wgate_hca, hca_cmp_ape_hca, hca_cmp_norm_w_hca,
                    hca_compress_state_hca,
                    hca_compress_state_block_table,
                    kv_cache_layer,
                    cmp_kv_hca,
                    hca_cmp_block_table,
                    segment_starts_t, segment_active_lengths,
                    owner_segments_t,
                    predecessor_segments,
                    hca_history_slots,
                    hca_history_positions,
                    hca_history_cmp_rows,
                    query_position_ids,
                    overlay_active_lengths,
                    segment_tail_positions,
                    snapshot_positions,
                    snapshot_valid,
                    final_segment_t,
                    reverse_index,
                    owner_rank_table,
                    owner_part_table,
                    final_win_seg_src, final_win_row_src,
                    final_slot_mapping,
                    hidden_tail_window,
                    tail_ready, tail_consumed,
                    cmp_window, cmp_meta_window,
                    state_window,
                    state_meta_window,
                    hca_compact_ready,
                    hca_compact_consumed,
                    attn_sink_layer,
                    wo_a_layer, wo_b_layer, wo_b_scale_layer,
                    x_attn,
                    pl.read(cache_owner_rank_t, [0]),
                    my_rank,
                    tail_comm_epoch,
                    type_index,
                )
            if layer_index < 2:
                attention_tid = _fwd_attention_stage_barrier_from_completion(attention_completion)
                attention_stage_deps[0] = attention_tid
            elif layer_index % 2 == 0:
                attention_tid = _fwd_attention_stage_barrier_from_completion(attention_completion)
                attention_stage_deps[0] = attention_tid
            else:
                attention_tid = _fwd_attention_stage_barrier_from_x_attn(x_attn)
                attention_stage_deps[0] = attention_tid
        attention_done = attention_stage_deps[0]
        if layer_index > 0:
            moe_ready = _fwd_wait_previous_moe(attention_done, moe_completion)
        else:
            moe_ready = attention_done
        with pl.scope():
            _fwd_moe_tail(
                x_attn,
                overlay_active_lengths,
                input_ids,
                hc_ffn_fn_layer, hc_ffn_scale_layer, hc_ffn_base_layer,
                norm_w_layer,
                gate_w_layer,
                gate_bias_layer,
                tid2eid_layer,
                routed_w1_layer, routed_w1_scale_layer,
                routed_w3_layer, routed_w3_scale_layer,
                routed_w2_layer, routed_w2_scale_layer,
                shared_w1_layer, shared_w1_scale_layer,
                shared_w3_layer, shared_w3_scale_layer,
                shared_w2_layer, shared_w2_scale_layer,
                moe_x_mixed,
                moe_post_ffn,
                moe_comb_ffn,
                moe_ffn_out,
                moe_dense_scale,
                moe_returned_y,
                count_target,
                count_signal,
                prefill_moe_x_target,
                prefill_moe_x_signal,
                prefill_moe_scale_target,
                prefill_moe_reverse_target,
                prefill_moe_reverse_signal,
                layer_output,
                moe_completion,
                moe_ready,
                layer_index,
            )
        layer_hidden = layer_output

    active_flat = pl.reshape(overlay_active_lengths, [NUM_ATTN_TILES, OVERLAY_SOURCES])
    pre_hc_hidden_out_flat = pl.reshape(pre_hc_hidden_out, [CP_LOCAL_ROWS, HC_MULT, D])
    publish_src_flat = pl.reshape(layer_output, [CP_LOCAL_ROWS, HC_MULT, D])
    publish_anchor = moe_completion

    # Retire communication credits, publish hidden states and apply HC/RMSNorm.
    with pl.scope():
        # Serving retains these windows without a host reset. Layer epochs
        # restart in each request, so retire all attention credits after the
        # final MoE and before the next HOST dispatch can reuse the windows.
        tail_completed = pl.cast(FWD_NUM_LAYERS, pl.INT32)
        hca_completed = pl.cast(HCA_NUM_LAYERS, pl.INT32)
        csa_completed = pl.cast(CSA_NUM_LAYERS, pl.INT32)
        retire_prefix = pl.read(segment_starts_t, [0])
        if retire_prefix > 0:
            # Each SWA adds one raw-history phase. Each compressed layer adds
            # one raw/state phase plus enough windows for the full history.
            tail_completed = pl.cast(FWD_NUM_LAYERS + 2, pl.INT32)
            hca_completed = pl.cast(
                HCA_NUM_LAYERS * (2 + (retire_prefix // HCA_COMPRESS_RATIO + CMP_WINDOW_ROWS - 1) // CMP_WINDOW_ROWS),
                pl.INT32,
            )
            csa_completed = pl.cast(
                CSA_NUM_LAYERS * (2 + (retire_prefix // CSA_COMPRESS_RATIO + RECORDS_PER_WINDOW - 1) // RECORDS_PER_WINDOW),
                pl.INT32,
            )
        _clear_prefill_cp_exchange_signals(
            publish_anchor, tail_ready, tail_consumed,
            tail_completed, my_rank,
        )
        _clear_prefill_cp_exchange_signals(
            publish_anchor, hca_compact_ready, hca_compact_consumed,
            hca_completed, my_rank,
        )
        _clear_prefill_cp_exchange_signals(
            publish_anchor, csa_compact_ready, csa_compact_consumed,
            csa_completed, my_rank,
        )
        clear_prefill_moe_signals(publish_anchor, count_signal, prefill_moe_x_signal, prefill_moe_reverse_signal)

        tile_blocks = (ATTN_TILE_ROWS // PRE_HC_COPY_TOKEN_TILE) * HC_MULT
        with pl.spmd(NUM_ATTN_TILES * tile_blocks, name_hint="publish_pre_hc_hidden_out"):
            block = pl.tile.get_block_idx()
            tile = block // tile_blocks
            tile_block = block % tile_blocks
            token_block = tile_block // HC_MULT
            hc_lane = tile_block % HC_MULT
            token0 = token_block * PRE_HC_COPY_TOKEN_TILE
            active = pl.read(active_flat, [tile, 1])
            for dt in pl.range(PRE_HC_COPY_TOKEN_TILE):
                token = token0 + dt
                row = tile * ATTN_TILE_ROWS + token
                if token < active:
                    pre_hc_hidden_out_flat[
                        row : row + 1,
                        hc_lane : hc_lane + 1,
                        0:D,
                    ] = pl.slice(publish_src_flat, [1, 1, D], [row, hc_lane, 0])
                else:
                    pre_hc_hidden_out_flat[
                        row : row + 1,
                        hc_lane : hc_lane + 1,
                        0:D,
                    ] = pl.full([1, 1, D], dtype=pl.FP32, value=0.0)

        # HC head + final RMSNorm: collapse the [HC_MULT, D] hyper-connection
        # mix to one [D] row and normalize into the BF16 hidden_out. The
        # pre_hc_hidden_out slab is MOE_ROWS == LOCAL_ROWS, so the
        # hc_head's T_DYN extent binds to LOCAL_ROWS. The intermediate
        # hidden_head is the hc_head BF16 output and the rms_norm input.
        pre_hc_view = pl.reshape(pre_hc_hidden_out_flat, [CP_LOCAL_ROWS, HC_MULT, D])
        hidden_head = pl.create_tensor([CP_LOCAL_ROWS, D], dtype=pl.BF16)
        with pl.scope():
            hc_head(pre_hc_view, hc_head_fn, hc_head_scale, hc_head_base, hidden_head)
            rms_norm(hidden_head, final_norm_w, hidden_out)
    return hidden_out


# CP needs larger packed root, attention, and sparse-reader workspaces.
PREFILL_RING_HEAP = tuple(
    gib * 1024**3
    for gib in ((2, 2, 2 if CP_SIZE == 8 else 1, 2) if CP_SIZE > 1 else (1, 1, 1, 1))
)

from prefill_cp_exchange import (
    CP_REQUEST_CAPACITY as CP_EXCHANGE_CP_REQUEST_CAPACITY,
    CP_REQUEST_HC_DIM as CP_EXCHANGE_CP_REQUEST_HC_DIM,
    CP_REQUEST_INNER_TABLE_COLS as CP_EXCHANGE_CP_REQUEST_INNER_TABLE_COLS,
    CP_REQUEST_MAIN_TABLE_COLS as CP_EXCHANGE_CP_REQUEST_MAIN_TABLE_COLS,
    CP_REQUEST_TABLE_COLS as CP_EXCHANGE_CP_REQUEST_TABLE_COLS,
    CP_SIZE as CP_EXCHANGE_CP_SIZE,
    D as CP_EXCHANGE_D,
    HCA_STATE_MAX_BLOCKS as CP_EXCHANGE_HCA_STATE_MAX_BLOCKS,
    LOCAL_ROWS as CP_EXCHANGE_LOCAL_ROWS,
    NUM_SEGMENTS as CP_EXCHANGE_NUM_SEGMENTS,
    PREFILL_CMP_MAX_BLOCKS as CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_MAX_BLOCKS as CP_EXCHANGE_PREFILL_ORI_MAX_BLOCKS,
    TAIL_ROWS as CP_EXCHANGE_TAIL_ROWS,
)
from prefill_cp_exchange import (
    _prefill_cp_request_header as prepare_cp_request,
    _prefill_cp_scatter_request as scatter_cp_request,
    _prefill_cp_gather_hidden as gather_cp_hidden,
)


@pl.jit(auto_scope=False)
def _prefill_request(
    x_hc: pl.Tensor[[FWD_TOKENS_DYN, HC_MULT, D], pl.FP32],
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
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    x_out: pl.Out[pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16]],
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
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    cp_hidden_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16],
    cp_tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_cmp_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    cp_cmp_meta_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32],
    cp_state_window: pld.DistributedTensor[[STATE_WINDOW_ROWS, HCA_COMPRESS_STATE_DIM], pl.FP32],
    cp_state_meta_window: pld.DistributedTensor[[CP_SIZE, STATE_META_DIM], pl.INT32],
    cp_hca_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_hca_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_main_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, CSA_MAIN_OUT_DIM], pl.BF16],
    cp_idx_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, IDX_HEAD_DIM], pl.INT8],
    cp_scale_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP16],
    cp_record_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, META_DIM], pl.INT32],
    cp_main_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, CSA_COMPRESS_STATE_DIM], pl.FP32],
    cp_main_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32],
    cp_inner_state_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32],
    cp_inner_state_meta_window: pld.DistributedTensor[[STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32],
    cp_csa_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_csa_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    cp_count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    cp_prefill_moe_x_target: pld.DistributedTensor[[CP_MOE_TOTAL_CAP, D], pl.INT8],
    cp_prefill_moe_x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    cp_prefill_moe_scale_target: pld.DistributedTensor[[CP_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    cp_prefill_moe_reverse_target: pld.DistributedTensor[[CP_MOE_TOTAL_CAP, D], pl.BF16],
    cp_prefill_moe_reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    entry_header_window: pld.DistributedTensor[[1, 16], pl.INT32],
    entry_input_window: pld.DistributedTensor[[CP_EXCHANGE_LOCAL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], pl.FP32],
    entry_ids_window: pld.DistributedTensor[[1, CP_EXCHANGE_LOCAL_ROWS * 2], pl.INT32],
    entry_tables_window: pld.DistributedTensor[[7, CP_EXCHANGE_CP_REQUEST_TABLE_COLS], pl.INT32],
    entry_ready: pld.DistributedTensor[[CP_EXCHANGE_CP_SIZE, 1], pl.INT32],
    entry_hidden_window: pld.DistributedTensor[[CP_EXCHANGE_CP_REQUEST_CAPACITY, CP_EXCHANGE_D], pl.BF16],
    entry_pre_hc_tail_window: pld.DistributedTensor[[CP_EXCHANGE_NUM_SEGMENTS * CP_EXCHANGE_TAIL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], pl.FP32],
    entry_complete: pld.DistributedTensor[[CP_EXCHANGE_CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    request_rows = pl.tensor.dim(x_hc, 0)
    header = pl.create_tensor([1, 16], dtype=pl.INT32)
    control = pl.create_tensor([1, 16], dtype=pl.INT32)
    input_x_flat = pl.reshape(x_hc, [request_rows, HC_DIM])
    input_ids_row = pl.reshape(input_ids, [1, request_rows])
    ori_block_table_row = pl.reshape(ori_block_table, [1, CP_EXCHANGE_PREFILL_ORI_MAX_BLOCKS])
    hca_cmp_block_table_row = pl.reshape(hca_cmp_block_table, [1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
    csa_cmp_block_table_row = pl.reshape(csa_cmp_block_table, [1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
    idx_block_table_row = pl.reshape(idx_block_table, [1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
    hca_compress_state_block_table_row = pl.reshape(hca_compress_state_block_table, [1, CP_EXCHANGE_HCA_STATE_MAX_BLOCKS])
    csa_compress_state_block_table_row = pl.reshape(csa_compress_state_block_table, [1, CP_EXCHANGE_CP_REQUEST_MAIN_TABLE_COLS])
    csa_inner_compress_state_block_table_row = pl.reshape(csa_inner_compress_state_block_table, [1, CP_EXCHANGE_CP_REQUEST_INNER_TABLE_COLS])
    local_x_hc = pl.create_tensor([CP_EXCHANGE_LOCAL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
    local_input_ids = pl.create_tensor([1, CP_EXCHANGE_LOCAL_ROWS], dtype=pl.INT64)
    local_ori_block_table = pl.create_tensor([1, CP_EXCHANGE_PREFILL_ORI_MAX_BLOCKS], dtype=pl.INT32)
    local_hca_cmp_block_table = pl.create_tensor([1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
    local_csa_cmp_block_table = pl.create_tensor([1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
    local_idx_block_table = pl.create_tensor([1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
    local_hca_compress_state_block_table = pl.create_tensor([1, CP_EXCHANGE_HCA_STATE_MAX_BLOCKS], dtype=pl.INT32)
    local_csa_compress_state_block_table = pl.create_tensor([1, CP_EXCHANGE_CP_REQUEST_MAIN_TABLE_COLS], dtype=pl.INT32)
    local_csa_inner_compress_state_block_table = pl.create_tensor([1, CP_EXCHANGE_CP_REQUEST_INNER_TABLE_COLS], dtype=pl.INT32)
    header = prepare_cp_request(num_tokens_per_owner, position_ids, header, my_rank)
    (control, local_x_hc, local_input_ids, local_ori_block_table, local_hca_cmp_block_table, local_csa_cmp_block_table, local_idx_block_table, local_hca_compress_state_block_table, local_csa_compress_state_block_table, local_csa_inner_compress_state_block_table) = scatter_cp_request(
        header,
        input_x_flat,
        input_ids_row,
        ori_block_table_row,
        hca_cmp_block_table_row,
        csa_cmp_block_table_row,
        idx_block_table_row,
        hca_compress_state_block_table_row,
        csa_compress_state_block_table_row,
        csa_inner_compress_state_block_table_row,
        entry_header_window,
        entry_input_window,
        entry_ids_window,
        entry_tables_window,
        entry_ready,
        control,
        local_x_hc,
        local_input_ids,
        local_ori_block_table,
        local_hca_cmp_block_table,
        local_csa_cmp_block_table,
        local_idx_block_table,
        local_hca_compress_state_block_table,
        local_csa_compress_state_block_table,
        local_csa_inner_compress_state_block_table,
        my_rank,
    )
    mode = pl.read(control, [0, 0])
    if mode == 1:
        cp_ori_block_table = pl.reshape(local_ori_block_table, [CP_EXCHANGE_PREFILL_ORI_MAX_BLOCKS])
        cp_hca_cmp_block_table = pl.reshape(local_hca_cmp_block_table, [CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
        cp_csa_cmp_block_table = pl.reshape(local_csa_cmp_block_table, [CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
        cp_idx_block_table = pl.reshape(local_idx_block_table, [CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
        cp_hca_compress_state_block_table = pl.reshape(local_hca_compress_state_block_table, [CP_EXCHANGE_HCA_STATE_MAX_BLOCKS])
        cp_csa_compress_state_block_table = pl.reshape(local_csa_compress_state_block_table, [CP_EXCHANGE_CP_REQUEST_MAIN_TABLE_COLS])
        cp_csa_inner_compress_state_block_table = pl.reshape(local_csa_inner_compress_state_block_table, [CP_EXCHANGE_CP_REQUEST_INNER_TABLE_COLS])
        segment_starts_t = pl.create_tensor([NUM_SEGMENTS], dtype=pl.INT32)
        predecessor_segments = pl.create_tensor([LOCAL_PARTS], dtype=pl.INT32)
        query_position_ids = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], dtype=pl.INT32)
        query_token_to_request = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], dtype=pl.INT32)
        overlay_position_ids = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], dtype=pl.INT32)
        overlay_token_to_request = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], dtype=pl.INT32)
        overlay_active_lengths = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], dtype=pl.INT32)
        swa_indices = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], dtype=pl.INT32)
        reverse_index = pl.create_tensor([NUM_SEGMENTS], dtype=pl.INT32)
        owner_rank_table = pl.create_tensor([NUM_SEGMENTS], dtype=pl.INT32)
        final_win_seg_src = pl.create_tensor([TAIL_ROWS], dtype=pl.INT32)
        final_win_row_src = pl.create_tensor([TAIL_ROWS], dtype=pl.INT32)
        final_slot_mapping = pl.create_tensor([TAIL_ROWS], dtype=pl.INT32)
        segment_active_lengths = pl.create_tensor([LOCAL_PARTS], dtype=pl.INT32)
        cache_owner_rank_t = pl.create_tensor([1], dtype=pl.INT32)
        owner_segments_t = pl.create_tensor([LOCAL_PARTS], dtype=pl.INT32)
        final_segment_t = pl.create_tensor([1], dtype=pl.INT32)
        segment_tail_positions = pl.create_tensor([NUM_SEGMENTS, TAIL_ROWS], dtype=pl.INT32)
        snapshot_positions = pl.create_tensor([LOCAL_PARTS, TAIL_ROWS], dtype=pl.INT32)
        snapshot_valid = pl.create_tensor([LOCAL_PARTS], dtype=pl.INT32)
        owner_part_table = pl.create_tensor([NUM_SEGMENTS], dtype=pl.INT32)
        segment_lengths_t = pl.create_tensor([NUM_SEGMENTS], dtype=pl.INT32)
        leaf_positions_input = pl.create_tensor([LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], dtype=pl.INT32)
        leaf_main_slots_input = pl.create_tensor([LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], dtype=pl.INT64)
        leaf_idx_slots_input = pl.create_tensor([LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], dtype=pl.INT64)
        leaf_main_state_slots_input = pl.create_tensor([LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], dtype=pl.INT64)
        leaf_inner_state_slots_input = pl.create_tensor([LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], dtype=pl.INT64)
        leaf_num_tokens_input = pl.create_tensor([LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES], dtype=pl.INT32)
        hca_history_slots = pl.create_tensor([LOCAL_PARTS, TAIL_ROWS], dtype=pl.INT32)
        hca_history_positions = pl.create_tensor([LOCAL_PARTS, TAIL_ROWS], dtype=pl.INT32)
        csa_history_slots = pl.create_tensor([TAIL_ROWS], dtype=pl.INT32)
        _prefill_cp_metadata(
            control,
            cp_ori_block_table, cp_csa_compress_state_block_table, cp_csa_inner_compress_state_block_table,
            cp_idx_block_table,
            segment_starts_t,
            predecessor_segments,
            query_position_ids, query_token_to_request,
            overlay_position_ids, overlay_token_to_request, overlay_active_lengths,
            swa_indices,
            reverse_index,
            owner_rank_table,
            final_win_seg_src, final_win_row_src,
            final_slot_mapping,
            segment_active_lengths,
            cache_owner_rank_t,
            owner_segments_t,
            final_segment_t,
            segment_tail_positions,
            snapshot_positions,
            snapshot_valid,
            owner_part_table,
            segment_lengths_t,
            leaf_positions_input,
            leaf_main_slots_input,
            leaf_idx_slots_input,
            leaf_main_state_slots_input,
            leaf_inner_state_slots_input,
            leaf_num_tokens_input,
            hca_history_slots,
            hca_history_positions,
            csa_history_slots,
            my_rank,
        )
        cp_x_hc = pl.reshape(local_x_hc, [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D])
        cp_input_ids = pl.reshape(local_input_ids, [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS])
        cp_pre_hc_hidden_out = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], dtype=pl.FP32)
        cp_hidden_out = pl.create_tensor([CP_LOCAL_ROWS, D], dtype=pl.BF16)
        cp_hidden_out = prefill_fwd(
            cp_x_hc,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w,
            wq_a, wq_b, wq_b_scale,
            wkv,
            gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin,
            kv_cache,
            hca_cmp_kv,
            csa_cmp_kv,
            attn_sink,
            wo_a, wo_b, wo_b_scale,
            segment_starts_t,
            predecessor_segments,
            query_position_ids, query_token_to_request,
            overlay_position_ids, overlay_token_to_request, overlay_active_lengths,
            swa_indices,
            reverse_index,
            owner_rank_table,
            final_win_seg_src, final_win_row_src,
            final_slot_mapping,
            segment_active_lengths,
            cache_owner_rank_t,
            owner_segments_t,
            final_segment_t,
            hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
            hca_compress_state,
            cp_hca_compress_state_block_table,
            segment_tail_positions,
            snapshot_positions,
            snapshot_valid,
            owner_part_table,
            csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
            csa_hadamard_idx,
            csa_idx_wq_b,
            csa_idx_wq_b_scale,
            csa_weights_proj,
            csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
            csa_compress_state,
            csa_inner_compress_state,
            idx_kv_cache, idx_kv_scale,
            cp_csa_compress_state_block_table, cp_csa_inner_compress_state_block_table, cp_idx_block_table,
            segment_lengths_t,
            leaf_positions_input,
            leaf_main_slots_input,
            leaf_idx_slots_input,
            leaf_main_state_slots_input,
            leaf_inner_state_slots_input,
            leaf_num_tokens_input,
            hca_history_slots, hca_history_positions, csa_history_slots,
            cp_hca_cmp_block_table, cp_csa_cmp_block_table, cp_hidden_tail_window,
            cp_tail_ready, cp_tail_consumed, cp_cmp_window, cp_cmp_meta_window, cp_state_window,
            cp_state_meta_window, cp_hca_compact_ready, cp_hca_compact_consumed, cp_main_window, cp_idx_window,
            cp_scale_window, cp_record_window, cp_main_state_window, cp_main_state_meta_window,
            cp_inner_state_window, cp_inner_state_meta_window, cp_csa_compact_ready, cp_csa_compact_consumed,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w,
            gate_w,
            gate_bias,
            tid2eid,
            cp_input_ids,
            routed_w1, routed_w1_scale,
            routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale,
            shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            cp_count_target, cp_count_signal, cp_prefill_moe_x_target, cp_prefill_moe_x_signal,
            cp_prefill_moe_scale_target, cp_prefill_moe_reverse_target, cp_prefill_moe_reverse_signal,
            hc_head_fn, hc_head_scale, hc_head_base,
            final_norm_w,
            cp_pre_hc_hidden_out, cp_hidden_out,
            my_rank,
        )
        cp_pre_hc_flat = pl.reshape(cp_pre_hc_hidden_out, [CP_LOCAL_ROWS, HC_DIM])
        output_pre_hc_flat = pl.reshape(pre_hc_hidden_out, [T, HC_DIM])
        x_out, output_pre_hc_flat = gather_cp_hidden(
            control,
            cp_hidden_out, cp_pre_hc_flat,
            entry_hidden_window,
            entry_pre_hc_tail_window,
            entry_complete,
            x_out,
            output_pre_hc_flat,
            my_rank,
        )
    else:
        # Unsupported request metadata must not leave allocator residue as output.
        # The caller contract requires a cold single-owner request within capacity.
        for row in pl.spmd(request_rows, name_hint="prefill_unsupported_request"):
            invalid_bits = pl.tile.full([1, D], value=0x7FC0, dtype=pl.INT16)
            invalid = pl.tile.reinterpret_view(invalid_bits, pl.BF16)
            pl.store(invalid, [row, 0], x_out)
        for row in pl.spmd(T, name_hint="prefill_unsupported_tail"):
            invalid_tail_bits = pl.tile.full([1, 1, D], value=0x7FC00000, dtype=pl.INT32)
            invalid_tail = pl.tile.reinterpret_view(invalid_tail_bits, pl.FP32)
            for stream in pl.range(HC_MULT):
                pl.store(invalid_tail, [row, stream, 0], pre_hc_hidden_out)
    return x_out


@pl.jit.host
def l3_prefill_fwd(
    x_hc: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN, HC_MULT, D], pl.FP32],
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
    csa_weights_proj: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[N_RANKS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[N_RANKS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
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
    pre_hc_hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, FWD_TOKENS_DYN, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
):

    pl.static_assert(CP_SIZE == N_RANKS and CP_SIZE > 1, "Prefill requires CP=EP>1; CP1 is deferred.")
    x_hc.bind_dynamic(1, FWD_TOKENS_DYN)
    hidden_out.bind_dynamic(1, FWD_TOKENS_DYN)
    ori_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    position_ids.bind_dynamic(1, FWD_TOKENS_DYN)
    input_ids.bind_dynamic(1, FWD_TOKENS_DYN)
    hca_cmp_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    hca_state_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    csa_cmp_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    csa_idx_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    csa_state_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    csa_inner_state_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)

    # The LM head owns every window and counter it touches: a peer routes into
    # logits_window while still reading its own hidden_window, and the barrier
    # counters stay independent of the MoE epoch protocol.
    lm_head_hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    lm_head_logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * LM_HEAD_VOCAB * 4)
    lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)

    cp_hidden_tail_window_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    cp_tail_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    cp_tail_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    cp_cmp_window_buf = pld.alloc_window_buffer([CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    cp_cmp_meta_window_buf = pld.alloc_window_buffer([CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32)
    cp_state_window_buf = pld.alloc_window_buffer([STATE_WINDOW_ROWS, HCA_COMPRESS_STATE_DIM], dtype=pl.FP32)
    cp_state_meta_window_buf = pld.alloc_window_buffer([CP_SIZE, STATE_META_DIM], dtype=pl.INT32)
    cp_hca_compact_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    cp_hca_compact_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    cp_main_window_buf = pld.alloc_window_buffer([RECORDS_PER_WINDOW, CSA_MAIN_OUT_DIM], dtype=pl.BF16)
    cp_idx_window_buf = pld.alloc_window_buffer([RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8)
    cp_scale_window_buf = pld.alloc_window_buffer([RECORDS_PER_WINDOW, SCALE_TILE_COLS], dtype=pl.FP16)
    cp_record_window_buf = pld.alloc_window_buffer([RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32)
    cp_main_state_window_buf = pld.alloc_window_buffer([STATE_RECORDS_PER_WINDOW, CSA_COMPRESS_STATE_DIM], dtype=pl.FP32)
    cp_main_state_meta_window_buf = pld.alloc_window_buffer([STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32)
    cp_inner_state_window_buf = pld.alloc_window_buffer([STATE_RECORDS_PER_WINDOW, CSA_INNER_COMPRESS_STATE_DIM], dtype=pl.FP32)
    cp_inner_state_meta_window_buf = pld.alloc_window_buffer([STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32)
    cp_csa_compact_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    cp_csa_compact_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    cp_count_target_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    cp_count_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    cp_prefill_moe_x_target_buf = pld.alloc_window_buffer([CP_MOE_TOTAL_CAP, D], dtype=pl.INT8)
    cp_prefill_moe_x_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    cp_prefill_moe_scale_target_buf = pld.alloc_window_buffer([CP_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], dtype=pl.FP32)
    cp_prefill_moe_reverse_target_buf = pld.alloc_window_buffer([CP_MOE_TOTAL_CAP, D], dtype=pl.BF16)
    cp_prefill_moe_reverse_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    entry_header_window_buf = pld.alloc_window_buffer([1, 16], dtype=pl.INT32)
    entry_input_window_buf = pld.alloc_window_buffer([CP_EXCHANGE_LOCAL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
    entry_ids_window_buf = pld.alloc_window_buffer([1, CP_EXCHANGE_LOCAL_ROWS * 2], dtype=pl.INT32)
    entry_tables_window_buf = pld.alloc_window_buffer([7, CP_EXCHANGE_CP_REQUEST_TABLE_COLS], dtype=pl.INT32)
    entry_ready_buf = pld.alloc_window_buffer([CP_EXCHANGE_CP_SIZE, 1], dtype=pl.INT32)
    entry_hidden_window_buf = pld.alloc_window_buffer([CP_EXCHANGE_CP_REQUEST_CAPACITY, CP_EXCHANGE_D], dtype=pl.BF16)
    entry_pre_hc_tail_window_buf = pld.alloc_window_buffer([CP_EXCHANGE_NUM_SEGMENTS * CP_EXCHANGE_TAIL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
    entry_complete_buf = pld.alloc_window_buffer([CP_EXCHANGE_CP_SIZE, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        cp_hidden_tail_window = pld.window(cp_hidden_tail_window_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
        cp_tail_ready = pld.window(cp_tail_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        cp_tail_consumed = pld.window(cp_tail_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        cp_cmp_window = pld.window(cp_cmp_window_buf, [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
        cp_cmp_meta_window = pld.window(cp_cmp_meta_window_buf, [CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32)
        cp_state_window = pld.window(cp_state_window_buf, [STATE_WINDOW_ROWS, HCA_COMPRESS_STATE_DIM], dtype=pl.FP32)
        cp_state_meta_window = pld.window(cp_state_meta_window_buf, [CP_SIZE, STATE_META_DIM], dtype=pl.INT32)
        cp_hca_compact_ready = pld.window(cp_hca_compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        cp_hca_compact_consumed = pld.window(cp_hca_compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        cp_main_window = pld.window(cp_main_window_buf, [RECORDS_PER_WINDOW, CSA_MAIN_OUT_DIM], dtype=pl.BF16)
        cp_idx_window = pld.window(cp_idx_window_buf, [RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8)
        cp_scale_window = pld.window(cp_scale_window_buf, [RECORDS_PER_WINDOW, SCALE_TILE_COLS], dtype=pl.FP16)
        cp_record_window = pld.window(cp_record_window_buf, [RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32)
        cp_main_state_window = pld.window(cp_main_state_window_buf, [STATE_RECORDS_PER_WINDOW, CSA_COMPRESS_STATE_DIM], dtype=pl.FP32)
        cp_main_state_meta_window = pld.window(cp_main_state_meta_window_buf, [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32)
        cp_inner_state_window = pld.window(cp_inner_state_window_buf, [STATE_RECORDS_PER_WINDOW, CSA_INNER_COMPRESS_STATE_DIM], dtype=pl.FP32)
        cp_inner_state_meta_window = pld.window(cp_inner_state_meta_window_buf, [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32)
        cp_csa_compact_ready = pld.window(cp_csa_compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        cp_csa_compact_consumed = pld.window(cp_csa_compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        cp_count_target = pld.window(cp_count_target_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        cp_count_signal = pld.window(cp_count_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        cp_prefill_moe_x_target = pld.window(cp_prefill_moe_x_target_buf, [CP_MOE_TOTAL_CAP, D], dtype=pl.INT8)
        cp_prefill_moe_x_signal = pld.window(cp_prefill_moe_x_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        cp_prefill_moe_scale_target = pld.window(cp_prefill_moe_scale_target_buf, [CP_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], dtype=pl.FP32)
        cp_prefill_moe_reverse_target = pld.window(cp_prefill_moe_reverse_target_buf, [CP_MOE_TOTAL_CAP, D], dtype=pl.BF16)
        cp_prefill_moe_reverse_signal = pld.window(cp_prefill_moe_reverse_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        entry_header_window = pld.window(entry_header_window_buf, [1, 16], dtype=pl.INT32)
        entry_input_window = pld.window(entry_input_window_buf, [CP_EXCHANGE_LOCAL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
        entry_ids_window = pld.window(entry_ids_window_buf, [1, CP_EXCHANGE_LOCAL_ROWS * 2], dtype=pl.INT32)
        entry_tables_window = pld.window(entry_tables_window_buf, [7, CP_EXCHANGE_CP_REQUEST_TABLE_COLS], dtype=pl.INT32)
        entry_ready = pld.window(entry_ready_buf, [CP_EXCHANGE_CP_SIZE, 1], dtype=pl.INT32)
        entry_hidden_window = pld.window(entry_hidden_window_buf, [CP_EXCHANGE_CP_REQUEST_CAPACITY, CP_EXCHANGE_D], dtype=pl.BF16)
        entry_pre_hc_tail_window = pld.window(entry_pre_hc_tail_window_buf, [CP_EXCHANGE_NUM_SEGMENTS * CP_EXCHANGE_TAIL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
        entry_complete = pld.window(entry_complete_buf, [CP_EXCHANGE_CP_SIZE, 1], dtype=pl.INT32)
        x_hc_rank = x_hc[r]
        hidden_rank = hidden_out[r]
        position_ids_rank = position_ids[r]
        input_ids_rank = input_ids[r]
        _prefill_request(
            x_hc_rank,
            hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r],
            attn_norm_w[r],
            wq_a[r], wq_b[r], wq_b_scale[r],
            wkv[r],
            gamma_cq[r], gamma_ckv[r],
            kv_cache[r],
            attn_sink[r],
            wo_a[r], wo_b[r], wo_b_scale[r],
            hca_cmp_kv[r],
            csa_cmp_kv[r],
            hca_cmp_wkv[r], hca_cmp_wgate[r], hca_cmp_ape[r], hca_cmp_norm_w[r],
            hca_compress_state[r],
            csa_cmp_wkv[r], csa_cmp_wgate[r], csa_cmp_ape[r], csa_cmp_norm_w[r],
            csa_compress_state[r],
            csa_hadamard_idx[r],
            csa_idx_wq_b[r],
            csa_idx_wq_b_scale[r],
            csa_weights_proj[r],
            csa_inner_wkv[r], csa_inner_wgate[r], csa_inner_ape[r], csa_inner_norm_w[r],
            csa_inner_compress_state[r],
            idx_kv_cache[r], idx_kv_scale[r],
            hca_compress_state_block_table[r],
            csa_compress_state_block_table[r],
            csa_inner_compress_state_block_table[r],
            freqs_cos[r], freqs_sin[r],
            ori_block_table[r],
            hca_cmp_block_table[r],
            csa_cmp_block_table[r],
            idx_block_table[r],
            position_ids_rank,
            input_ids_rank,
            hc_head_fn[r], hc_head_scale[r], hc_head_base[r],
            final_norm_w[r],
            pre_hc_hidden_out[r],
            hidden_rank,
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r],
            gate_w[r],
            gate_bias[r],
            tid2eid[r],
            routed_w1[r], routed_w1_scale[r],
            routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r],
            shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            num_tokens_per_owner,
            cp_hidden_tail_window, cp_tail_ready, cp_tail_consumed, cp_cmp_window,
            cp_cmp_meta_window, cp_state_window, cp_state_meta_window, cp_hca_compact_ready,
            cp_hca_compact_consumed, cp_main_window, cp_idx_window, cp_scale_window, cp_record_window,
            cp_main_state_window, cp_main_state_meta_window, cp_inner_state_window, cp_inner_state_meta_window,
            cp_csa_compact_ready, cp_csa_compact_consumed, cp_count_target, cp_count_signal,
            cp_prefill_moe_x_target, cp_prefill_moe_x_signal, cp_prefill_moe_scale_target,
            cp_prefill_moe_reverse_target, cp_prefill_moe_reverse_signal,
            entry_header_window,
            entry_input_window,
            entry_ids_window,
            entry_tables_window,
            entry_ready,
            entry_hidden_window,
            entry_pre_hc_tail_window,
            entry_complete,
            r,
            device=r,
        )

    # Grouped LM head: the N_RANKS DP world is cut into N_RANKS // LM_HEAD_TP_SIZE
    # groups. Every card is both an owner and a TP rank, so the single lm_head
    # dispatch runs on the full world and every peer stays inside its own group.
    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(lm_head_logits_window_buf, [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
        logits_done = pld.window(lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        lm_head_test(
            hidden_out[r], lm_head_weight[r], logit_row_indices[r], logits[r],
            hidden_window, hidden_done, logits_window, logits_done,
            r // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE, r % LM_HEAD_TP_SIZE,
            LM_HEAD_COMM_EPOCH, device=r,
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


def _make_stacked_spec(name, base_specs, cache_block_nums=None):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    count = _layer_count(name)
    packed_shape = [spec.shape[0], count * spec.shape[1], *spec.shape[2:]]

    def init_value():
        if cache_block_nums and name in cache_block_nums:
            return torch.zeros(packed_shape, dtype=spec.dtype)
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
    return TensorSpec(name, packed_shape, spec.dtype, init_value=init_value)


def _make_shared_spec(name, base_specs, start_pos, cache_block_nums=None):
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
            return ranked(pos.to(torch.int64))
        if name in ("hca_state_slot_mapping", "csa_state_slot_mapping", "csa_inner_state_slot_mapping"):
            # State mappings carry the physical row, not the logical position.
            block_size, max_blocks, state_name = {
                "hca_state_slot_mapping": (
                    HCA_STATE_BLOCK_SIZE, HCA_STATE_MAX_BLOCKS, "hca_compress_state"),
                "csa_state_slot_mapping": (
                    CSA_STATE_BLOCK_SIZE, CSA_STATE_MAX_BLOCKS, "csa_compress_state"),
                "csa_inner_state_slot_mapping": (
                    INNER_STATE_BLOCK_SIZE, INNER_STATE_MAX_BLOCKS, "csa_inner_compress_state"),
            }[name]
            physical_blocks = cache_block_nums[state_name]
            block = pos // block_size
            row = (block % physical_blocks) * block_size + pos % block_size
            addressable = (pos >= 0) & (pos < MAX_SEQ_LEN) & (block < max_blocks)
            return ranked(torch.where(addressable, row, torch.full_like(row, -1)))
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
        if name in (
            "ori_block_table",
            "hca_cmp_block_table",
            "csa_cmp_block_table",
            "idx_block_table",
        ):
            physical_pages = {
                "ori_block_table": cache_block_nums["kv_cache"],
                "hca_cmp_block_table": cache_block_nums["hca_cmp_kv"],
                "csa_cmp_block_table": cache_block_nums["csa_cmp_kv"],
                "idx_block_table": cache_block_nums["idx_kv_cache"],
            }[name]
            out = torch.arange(spec.shape[-1], dtype=spec.dtype) % physical_pages
            return ranked(out)
        if name in ("hca_compress_state_block_table", "csa_compress_state_block_table",
                    "csa_inner_compress_state_block_table"):
            state_name = {
                "hca_compress_state_block_table": "hca_compress_state",
                "csa_compress_state_block_table": "csa_compress_state",
                "csa_inner_compress_state_block_table": "csa_inner_compress_state",
            }[name]
            out = torch.arange(spec.shape[-1], dtype=spec.dtype) % cache_block_nums[state_name]
            return ranked(out)
        # Any remaining shared metadata: smoke zeros.
        return torch.zeros(list(spec.shape), dtype=spec.dtype)

    return TensorSpec(name, list(spec.shape), spec.dtype, init_value=init_value)


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


def _build_forward_base_specs(num_tokens):
    """Compose one layer of fixture shapes and weights from the maintained stages."""
    import torch
    from golden import TensorSpec
    from prefill_compressor_ratio128 import build_tensor_specs as hca_specs
    from prefill_compressor_ratio4 import build_tensor_specs as csa_specs
    from prefill_indexer import build_tensor_specs as indexer_specs

    single = {
        s.name: s for s in build_swa_attention_tensor_specs(start_pos=0, num_tokens=num_tokens)
        if isinstance(s, TensorSpec) and s.name != "x_out"
    }
    single["ori_block_table"] = single.pop("block_table")
    for prefix, build in (("hca", hca_specs), ("csa", csa_specs)):
        source = {s.name: s for s in build(start_pos=0) if isinstance(s, TensorSpec)}
        for name in ("wkv", "wgate", "ape", "norm_w"):
            single[f"{prefix}_cmp_{name}"] = source[name]
        for name in ("compress_state", "compress_state_block_table", "cmp_kv",
                     "cmp_slot_mapping", "state_slot_mapping"):
            single[f"{prefix}_{name}"] = source[name]
        single[f"{prefix}_cmp_block_table"] = TensorSpec(
            f"{prefix}_cmp_block_table", [SPARSE_CMP_MAX_BLOCKS], torch.int32,
        )

    indexer = {s.name: s for s in indexer_specs(start_pos=0, num_tokens=num_tokens) if isinstance(s, TensorSpec)}
    for name, target in (("hadamard", "csa_hadamard_idx"), ("wq_b", "csa_idx_wq_b"),
                         ("wq_b_scale", "csa_idx_wq_b_scale"), ("weights_proj", "csa_weights_proj"),
                         ("idx_slot_mapping", "csa_idx_slot_mapping")):
        single[target] = indexer[name]
    for name in ("inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
                 "inner_compress_state", "inner_compress_state_block_table", "inner_state_slot_mapping"):
        single[f"csa_{name}"] = indexer[name]
    for name in ("idx_kv_cache", "idx_kv_scale", "idx_block_table"):
        single[name] = indexer[name]
    for name in ("freqs_cos", "freqs_sin"):
        raw = single[name]
        compressed = indexer[name]

        def init_rope(raw=raw, compressed=compressed):
            return torch.stack((raw.create_tensor(), compressed.create_tensor()), dim=0)

        single[name] = TensorSpec(name, [2, *raw.shape], raw.dtype, init_value=init_rope)

    # The full-forward builder owns request metadata and cache pool geometry.
    # Stage-local write mappings above supply only shapes/dtypes; their initializers
    # are replaced by _make_shared_spec and _make_stacked_spec below.
    specs = {
        name: TensorSpec(name, [N_RANKS, *src.shape], src.dtype, init_value=_ranked_init(src, N_RANKS, torch))
        for name, src in single.items()
    }
    specs.update({s.name: s for s in build_moe_tensor_specs(layer_id=0, token_capacity=T)
                  if isinstance(s, TensorSpec) and s.name not in {"x_hc", "x_next"}})
    return specs


def build_tensor_specs(
    start_pos=0,
    num_tokens=T,
    num_tiles=1,
    ori_block_num=CSA_ORI_BLOCK_NUM,
    cmp_block_num=CSA_CMP_BLOCK_NUM,
    idx_block_num=PREFILL_IDX_BLOCK_NUM,
    hca_state_block_num=HCA_STATE_BLOCK_NUM,
    csa_state_block_num=CSA_STATE_BLOCK_NUM,
    inner_state_block_num=INNER_STATE_BLOCK_NUM,
    active_ranks=1,
):
    import torch
    from golden import TensorSpec

    if CP_SIZE != N_RANKS or CP_SIZE <= 1:
        raise ValueError("Prefill requires CP=EP>1; CP1 is deferred.")
    if start_pos < 0:
        raise ValueError(f"Prefill start position must be non-negative, got {start_pos}.")
    if start_pos % BLOCK_SIZE:
        raise ValueError(
            f"Prefill continuation starts on a cache page boundary, got {start_pos} "
            f"which is not a multiple of {BLOCK_SIZE}."
        )
    if active_ranks != 1:
        raise ValueError("Prefill requires one active request owner.")
    if not 1 <= num_tokens <= CP_EXCHANGE_CP_REQUEST_CAPACITY:
        raise ValueError(f"Prefill token count must be in [1, {CP_EXCHANGE_CP_REQUEST_CAPACITY}].")
    if num_tiles < 1 or num_tiles * T < num_tokens:
        raise ValueError("Prefill input storage must cover all active tokens.")

    def init_lm_head_weight():
        shards = (torch.randn(LM_HEAD_TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)
        return torch.stack([shards[r % LM_HEAD_TP_SIZE] for r in range(N_RANKS)], dim=0)

    # One owner carries the request; peers carry zero tokens and -1 logit indices.
    def init_logit_row_indices():
        indices = torch.full((N_RANKS, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:active_ranks, 0] = max(min(num_tokens, num_tiles * T), 1) - 1
        return indices

    def init_num_tokens_per_owner():
        counts = torch.zeros(N_RANKS, dtype=torch.int32)
        counts[:active_ranks] = num_tokens
        return counts

    first_tile_tokens = max(1, min(num_tokens, T))
    base_specs = _build_forward_base_specs(first_tile_tokens)

    ordered_names = [
        "x_hc",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
        "hca_cmp_kv", "csa_cmp_kv",
        "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
        "hca_compress_state",
        "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
        "csa_compress_state",
        "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
        "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
        "csa_inner_compress_state", "idx_kv_cache", "idx_kv_scale",
        "hca_compress_state_block_table", "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
        "freqs_cos", "freqs_sin",
        "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
        "ori_slot_mapping", "position_ids", "input_ids",
        "hca_cmp_slot_mapping", "hca_state_slot_mapping",
        "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
        "csa_state_slot_mapping", "csa_inner_state_slot_mapping",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
        "hc_head_fn", "hc_head_scale", "hc_head_base",
        "final_norm_w",
    ]

    cache_block_nums = {
        "kv_cache": ori_block_num,
        "hca_cmp_kv": cmp_block_num,
        "csa_cmp_kv": cmp_block_num,
        "idx_kv_cache": idx_block_num,
        "idx_kv_scale": idx_block_num,
        "hca_compress_state": hca_state_block_num,
        "csa_compress_state": csa_state_block_num,
        "csa_inner_compress_state": inner_state_block_num,
    }
    TILED_NAMES = {
        "ori_slot_mapping", "position_ids", "input_ids",
        "hca_cmp_slot_mapping", "hca_state_slot_mapping",
        "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
        "csa_state_slot_mapping", "csa_inner_state_slot_mapping",
    }

    def make_tiled_shared_spec(name):
        per_tile = [
            _make_shared_spec(name, base_specs, start_pos + tile * T, cache_block_nums)
            for tile in range(num_tiles)
        ]
        head = per_tile[0]
        if num_tiles == 1:
            return head
        shape = list(head.shape)
        shape[1] = num_tiles * T

        def init_joined(parts=per_tile):
            return torch.cat([p.init_value() if callable(p.init_value) else p.init_value for p in parts], dim=1)

        return TensorSpec(name, shape, head.dtype, init_value=init_joined)

    specs = []
    for name in ordered_names:
        if name == "x_hc":
            base = base_specs[name]
            x_hc_shape = list(base.shape)
            x_hc_shape[1] = num_tiles * T

            def init_x_hc(shape=x_hc_shape, dtype=base.dtype):
                return (torch.randn(shape) * 0.05).to(dtype)

            specs.append(TensorSpec(name, x_hc_shape, base.dtype, init_value=init_x_hc))
        elif name in TILED_NAMES:
            specs.append(make_tiled_shared_spec(name))
        elif name in SHARED_NAMES:
            specs.append(_make_shared_spec(name, base_specs, start_pos, cache_block_nums))
        elif name in HC_HEAD_NAMES:
            specs.append(_make_hc_head_spec(name))
        elif name in FINAL_NORM_NAMES:
            specs.append(_make_final_norm_spec(name))
        else:
            specs.append(_make_stacked_spec(name, base_specs, cache_block_nums))

    # Shard the static weight parameters per rank and keep them device-resident
    # (child_memory): each shard uploaded once to its card and reused across
    # dispatches, skipping per-dispatch H2D/D2H. RESIDENT_WEIGHT_NAMES are static
    # weights; RESIDENT_CACHE_NAMES are the KV/state caches (the written kv_cache
    # is also an InOut, read back at the end via RESIDENT_CACHE_OUTPUT_NAMES).
    for spec in specs:
        if spec.name in RESIDENT_WEIGHT_NAMES or spec.name in RESIDENT_CACHE_NAMES:
            spec.resident = "stacked"

    specs.append(TensorSpec("pre_hc_hidden_out", [N_RANKS, T, HC_MULT, D], torch.float32))
    specs.append(TensorSpec(
        "lm_head_weight",
        [N_RANKS, VOCAB_PER_TP, D],
        torch.bfloat16,
        init_value=init_lm_head_weight,
        resident="stacked",
    ))
    specs.append(TensorSpec("hidden_out", [N_RANKS, num_tiles * T, D], torch.bfloat16))
    specs.append(TensorSpec("logits", [N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], torch.float32))
    specs.append(TensorSpec("num_tokens_per_owner", [N_RANKS], torch.int32, init_value=init_num_tokens_per_owner))
    specs.append(TensorSpec(
        "logit_row_indices",
        [N_RANKS, MAX_LOGIT_ROWS],
        torch.int32,
        init_value=init_logit_row_indices,
    ))
    return specs


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Flash packed-prefill forward driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe).")
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=[N_RANKS],
                        help="Context parallel group size: the full EP world.")
    parser.add_argument("--tp", type=int, default=LM_HEAD_TP_SIZE, choices=[2, 4, 8, 16],
                        help="LM-head TP world size; must be <= --ep.")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--active-ranks", type=int, default=1,
                        help="Ranks carrying tokens; the rest stay idle as in single-request serving.")
    parser.add_argument("--num-tokens", type=int, default=T // 2,
                        help=f"Active token rows for MoE routing/combine; default is T // 2={T // 2}.")
    parser.add_argument("--ori-block-num", type=int, default=CSA_ORI_BLOCK_NUM)
    parser.add_argument("--cmp-block-num", type=int, default=CSA_CMP_BLOCK_NUM)
    parser.add_argument("--idx-block-num", type=int, default=PREFILL_IDX_BLOCK_NUM)
    parser.add_argument("--hca-state-block-num", type=int, default=HCA_STATE_BLOCK_NUM)
    parser.add_argument("--csa-state-block-num", type=int, default=CSA_STATE_BLOCK_NUM)
    parser.add_argument("--inner-state-block-num", type=int, default=INNER_STATE_BLOCK_NUM)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)

    parser.add_argument("--num-tiles", type=int, default=1,
                        help="fixed-T tiles in one submitted request; 1 reproduces the packed graph.")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"
    assert args.tp <= args.ep, f"--tp must be <= --ep, got tp={args.tp}, ep={args.ep}"
    assert args.ep % args.tp == 0, (
        f"grouped LM head needs --ep % --tp == 0, got ep={args.ep}, tp={args.tp}"
    )
    assert LM_HEAD_TP_SIZE == args.tp, (
        f"import-time LM_HEAD_TP_SIZE must match --tp, got {LM_HEAD_TP_SIZE} vs {args.tp}"
    )
    assert N_RANKS == args.ep, f"import-time N_RANKS must match --ep, got {N_RANKS} vs {args.ep}"

    specs = build_tensor_specs(
        start_pos=args.start_pos,
        num_tokens=args.num_tokens,
        ori_block_num=args.ori_block_num,
        cmp_block_num=args.cmp_block_num,
        idx_block_num=args.idx_block_num,
        hca_state_block_num=args.hca_state_block_num,
        csa_state_block_num=args.csa_state_block_num,
        inner_state_block_num=args.inner_state_block_num,
        active_ranks=args.active_ranks,
        num_tiles=args.num_tiles,
    )

    result = run(
        fn=l3_prefill_fwd,
        specs=specs,
        golden_fn=None,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=False,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_scope_stats=args.enable_scope_stats,
            ring_heap=PREFILL_RING_HEAP,
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
