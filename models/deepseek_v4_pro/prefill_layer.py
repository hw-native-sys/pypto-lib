# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2 # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 packed (request-aware) chunked prefill single layer with MoE EP2.

This is the Qwen-style packed-prefill variant of ``prefill_layer.py``. The packed
``prefill_layer_core`` expands the batch and sequence dimensions internally: it
loops over requests and over fixed-``T`` token tiles, builds tile-local ``[T, ...]``
inputs from the packed buffers, and directly calls the existing fixed-``T`` child
kernels (``prefill_attention_{swa,hca,csa}`` and ``moe``) per tile, scattering valid
rows back into the packed output.

Coordinate system (shared by JIT and golden, see issue #591):
  * packed token buffers : global packed row (``chunk_offsets[r] + tile_id*T + t``)
  * cache/state pools    : global physical rows allocated through request metadata
  * block tables         : request-local views containing global physical block ids
  * sparse-index overlay : tile-local (``WIN + t`` for the current tile's tokens)
  * position_ids         : absolute position (``chunk_start + tile_id*T + t``)

All tile-local metadata (slot mappings, sparse indices, positions) is produced on
the host by reusing the child ``build_*_tensor_specs(start_pos=tile_ctx,
num_tokens=valid)`` builders, which already encode the absolute-position ring /
overlay / compressed / state formulas for a single ``[T]`` tile. The kernel just
gathers the precomputed packed metadata per tile.
"""

import itertools

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

# The prefill path routes PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe (which freezes recv shapes and derives RECV_MAX = EP * MOE_TOKENS at import).
import config
config.MOE_TOKENS = config.PREFILL_TOKENS
# Import moe first. It applies the EP2 PRO override before dependent
# modules bake config-derived MoE shapes.
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
    SIGNAL_PAD,
    T,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    golden_moe,
    moe,
)
from config import ACTIVE as MODEL_CONFIG
from prefill_attention_swa import (
    BLOCK_NUM as SWA_ORI_BLOCK_NUM,
    BLOCK_SIZE as SWA_BLOCK_SIZE,
    build_tensor_specs as build_swa_attention_tensor_specs,
    golden_prefill_attention_swa,
    prefill_attention_swa,
)
from prefill_attention_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    HCA_CMP_BLOCK_NUM,
    HCA_ORI_BLOCK_NUM,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    build_tensor_specs as build_hca_attention_tensor_specs,
    golden_prefill_attention_hca,
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
    build_tensor_specs as build_csa_attention_tensor_specs,
    golden_prefill_attention_csa,
    prefill_attention_csa,
)
assert SWA_BLOCK_SIZE == BLOCK_SIZE, "SWA/HCA/CSA must share the PyPTO block size"
assert SWA_ORI_BLOCK_NUM == HCA_ORI_BLOCK_NUM == CSA_ORI_BLOCK_NUM
assert HCA_CMP_BLOCK_NUM == CSA_CMP_BLOCK_NUM

# ``T`` is the fixed child-kernel token-tile capacity (Qwen's ``TOK_TILE``). It is
# NOT the packed token total. The packed prefill only ever feeds the children a
# fixed ``[T, ...]`` tile at a time.
# One prefill layer overflows the runtime's default 256 MiB-per-ring output
# heap and fails with `orch_error_code=2 HEAP_RING_DEADLOCK`. Size every ring:
# ring 2 alone, which prefill_fwd.py sets, does not clear it.
LAYER_RING_HEAP = 1024 * 1024 * 1024

TOK_TILE = T
PREFILL_CHUNK_TOKENS = T
DEFAULT_CHUNK_LENS = (T, T + T // 2)
DEFAULT_USER_BATCH = len(DEFAULT_CHUNK_LENS)

# Standalone fixture capacities used to size the self-contained global pools.
# Cache/state tensors keep the production global-pool ABI; request-local block
# tables carry allocator-assigned physical ids into those pools.
ORI_CACHE_BLOCKS = CSA_ORI_BLOCK_NUM
CMP_CACHE_BLOCKS = CSA_CMP_BLOCK_NUM
IDX_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM
ORI_TABLE_BLOCKS = SPARSE_ORI_MAX_BLOCKS
CMP_TABLE_BLOCKS = SPARSE_CMP_MAX_BLOCKS
IDX_TABLE_BLOCKS = IDX_CACHE_MAX_BLOCKS

# ---------------------------------------------------------------------------
# Layer schedule, derived from the active preset (never hard-coded).
#
# ``compress_ratios`` holds ``num_hidden_layers + 1`` entries; the trailing one
# describes the MTP layer, which ``prefill_mtp.py`` owns. The main-model
# schedule is a uniform prefix followed by a strict CSA(4)/HCA(128) alternation:
#   flash: 43 layers = 2 SWA (ratio 0) prefix + 21 CSA + 20 HCA
#   pro:   61 layers = 0 SWA, HCA (ratio 128) prefix + 30 CSA + 31 HCA
# ``prefill_layer_core`` dispatches on the runtime ``layer_id`` scalar, so it can
# only reproduce the schedule with integer arithmetic; these three constants are
# exactly what that arithmetic needs. ``_kernel_attention_kind`` below
# cross-checks them against ``compress_ratios`` at import.
# ---------------------------------------------------------------------------
_MAIN_RATIOS = tuple(MODEL_CONFIG.compress_ratios[: MODEL_CONFIG.num_hidden_layers])
NUM_SWA_LAYERS = sum(1 for r in _MAIN_RATIOS if r == 0)   # leading SWA run: flash 2, pro 0
FIRST_CSA_LAYER = _MAIN_RATIOS.index(4)                   # alternation start: 2 for both
CSA_PARITY = FIRST_CSA_LAYER % 2                          # CSA layers carry this parity
assert _MAIN_RATIOS[:NUM_SWA_LAYERS] == (0,) * NUM_SWA_LAYERS, \
    "SWA layers must form the leading prefix of compress_ratios"
assert all(r == 128 for r in _MAIN_RATIOS[NUM_SWA_LAYERS:FIRST_CSA_LAYER]), \
    "layers between the SWA prefix and the first CSA layer must be HCA (ratio 128)"
assert all(r == (4 if (i - FIRST_CSA_LAYER) % 2 == 0 else 128)
           for i, r in enumerate(_MAIN_RATIOS) if i >= FIRST_CSA_LAYER), \
    "compress_ratios must alternate CSA(4)/HCA(128) from FIRST_CSA_LAYER onwards"

# Dynamic (batch-dependent) kernel-signature dims. Global-pool symbols use the
# same names as the fixed-T children so a packed allocator capacity propagates
# through every inlined cache/state consumer (issue #591 §1).
USER_BATCH_DYN = pl.dynamic("DEEPSEEK_PREFILL_USER_BATCH_DYN")
PREFILL_TOKENS_DYN = pl.dynamic("DEEPSEEK_PREFILL_TOKENS_DYN")

PREFILL_ORI_CACHE_BLOCKS_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
PREFILL_CMP_CACHE_BLOCKS_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
PREFILL_IDX_CACHE_BLOCKS_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")

PREFILL_ORI_BLOCK_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_ORI_BLOCK_TABLE_DYN")
PREFILL_CMP_BLOCK_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_CMP_BLOCK_TABLE_DYN")
PREFILL_IDX_BLOCK_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_IDX_BLOCK_TABLE_DYN")

PREFILL_HCA_STATE_BLOCKS_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")
PREFILL_CSA_STATE_BLOCKS_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
PREFILL_INNER_STATE_BLOCKS_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")
PREFILL_HCA_STATE_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_HCA_STATE_TABLE_DYN")
PREFILL_CSA_STATE_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_CSA_STATE_TABLE_DYN")
PREFILL_INNER_STATE_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_INNER_STATE_TABLE_DYN")


@pl.jit
def prefill_layer_core(
    x_hc: pl.Tensor[[PREFILL_TOKENS_DYN, HC_MULT, D], pl.FP32],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_tile_offsets: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[
        [PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, 2 * HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_compress_state_block_table: pl.Tensor[[PREFILL_HCA_STATE_TABLE_DYN], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[[PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, 2 * CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[[PREFILL_CSA_STATE_TABLE_DYN], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[[PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, 2 * INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[PREFILL_INNER_STATE_TABLE_DYN], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[PREFILL_ORI_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[PREFILL_ORI_BLOCK_TABLE_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[PREFILL_CMP_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_BLOCK_TABLE_DYN], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[PREFILL_IDX_BLOCK_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
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
    x_next: pl.InOut[pl.Tensor[[PREFILL_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pl.InOut[pld.DistributedTensor[[N_RANKS, N_LOCAL, SIGNAL_PAD], pl.INT32]],
    consumed: pl.InOut[pld.DistributedTensor[[N_RANKS, SIGNAL_PAD], pl.INT32]],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_TOKENS_DYN, HC_MULT, D], pl.FP32]:
    x_hc.bind_dynamic(0, PREFILL_TOKENS_DYN)
    ori_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    position_ids.bind_dynamic(0, PREFILL_TOKENS_DYN)
    hca_cmp_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    hca_state_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    csa_cmp_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    csa_idx_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    csa_state_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    csa_inner_state_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    input_ids.bind_dynamic(0, PREFILL_TOKENS_DYN)
    x_next.bind_dynamic(0, PREFILL_TOKENS_DYN)
    seq_lens.bind_dynamic(0, USER_BATCH_DYN)
    chunk_lens.bind_dynamic(0, USER_BATCH_DYN)
    chunk_offsets.bind_dynamic(0, USER_BATCH_DYN)
    chunk_tile_offsets.bind_dynamic(0, USER_BATCH_DYN)
    kv_cache.bind_dynamic(0, PREFILL_ORI_CACHE_BLOCKS_DYN)
    ori_block_table.bind_dynamic(0, PREFILL_ORI_BLOCK_TABLE_DYN)
    cmp_kv.bind_dynamic(0, PREFILL_CMP_CACHE_BLOCKS_DYN)
    cmp_block_table.bind_dynamic(0, PREFILL_CMP_BLOCK_TABLE_DYN)
    idx_kv_cache.bind_dynamic(0, PREFILL_IDX_CACHE_BLOCKS_DYN)
    idx_kv_scale.bind_dynamic(0, PREFILL_IDX_CACHE_BLOCKS_DYN)
    idx_block_table.bind_dynamic(0, PREFILL_IDX_BLOCK_TABLE_DYN)
    hca_compress_state.bind_dynamic(0, PREFILL_HCA_STATE_BLOCKS_DYN)
    hca_compress_state_block_table.bind_dynamic(0, PREFILL_HCA_STATE_TABLE_DYN)
    csa_compress_state.bind_dynamic(0, PREFILL_CSA_STATE_BLOCKS_DYN)
    csa_compress_state_block_table.bind_dynamic(0, PREFILL_CSA_STATE_TABLE_DYN)
    csa_inner_compress_state.bind_dynamic(0, PREFILL_INNER_STATE_BLOCKS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(0, PREFILL_INNER_STATE_TABLE_DYN)
    user_batch = pl.tensor.dim(seq_lens, 0)
    for request_id in pl.range(user_batch):
        chunk_len_b = pl.tensor.read(chunk_lens, [request_id])
        chunk_base = pl.cast(pl.tensor.read(chunk_offsets, [request_id]), pl.INDEX)
        tile_ord_base = pl.cast(pl.tensor.read(chunk_tile_offsets, [request_id]), pl.INDEX)
        tok_blocks = (chunk_len_b + TOK_TILE - 1) // TOK_TILE
        ridx = pl.cast(request_id, pl.INDEX)

        # All requests address the same fixed-capacity global physical pools.
        # Request ownership is carried exclusively by the request-local block
        # tables and lowered slot mappings below.
        kv_cache_req = kv_cache
        ori_block_table_req = pl.slice(ori_block_table, [ORI_TABLE_BLOCKS], [ridx * ORI_TABLE_BLOCKS])
        cmp_kv_req = cmp_kv
        cmp_block_table_req = pl.slice(cmp_block_table, [CMP_TABLE_BLOCKS], [ridx * CMP_TABLE_BLOCKS])
        idx_kv_cache_req = idx_kv_cache
        idx_kv_scale_req = idx_kv_scale
        idx_block_table_req = pl.slice(idx_block_table, [IDX_TABLE_BLOCKS], [ridx * IDX_TABLE_BLOCKS])
        hca_compress_state_req = hca_compress_state
        hca_state_table_req = pl.slice(hca_compress_state_block_table, [HCA_STATE_MAX_BLOCKS],
                                       [ridx * HCA_STATE_MAX_BLOCKS])
        csa_compress_state_req = csa_compress_state
        csa_state_table_req = pl.slice(csa_compress_state_block_table, [CSA_STATE_MAX_BLOCKS],
                                       [ridx * CSA_STATE_MAX_BLOCKS])
        csa_inner_compress_state_req = csa_inner_compress_state
        csa_inner_state_table_req = pl.slice(csa_inner_compress_state_block_table, [INNER_STATE_MAX_BLOCKS],
                                             [ridx * INNER_STATE_MAX_BLOCKS])

        for tile_id in pl.range(tok_blocks):
            p0 = tile_id * TOK_TILE
            tile_base = chunk_base + p0
            valid_tok = pl.min(TOK_TILE, chunk_len_b - p0)
            valid_n = pl.cast(valid_tok, pl.INT32)  # child num_tokens scalar (INT32)
            # Global execution ordinal of this MoE call (1-based). The readiness
            # windows hold monotonic epochs, so each serial MoE call needs a
            # unique, gap-free epoch equal to its execution order; chunk_tile_offsets
            # is the exclusive prefix sum of tok_blocks over requests.
            moe_epoch = pl.cast(tile_ord_base + tile_id + 1, pl.INT32)

            # Tile-local fixed-[T] inputs gathered from the packed buffers. The
            # children only read the leading ``valid_tok`` rows, so the padded
            # tail (when valid_tok < T) is ignored. No explicit per-tile pl.scope:
            # rely on auto_scope + pl.range's sequential semantics so the
            # global cache/state RAW dependency is carried across tiles
            # (tile N's writeback ordered before tile N+1's gather).
            # Keep all child inputs at their fixed [T] capacity. In particular,
            # x_hc / position_ids feed T_DYN children whose sibling tensors are
            # full-[T], so narrowing them would bind T_DYN to two extents.
            # num_tokens gates every padded tail row in the child kernels.
            x_hc_tile = pl.slice(x_hc, [TOK_TILE, HC_MULT, D], [tile_base, 0, 0])
            ori_slot_tile = pl.slice(ori_slot_mapping, [TOK_TILE], [tile_base])
            position_ids_tile = pl.slice(position_ids, [TOK_TILE], [tile_base])
            hca_cmp_slot_tile = pl.slice(hca_cmp_slot_mapping, [TOK_TILE], [tile_base])
            hca_state_slot_tile = pl.slice(hca_state_slot_mapping, [TOK_TILE], [tile_base])
            csa_cmp_slot_tile = pl.slice(csa_cmp_slot_mapping, [TOK_TILE], [tile_base])
            csa_idx_slot_tile = pl.slice(csa_idx_slot_mapping, [TOK_TILE], [tile_base])
            csa_state_slot_tile = pl.slice(csa_state_slot_mapping, [TOK_TILE], [tile_base])
            csa_inner_state_slot_tile = pl.slice(csa_inner_state_slot_mapping, [TOK_TILE], [tile_base])
            input_ids_tile = pl.slice(input_ids, [TOK_TILE], [tile_base])

            x_attn_tile = pl.create_tensor([TOK_TILE, HC_MULT, D], dtype=pl.FP32)
            if layer_id < NUM_SWA_LAYERS:
                prefill_attention_swa(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    kv_cache_req, ori_block_table_req, ori_slot_tile,
                    position_ids_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )
            elif layer_id < FIRST_CSA_LAYER or layer_id % 2 != CSA_PARITY:
                prefill_attention_hca(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
                    hca_compress_state_req, hca_state_table_req,
                    kv_cache_req, ori_slot_tile, ori_block_table_req,
                    cmp_kv_req, cmp_block_table_req,
                    position_ids_tile, hca_cmp_slot_tile, hca_state_slot_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )
            else:
                prefill_attention_csa(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
                    csa_compress_state_req, csa_state_table_req,
                    csa_hadamard_idx,
                    csa_idx_wq_b, csa_idx_wq_b_scale, csa_weights_proj,
                    csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
                    csa_inner_compress_state_req, csa_inner_state_table_req,
                    kv_cache_req, ori_block_table_req, ori_slot_tile,
                    cmp_kv_req, cmp_block_table_req, idx_kv_cache_req, idx_kv_scale_req, idx_block_table_req,
                    position_ids_tile, csa_cmp_slot_tile, csa_idx_slot_tile,
                    csa_state_slot_tile, csa_inner_state_slot_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )

            x_next_tile = pl.create_tensor([TOK_TILE, HC_MULT, D], dtype=pl.FP32)
            moe(
                x_attn_tile,
                hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                norm_w, gate_w, gate_bias, tid2eid, input_ids_tile,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                x_next_tile,
                recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                routed_y_buf, combine_arrived, consumed,
                layer_id, valid_n, my_rank, moe_epoch,
            )

            # Scatter the tile back into the padded physical output. Each
            # request's physical span is rounded up to whole-T tiles, so a
            # full-T write is safe even for a partial logical tail tile.
            x_next = pl.assemble(x_next, x_next_tile, [tile_base, 0, 0])

    # Quiesce every rank's final reduction marker before resetting this rank's
    # inbound epoch slots for the next persistent dispatch. Once all consumed
    # markers are visible, no meta/payload/result publisher from this dispatch
    # can still write these local signal windows.
    last_req = user_batch - 1
    last_tiles = (pl.tensor.read(chunk_lens, [last_req]) + TOK_TILE - 1) // TOK_TILE
    total_epochs = pl.cast(pl.tensor.read(chunk_tile_offsets, [last_req]), pl.INDEX) \
        + pl.cast(last_tiles, pl.INDEX)
    final_epoch = pl.cast(total_epochs, pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_retire"):
        _x_next_anchor = pl.read(x_next, [0, 0, 0])
        for src in pl.range(N_RANKS):
            pld.system.wait(signal=consumed, offsets=[src, 0], expected=final_epoch, cmp=pld.WaitCmp.Ge)
        for src in pl.range(N_RANKS):
            pld.system.notify(target=arrived, peer=my_rank, offsets=[src, 0], value=0, op=pld.NotifyOp.Set)
            pld.system.notify(target=consumed, peer=my_rank, offsets=[src, 0], value=0, op=pld.NotifyOp.Set)
            for e in pl.range(N_LOCAL):
                pld.system.notify(
                    target=data_arrived, peer=my_rank, offsets=[src, e, 0],
                    value=0, op=pld.NotifyOp.Set,
                )
                pld.system.notify(
                    target=combine_arrived, peer=my_rank, offsets=[src, e, 0],
                    value=0, op=pld.NotifyOp.Set,
                )
    return x_next


@pl.jit.host
def l3_prefill_layer(
    x_hc: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, HC_MULT, D], pl.FP32],
    seq_lens: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    chunk_tile_offsets: pl.Tensor[[N_RANKS, USER_BATCH_DYN], pl.INT32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[
        [N_RANKS, PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, 2 * HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_HCA_STATE_TABLE_DYN], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, 2 * CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_TABLE_DYN], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, 2 * INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_TABLE_DYN], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_ORI_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, PREFILL_ORI_BLOCK_TABLE_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_CMP_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, PREFILL_CMP_BLOCK_TABLE_DYN], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[N_RANKS, PREFILL_IDX_BLOCK_TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, SIGNAL_PAD], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32)
    consumed_buf = pld.alloc_window_buffer([N_RANKS, SIGNAL_PAD], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, SIGNAL_PAD], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, N_LOCAL, SIGNAL_PAD], dtype=pl.INT32)
        consumed = pld.window(consumed_buf, [N_RANKS, SIGNAL_PAD], dtype=pl.INT32)
        prefill_layer_core(
            x_hc[rank],
            seq_lens[rank], chunk_lens[rank], chunk_offsets[rank], chunk_tile_offsets[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank], hca_cmp_norm_w[rank],
            hca_compress_state[rank], hca_compress_state_block_table[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank], csa_cmp_norm_w[rank],
            csa_compress_state[rank], csa_compress_state_block_table[rank],
            csa_hadamard_idx[rank],
            csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank], csa_weights_proj[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank], csa_inner_norm_w[rank],
            csa_inner_compress_state[rank],
            csa_inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping[rank],
            cmp_kv[rank], cmp_block_table[rank],
            idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank],
            position_ids[rank],
            hca_cmp_slot_mapping[rank], hca_state_slot_mapping[rank],
            csa_cmp_slot_mapping[rank], csa_idx_slot_mapping[rank],
            csa_state_slot_mapping[rank], csa_inner_state_slot_mapping[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_next[rank],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived, consumed,
            layer_id, rank,
            device=rank,
        )


HOST_TENSOR_ORDER = (
    "x_hc",
    "seq_lens",
    "chunk_lens",
    "chunk_offsets",
    "chunk_tile_offsets",
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
    "hca_compress_state",
    "hca_compress_state_block_table",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "csa_inner_compress_state_block_table",
    "kv_cache",
    "ori_block_table",
    "ori_slot_mapping",
    "cmp_kv",
    "cmp_block_table",
    "idx_kv_cache",
    "idx_kv_scale",
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


# ---------------------------------------------------------------------------
# Host-side packed metadata builder, tensor specs, and golden reference.
# ---------------------------------------------------------------------------

_KIND_BUILDER = {
    "swa": build_swa_attention_tensor_specs,
    "hca": build_hca_attention_tensor_specs,
    "csa": build_csa_attention_tensor_specs,
}

# Child-local token-metadata tensors (gathered per tile from the packed buffers).
_TOKEN_META_NAMES = {
    "position_ids", "ori_slot_mapping",
    "cmp_slot_mapping", "state_slot_mapping", "idx_slot_mapping", "inner_state_slot_mapping",
}
# Child cache/state pools plus request-local table views (persist across tiles).
_CACHE_STATE_NAMES = {
    "kv_cache", "block_table", "ori_block_table", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "compress_state", "compress_state_block_table",
    "inner_compress_state", "inner_compress_state_block_table",
}

# Global cache/state pools and packed per-request tables (packed-name ->
# child-local name, or source-kind mappings for namespaced state tensors).
_PACKED_CACHE_SPECS = {
    "kv_cache": "kv_cache",
    "ori_block_table": "ori_block_table",
    "cmp_kv": "cmp_kv",
    "cmp_block_table": "cmp_block_table",
    "idx_kv_cache": "idx_kv_cache",
    "idx_kv_scale": "idx_kv_scale",
    "idx_block_table": "idx_block_table",
    "hca_compress_state": ("hca", "compress_state"),
    "hca_compress_state_block_table": ("hca", "compress_state_block_table"),
    "csa_compress_state": ("csa", "compress_state"),
    "csa_compress_state_block_table": ("csa", "compress_state_block_table"),
    "csa_inner_compress_state": ("csa", "inner_compress_state"),
    "csa_inner_compress_state_block_table": ("csa", "inner_compress_state_block_table"),
}

_GLOBAL_POOL_NAMES = {
    "kv_cache", "cmp_kv", "idx_kv_cache", "idx_kv_scale",
    "hca_compress_state", "csa_compress_state", "csa_inner_compress_state",
}

# Standalone builders allocate one fixed-capacity physical pool. The packed
# fixture concatenates those capacities into one allocator-owned global pool;
# every request's table and direct slot mapping is rebased into its own range.
_GLOBAL_TABLE_BLOCK_STRIDES = {
    "ori_block_table": ORI_CACHE_BLOCKS,
    "cmp_block_table": CMP_CACHE_BLOCKS,
    "idx_block_table": IDX_CACHE_BLOCKS,
    "hca_compress_state_block_table": HCA_STATE_BLOCK_NUM,
    "csa_compress_state_block_table": CSA_STATE_BLOCK_NUM,
    "csa_inner_compress_state_block_table": INNER_STATE_BLOCK_NUM,
}


def _req_block_count(kind, child_name):
    """Per-request dim0 of a child-local cache/state/table tensor."""
    if child_name == "kv_cache":
        return ORI_CACHE_BLOCKS
    if child_name in ("block_table", "ori_block_table"):
        return ORI_TABLE_BLOCKS
    if child_name == "cmp_kv":
        return CMP_CACHE_BLOCKS
    if child_name == "cmp_block_table":
        return CMP_TABLE_BLOCKS
    if child_name in ("idx_kv_cache", "idx_kv_scale"):
        return IDX_CACHE_BLOCKS
    if child_name == "idx_block_table":
        return IDX_TABLE_BLOCKS
    if child_name == "compress_state":
        return HCA_STATE_BLOCK_NUM if kind == "hca" else CSA_STATE_BLOCK_NUM
    if child_name == "compress_state_block_table":
        return HCA_STATE_MAX_BLOCKS if kind == "hca" else CSA_STATE_MAX_BLOCKS
    if child_name == "inner_compress_state":
        return INNER_STATE_BLOCK_NUM
    if child_name == "inner_compress_state_block_table":
        return INNER_STATE_MAX_BLOCKS
    raise KeyError(child_name)


def _child_to_packed(kind, child_name):
    """Map a child-local cache/state name to its packed-buffer name for this kind."""
    if child_name in ("block_table", "ori_block_table"):
        return "ori_block_table"
    if child_name in ("kv_cache", "cmp_kv", "cmp_block_table", "idx_kv_cache", "idx_kv_scale", "idx_block_table"):
        return child_name
    prefix = "hca_" if kind == "hca" else "csa_"
    return prefix + child_name


def _spec_value(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def _attention_kind_for_layer(layer_id):
    if not 0 <= layer_id < MODEL_CONFIG.num_hidden_layers:
        raise ValueError(
            f"layer_id must be in [0, {MODEL_CONFIG.num_hidden_layers - 1}] for the "
            f"{MODEL_CONFIG.name} main model, got {layer_id}; the trailing MTP layer "
            f"({MODEL_CONFIG.num_hidden_layers}) is owned by prefill_mtp.py"
        )
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeek V4 attention compress ratio {ratio} at layer {layer_id}")


def _kernel_attention_kind(layer_id):
    """Host mirror of the integer dispatch inside ``prefill_layer_core``."""
    if layer_id < NUM_SWA_LAYERS:
        return "swa"
    if layer_id < FIRST_CSA_LAYER or layer_id % 2 != CSA_PARITY:
        return "hca"
    return "csa"


assert all(_kernel_attention_kind(i) == _attention_kind_for_layer(i)
           for i in range(MODEL_CONFIG.num_hidden_layers)), \
    "prefill_layer_core dispatch disagrees with MODEL_CONFIG.compress_ratios"


def _tile_token_meta(kind, context_len, valid_tok, torch, request_specs=None):
    """Child-local [T] token metadata for one tile, via the fixed-T child builder.

    Reuses the existing single-tile builders, which already encode the
    absolute-position paged-cache/state coordinate logic. ``context_len``
    is the tile's absolute start position; ``valid_tok`` its active token count.
    """
    from golden import TensorSpec

    if request_specs is None:
        specs = {
            s.name: s
            for s in _KIND_BUILDER[kind](start_pos=context_len, num_tokens=valid_tok)
            if isinstance(s, TensorSpec)
        }
    else:
        specs = request_specs(kind, context_len, valid_tok)
    meta = {name: _spec_value(specs[name], torch) for name in specs if name in _TOKEN_META_NAMES}
    return meta


def _iter_request_tiles(seq_lens_v, chunk_lens_v, chunk_offsets_v):
    """Yield (request_id, tile_id, context_len, valid_tok, physical_base) per tile."""
    for r in range(len(chunk_lens_v)):
        seq_len = int(seq_lens_v[r])
        chunk_len = int(chunk_lens_v[r])
        base = int(chunk_offsets_v[r])
        chunk_start = seq_len - chunk_len
        tok_blocks = (chunk_len + T - 1) // T
        for tile_id in range(tok_blocks):
            p0 = tile_id * T
            valid = min(T, chunk_len - p0)
            yield r, tile_id, chunk_start + p0, valid, base + p0


def _check_allocator_range(name, values, lower, upper):
    """Reject invalid sentinels or physical ids outside one request's allocation."""
    invalid_negative = values < -1
    if bool(invalid_negative.any()):
        bad = int(values[invalid_negative].min())
        raise ValueError(f"{name} contains invalid negative physical id {bad}; only -1 is a sentinel")
    valid = values >= 0
    if not bool(valid.any()):
        return
    valid_values = values[valid]
    actual_min = int(valid_values.min())
    actual_max = int(valid_values.max())
    if actual_min < lower or actual_max >= upper:
        raise ValueError(
            f"{name} physical ids [{actual_min}, {actual_max}] escape "
            f"request allocation [{lower}, {upper})"
        )


def _rebase_nonnegative(values, offset, *, extent, name):
    """Clone and shift valid physical ids into one request allocation."""
    rebased = values.clone()
    valid = rebased >= 0
    rebased[valid] += offset
    _check_allocator_range(name, rebased, offset, offset + extent)
    return rebased


def _packed_token_metadata(
    kind,
    seq_lens_v,
    chunk_lens_v,
    chunk_offsets_v,
    total_tokens,
    torch,
    request_specs=None,
):
    """Assemble rank-shared padded-physical [total_tokens, ...] metadata tensors."""
    pos = torch.zeros(total_tokens, dtype=torch.int32)
    ori_slot = torch.full((total_tokens,), -1, dtype=torch.int64)
    hca_cmp = torch.full((total_tokens,), -1, dtype=torch.int64)
    hca_state = torch.full((total_tokens,), -1, dtype=torch.int64)
    csa_cmp = torch.full((total_tokens,), -1, dtype=torch.int64)
    csa_idx = torch.full((total_tokens,), -1, dtype=torch.int64)
    csa_state = torch.full((total_tokens,), -1, dtype=torch.int64)
    csa_inner = torch.full((total_tokens,), -1, dtype=torch.int64)

    for request_id, _tid, ctx, valid, base in _iter_request_tiles(
        seq_lens_v, chunk_lens_v, chunk_offsets_v
    ):
        m = _tile_token_meta(kind, ctx, valid, torch, request_specs=request_specs)
        pos[base:base + T] = m["position_ids"][:T]
        ori_slot[base:base + T] = _rebase_nonnegative(
            m["ori_slot_mapping"][:T],
            request_id * ORI_CACHE_BLOCKS * BLOCK_SIZE,
            extent=ORI_CACHE_BLOCKS * BLOCK_SIZE,
            name=f"request {request_id} ori_slot_mapping",
        )
        if kind == "hca":
            hca_cmp[base:base + T] = _rebase_nonnegative(
                m["cmp_slot_mapping"][:T],
                request_id * CMP_CACHE_BLOCKS * BLOCK_SIZE,
                extent=CMP_CACHE_BLOCKS * BLOCK_SIZE,
                name=f"request {request_id} hca_cmp_slot_mapping",
            )
            hca_state[base:base + T] = _rebase_nonnegative(
                m["state_slot_mapping"][:T],
                request_id * HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE,
                extent=HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE,
                name=f"request {request_id} hca_state_slot_mapping",
            )
        elif kind == "csa":
            csa_cmp[base:base + T] = _rebase_nonnegative(
                m["cmp_slot_mapping"][:T],
                request_id * CMP_CACHE_BLOCKS * BLOCK_SIZE,
                extent=CMP_CACHE_BLOCKS * BLOCK_SIZE,
                name=f"request {request_id} csa_cmp_slot_mapping",
            )
            csa_idx[base:base + T] = _rebase_nonnegative(
                m["idx_slot_mapping"][:T],
                request_id * IDX_CACHE_BLOCKS * BLOCK_SIZE,
                extent=IDX_CACHE_BLOCKS * BLOCK_SIZE,
                name=f"request {request_id} csa_idx_slot_mapping",
            )
            csa_state[base:base + T] = _rebase_nonnegative(
                m["state_slot_mapping"][:T],
                request_id * CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE,
                extent=CSA_STATE_BLOCK_NUM * CSA_STATE_BLOCK_SIZE,
                name=f"request {request_id} csa_state_slot_mapping",
            )
            csa_inner[base:base + T] = _rebase_nonnegative(
                m["inner_state_slot_mapping"][:T],
                request_id * INNER_STATE_BLOCK_NUM * INNER_STATE_BLOCK_SIZE,
                extent=INNER_STATE_BLOCK_NUM * INNER_STATE_BLOCK_SIZE,
                name=f"request {request_id} csa_inner_state_slot_mapping",
            )

    return {
        "position_ids": pos,
        "ori_slot_mapping": ori_slot,
        "hca_cmp_slot_mapping": hca_cmp,
        "hca_state_slot_mapping": hca_state,
        "csa_cmp_slot_mapping": csa_cmp,
        "csa_idx_slot_mapping": csa_idx,
        "csa_state_slot_mapping": csa_state,
        "csa_inner_state_slot_mapping": csa_inner,
    }


def _resolve_batch(chunk_lens, start_positions, torch):
    """Normalize batch config into logical lengths and padded physical offsets."""
    chunk_lens_v = [int(c) for c in chunk_lens]
    if not chunk_lens_v:
        raise ValueError("chunk_lens must contain at least one request")
    for c in chunk_lens_v:
        if c <= 0:
            raise ValueError(f"chunk_lens must be positive, got {chunk_lens_v}")
    if start_positions is None:
        start_positions = [0] * len(chunk_lens_v)
    start_positions = [int(s) for s in start_positions]
    if len(start_positions) != len(chunk_lens_v):
        raise ValueError(
            f"start_positions length must match chunk_lens length, "
            f"got {len(start_positions)} and {len(chunk_lens_v)}"
        )
    if any(s < 0 for s in start_positions):
        raise ValueError(f"start_positions must be non-negative, got {start_positions}")
    seq_lens_v = [start_positions[i] + chunk_lens_v[i] for i in range(len(chunk_lens_v))]
    if any(seq_len > MAX_SEQ_LEN for seq_len in seq_lens_v):
        raise ValueError(
            f"start_position + chunk_len must not exceed MAX_SEQ_LEN={MAX_SEQ_LEN}, "
            f"got seq_lens={seq_lens_v}"
        )
    tile_counts_v = [(c + T - 1) // T for c in chunk_lens_v]
    padded_lens_v = [tc * T for tc in tile_counts_v]
    chunk_offsets_v, acc = [], 0
    for c in padded_lens_v:
        chunk_offsets_v.append(acc)
        acc += c
    total_tokens = acc
    int32_max = torch.iinfo(torch.int32).max
    if total_tokens > int32_max:
        raise ValueError(f"padded token count {total_tokens} exceeds INT32 capacity {int32_max}")
    tile_offsets_v, tacc = [], 0
    for tc in tile_counts_v:
        tile_offsets_v.append(tacc)
        tacc += tc
    if tacc > int32_max:
        raise ValueError(f"packed tile count {tacc} exceeds INT32 capacity {int32_max}")
    return (torch.tensor(seq_lens_v, dtype=torch.int32),
            torch.tensor(chunk_lens_v, dtype=torch.int32),
            torch.tensor(chunk_offsets_v, dtype=torch.int32),
            torch.tensor(tile_offsets_v, dtype=torch.int32),
            total_tokens)


def build_tensor_specs(layer_id=2, chunk_lens=DEFAULT_CHUNK_LENS, start_positions=None):
    """Packed batch tensor specs for the chunked prefill layer.

    ``chunk_lens`` lists the current-chunk length per request;
    the default covers ``DEFAULT_USER_BATCH`` requests with chunk lengths
    ``T`` and ``T + T//2``.
    ``start_positions`` is the prior context length per request (default 0 =
    fresh prefill, no cache history). Token tensors are physically padded to
    whole ``T`` tiles. Mutable cache/state storage is global; block tables are
    concatenated request-local views containing global physical block ids.
    """
    import torch
    from golden import ScalarSpec, TensorSpec

    kind = _attention_kind_for_layer(layer_id)
    seq_lens_t, chunk_lens_t, chunk_offsets_t, tile_offsets_t, total_tokens = _resolve_batch(
        chunk_lens, start_positions, torch)
    batch = len(chunk_lens_t)
    seq_lens_list = [int(v) for v in seq_lens_t]
    chunk_lens_list = [int(v) for v in chunk_lens_t]

    # Reuse one child builder per request geometry. In particular, CSA's INT8
    # index cache and FP32 scale specs close over one shared historical sample;
    # rebuilding them independently produces a cache/scale pair from different
    # random draws.
    request_spec_cache = {}

    def request_specs(request_kind, context_len, valid_tokens=T):
        key = (request_kind, int(context_len), int(valid_tokens))
        if key not in request_spec_cache:
            request_spec_cache[key] = {
                s.name: s
                for s in _KIND_BUILDER[request_kind](
                    start_pos=int(context_len),
                    num_tokens=int(valid_tokens),
                )
                if isinstance(s, TensorSpec)
            }
        return request_spec_cache[key]

    swa = request_specs("swa", 0)
    hca = request_specs("hca", 0)
    csa = request_specs("csa", 0)
    active = {"swa": swa, "hca": hca, "csa": csa}[kind]
    src_by_kind = {"swa": swa, "hca": hca, "csa": csa}

    def ranked_init(src):
        def init():
            return torch.stack([_spec_value(src, torch) for _ in range(N_RANKS)], dim=0).contiguous()
        return init

    def replicate(values):
        def init():
            return torch.stack([values.clone() for _ in range(N_RANKS)], dim=0).contiguous()
        return init

    # Per-rank weight tensors (same selection as prefill_layer.py minus token +
    # cache/state tensors, which are rebuilt as packed/per-request below).
    weight_specs = [
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
        ("csa_cmp_wkv", csa["cmp_wkv"]),
        ("csa_cmp_wgate", csa["cmp_wgate"]),
        ("csa_cmp_ape", csa["cmp_ape"]),
        ("csa_cmp_norm_w", csa["cmp_norm_w"]),
        ("csa_hadamard_idx", csa["hadamard_idx"]),
        ("csa_idx_wq_b", csa["idx_wq_b"]),
        ("csa_idx_wq_b_scale", csa["idx_wq_b_scale"]),
        ("csa_weights_proj", csa["idx_weights_proj"]),
        ("csa_inner_wkv", csa["inner_wkv"]),
        ("csa_inner_wgate", csa["inner_wgate"]),
        ("csa_inner_ape", csa["inner_ape"]),
        ("csa_inner_norm_w", csa["inner_norm_w"]),
        ("attn_sink", active["attn_sink"]),
        ("wo_a", active["wo_a"]),
        ("wo_b", active["wo_b"]),
        ("wo_b_scale", active["wo_b_scale"]),
    ]

    tensor_specs = [TensorSpec(name, [N_RANKS, *src.shape], src.dtype, init_value=ranked_init(src))
                    for name, src in weight_specs]

    # Packed token tensors. Metadata is rank-shared; x_hc/input_ids carry per-rank data.
    meta = _packed_token_metadata(kind, seq_lens_list, chunk_lens_list,
                                  [int(c) for c in chunk_offsets_t], total_tokens, torch,
                                  request_specs=request_specs)
    chunk_offsets_list = [int(c) for c in chunk_offsets_t]

    def init_x_hc():
        x = torch.zeros(N_RANKS, total_tokens, HC_MULT, D, dtype=torch.float32)
        for base, chunk_len in zip(chunk_offsets_list, chunk_lens_list, strict=True):
            # Keep active rows at the same O(1) scale as the standalone
            # attention fixtures. A tiny residual makes the expected MX
            # quantization error dominate the layer-composition comparison.
            x[:, base:base + chunk_len] = torch.empty(
                N_RANKS, chunk_len, HC_MULT, D, dtype=torch.float32
            ).uniform_(-1.0, 1.0)
        return x

    def init_input_ids():
        ids = torch.zeros(N_RANKS, total_tokens, dtype=torch.int64)
        for rank in range(N_RANKS):
            for base, chunk_len in zip(chunk_offsets_list, chunk_lens_list, strict=True):
                ids[rank, base:base + chunk_len] = (torch.arange(chunk_len, dtype=torch.int64) + base + rank) % VOCAB
        return ids.contiguous()

    tensor_specs.append(TensorSpec("x_hc", [N_RANKS, total_tokens, HC_MULT, D], torch.float32, init_value=init_x_hc))
    tensor_specs.append(TensorSpec("input_ids", [N_RANKS, total_tokens], torch.int64, init_value=init_input_ids))
    tensor_specs.append(TensorSpec("position_ids", [N_RANKS, total_tokens], torch.int32,
                                   init_value=replicate(meta["position_ids"])))
    tensor_specs.append(TensorSpec("ori_slot_mapping", [N_RANKS, total_tokens], torch.int64,
                                   init_value=replicate(meta["ori_slot_mapping"])))
    for name in ("hca_cmp_slot_mapping", "hca_state_slot_mapping", "csa_cmp_slot_mapping",
                 "csa_idx_slot_mapping", "csa_state_slot_mapping", "csa_inner_state_slot_mapping"):
        tensor_specs.append(TensorSpec(name, [N_RANKS, total_tokens], torch.int64, init_value=replicate(meta[name])))

    def resolve_cache_src(packed_name, info):
        """Resolve (source spec, source kind, child-local name) for a packed cache."""
        if isinstance(info, tuple):
            sk, cn = info
            return src_by_kind[sk][cn], sk, cn
        cn = info
        if cn == "ori_block_table":
            return (active.get("ori_block_table") or swa["block_table"]), kind, cn
        if cn in ("cmp_kv", "cmp_block_table"):
            return (active.get(cn) or csa[cn]), kind, cn
        if cn in ("idx_kv_cache", "idx_kv_scale", "idx_block_table"):
            return csa[cn], kind, cn
        return active[cn], kind, cn  # kv_cache

    # Allocator-owned global cache/state pools plus request-local block tables.
    # The standalone fixture capacity is repeated per request, and all physical
    # ids are rebased into the corresponding segment of the one global tensor.
    for packed_name, info in _PACKED_CACHE_SPECS.items():
        src, src_kind, child_name = resolve_cache_src(packed_name, info)
        per_req = _spec_value(src, torch)
        is_global_pool = packed_name in _GLOBAL_POOL_NAMES

        def make_init(
            packed_name=packed_name,
            per_req=per_req,
            is_global_pool=is_global_pool,
            src_kind=src_kind,
            child_name=child_name,
        ):
            def init():
                blocks = []
                for r in range(batch):
                    cs = seq_lens_list[r] - chunk_lens_list[r]
                    first_tile_tokens = min(T, chunk_lens_list[r])
                    rspec = request_specs(src_kind, cs, first_tile_tokens).get(child_name)
                    block = _spec_value(rspec, torch) if rspec is not None else per_req.clone()
                    if packed_name in _GLOBAL_TABLE_BLOCK_STRIDES:
                        block_stride = _GLOBAL_TABLE_BLOCK_STRIDES[packed_name]
                        block = _rebase_nonnegative(
                            block,
                            r * block_stride,
                            extent=block_stride,
                            name=f"request {r} {packed_name}",
                        )
                    blocks.append(block)
                pool = torch.cat(blocks, dim=0).contiguous()
                return torch.stack([pool.clone() for _ in range(N_RANKS)], dim=0).contiguous()
            return init

        dim0 = batch * src.shape[0]
        tensor_specs.append(TensorSpec(packed_name, [N_RANKS, dim0, *src.shape[1:]],
                                       src.dtype, init_value=make_init()))

    # Batch metadata.
    tensor_specs.append(TensorSpec("seq_lens", [N_RANKS, batch], torch.int32, init_value=replicate(seq_lens_t)))
    tensor_specs.append(TensorSpec("chunk_lens", [N_RANKS, batch], torch.int32, init_value=replicate(chunk_lens_t)))
    tensor_specs.append(TensorSpec("chunk_offsets", [N_RANKS, batch], torch.int32, init_value=replicate(chunk_offsets_t)))
    tensor_specs.append(TensorSpec("chunk_tile_offsets", [N_RANKS, batch], torch.int32,
                                   init_value=replicate(tile_offsets_t)))

    # MoE weight tensors (per rank). tid2eid keeps its hash-table init.
    for spec in build_moe_tensor_specs(layer_id=layer_id):
        if not isinstance(spec, TensorSpec) or spec.name in {"x_hc", "x_next", "input_ids"}:
            continue
        if spec.name == "tid2eid":
            def init_tid2eid(spec=spec):
                _, vocab, topk = spec.shape
                ids = torch.arange(vocab, dtype=torch.int64).view(vocab, 1)
                ks = torch.arange(topk, dtype=torch.int64).view(1, topk)
                table = ((ids * topk + ks) % N_EXPERTS_GLOBAL).to(dtype=spec.dtype)
                return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

            tensor_specs.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_tid2eid))
        else:
            tensor_specs.append(spec)

    # InOut, not Out: the kernel writes only the packed chunk rows, and the host zeros
    # must reach the device so valid_ratio_reldiff can check the pad rows are untouched.
    tensor_specs.append(TensorSpec("x_next", [N_RANKS, total_tokens, HC_MULT, D], torch.float32,
                                   init_value=torch.zeros))

    # Keep static weight parameters device-resident (child_memory), sharded per
    # rank. Dynamic cache/state/table tensors must stay as host tensors because
    # the generated orchestration reads their ``.shape`` to bind dynamic sizes.
    RESIDENT_WEIGHT_NAMES = frozenset([
        # Attention core weights + RoPE tables
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin",
        # HCA / CSA compressor + indexer weights (states/block tables excluded)
        "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
        "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
        "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
        "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
        # Attention output projection
        "attn_sink", "wo_a", "wo_b", "wo_b_scale",
        # MoE FFN / gate / experts + static route table
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
    ])
    for spec in tensor_specs:
        if spec.name in RESIDENT_WEIGHT_NAMES:
            spec.resident = "stacked"

    tensor_by_name = {spec.name: spec for spec in tensor_specs}
    missing = [name for name in HOST_TENSOR_ORDER if name not in tensor_by_name]
    if missing:
        raise ValueError(f"missing packed prefill layer tensor specs: {missing}")
    return [tensor_by_name[name] for name in HOST_TENSOR_ORDER] + [
        ScalarSpec("layer_id", torch.int32, layer_id),
    ]


def golden_prefill_layer(tensors):
    """Reference for packed chunked prefill: request/tile loop mirroring the kernel.

    For each request, the global cache/state pools are reused across its tiles
    (so a tile reads what earlier tiles wrote) while tables remain request-local. Each tile
    runs the active attention child golden per rank, then one collective MoE golden,
    then scatters the valid rows back into the packed ``x_next``.
    """
    import torch
    from golden import TensorSpec

    layer_id = int(tensors["layer_id"])
    kind = _attention_kind_for_layer(layer_id)
    chunk_lens = tensors["chunk_lens"][0]
    chunk_offsets = tensors["chunk_offsets"][0]
    batch = chunk_lens.shape[0]

    # Map child-local attention tensor names -> packed names (un-namespacing).
    mapped = dict(tensors)
    if kind == "swa":
        mapped["block_table"] = tensors["ori_block_table"]
        attention_golden = golden_prefill_attention_swa
    elif kind == "hca":
        mapped.update({
            "cmp_wkv": tensors["hca_cmp_wkv"], "cmp_wgate": tensors["hca_cmp_wgate"],
            "cmp_ape": tensors["hca_cmp_ape"], "cmp_norm_w": tensors["hca_cmp_norm_w"],
            "compress_state": tensors["hca_compress_state"],
            "compress_state_block_table": tensors["hca_compress_state_block_table"],
            "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"], "state_slot_mapping": tensors["hca_state_slot_mapping"],
        })
        attention_golden = golden_prefill_attention_hca
    else:
        mapped.update({
            "cmp_wkv": tensors["csa_cmp_wkv"], "cmp_wgate": tensors["csa_cmp_wgate"],
            "cmp_ape": tensors["csa_cmp_ape"], "cmp_norm_w": tensors["csa_cmp_norm_w"],
            "compress_state": tensors["csa_compress_state"],
            "compress_state_block_table": tensors["csa_compress_state_block_table"],
            "hadamard_idx": tensors["csa_hadamard_idx"], "idx_wq_b": tensors["csa_idx_wq_b"],
            "idx_wq_b_scale": tensors["csa_idx_wq_b_scale"], "idx_weights_proj": tensors["csa_weights_proj"],
            "inner_wkv": tensors["csa_inner_wkv"], "inner_wgate": tensors["csa_inner_wgate"],
            "inner_ape": tensors["csa_inner_ape"], "inner_norm_w": tensors["csa_inner_norm_w"],
            "inner_compress_state": tensors["csa_inner_compress_state"],
            "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"],
            "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"], "idx_slot_mapping": tensors["csa_idx_slot_mapping"],
            "state_slot_mapping": tensors["csa_state_slot_mapping"],
            "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping"],
        })
        attention_golden = golden_prefill_attention_csa

    attn_specs = _KIND_BUILDER[kind](start_pos=0, num_tokens=T)
    x_next = tensors["x_next"]

    def tile_buffer(packed_per_rank, rank, base, _valid, feature_shape, dtype):
        buf = torch.zeros((T, *feature_shape), dtype=dtype)
        buf[:] = packed_per_rank[rank, base:base + T]
        return buf

    for request_id in range(batch):
        chunk_len = int(chunk_lens[request_id])
        chunk_base = int(chunk_offsets[request_id])
        tok_blocks = (chunk_len + T - 1) // T

        # Global mutable pool views plus request-local table views.
        req_views = {}
        for packed_name, info in _PACKED_CACHE_SPECS.items():
            if packed_name in _GLOBAL_POOL_NAMES:
                req_views[packed_name] = tensors[packed_name]
                continue
            child_name = info[1] if isinstance(info, tuple) else info
            source_kind = info[0] if isinstance(info, tuple) else kind
            cnt = _req_block_count(source_kind, child_name)
            req_views[packed_name] = tensors[packed_name][:, request_id * cnt:(request_id + 1) * cnt]

        for tile_id in range(tok_blocks):
            p0 = tile_id * T
            valid = min(T, chunk_len - p0)
            base = chunk_base + p0

            x_attn_tile = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
            for rank in range(N_RANKS):
                attn_tensors = {}
                for spec in attn_specs:
                    if not isinstance(spec, TensorSpec):
                        continue  # scalar (num_tokens) set explicitly below
                    name = spec.name
                    if name == "x_out":
                        attn_tensors[name] = x_attn_tile[rank]
                    elif name == "x_hc":
                        attn_tensors[name] = tile_buffer(tensors["x_hc"], rank, base, valid, (HC_MULT, D), torch.float32)
                    elif name in _TOKEN_META_NAMES:
                        packed = mapped[name]
                        attn_tensors[name] = tile_buffer(packed, rank, base, valid, tuple(packed.shape[2:]), packed.dtype)
                    elif name in _CACHE_STATE_NAMES:
                        attn_tensors[name] = req_views[_child_to_packed(kind, name)][rank]
                    else:
                        attn_tensors[name] = mapped[name][rank]
                attn_tensors["num_tokens"] = valid
                attention_golden(attn_tensors)
                x_attn_tile[rank] = attn_tensors["x_out"]

            moe_tensors = dict(tensors)
            moe_tensors["x_hc"] = x_attn_tile
            input_ids_tile = torch.zeros(N_RANKS, T, dtype=torch.int64)
            input_ids_tile[:, :valid] = tensors["input_ids"][:, base:base + valid]
            moe_tensors["input_ids"] = input_ids_tile
            moe_tensors["num_tokens"] = valid
            # Match the FP32 device-side MoE output tile. Staging this buffer as
            # BF16 truncates the golden result before it is scattered to x_next.
            x_next_tile = torch.zeros_like(x_attn_tile)
            moe_tensors["x_next"] = x_next_tile
            golden_moe(moe_tensors)

            x_next[:, base:base + valid] = x_next_tile[:, :valid]


def valid_ratio_reldiff(
    diff_thd,
    pct_thd,
    *,
    max_abs_diff=float("inf"),
):
    """Validate logical token rows and require padded physical rows to stay exact."""
    import torch
    from golden import ratio_reldiff

    base_cmp = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd)

    def cmp(actual, expected, **kwargs):
        if actual.shape != expected.shape or actual.ndim < 2:
            return False, (
                f"    output shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        actual_f = actual.cpu().to(torch.float32)
        expected_f = expected.cpu().to(torch.float32)
        for label, values in (("actual", actual_f), ("expected", expected_f)):
            nonfinite = ~torch.isfinite(values)
            if nonfinite.any().item():
                return False, (
                    f"    illegal values in {label}: "
                    f"count={int(nonfinite.count_nonzero().item())}"
                )

        inputs = kwargs.get("inputs", {})
        chunk_lens = inputs.get("chunk_lens")
        chunk_offsets = inputs.get("chunk_offsets")
        if chunk_lens is None or chunk_offsets is None:
            return False, "    compare_fn requires chunk_lens and chunk_offsets inputs"
        chunk_lens = chunk_lens.cpu().to(torch.int64)
        chunk_offsets = chunk_offsets.cpu().to(torch.int64)
        rank_count, physical_rows = actual.shape[:2]
        if (
            chunk_lens.ndim != 2
            or chunk_offsets.shape != chunk_lens.shape
            or chunk_lens.shape[0] != rank_count
            or chunk_lens.shape[1] == 0
        ):
            return False, (
                f"    invalid packed metadata shapes: chunk_lens={tuple(chunk_lens.shape)} "
                f"chunk_offsets={tuple(chunk_offsets.shape)} ranks={rank_count}"
            )
        if not torch.equal(chunk_lens, chunk_lens[:1].expand_as(chunk_lens)):
            return False, "    chunk_lens differ across ranks, but golden uses shared metadata"
        if not torch.equal(chunk_offsets, chunk_offsets[:1].expand_as(chunk_offsets)):
            return False, "    chunk_offsets differ across ranks, but golden uses shared metadata"

        lens = chunk_lens[0]
        offsets = chunk_offsets[0]
        invalid_lens = lens <= 0
        invalid_offsets = offsets < 0
        ends = offsets + lens
        out_of_range = ends > physical_rows
        if invalid_lens.any().item() or invalid_offsets.any().item() or out_of_range.any().item():
            return False, (
                f"    invalid packed metadata for physical_rows={physical_rows}: "
                f"chunk_lens={lens.tolist()} chunk_offsets={offsets.tolist()}"
            )

        intervals = sorted(zip(offsets.tolist(), ends.tolist(), strict=True))
        for (_, previous_end), (next_start, _) in itertools.pairwise(intervals):
            if next_start < previous_end:
                return False, f"    overlapping packed token intervals: {intervals}"

        mask = torch.zeros(physical_rows, dtype=torch.bool)
        for base, end in intervals:
            mask[base:end] = True
        if not torch.equal(actual[:, ~mask], expected[:, ~mask]):
            changed = int((actual[:, ~mask] != expected[:, ~mask]).count_nonzero().item())
            return False, f"    padded physical token rows changed: changed_values={changed}"

        actual_valid = actual[:, mask]
        expected_valid = expected[:, mask]
        ok, detail = base_cmp(actual_valid, expected_valid, **kwargs)
        if not ok:
            return ok, detail
        if actual_valid.numel() > 0:
            worst_abs = float(
                (actual_valid.float() - expected_valid.float()).abs().max().item()
            )
            if worst_abs > max_abs_diff:
                return False, (
                    f"    worst absolute diff={worst_abs:.6g} exceeds "
                    f"max_abs_diff={max_abs_diff:.6g}"
                )
        return True, ""

    cmp.__name__ = "valid_ratio_reldiff"
    return cmp


def mapped_pool_ratio_allclose(
    mapping_names,
    *,
    atol,
    rtol,
    max_error_ratio,
):
    """Validate allocator-mapped rows without a whole-pool ratio budget.

    Cache/state pools grow with the request batch, while each layer only writes
    rows selected by its slot mappings. Numerical tolerance is therefore
    applied to the union of those rows. Every other physical row must remain
    exactly equal to golden, which also detects writes outside the allocation.
    """
    import torch

    from golden import ratio_allclose

    if isinstance(mapping_names, str):
        mapping_names = (mapping_names,)
    else:
        mapping_names = tuple(mapping_names)
    written_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
    )

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, (
                f"    pool shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if actual.ndim < 3:
            return False, f"    mapped pool must have rank >= 3, got {tuple(actual.shape)}"

        rank_count = actual.shape[0]
        actual_rows = actual.reshape(rank_count, -1, actual.shape[-1])
        expected_rows = expected.reshape(rank_count, -1, expected.shape[-1])
        row_count = actual_rows.shape[1]
        for label, rows in (("actual", actual_rows), ("expected", expected_rows)):
            if torch.is_floating_point(rows):
                nonfinite = ~torch.isfinite(rows)
                if nonfinite.any().item():
                    return False, (
                        f"    {label} pool contains "
                        f"{int(nonfinite.count_nonzero().item())} non-finite value(s)"
                    )
        written_rows = torch.zeros((rank_count, row_count), dtype=torch.bool)
        inputs = kwargs.get("inputs", {})

        for mapping_name in mapping_names:
            mapping = inputs.get(mapping_name)
            if mapping is None:
                return False, f"    compare_fn misconfigured: missing input '{mapping_name}'"
            mapping = mapping.cpu().to(torch.int64)
            if mapping.ndim != 2 or mapping.shape[0] != rank_count:
                return False, (
                    f"    '{mapping_name}' shape {tuple(mapping.shape)} does not match "
                    f"ranked pool shape {tuple(actual.shape)}"
                )
            invalid_negative = mapping < -1
            if invalid_negative.any().item():
                first = invalid_negative.nonzero(as_tuple=False)[0]
                rank = int(first[0].item())
                token = int(first[1].item())
                row = int(mapping[rank, token].item())
                return False, (
                    f"    '{mapping_name}'[{rank}, {token}]={row} is invalid; "
                    "only -1 is a negative sentinel"
                )
            valid = mapping >= 0
            out_of_range = valid & (mapping >= row_count)
            if out_of_range.any().item():
                first = out_of_range.nonzero(as_tuple=False)[0]
                rank = int(first[0].item())
                token = int(first[1].item())
                row = int(mapping[rank, token].item())
                return False, (
                    f"    '{mapping_name}'[{rank}, {token}]={row} is outside "
                    f"physical row range [0, {row_count})"
                )
            for rank in range(rank_count):
                rank_mapping = mapping[rank, valid[rank]]
                if rank_mapping.numel() != torch.unique(rank_mapping).numel():
                    return False, (
                        f"    '{mapping_name}' contains duplicate rows on rank {rank}"
                    )
                already_written = written_rows[rank, rank_mapping]
                if already_written.any().item():
                    return False, (
                        f"    mappings {mapping_names} overlap on rank {rank}"
                    )
                written_rows[rank, rank_mapping] = True

        equal_rows = (actual_rows == expected_rows).all(dim=-1)
        stray_rows = ~written_rows & ~equal_rows
        if stray_rows.any().item():
            first = stray_rows.nonzero(as_tuple=False)[0]
            rank = int(first[0].item())
            row = int(first[1].item())
            changed_values = int(
                (actual_rows[rank, row] != expected_rows[rank, row]).sum().item()
            )
            return False, (
                f"    unmapped physical row changed: rank={rank} row={row} "
                f"changed_values={changed_values} mappings={mapping_names}"
            )

        ok, detail = written_compare(
            actual_rows[written_rows],
            expected_rows[written_rows],
            **kwargs,
        )
        if ok:
            return True, ""
        return False, f"    mapped rows from {mapping_names}:\n{detail}"

    compare.__name__ = (
        f"mapped_pool_ratio_allclose(mappings={mapping_names}, atol={atol}, "
        f"rtol={rtol}, max_error_ratio={max_error_ratio})"
    )
    return compare


if __name__ == "__main__":
    import argparse
    import torch

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe)")
    parser.add_argument("-d", "--device", type=str,
                        default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--layer-id", type=int, default=2,
                        help="Layer id selects attention by MODEL_CONFIG.compress_ratios[layer_id].")
    parser.add_argument("--chunk-lens", type=str, default=",".join(str(c) for c in DEFAULT_CHUNK_LENS),
                        help="Comma-separated per-request logical chunk lengths.")
    parser.add_argument("--start-positions", type=str, default=None,
                        help="Comma-separated per-request prior context lengths; defaults to all zeros.")
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="persist inputs and golden outputs for replay")
    parser.add_argument("--golden-data", type=str, default=None,
                        help="directory containing cached in/ and out/ tensors")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible inputs and golden")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--weights", type=str, default=None,
                        help="HF checkpoint dir: inject this layer's real DeepSeek-V4-Flash weights "
                             "(golden recomputes with them, so validation runs on real weights).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"
    chunk_lens = tuple(int(x) for x in args.chunk_lens.split(","))
    start_positions = None if args.start_positions is None else tuple(int(x) for x in args.start_positions.split(","))
    compare_fn = {
        # Pro EP2 measurements place 3.3-5.1% of the composed HCA/CSA output
        # above 1e-2 across packed chunk boundaries. Keep less than one point
        # of margin while child kernels retain their stricter precision gates.
        "x_next": valid_ratio_reldiff(
            diff_thd=0.01,
            pct_thd=0.06,
            max_abs_diff=0.25,
        ),
        # Apply ratio budgets only to allocator-mapped writes. Unmapped rows
        # must remain exact, so expanding a global pool cannot hide an error.
        "kv_cache": mapped_pool_ratio_allclose(
            "ori_slot_mapping",
            atol=1e-4,
            rtol=1.0 / 128,
            max_error_ratio=0.005,
        ),
        "cmp_kv": mapped_pool_ratio_allclose(
            ("hca_cmp_slot_mapping", "csa_cmp_slot_mapping"),
            atol=1e-4,
            rtol=1.0 / 128,
            max_error_ratio=0.005,
        ),
        "idx_kv_cache": mapped_pool_ratio_allclose(
            "csa_idx_slot_mapping",
            atol=1,
            rtol=0,
            max_error_ratio=0.01,
        ),
        "idx_kv_scale": mapped_pool_ratio_allclose(
            "csa_idx_slot_mapping",
            atol=1e-4,
            rtol=1.0 / 128,
            max_error_ratio=0.01,
        ),
        "hca_compress_state": mapped_pool_ratio_allclose(
            "hca_state_slot_mapping",
            atol=1e-3,
            rtol=1e-3,
            max_error_ratio=0.005,
        ),
        "csa_compress_state": mapped_pool_ratio_allclose(
            "csa_state_slot_mapping",
            atol=1e-3,
            rtol=1e-3,
            max_error_ratio=0.005,
        ),
        "csa_inner_compress_state": mapped_pool_ratio_allclose(
            "csa_inner_state_slot_mapping",
            atol=1e-3,
            rtol=1e-3,
            max_error_ratio=0.005,
        ),
    }

    specs = build_tensor_specs(
        layer_id=args.layer_id,
        chunk_lens=chunk_lens,
        start_positions=start_positions,
    )
    if args.weights is not None:
        from utils import apply_real_layer_weights

        count = apply_real_layer_weights(specs, args.weights, layer_id=args.layer_id, ep=N_RANKS)
        print(f"[RUN] real weights: layer {args.layer_id}, {count} tensors from {args.weights}", flush=True)

    result = run(
        fn=l3_prefill_layer,
        specs=specs,
        golden_fn=golden_prefill_layer,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            ring_heap=LAYER_RING_HEAP,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=compare_fn,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
