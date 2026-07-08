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
  * cache/state/tables   : request-local (each request owns a contiguous slice)
  * sparse-index overlay : tile-local (``WIN + t`` for the current tile's tokens)
  * position_ids         : absolute position (``chunk_start + tile_id*T + t``)

All tile-local metadata (slot mappings, sparse indices, positions) is produced on
the host by reusing the child ``build_*_tensor_specs(start_pos=tile_ctx,
num_tokens=valid)`` builders, which already encode the absolute-position ring /
overlay / compressed / state formulas for a single ``[T]`` tile. The kernel just
gathers the precomputed packed metadata per tile.
"""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

# The prefill path routes PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe (which freezes recv shapes and derives RECV_MAX = EP * MOE_TOKENS at import).
import config
config.MOE_TOKENS = config.PREFILL_TOKENS
# Import moe first. It applies the EP2 FLASH override before dependent
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
    T,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    golden_moe,
    moe,
)
from config import FLASH as MODEL_CONFIG
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
    SPARSE_TOPK,
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
TOK_TILE = T
PREFILL_CHUNK_TOKENS = T
DEFAULT_CHUNK_LENS = (T, T + T // 2)
DEFAULT_USER_BATCH = len(DEFAULT_CHUNK_LENS)

# Per-request contiguous block/state counts (each request owns one such slice; the
# packed buffer dim0 is ``user_batch * <count>``). The cache/state block tables are
# request-local so the children see block ids starting at 0 inside their slice.
ORI_CACHE_BLOCKS = CSA_ORI_BLOCK_NUM
CMP_CACHE_BLOCKS = CSA_CMP_BLOCK_NUM
IDX_CACHE_BLOCKS = PREFILL_IDX_BLOCK_NUM
ORI_TABLE_BLOCKS = SPARSE_ORI_MAX_BLOCKS
CMP_TABLE_BLOCKS = SPARSE_CMP_MAX_BLOCKS
IDX_TABLE_BLOCKS = IDX_CACHE_MAX_BLOCKS

# Dynamic (batch-dependent) kernel-signature dims. Kept local to this file so
# config.py and the fixed-T child kernels stay untouched (issue #591 §1).
USER_BATCH_DYN = pl.dynamic("DEEPSEEK_PREFILL_USER_BATCH_DYN")
PREFILL_TOKENS_DYN = pl.dynamic("DEEPSEEK_PREFILL_TOKENS_DYN")

PREFILL_ORI_CACHE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_ORI_CACHE_BLOCKS_DYN")
PREFILL_CMP_CACHE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_CMP_CACHE_BLOCKS_DYN")
PREFILL_IDX_CACHE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_IDX_CACHE_BLOCKS_DYN")

PREFILL_ORI_BLOCK_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_ORI_BLOCK_TABLE_DYN")
PREFILL_CMP_BLOCK_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_CMP_BLOCK_TABLE_DYN")
PREFILL_IDX_BLOCK_TABLE_DYN = pl.dynamic("DEEPSEEK_PREFILL_IDX_BLOCK_TABLE_DYN")

PREFILL_HCA_STATE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_HCA_STATE_BLOCKS_DYN")
PREFILL_CSA_STATE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_CSA_STATE_BLOCKS_DYN")
PREFILL_INNER_STATE_BLOCKS_DYN = pl.dynamic("DEEPSEEK_PREFILL_INNER_STATE_BLOCKS_DYN")


@pl.jit
def prefill_layer_core(
    x_hc: pl.Tensor[[PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16],
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
    hca_cmp_kv_state: pl.InOut[pl.Tensor[
        [PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_cmp_score_state: pl.InOut[pl.Tensor[
        [PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_compress_state_block_table: pl.Tensor[[PREFILL_HCA_STATE_BLOCKS_DYN], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.InOut[
        pl.Tensor[[PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_cmp_score_state: pl.InOut[
        pl.Tensor[[PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[[PREFILL_CSA_STATE_BLOCKS_DYN], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.InOut[
        pl.Tensor[[PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_score_state: pl.InOut[
        pl.Tensor[[PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[PREFILL_INNER_STATE_BLOCKS_DYN], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[PREFILL_ORI_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[PREFILL_ORI_BLOCK_TABLE_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[PREFILL_CMP_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_BLOCK_TABLE_DYN], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[PREFILL_TOKENS_DYN, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
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
    x_next: pl.Out[pl.Tensor[[PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16]:
    x_hc.bind_dynamic(0, PREFILL_TOKENS_DYN)
    ori_slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    cmp_sparse_indices.bind_dynamic(0, PREFILL_TOKENS_DYN)
    cmp_sparse_lens.bind_dynamic(0, PREFILL_TOKENS_DYN)
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
    idx_block_table.bind_dynamic(0, PREFILL_IDX_BLOCK_TABLE_DYN)
    hca_cmp_kv_state.bind_dynamic(0, PREFILL_HCA_STATE_BLOCKS_DYN)
    hca_cmp_score_state.bind_dynamic(0, PREFILL_HCA_STATE_BLOCKS_DYN)
    hca_compress_state_block_table.bind_dynamic(0, PREFILL_HCA_STATE_BLOCKS_DYN)
    csa_cmp_kv_state.bind_dynamic(0, PREFILL_CSA_STATE_BLOCKS_DYN)
    csa_cmp_score_state.bind_dynamic(0, PREFILL_CSA_STATE_BLOCKS_DYN)
    csa_compress_state_block_table.bind_dynamic(0, PREFILL_CSA_STATE_BLOCKS_DYN)
    csa_inner_kv_state.bind_dynamic(0, PREFILL_INNER_STATE_BLOCKS_DYN)
    csa_inner_score_state.bind_dynamic(0, PREFILL_INNER_STATE_BLOCKS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(0, PREFILL_INNER_STATE_BLOCKS_DYN)
    user_batch = pl.tensor.dim(seq_lens, 0)
    for request_id in pl.range(user_batch):
        chunk_len_b = pl.tensor.read(chunk_lens, [request_id])
        chunk_base = pl.cast(pl.tensor.read(chunk_offsets, [request_id]), pl.INDEX)
        tile_ord_base = pl.cast(pl.tensor.read(chunk_tile_offsets, [request_id]), pl.INDEX)
        tok_blocks = (chunk_len_b + TOK_TILE - 1) // TOK_TILE
        ridx = pl.cast(request_id, pl.INDEX)

        # Request-local cache/state/table views (one slice per request; persist
        # across the request's tiles via in-place mutation by the children).
        kv_cache_req = pl.slice(kv_cache, [ORI_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
                                [ridx * ORI_CACHE_BLOCKS, 0, 0, 0])
        ori_block_table_req = pl.slice(ori_block_table, [ORI_TABLE_BLOCKS], [ridx * ORI_TABLE_BLOCKS])
        cmp_kv_req = pl.slice(cmp_kv, [CMP_CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
                              [ridx * CMP_CACHE_BLOCKS, 0, 0, 0])
        cmp_block_table_req = pl.slice(cmp_block_table, [CMP_TABLE_BLOCKS], [ridx * CMP_TABLE_BLOCKS])
        idx_kv_cache_req = pl.slice(idx_kv_cache, [IDX_CACHE_BLOCKS, BLOCK_SIZE, 1, IDX_HEAD_DIM],
                                    [ridx * IDX_CACHE_BLOCKS, 0, 0, 0])
        idx_block_table_req = pl.slice(idx_block_table, [IDX_TABLE_BLOCKS], [ridx * IDX_TABLE_BLOCKS])
        hca_kv_state_req = pl.slice(hca_cmp_kv_state, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
                                    [ridx * HCA_STATE_BLOCK_NUM, 0, 0])
        hca_score_state_req = pl.slice(hca_cmp_score_state, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
                                       [ridx * HCA_STATE_BLOCK_NUM, 0, 0])
        hca_state_table_req = pl.slice(hca_compress_state_block_table, [HCA_STATE_MAX_BLOCKS],
                                       [ridx * HCA_STATE_MAX_BLOCKS])
        csa_kv_state_req = pl.slice(csa_cmp_kv_state, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM],
                                    [ridx * CSA_STATE_BLOCK_NUM, 0, 0])
        csa_score_state_req = pl.slice(csa_cmp_score_state, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM],
                                       [ridx * CSA_STATE_BLOCK_NUM, 0, 0])
        csa_state_table_req = pl.slice(csa_compress_state_block_table, [CSA_STATE_MAX_BLOCKS],
                                       [ridx * CSA_STATE_MAX_BLOCKS])
        csa_inner_kv_state_req = pl.slice(csa_inner_kv_state, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM],
                                          [ridx * INNER_STATE_BLOCK_NUM, 0, 0])
        csa_inner_score_state_req = pl.slice(csa_inner_score_state, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM],
                                             [ridx * INNER_STATE_BLOCK_NUM, 0, 0])
        csa_inner_state_table_req = pl.slice(csa_inner_compress_state_block_table, [INNER_STATE_MAX_BLOCKS],
                                             [ridx * INNER_STATE_MAX_BLOCKS])

        for tile_id in pl.range(tok_blocks):
            p0 = tile_id * TOK_TILE
            tile_base = chunk_base + p0
            valid_tok = pl.min(TOK_TILE, chunk_len_b - p0)
            valid_n = pl.cast(valid_tok, pl.INT32)  # child num_tokens scalar (INT32)
            # Global execution ordinal of this MoE call (1-based). The done
            # windows are monotonic counters, so each serial MoE call needs a
            # unique, gap-free epoch == its execution order; chunk_tile_offsets
            # is the exclusive prefix sum of tok_blocks over requests.
            moe_epoch = pl.cast(tile_ord_base + tile_id + 1, pl.INT32)

            # Tile-local fixed-[T] inputs gathered from the packed buffers. The
            # children only read the leading ``valid_tok`` rows, so the padded
            # tail (when valid_tok < T) is ignored. No explicit per-tile pl.scope:
            # rely on auto_scope + pl.range's sequential semantics so the
            # request-local cache/state RAW dependency is carried across tiles
            # (tile N's writeback ordered before tile N+1's gather).
            x_hc_tile = pl.slice(x_hc, [TOK_TILE, HC_MULT, D], [tile_base, 0, 0],
                                 valid_shape=[valid_tok, HC_MULT, D])
            ori_slot_tile = pl.slice(ori_slot_mapping, [TOK_TILE], [tile_base], valid_shape=[valid_tok])
            cmp_sparse_indices_tile = pl.slice(cmp_sparse_indices, [TOK_TILE, SPARSE_TOPK], [tile_base, 0],
                                               valid_shape=[valid_tok, SPARSE_TOPK])
            cmp_sparse_lens_tile = pl.slice(cmp_sparse_lens, [TOK_TILE], [tile_base], valid_shape=[valid_tok])
            position_ids_tile = pl.slice(position_ids, [TOK_TILE], [tile_base], valid_shape=[valid_tok])
            hca_cmp_slot_tile = pl.slice(hca_cmp_slot_mapping, [TOK_TILE], [tile_base], valid_shape=[valid_tok])
            hca_state_slot_tile = pl.slice(hca_state_slot_mapping, [TOK_TILE], [tile_base], valid_shape=[valid_tok])
            csa_cmp_slot_tile = pl.slice(csa_cmp_slot_mapping, [TOK_TILE], [tile_base], valid_shape=[valid_tok])
            csa_idx_slot_tile = pl.slice(csa_idx_slot_mapping, [TOK_TILE], [tile_base], valid_shape=[valid_tok])
            csa_state_slot_tile = pl.slice(csa_state_slot_mapping, [TOK_TILE], [tile_base], valid_shape=[valid_tok])
            csa_inner_state_slot_tile = pl.slice(csa_inner_state_slot_mapping, [TOK_TILE], [tile_base],
                                                 valid_shape=[valid_tok])
            input_ids_tile = pl.slice(input_ids, [TOK_TILE], [tile_base], valid_shape=[valid_tok])

            x_attn_tile = pl.create_tensor([TOK_TILE, HC_MULT, D], dtype=pl.BF16)
            if layer_id < 2:
                prefill_attention_swa(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    kv_cache_req, ori_block_table_req, ori_slot_tile,
                    cmp_sparse_indices_tile, cmp_sparse_lens_tile, position_ids_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )
            elif layer_id % 2 == 1:
                prefill_attention_hca(
                    x_hc_tile, hc_attn_fn, hc_attn_scale, hc_attn_base,
                    attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
                    freqs_cos, freqs_sin,
                    hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
                    hca_kv_state_req, hca_score_state_req, hca_state_table_req,
                    kv_cache_req, ori_slot_tile, ori_block_table_req,
                    cmp_kv_req, cmp_block_table_req, cmp_sparse_indices_tile, cmp_sparse_lens_tile,
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
                    csa_kv_state_req, csa_score_state_req, csa_state_table_req,
                    csa_hadamard_idx,
                    csa_idx_wq_b, csa_idx_wq_b_scale, csa_weights_proj,
                    csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
                    csa_inner_kv_state_req, csa_inner_score_state_req, csa_inner_state_table_req,
                    kv_cache_req, ori_block_table_req, ori_slot_tile,
                    cmp_kv_req, cmp_block_table_req, idx_kv_cache_req, idx_block_table_req,
                    position_ids_tile, csa_cmp_slot_tile, csa_idx_slot_tile,
                    csa_state_slot_tile, csa_inner_state_slot_tile,
                    attn_sink, wo_a, wo_b, wo_b_scale,
                    x_attn_tile, valid_n,
                )

            x_next_tile = pl.create_tensor([TOK_TILE, HC_MULT, D], dtype=pl.BF16)
            moe(
                x_attn_tile,
                hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
                norm_w, gate_w, gate_bias, tid2eid, input_ids_tile,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                x_next_tile,
                recv_meta, recv_x, recv_aux, recv_route, arrived,
                routed_y_buf, combine_arrived,
                layer_id, valid_n, my_rank, moe_epoch,
            )

            # Scatter the tile back into the padded physical output. Each
            # request's physical span is rounded up to whole-T tiles, so a
            # full-T write is safe even for a partial logical tail tile.
            x_next = pl.assemble(x_next, x_next_tile, [tile_base, 0, 0])
    return x_next


@pl.jit.host
def l3_prefill_layer(
    x_hc: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16],
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
    hca_cmp_kv_state: pl.InOut[pl.Tensor[
        [N_RANKS, PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_cmp_score_state: pl.InOut[pl.Tensor[
        [N_RANKS, PREFILL_HCA_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_HCA_STATE_BLOCKS_DYN], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    csa_cmp_kv_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_cmp_score_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCKS_DYN, CSA_STATE_BLOCK_SIZE, CSA_MAIN_OUT_DIM], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_CSA_STATE_BLOCKS_DYN], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_kv_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_score_state: pl.InOut[
        pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCKS_DYN, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, PREFILL_INNER_STATE_BLOCKS_DYN], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_ORI_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, PREFILL_ORI_BLOCK_TABLE_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_CMP_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, PREFILL_CMP_BLOCK_TABLE_DYN], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, PREFILL_IDX_CACHE_BLOCKS_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
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
    x_next: pl.Out[pl.Tensor[[N_RANKS, PREFILL_TOKENS_DYN, HC_MULT, D], pl.BF16]],
    layer_id: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer(N_RANKS * N_LOCAL * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_aux_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * AUX_PAD * 4)
    recv_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_arrived_buf = pld.alloc_window_buffer(N_RANKS * 4)

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        prefill_layer_core(
            x_hc[rank],
            seq_lens[rank], chunk_lens[rank], chunk_offsets[rank], chunk_tile_offsets[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank], freqs_cos[rank], freqs_sin[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank], hca_cmp_norm_w[rank],
            hca_cmp_kv_state[rank], hca_cmp_score_state[rank], hca_compress_state_block_table[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank], csa_cmp_norm_w[rank],
            csa_cmp_kv_state[rank], csa_cmp_score_state[rank], csa_compress_state_block_table[rank],
            csa_hadamard_idx[rank],
            csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank], csa_weights_proj[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank], csa_inner_norm_w[rank],
            csa_inner_kv_state[rank], csa_inner_score_state[rank],
            csa_inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping[rank],
            cmp_kv[rank], cmp_block_table[rank],
            cmp_sparse_indices[rank], cmp_sparse_lens[rank],
            idx_kv_cache[rank], idx_block_table[rank],
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
            recv_meta, recv_x, recv_aux, recv_route, arrived,
            routed_y_buf, combine_arrived,
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
    "position_ids", "ori_slot_mapping", "cmp_sparse_indices", "cmp_sparse_lens",
    "cmp_slot_mapping", "state_slot_mapping", "idx_slot_mapping", "inner_state_slot_mapping",
}
# Child-local cache/state/table tensors (request-local slices, persist across tiles).
_CACHE_STATE_NAMES = {
    "kv_cache", "block_table", "ori_block_table", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_block_table",
    "cmp_kv_state", "cmp_score_state", "compress_state_block_table",
    "inner_kv_state", "inner_score_state", "inner_compress_state_block_table",
}

# Packed per-request cache/state/table tensors (packed-name -> child-local name,
# or (source-kind, child-local name) for the namespaced state tensors).
_PACKED_CACHE_SPECS = {
    "kv_cache": "kv_cache",
    "ori_block_table": "ori_block_table",
    "cmp_kv": "cmp_kv",
    "cmp_block_table": "cmp_block_table",
    "idx_kv_cache": "idx_kv_cache",
    "idx_block_table": "idx_block_table",
    "hca_cmp_kv_state": ("hca", "cmp_kv_state"),
    "hca_cmp_score_state": ("hca", "cmp_score_state"),
    "hca_compress_state_block_table": ("hca", "compress_state_block_table"),
    "csa_cmp_kv_state": ("csa", "cmp_kv_state"),
    "csa_cmp_score_state": ("csa", "cmp_score_state"),
    "csa_compress_state_block_table": ("csa", "compress_state_block_table"),
    "csa_inner_kv_state": ("csa", "inner_kv_state"),
    "csa_inner_score_state": ("csa", "inner_score_state"),
    "csa_inner_compress_state_block_table": ("csa", "inner_compress_state_block_table"),
}

_HISTORY_CACHE_NAMES = {
    "kv_cache", "cmp_kv", "idx_kv_cache",
    "hca_cmp_kv_state", "hca_cmp_score_state",
    "csa_cmp_kv_state", "csa_cmp_score_state",
    "csa_inner_kv_state", "csa_inner_score_state",
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
    if child_name == "idx_kv_cache":
        return IDX_CACHE_BLOCKS
    if child_name == "idx_block_table":
        return IDX_TABLE_BLOCKS
    if child_name in ("cmp_kv_state", "cmp_score_state"):
        return HCA_STATE_BLOCK_NUM if kind == "hca" else CSA_STATE_BLOCK_NUM
    if child_name == "compress_state_block_table":
        return HCA_STATE_MAX_BLOCKS if kind == "hca" else CSA_STATE_MAX_BLOCKS
    if child_name in ("inner_kv_state", "inner_score_state"):
        return INNER_STATE_BLOCK_NUM
    if child_name == "inner_compress_state_block_table":
        return INNER_STATE_MAX_BLOCKS
    raise KeyError(child_name)


def _child_to_packed(kind, child_name):
    """Map a child-local cache/state name to its packed-buffer name for this kind."""
    if child_name in ("block_table", "ori_block_table"):
        return "ori_block_table"
    if child_name in ("kv_cache", "cmp_kv", "cmp_block_table", "idx_kv_cache", "idx_block_table"):
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
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeek V4 attention compress ratio {ratio} at layer {layer_id}")


def _tile_token_meta(kind, context_len, valid_tok, torch):
    """Child-local [T] token metadata for one tile, via the fixed-T child builder.

    Reuses the existing single-tile builders, which already encode the
    absolute-position ring/overlay/compressed/state coordinate logic. ``context_len``
    is the tile's absolute start position; ``valid_tok`` its active token count.
    """
    from golden import TensorSpec

    specs = {s.name: s for s in _KIND_BUILDER[kind](start_pos=context_len, num_tokens=valid_tok)
             if isinstance(s, TensorSpec)}
    meta = {name: _spec_value(specs[name], torch) for name in specs if name in _TOKEN_META_NAMES}
    if "cmp_sparse_indices" not in meta:
        # CSA ignores cmp_sparse_indices/lens; provide window-only SWA values so the
        # shared packed tensors are well-formed.
        swa = {s.name: s for s in build_swa_attention_tensor_specs(start_pos=context_len, num_tokens=valid_tok)
               if isinstance(s, TensorSpec)}
        meta["cmp_sparse_indices"] = _spec_value(swa["cmp_sparse_indices"], torch)
        meta["cmp_sparse_lens"] = _spec_value(swa["cmp_sparse_lens"], torch)
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


def _packed_token_metadata(kind, seq_lens_v, chunk_lens_v, chunk_offsets_v, total_tokens, torch):
    """Assemble rank-shared padded-physical [total_tokens, ...] metadata tensors."""
    pos = torch.zeros(total_tokens, dtype=torch.int32)
    ori_slot = torch.full((total_tokens,), -1, dtype=torch.int64)
    sparse_idx = torch.full((total_tokens, SPARSE_TOPK), -1, dtype=torch.int32)
    sparse_lens = torch.zeros(total_tokens, dtype=torch.int32)
    hca_cmp = torch.full((total_tokens,), -1, dtype=torch.int64)
    hca_state = torch.full((total_tokens,), -1, dtype=torch.int64)
    csa_cmp = torch.full((total_tokens,), -1, dtype=torch.int64)
    csa_idx = torch.full((total_tokens,), -1, dtype=torch.int64)
    csa_state = torch.full((total_tokens,), -1, dtype=torch.int64)
    csa_inner = torch.full((total_tokens,), -1, dtype=torch.int64)

    for _r, _tid, ctx, valid, base in _iter_request_tiles(seq_lens_v, chunk_lens_v, chunk_offsets_v):
        m = _tile_token_meta(kind, ctx, valid, torch)
        pos[base:base + T] = m["position_ids"][:T]
        ori_slot[base:base + T] = m["ori_slot_mapping"][:T]
        sparse_idx[base:base + T] = m["cmp_sparse_indices"][:T]
        sparse_lens[base:base + T] = m["cmp_sparse_lens"][:T]
        if kind == "hca":
            hca_cmp[base:base + T] = m["cmp_slot_mapping"][:T]
            hca_state[base:base + T] = m["state_slot_mapping"][:T]
        elif kind == "csa":
            csa_cmp[base:base + T] = m["cmp_slot_mapping"][:T]
            csa_idx[base:base + T] = m["idx_slot_mapping"][:T]
            csa_state[base:base + T] = m["state_slot_mapping"][:T]
            csa_inner[base:base + T] = m["inner_state_slot_mapping"][:T]

    return {
        "position_ids": pos,
        "ori_slot_mapping": ori_slot,
        "cmp_sparse_indices": sparse_idx,
        "cmp_sparse_lens": sparse_lens,
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
    seq_lens_v = [start_positions[i] + chunk_lens_v[i] for i in range(len(chunk_lens_v))]
    tile_counts_v = [(c + T - 1) // T for c in chunk_lens_v]
    padded_lens_v = [tc * T for tc in tile_counts_v]
    chunk_offsets_v, acc = [], 0
    for c in padded_lens_v:
        chunk_offsets_v.append(acc)
        acc += c
    total_tokens = acc
    tile_offsets_v, tacc = [], 0
    for tc in tile_counts_v:
        tile_offsets_v.append(tacc)
        tacc += tc
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
    ``start_positions`` the prior context length per request (default 0 = fresh
    prefill, no cache history). Token tensors are physically padded to whole
    ``T`` tiles; cache/state/tables are ``user_batch``-concatenated
    request-local slices.
    """
    import torch
    from golden import ScalarSpec, TensorSpec

    kind = _attention_kind_for_layer(layer_id)
    seq_lens_t, chunk_lens_t, chunk_offsets_t, tile_offsets_t, total_tokens = _resolve_batch(
        chunk_lens, start_positions, torch)
    batch = len(chunk_lens_t)
    seq_lens_list = [int(v) for v in seq_lens_t]
    chunk_lens_list = [int(v) for v in chunk_lens_t]

    def kind_specs(build_fn):
        return {s.name: s for s in build_fn(start_pos=0, num_tokens=T) if isinstance(s, TensorSpec)}

    swa = kind_specs(build_swa_attention_tensor_specs)
    hca = kind_specs(build_hca_attention_tensor_specs)
    csa = kind_specs(build_csa_attention_tensor_specs)
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
                                  [int(c) for c in chunk_offsets_t], total_tokens, torch)
    chunk_offsets_list = [int(c) for c in chunk_offsets_t]

    def init_x_hc():
        x = torch.zeros(N_RANKS, total_tokens, HC_MULT, D, dtype=torch.bfloat16)
        for base, chunk_len in zip(chunk_offsets_list, chunk_lens_list):
            x[:, base:base + chunk_len] = ((torch.rand(N_RANKS, chunk_len, HC_MULT, D) - 0.5) / 10.0).to(
                torch.bfloat16)
        return x

    def init_input_ids():
        ids = torch.zeros(N_RANKS, total_tokens, dtype=torch.int64)
        for rank in range(N_RANKS):
            for base, chunk_len in zip(chunk_offsets_list, chunk_lens_list):
                ids[rank, base:base + chunk_len] = (torch.arange(chunk_len, dtype=torch.int64) + base + rank) % VOCAB
        return ids.contiguous()

    tensor_specs.append(TensorSpec("x_hc", [N_RANKS, total_tokens, HC_MULT, D], torch.bfloat16, init_value=init_x_hc))
    tensor_specs.append(TensorSpec("input_ids", [N_RANKS, total_tokens], torch.int64, init_value=init_input_ids))
    tensor_specs.append(TensorSpec("position_ids", [N_RANKS, total_tokens], torch.int32,
                                   init_value=replicate(meta["position_ids"])))
    tensor_specs.append(TensorSpec("ori_slot_mapping", [N_RANKS, total_tokens], torch.int64,
                                   init_value=replicate(meta["ori_slot_mapping"])))
    tensor_specs.append(TensorSpec("cmp_sparse_indices", [N_RANKS, total_tokens, SPARSE_TOPK], torch.int32,
                                   init_value=replicate(meta["cmp_sparse_indices"])))
    tensor_specs.append(TensorSpec("cmp_sparse_lens", [N_RANKS, total_tokens], torch.int32,
                                   init_value=replicate(meta["cmp_sparse_lens"])))
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
        if cn in ("idx_kv_cache", "idx_block_table"):
            return csa[cn], kind, cn
        return active[cn], kind, cn  # kv_cache

    # Packed per-request cache/state/table tensors. With start_positions=0 the caches
    # and states are zero (no history); the block tables are the child's request-local
    # tables tiled per request. Requests with prior context populate their own history.
    for packed_name, info in _PACKED_CACHE_SPECS.items():
        src, src_kind, child_name = resolve_cache_src(packed_name, info)
        per_req = _spec_value(src, torch)
        is_history = packed_name in _HISTORY_CACHE_NAMES

        def make_init(per_req=per_req, is_history=is_history, src_kind=src_kind, child_name=child_name):
            def init():
                blocks = []
                for r in range(batch):
                    cs = seq_lens_list[r] - chunk_lens_list[r]
                    rspec = None
                    if is_history and cs > 0:
                        rspecs = {s.name: s for s in _KIND_BUILDER[src_kind](start_pos=cs, num_tokens=T)
                                  if isinstance(s, TensorSpec)}
                        rspec = rspecs.get(child_name)
                    blocks.append(_spec_value(rspec, torch) if rspec is not None else per_req.clone())
                packed = torch.cat(blocks, dim=0).contiguous()
                return torch.stack([packed.clone() for _ in range(N_RANKS)], dim=0).contiguous()
            return init

        tensor_specs.append(TensorSpec(packed_name, [N_RANKS, batch * src.shape[0], *src.shape[1:]],
                                       src.dtype, init_value=make_init(),
                                       is_output=src.is_output))

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

    tensor_specs.append(TensorSpec("x_next", [N_RANKS, total_tokens, HC_MULT, D], torch.bfloat16, is_output=True))

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

    For each request, request-local cache/state views are sliced once and reused
    across that request's tiles (so a tile reads what earlier tiles wrote). Each tile
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
            "cmp_kv_state": tensors["hca_cmp_kv_state"], "cmp_score_state": tensors["hca_cmp_score_state"],
            "compress_state_block_table": tensors["hca_compress_state_block_table"],
            "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"], "state_slot_mapping": tensors["hca_state_slot_mapping"],
        })
        attention_golden = golden_prefill_attention_hca
    else:
        mapped.update({
            "cmp_wkv": tensors["csa_cmp_wkv"], "cmp_wgate": tensors["csa_cmp_wgate"],
            "cmp_ape": tensors["csa_cmp_ape"], "cmp_norm_w": tensors["csa_cmp_norm_w"],
            "cmp_kv_state": tensors["csa_cmp_kv_state"], "cmp_score_state": tensors["csa_cmp_score_state"],
            "compress_state_block_table": tensors["csa_compress_state_block_table"],
            "hadamard_idx": tensors["csa_hadamard_idx"], "idx_wq_b": tensors["csa_idx_wq_b"],
            "idx_wq_b_scale": tensors["csa_idx_wq_b_scale"], "idx_weights_proj": tensors["csa_weights_proj"],
            "inner_wkv": tensors["csa_inner_wkv"], "inner_wgate": tensors["csa_inner_wgate"],
            "inner_ape": tensors["csa_inner_ape"], "inner_norm_w": tensors["csa_inner_norm_w"],
            "inner_kv_state": tensors["csa_inner_kv_state"], "inner_score_state": tensors["csa_inner_score_state"],
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

        # Request-local cache/state/table views (mutable; persist across tiles).
        req_views = {}
        for packed_name, info in _PACKED_CACHE_SPECS.items():
            child_name = info[1] if isinstance(info, tuple) else info
            cnt = _req_block_count(kind, child_name)
            req_views[packed_name] = tensors[packed_name][:, request_id * cnt:(request_id + 1) * cnt]

        for tile_id in range(tok_blocks):
            p0 = tile_id * T
            valid = min(T, chunk_len - p0)
            base = chunk_base + p0

            x_attn_tile = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.bfloat16)
            for rank in range(N_RANKS):
                attn_tensors = {}
                for spec in attn_specs:
                    if not isinstance(spec, TensorSpec):
                        continue  # scalar (num_tokens) set explicitly below
                    name = spec.name
                    if name == "x_out":
                        attn_tensors[name] = x_attn_tile[rank]
                    elif name == "x_hc":
                        attn_tensors[name] = tile_buffer(tensors["x_hc"], rank, base, valid, (HC_MULT, D), torch.bfloat16)
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
            x_next_tile = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.bfloat16)
            moe_tensors["x_next"] = x_next_tile
            golden_moe(moe_tensors)

            x_next[:, base:base + valid] = x_next_tile[:, :valid]


def valid_ratio_reldiff(diff_thd, pct_thd):
    """Relative-diff comparator over logical token rows, ignoring padded tails."""
    import torch
    from golden import ratio_reldiff

    base_cmp = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd)

    def cmp(actual, expected, **kwargs):
        inputs = kwargs.get("inputs", {})
        chunk_lens = inputs.get("chunk_lens")
        chunk_offsets = inputs.get("chunk_offsets")
        if chunk_lens is None or chunk_offsets is None:
            return base_cmp(actual, expected, **kwargs)

        lens = chunk_lens[0].cpu().to(torch.int64)
        offsets = chunk_offsets[0].cpu().to(torch.int64)
        mask = torch.zeros(actual.shape[1], dtype=torch.bool)
        for chunk_len, base in zip(lens.tolist(), offsets.tolist()):
            mask[base:base + chunk_len] = True
        return base_cmp(actual[:, mask], expected[:, mask], **kwargs)

    cmp.__name__ = "valid_ratio_reldiff"
    return cmp


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit

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
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"
    chunk_lens = tuple(int(x) for x in args.chunk_lens.split(","))
    start_positions = None if args.start_positions is None else tuple(int(x) for x in args.start_positions.split(","))

    result = run_jit(
        fn=l3_prefill_layer,
        specs=build_tensor_specs(
            layer_id=args.layer_id,
            chunk_lens=chunk_lens,
            start_positions=start_positions,
        ),
        golden_fn=golden_prefill_layer,
        compile_only=args.compile_only,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # Real-weight x_next over-thd fractions (frac>5e-3 / frac>1e-2):
            # --chunk-lens 128,192
            # swa(L0) 0.15% / 0.0006%, hca(L9) 1.1% / 0.45%, csa(L8) 3.89% / 0.63%.
            "x_next": valid_ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
