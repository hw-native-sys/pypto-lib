# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 CSA (Compressed Sparse Attention) decode orchestration.

This standalone harness targets the ratio-4 compression step used by the current
workflow checkpoint. It composes:

- hc_pre
- qkv_proj_rope
- main compressor (ratio=4, rotate=False)
- inner compressor (ratio=4, rotate=True)
- indexer
- sparse_attn_csa (with fused grouped o_proj)
- hc_post

The helper stack in this repo has already moved to the refreshed v4 contracts:
q_proj runs through the W8A8 path, sparse_attn_csa owns grouped o_proj, and the
indexer consumes a prepared `idx_kv_cache` instead of owning the inner
compressor itself. This file aligns to that stack instead of the older draft
surface.
"""


import sys

import config


_TP_CHOICES = (1, 2, 4)
_TP_DEFAULT = 2


def _parse_tp_argv():
    for index, arg in enumerate(sys.argv):
        if arg == "--tp" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--tp="):
            return int(arg.split("=", 1)[1])
    return _TP_DEFAULT


TP_SIZE = _parse_tp_argv()
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")
config.TP = TP_SIZE

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_PAGES_PER_REQUEST,
    CSA_STATE_ROWS_PER_REQUEST,
    CSA_INNER_STATE_BLOCK_SIZE,
    CSA_INNER_STATE_PAGES_PER_REQUEST,
    CSA_INNER_STATE_ROWS_PER_REQUEST,
    CSA_TOPK,
    CSA_MAX_QUERIES,
    CSA_CANDIDATES_PER_LEAF,
    CSA_TOPK_INVALID_TASK_SLOT,
    KV_ORI_BLOCK_NUM,
    KV_CMP_BLOCK_NUM,
    KV_ORI_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    SWA_SOURCE_OVERLAY_BASE,
)
from decode_compressor_ratio4 import compressor_ratio4
from decode_indexer_compressor import indexer_compressor
from decode_indexer import T_PAD, T_PAD_DYN, indexer
from decode_metadata import (
    PHASE_D_LEAF_FIELDS,
    PHASE_D_PAIR_FIELDS,
    PHASE_D_ROOT_FIELDS,
    PHASE_D_SINGLETON_FIELDS,
    PHASE_D_UPPER_FIELDS,
)
from decode_sparse_attn_csa import (
    T_PAD as CSA_HEAD_T_PAD,
    sparse_attn_csa,
    sparse_attn_csa_heads,
)
from decode_o_proj import (
    ATTENTION_WINDOW_ROWS,
    GROUP_T_PAD,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    LOCAL_T,
    LOCAL_T_PAD,
    O_WINDOW_ROWS,
    decode_o_proj,
    o_group_a2a,
    o_proj_reduce_scatter,
)
from hc_post import hc_post
from hc_pre import hc_pre
from qkv_proj_rope import qkv_proj_rope
from rope_interleave import _rope_interleave_active_body
from rmsnorm import rms_norm


# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")
T_DYN = pl.dynamic("T_DYN")
ROPE_ROWS_DYN = pl.dynamic("DCSA_ROPE_ROWS_DYN")
EVT_DYN = pl.dynamic("DCSA_EVENT_DYN")
MAIN_STATE_DYN = pl.dynamic("DCSA_MAIN_STATE_DYN")
INNER_STATE_DYN = pl.dynamic("DCSA_INNER_STATE_DYN")
MAIN_CACHE_DYN = pl.dynamic("DCSA_MAIN_CACHE_DYN")
IDX_ROWS_DYN = pl.dynamic("DCSA_IDX_ROWS_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("DCSA_ORI_BLOCK_NUM_DYN")
# Forest descriptors are chunk-major with a fixed 16-query task-array bound;
# the per-chunk ragged count is bound at axis 1. The global descriptors
# (query IDs, pages, page offsets, windows, epochs, RoPE) are full-active and
# reused per chunk inside the indexer.
LEAF_DYN = pl.dynamic("DCSA_LEAF_DYN")
PAIR_DYN = pl.dynamic("DCSA_PAIR_DYN")
SINGLETON_DYN = pl.dynamic("DCSA_SINGLETON_DYN")
UPPER_DYN = pl.dynamic("DCSA_UPPER_DYN")
GLOB_PAGE_DYN = pl.dynamic("DCSA_GLOB_PAGE_DYN")
GLOB_REQOFF_DYN = pl.dynamic("DCSA_GLOB_REQOFF_DYN")
# Indexer global ragged page/offset pools (separate physical pool from the
# sparse-attention csa_pages; both share the same logical candidate range).
IDX_PAGE_DYN = pl.dynamic("DCSA_IDX_PAGE_DYN")
IDX_REQOFF_DYN = pl.dynamic("DCSA_IDX_REQOFF_DYN")
# TP output wrapper uses the sparse module's dynamic cache/page symbols.
CSA_CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
CSA_PAGE_DYN = pl.dynamic("PAGE_DYN")
CSA_REQUEST_OFFSET_DYN = pl.dynamic("REQUEST_OFFSET_DYN")

# model config
B = DECODE_BATCH // config.DP
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local
COMPRESS_RATIO = 4
COFF = 2  # 1 + int(COMPRESS_RATIO == 4)
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM

# The chunk loop and fixed [4080, 1024] arena live inside
# decode_indexer.py::indexer; attention_csa only passes the
# chunk-major forest descriptors and per-chunk counts through.
CSA_INDEXER_CHUNK_T = CSA_MAX_QUERIES
CSA_INDEXER_MAX_CHUNKS = T // CSA_INDEXER_CHUNK_T
assert T % CSA_INDEXER_CHUNK_T == 0
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
RAW_COMMIT_TILE = 8
TP_FIXTURE_OUTPUT_SENTINEL = -7.0


@pl.jit
def decode_csa(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    current_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    swa_sources: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    csa_pages: pl.Tensor[[CSA_PAGE_DYN, 2], pl.INT32],
    csa_page_offsets: pl.Tensor[[CSA_REQUEST_OFFSET_DYN], pl.INT32],
    csa_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    csa_request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    idx_topk: pl.Tensor[[T_DYN, CSA_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    o_local: pl.InOut[pl.Tensor[[LOCAL_T_PAD, D], pl.BF16]],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Run rank-local CSA heads, A2A, sharded O projection, and RS."""
    q.bind_dynamic(0, T_DYN)
    ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    current_kv.bind_dynamic(0, T_DYN)
    swa_sources.bind_dynamic(0, T_DYN)
    cmp_kv.bind_dynamic(0, CSA_CMP_BLOCK_NUM_DYN)
    query_request_ids.bind_dynamic(0, T_DYN)
    csa_pages.bind_dynamic(0, CSA_PAGE_DYN)
    csa_page_offsets.bind_dynamic(0, CSA_REQUEST_OFFSET_DYN)
    csa_windows.bind_dynamic(0, B_DYN)
    csa_request_epochs.bind_dynamic(0, B_DYN)
    idx_topk.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)

    sparse_grouped = pl.create_tensor(
        [O_GROUPS * CSA_HEAD_T_PAD, O_GROUP_IN], dtype=pl.BF16
    )
    ready_dep = pl.system.task_dummy(deps=[])
    sparse_grouped, _heads_dep = sparse_attn_csa_heads(
        q, ori_kv, current_kv, swa_sources,
        cmp_kv, query_request_ids,
        csa_pages, csa_page_offsets, csa_windows, csa_request_epochs,
        idx_topk, attn_sink, freqs_cos, freqs_sin,
        sparse_grouped, ready_dep,
    )
    packed_grouped = pl.create_tensor(
        [O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], dtype=pl.BF16
    )
    with pl.spmd(
        O_GROUPS * LOCAL_T_PAD,
        name_hint="csa_pack_grouped_heads",
        deps=[_heads_dep],
    ) as _pack_tid:
        packed_row = pl.tile.get_block_idx()
        packed_group = packed_row // LOCAL_T_PAD
        packed_token = packed_row - packed_group * LOCAL_T_PAD
        if packed_token < local_t:
            sparse_row = packed_group * CSA_HEAD_T_PAD + packed_token
            packed_grouped[packed_row : packed_row + 1, :] = sparse_grouped[
                sparse_row : sparse_row + 1, :
            ]

    attention_local_flat = pl.create_tensor(
        [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16
    )
    attention_local_flat, attention_signal = o_group_a2a(
        packed_grouped, attention_local_flat,
        attention_window, attention_signal,
        group_base, tp_rank, local_t,
    )

    attention_local_groups = pl.reshape(
        attention_local_flat, [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN]
    )
    o_partial = pl.create_tensor([GROUP_T_PAD, D], dtype=pl.FP32)
    o_partial, projection_tid = decode_o_proj(
        attention_local_groups, wo_a, wo_b, wo_b_scale, local_t, o_partial
    )
    o_local, o_signal = o_proj_reduce_scatter(
        o_partial, o_local, o_window, o_signal,
        group_base, tp_rank, local_t, projection_tid,
    )
    return o_local, attention_signal, o_signal


@pl.jit.host
def l3_decode_csa(
    q: pl.Tensor[[TP_SIZE, T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    current_kv: pl.Tensor[[TP_SIZE, T_DYN, HEAD_DIM], pl.BF16],
    swa_sources: pl.Tensor[[TP_SIZE, T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[TP_SIZE, CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    csa_pages: pl.Tensor[[TP_SIZE, CSA_PAGE_DYN, 2], pl.INT32],
    csa_page_offsets: pl.Tensor[[TP_SIZE, CSA_REQUEST_OFFSET_DYN], pl.INT32],
    csa_windows: pl.Tensor[[TP_SIZE, B_DYN, 3], pl.INT32],
    csa_request_epochs: pl.Tensor[[TP_SIZE, B_DYN], pl.INT32],
    idx_topk: pl.Tensor[[TP_SIZE, T_DYN, CSA_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    freqs_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    o_local: pl.InOut[pl.Tensor[[TP_SIZE, LOCAL_T_PAD, D], pl.BF16]],
    local_t: pl.Scalar[pl.INT32],
):
    """Submit exactly one rank-local CSA L2 per physical TP rank."""
    q.bind_dynamic(1, T_DYN)
    ori_kv.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    current_kv.bind_dynamic(1, T_DYN)
    swa_sources.bind_dynamic(1, T_DYN)
    cmp_kv.bind_dynamic(1, CSA_CMP_BLOCK_NUM_DYN)
    query_request_ids.bind_dynamic(1, T_DYN)
    csa_pages.bind_dynamic(1, CSA_PAGE_DYN)
    csa_page_offsets.bind_dynamic(1, CSA_REQUEST_OFFSET_DYN)
    csa_windows.bind_dynamic(1, B_DYN)
    csa_request_epochs.bind_dynamic(1, B_DYN)
    idx_topk.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer(
        [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16
    )
    attention_signal_buf = pld.alloc_window_buffer(
        [TP_SIZE, 1], dtype=pl.INT32
    )
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(
            attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16
        )
        attention_signal = pld.window(
            attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32
        )
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        decode_csa(
            q[rank], ori_kv[rank], current_kv[rank], swa_sources[rank],
            cmp_kv[rank], query_request_ids[rank],
            csa_pages[rank], csa_page_offsets[rank], csa_windows[rank],
            csa_request_epochs[rank], idx_topk[rank], attn_sink[rank],
            freqs_cos[rank], freqs_sin[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank], o_local[rank],
            attention_window, attention_signal, o_window, o_signal,
            0, rank, local_t, device=rank,
        )


@pl.jit.inline(auto_scope=False)
def attention_csa(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    main_state: pl.Tensor[
        [MAIN_STATE_DYN, CSA_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32
    ],
    main_state_page_ids: pl.Tensor[
        [B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    main_state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    main_state_page_epochs: pl.Tensor[
        [B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    compressor_request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    event_query_ids: pl.Tensor[[EVT_DYN], pl.INT32],
    main_event_write_slots: pl.Tensor[[EVT_DYN], pl.INT64],
    main_cache: pl.Tensor[
        [MAIN_CACHE_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ],
    main_state_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_state: pl.Tensor[
        [INNER_STATE_DYN, CSA_INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32
    ],
    inner_state_page_ids: pl.Tensor[
        [B_DYN, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    inner_state_page_epochs: pl.Tensor[
        [B_DYN, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_event_write_slots: pl.Tensor[[EVT_DYN], pl.INT64],
    inner_state_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    idx_kv_cache_flat: pl.Tensor[[IDX_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale_flat: pl.Tensor[[IDX_ROWS_DYN, 1], pl.FP32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    idx_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    query_vectors: pl.InOut[
        pl.Tensor[[T_PAD_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8]
    ],
    query_scales: pl.InOut[pl.Tensor[[T_PAD_DYN, IDX_N_HEADS], pl.FP32]],
    query_weights: pl.InOut[pl.Tensor[[T_PAD_DYN, IDX_N_HEADS], pl.FP32]],
    query_request_ids_padded: pl.Tensor[[T_PAD_DYN], pl.INT32],
    cos: pl.Tensor[[ROPE_ROWS_DYN, HALF_ROPE], pl.FP32],
    sin: pl.Tensor[[ROPE_ROWS_DYN, HALF_ROPE], pl.FP32],
    idx_pages: pl.Tensor[[IDX_PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[IDX_REQOFF_DYN], pl.INT32],
    idx_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    idx_request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32
    ],
    pair_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, PAIR_DYN, PHASE_D_PAIR_FIELDS], pl.INT32
    ],
    singleton_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS],
        pl.INT32,
    ],
    upper_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, UPPER_DYN, PHASE_D_UPPER_FIELDS], pl.INT32
    ],
    root_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, CSA_INDEXER_CHUNK_T, PHASE_D_ROOT_FIELDS],
        pl.INT32,
    ],
    pair_group_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    singleton_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    upper_merge_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    raw_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    swa_sources: pl.Tensor[[T_DYN, WIN], pl.INT32],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    csa_pages: pl.Tensor[[GLOB_PAGE_DYN, 2], pl.INT32],
    csa_page_offsets: pl.Tensor[[GLOB_REQOFF_DYN], pl.INT32],
    csa_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    csa_request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    """Compose one CSA attention step over the ragged cache ABI.

    HC-pre, RMSNorm, QKV projection, the main/inner compressors, the exact
    Top-512 forest, sparse value attention, and HC-post each run once over
    full active T.  Only the exact forest is micro-chunked into 16-query
    batches inside ``decode_indexer.py::indexer``. Token-local RoPE is derived
    once here via ``_rope_interleave_active_body`` and reused by the
    compressors. Raw KV commit is delayed until sparse attention finishes.
    """
    t_dim = pl.tensor.dim(x_hc, 0)

    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post_t,
        comb_t,
    )

    # Token-local + event-local interleaved/sign-folded RoPE, derived once
    # from the half-width cos/sin rows. The spec appends event-local rows
    # (at compression starts 4r) after the token-local rows, so the table
    # has the same row count as cos/sin (bound to ROPE_ROWS_DYN =
    # t_dim + event_count).
    # The indexer consumes the first t_dim rows; the compressors consume the
    # event-local rows via a t_dim offset (see csa_event_rope below).
    event_count = pl.tensor.dim(event_query_ids, 0)
    rope_rows = pl.tensor.dim(cos, 0)
    cos_il = pl.create_tensor([rope_rows, ROPE_HEAD_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([rope_rows, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_completion = pl.array.create(1, pl.TASK_ID)
    _rope_interleave_active_body(cos, sin, cos_il, sin_signed, rope_completion)
    # ``cos``/``sin`` also carry the event-local rows consumed by the
    # compressors.  The indexer ABI is token-local and must receive a view
    # whose first axis is exactly the active token count; passing the full
    # token+event table makes the stricter frontend bind two dynamic extents
    # to the same indexer symbol.
    token_cos_il = pl.slice(
        cos_il, [t_dim, ROPE_HEAD_DIM], [0, 0]
    )
    token_sin_signed = pl.slice(
        sin_signed, [t_dim, ROPE_HEAD_DIM], [0, 0]
    )

    x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_t)
    # Keep the token-local RoPE producer in the same scalar dependency chain as
    # the existing RMSNorm fence. Passing this stable scalar through the
    # inlined indexer avoids leaking an array-element SSA value across scopes.
    late_dep = pl.system.task_dummy(deps=[rms_tid, rope_completion[0]])

    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_done = qkv_proj_rope(
        x_normed_t,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        freqs_cos,
        freqs_sin,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        late_dep,
    )

    event_cos_il = pl.create_tensor([event_count, ROPE_HEAD_DIM], dtype=pl.FP32)
    event_sin_signed = pl.create_tensor(
        [event_count, ROPE_HEAD_DIM], dtype=pl.FP32
    )
    # Derive event-local RoPE from the interleaved cos_il/sin_signed table.
    # The event-local rows (at compression starts 4r) were appended after the
    # token-local rows in the spec, so they live at offset t_dim.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="csa_event_rope",
        deps=[late_dep],
        allow_early_resolve=True,
    ) as event_rope_tid:
        for event in pl.range(event_count):
            src = t_dim + event
            event_cos_il[event : event + 1, :] = cos_il[src : src + 1, :]
            event_sin_signed[event : event + 1, :] = sin_signed[src : src + 1, :]
    compressor_dep = pl.system.task_dummy(deps=[late_dep, event_rope_tid])

    main_overlay = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    main_cache_done, main_state_done = compressor_ratio4(
        x_normed_t,
        main_overlay,
        main_state,
        main_state_page_ids,
        main_state_valid_ranges,
        main_state_page_epochs,
        compressor_request_epochs,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        event_query_ids,
        event_cos_il,
        event_sin_signed,
        main_cache,
        main_event_write_slots,
        main_state_write_slots,
        compressor_dep,
    )

    inner_overlay = pl.create_tensor([t_dim, IDX_HEAD_DIM], dtype=pl.FP32)
    inner_cache_done, inner_state_done = indexer_compressor(
        x_normed_t,
        inner_overlay,
        inner_state,
        inner_state_page_ids,
        inner_state_valid_ranges,
        inner_state_page_epochs,
        compressor_request_epochs,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        event_query_ids,
        event_cos_il,
        event_sin_signed,
        inner_hadamard,
        idx_kv_cache_flat,
        idx_kv_scale_flat,
        inner_event_write_slots,
        inner_state_write_slots,
        compressor_dep,
    )
    inner_commit_done = pl.system.task_dummy(
        deps=[inner_cache_done, inner_state_done]
    )


    # The indexer owns the only 16-query micro-chunk boundary.  The CSA
    # wrapper passes the full active token set exactly once; no query projection
    # or HC/QKV/compressor work is repeated per chunk.  The InOut query tensors
    # are padded to a CSA_INDEXER_CHUNK_T multiple so the forest's fixed 16-row
    # pl.slice views never read out-of-bounds memory in the final partial
    # chunk (e.g. batch=1/S=8: t_dim=8 -> padded_dim=16).  Input tensors
    # (x_normed_t, qr, etc.) stay at t_dim; the projection SPMD keys on
    # ``query_count`` from x.dim(0).  Padded projection rows are not required
    # to carry a value: the padded descriptor prefixes contain no work for
    # those rows, and the commit guard never publishes them.
    # These buffers are direct entry-point arguments rather than local
    # temporaries: the strict inline specializer needs concrete metadata for
    # the indexer's dynamic T_PAD_DYN bindings.  They are allocated at the
    # compile-time upper bound and only the active query prefix is submitted.
    indexer_ready = pl.system.task_dummy(deps=[late_dep, qkv_done])
    topk_indices = pl.create_tensor([t_dim, CSA_TOPK], dtype=pl.INT32)
    index_completion = pl.array.create(1, pl.TASK_ID)
    indexer(
        x_normed_t, qr, qr_scale, idx_wq_b, idx_wq_b_scale, idx_weights_proj,
        token_cos_il, token_sin_signed, idx_hadamard, query_vectors, query_scales,
        query_weights, idx_kv_cache_flat, idx_kv_scale_flat,
        query_request_ids_padded, idx_pages, idx_page_offsets, idx_windows,
        idx_request_epochs, leaf_descriptors, pair_descriptors,
        singleton_descriptors, upper_descriptors, root_descriptors, topk_indices,
        pair_group_chunk_counts, singleton_chunk_counts,
        upper_merge_chunk_counts, inner_commit_done,
        indexer_ready, pl.cast(0, target_type=pl.INDEX), indexer_ready,
        index_completion,
    )
    sparse_ready = pl.system.task_dummy(
        deps=[main_cache_done, inner_commit_done, index_completion[0], late_dep, qkv_done]
    )

    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    value_done = sparse_attn_csa(
        q,
        kv_cache,
        kv,
        swa_sources,
        main_cache,
        query_request_ids,
        csa_pages,
        csa_page_offsets,
        csa_windows,
        csa_request_epochs,
        topk_indices,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        sparse_ready,
    )

    raw_blocks = pl.tensor.dim(kv_cache, 0)
    raw_flat = pl.reshape(kv_cache, [raw_blocks * BLOCK_SIZE, HEAD_DIM])
    with pl.spmd(
        t_dim // RAW_COMMIT_TILE,
        name_hint="csa_dcsa_raw_commit",
        deps=[value_done, main_state_done],
    ) as raw_commit_tid:
        token0 = pl.tile.get_block_idx() * RAW_COMMIT_TILE
        for local_token in pl.range(RAW_COMMIT_TILE):
            token = token0 + local_token
            row_i64 = pl.read(raw_write_slots, [token])
            if row_i64 >= 0:
                if row_i64 < raw_blocks * BLOCK_SIZE:
                    row = pl.cast(row_i64, target_type=pl.INDEX)
                    raw_flat[row : row + 1, :] = kv[token : token + 1, :]

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit(auto_scope=False)
def attention_csa_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    main_state: pl.InOut[
        pl.Tensor[[MAIN_STATE_DYN, CSA_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32]
    ],
    main_state_page_ids: pl.Tensor[
        [B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    main_state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    main_state_page_epochs: pl.Tensor[
        [B_DYN, CSA_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    compressor_request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    event_query_ids: pl.Tensor[[EVT_DYN], pl.INT32],
    main_event_write_slots: pl.Tensor[[EVT_DYN], pl.INT64],
    main_cache: pl.InOut[
        pl.Tensor[[MAIN_CACHE_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    main_state_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_state: pl.InOut[
        pl.Tensor[
            [INNER_STATE_DYN, CSA_INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_state_page_ids: pl.Tensor[
        [B_DYN, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_state_valid_ranges: pl.Tensor[[B_DYN, 2], pl.INT32],
    inner_state_page_epochs: pl.Tensor[
        [B_DYN, CSA_INNER_STATE_PAGES_PER_REQUEST], pl.INT32
    ],
    inner_event_write_slots: pl.Tensor[[EVT_DYN], pl.INT64],
    inner_state_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    idx_kv_cache_flat: pl.InOut[
        pl.Tensor[[IDX_ROWS_DYN, IDX_HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale_flat: pl.InOut[pl.Tensor[[IDX_ROWS_DYN, 1], pl.FP32]],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    idx_hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    query_vectors: pl.InOut[
        pl.Tensor[[T_PAD_DYN, IDX_N_HEADS, IDX_HEAD_DIM], pl.INT8]
    ],
    query_scales: pl.InOut[pl.Tensor[[T_PAD_DYN, IDX_N_HEADS], pl.FP32]],
    query_weights: pl.InOut[pl.Tensor[[T_PAD_DYN, IDX_N_HEADS], pl.FP32]],
    query_request_ids_padded: pl.Tensor[[T_PAD_DYN], pl.INT32],
    cos: pl.Tensor[[ROPE_ROWS_DYN, HALF_ROPE], pl.FP32],
    sin: pl.Tensor[[ROPE_ROWS_DYN, HALF_ROPE], pl.FP32],
    idx_pages: pl.Tensor[[IDX_PAGE_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[IDX_REQOFF_DYN], pl.INT32],
    idx_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    idx_request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    leaf_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, LEAF_DYN, PHASE_D_LEAF_FIELDS], pl.INT32
    ],
    pair_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, PAIR_DYN, PHASE_D_PAIR_FIELDS], pl.INT32
    ],
    singleton_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, SINGLETON_DYN, PHASE_D_SINGLETON_FIELDS],
        pl.INT32,
    ],
    upper_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, UPPER_DYN, PHASE_D_UPPER_FIELDS], pl.INT32
    ],
    root_descriptors: pl.Tensor[
        [CSA_INDEXER_MAX_CHUNKS, CSA_INDEXER_CHUNK_T, PHASE_D_ROOT_FIELDS],
        pl.INT32,
    ],
    pair_group_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    singleton_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    upper_merge_chunk_counts: pl.Tensor[[CSA_INDEXER_MAX_CHUNKS], pl.INT32],
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    raw_write_slots: pl.Tensor[[T_DYN], pl.INT64],
    swa_sources: pl.Tensor[[T_DYN, WIN], pl.INT32],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    csa_pages: pl.Tensor[[GLOB_PAGE_DYN, 2], pl.INT32],
    csa_page_offsets: pl.Tensor[[GLOB_REQOFF_DYN], pl.INT32],
    csa_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    csa_request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Standalone compile entry: bind dynamics and run the full pipeline."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, ROPE_ROWS_DYN)
    sin.bind_dynamic(0, ROPE_ROWS_DYN)
    main_state_write_slots.bind_dynamic(0, T_DYN)
    inner_state_write_slots.bind_dynamic(0, T_DYN)
    raw_write_slots.bind_dynamic(0, T_DYN)
    swa_sources.bind_dynamic(0, T_DYN)
    query_request_ids.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)
    main_state.bind_dynamic(0, MAIN_STATE_DYN)
    main_state_page_ids.bind_dynamic(0, B_DYN)
    main_state_valid_ranges.bind_dynamic(0, B_DYN)
    main_state_page_epochs.bind_dynamic(0, B_DYN)
    compressor_request_epochs.bind_dynamic(0, B_DYN)
    event_query_ids.bind_dynamic(0, EVT_DYN)
    main_event_write_slots.bind_dynamic(0, EVT_DYN)
    main_cache.bind_dynamic(0, MAIN_CACHE_DYN)
    inner_state.bind_dynamic(0, INNER_STATE_DYN)
    inner_state_page_ids.bind_dynamic(0, B_DYN)
    inner_state_valid_ranges.bind_dynamic(0, B_DYN)
    inner_state_page_epochs.bind_dynamic(0, B_DYN)
    inner_event_write_slots.bind_dynamic(0, EVT_DYN)
    idx_kv_cache_flat.bind_dynamic(0, IDX_ROWS_DYN)
    idx_kv_scale_flat.bind_dynamic(0, IDX_ROWS_DYN)
    query_vectors.bind_dynamic(0, T_PAD_DYN)
    query_scales.bind_dynamic(0, T_PAD_DYN)
    query_weights.bind_dynamic(0, T_PAD_DYN)
    query_request_ids_padded.bind_dynamic(0, T_PAD_DYN)
    idx_pages.bind_dynamic(0, IDX_PAGE_DYN)
    idx_page_offsets.bind_dynamic(0, IDX_REQOFF_DYN)
    idx_windows.bind_dynamic(0, B_DYN)
    idx_request_epochs.bind_dynamic(0, B_DYN)
    leaf_descriptors.bind_dynamic(1, LEAF_DYN)
    pair_descriptors.bind_dynamic(1, PAIR_DYN)
    singleton_descriptors.bind_dynamic(1, SINGLETON_DYN)
    upper_descriptors.bind_dynamic(1, UPPER_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    csa_pages.bind_dynamic(0, GLOB_PAGE_DYN)
    csa_page_offsets.bind_dynamic(0, GLOB_REQOFF_DYN)
    csa_windows.bind_dynamic(0, B_DYN)
    csa_request_epochs.bind_dynamic(0, B_DYN)

    # ``attention_csa`` opts out of compiler-inserted scopes because its exact
    # Top-512 indexer owns nested per-chunk scopes. Keep the baseline's single
    # attention scope at the executable boundary so ordinary tensor producer /
    # consumer edges (notably output projection -> HC-post) remain automatic.
    with pl.scope():
        attention_csa(
            x_hc,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin,
            cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
            main_state, main_state_page_ids, main_state_valid_ranges,
            main_state_page_epochs, compressor_request_epochs,
            event_query_ids,
            main_event_write_slots, main_cache, main_state_write_slots,
            inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_hadamard,
            inner_state, inner_state_page_ids, inner_state_valid_ranges,
            inner_state_page_epochs, inner_event_write_slots,
            inner_state_write_slots,
            idx_kv_cache_flat, idx_kv_scale_flat,
            idx_wq_b, idx_wq_b_scale, idx_weights_proj, idx_hadamard,
            query_vectors, query_scales, query_weights,
            query_request_ids_padded,
            cos, sin,
            idx_pages, idx_page_offsets, idx_windows,
            idx_request_epochs,
            leaf_descriptors, pair_descriptors, singleton_descriptors,
            upper_descriptors, root_descriptors,
            pair_group_chunk_counts, singleton_chunk_counts,
            upper_merge_chunk_counts,
            kv_cache, raw_write_slots, swa_sources, query_request_ids,
            csa_pages, csa_page_offsets, csa_windows, csa_request_epochs,
            attn_sink, wo_a, wo_b, wo_b_scale,
            x_out,
        )
    return x_out


def golden_attention_csa(tensors):
    """Torch reference: compose the component goldens over the ragged ABI."""
    import torch

    from decode_compressor_ratio4 import golden_compressor as golden_main
    from decode_indexer import golden_indexer
    from decode_indexer_compressor import (
        golden_compressor as golden_inner,
    )
    from decode_sparse_attn_csa import golden_sparse_attn
    from hc_post import golden_hc_post
    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm

    tokens = tensors["x_hc"].shape[0]
    batch = tokens // S
    positions = (
        tensors["main_state_valid_ranges"][:, 1].to(torch.int32)[:, None]
        + torch.arange(S, dtype=torch.int32)[None, :]
    ).reshape(-1)

    x_mixed = torch.zeros(tokens, D, dtype=torch.bfloat16)
    post = torch.zeros(tokens, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(tokens, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })

    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])

    rope_cos = tensors["freqs_cos"]
    rope_sin = tensors["freqs_sin"]
    q = torch.zeros(tokens, H, HEAD_DIM, dtype=torch.bfloat16)
    current_kv = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
    qr_i8 = torch.zeros(tokens, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(tokens, 1, dtype=torch.float32)
    golden_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": current_kv,
        "qr": qr_i8,
        "qr_scale": qr_scale,
    })

    event_query_ids = tensors["event_query_ids"].long()
    # Event-local RoPE at compression starts (4r). The spec appends these
    # half-width rows after the token-local rows in tensors["cos"]/["sin"],
    # so they live at offset t_dim. The golden compressors index by event
    # number, matching the device's event_cos_il/sin_signed row order.
    event_rope_cos = tensors["cos"][tokens:].float().clone()
    event_rope_sin = tensors["sin"][tokens:].float().clone()

    # Both compressors share one event set and reuse event-local RoPE rows.
    main_overlay = torch.zeros(tokens, HEAD_DIM, dtype=torch.float32)
    golden_main({
        "x": x_normed,
        "kv": main_overlay,
        "compress_state": tensors["main_state"],
        "state_page_ids": tensors["main_state_page_ids"],
        "state_valid_ranges": tensors["main_state_valid_ranges"],
        "state_page_epochs": tensors["main_state_page_epochs"],
        "request_epochs": tensors["compressor_request_epochs"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "event_query_ids": tensors["event_query_ids"],
        "cos": event_rope_cos,
        "sin": event_rope_sin,
        "cmp_kv_cache": tensors["main_cache"],
        "event_write_slots": tensors["main_event_write_slots"],
        "position_ids": positions,
        "state_slot_mapping": tensors["main_state_write_slots"],
    })

    inner_overlay = torch.zeros(tokens, IDX_HEAD_DIM, dtype=torch.float32)
    golden_inner({
        "x": x_normed,
        "kv": inner_overlay,
        "compress_state": tensors["inner_state"],
        "state_page_ids": tensors["inner_state_page_ids"],
        "state_valid_ranges": tensors["inner_state_valid_ranges"],
        "state_page_epochs": tensors["inner_state_page_epochs"],
        "request_epochs": tensors["compressor_request_epochs"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "event_query_ids": tensors["event_query_ids"],
        "cos": event_rope_cos,
        "sin": event_rope_sin,
        "hadamard": tensors["inner_hadamard"],
        "idx_kv_cache_flat": tensors["idx_kv_cache_flat"],
        "idx_kv_scale_flat": tensors["idx_kv_scale_flat"],
        "event_write_slots": tensors["inner_event_write_slots"],
        "position_ids": positions,
        "state_slot_mapping": tensors["inner_state_write_slots"],
    })

    # Exact packed forest: golden_indexer mirrors the projected-query and
    # paged-cache score path, then iterates the chunk-major leaf descriptors.
    # Only ``topk_indices`` is a device output (single-Out workaround for the
    # PyPTO phi-swap bug); ``topk_scores`` is no longer published.
    topk_indices = torch.zeros(tokens, CSA_TOPK, dtype=torch.int32)
    golden_indexer({
        "x": x_normed,
        "qr": qr_i8,
        "qr_scale": qr_scale,
        "wq_b": tensors["idx_wq_b"],
        "wq_b_scale": tensors["idx_wq_b_scale"],
        "weights_proj": tensors["idx_weights_proj"],
        "cos": tensors["cos"],
        "sin": tensors["sin"],
        "hadamard": tensors["idx_hadamard"],
        "idx_kv_cache_flat": tensors["idx_kv_cache_flat"],
        "idx_kv_scale_flat": tensors["idx_kv_scale_flat"],
        "query_request_ids": tensors["query_request_ids"],
        "idx_pages": tensors["idx_pages"],
        "idx_page_offsets": tensors["idx_page_offsets"],
        "idx_windows": tensors["idx_windows"],
        "request_epochs": tensors["idx_request_epochs"],
        "topk_indices": topk_indices,
        "leaf_descriptors": tensors["leaf_descriptors"],
    })


    attn_out = torch.zeros(tokens, D, dtype=torch.bfloat16)
    golden_sparse_attn({
        "q": q,
        "ori_kv": tensors["kv_cache"],
        "current_kv": current_kv,
        "swa_sources": tensors["swa_sources"],
        "cmp_kv": tensors["main_cache"],
        "query_request_ids": tensors["query_request_ids"],
        "csa_pages": tensors["csa_pages"],
        "csa_page_offsets": tensors["csa_page_offsets"],
        "csa_windows": tensors["csa_windows"],
        "request_epochs": tensors["csa_request_epochs"],
        "idx_topk": topk_indices,
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos,
        "freqs_sin": rope_sin,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # Delayed raw-KV ring commit: strictly after the sparse gather, matching
    # the device's deps=[value_done] ordering.
    kv_cache = tensors["kv_cache"]
    raw_write_slots = tensors["raw_write_slots"].to(torch.int64)
    for t in range(tokens):
        slot = int(raw_write_slots[t].item())
        if slot >= 0:
            kv_cache[slot // BLOCK_SIZE, slot % BLOCK_SIZE, 0] = current_kv[t]

    y = torch.zeros(tokens, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post,
        "comb": comb,
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tp_tensor_specs(local_t, start_pos=None):
    """Build TP output-path fixtures for one runtime context position.

    The TP wrapper consumes already-produced q/KV/indexer rows.  The fixture
    therefore keeps the same paged/overlay ABI as the full CSA path while
    varying only the visible decode position.  ``local_t`` is the active
    token prefix; the remaining compile-time capacity is poisoned and must
    stay untouched.
    """
    import torch

    from golden import ScalarSpec, TensorSpec
    from utils import (
        block_table,
        position_ids_from_starts,
        resolve_start_positions,
        swa_indices_and_lens,
    )

    if local_t < S or local_t > LOCAL_T or local_t % S != 0:
        raise ValueError(f"local_t must be a multiple of {S} in [{S}, {LOCAL_T}], got {local_t}")
    local_batch = local_t // S
    starts = resolve_start_positions(
        start_pos,
        batch=local_batch,
        seq=S,
        max_seq_len=MAX_SEQ_LEN,
        default_fn=lambda: torch.full((local_batch,), WIN - 1, dtype=torch.int32),
    ).to(torch.int64)
    positions = position_ids_from_starts(starts, seq=S).to(torch.int64)
    table = block_table(
        batch=local_batch,
        table_blocks=KV_ORI_MAX_BLOCKS,
        physical_blocks=KV_ORI_BLOCK_NUM,
    )
    sources, lens = swa_indices_and_lens(
        positions, table, block_size=BLOCK_SIZE, window=WIN,
    )
    for request in range(local_batch):
        for step in range(S):
            query = request * S + step
            begin = int(lens[query].item()) - step - 1
            for overlay_step in range(step + 1):
                sources[query, begin + overlay_step] = (
                    SWA_SOURCE_OVERLAY_BASE - (request * S + overlay_step)
                )

    bounds = ((positions + 1) // COMPRESS_RATIO).to(torch.int64)
    flat_bounds = bounds.reshape(-1)
    page_counts = [max(1, (int(bounds[r].max().item()) + BLOCK_SIZE - 1) // BLOCK_SIZE)
                   for r in range(local_batch)]
    page_offsets = [0]
    for count in page_counts:
        page_offsets.append(page_offsets[-1] + count)
    page_entries = torch.empty(page_offsets[-1], 2, dtype=torch.int32)
    request_epochs = torch.arange(local_batch, dtype=torch.int32) + 7
    windows = torch.zeros(local_batch, 3, dtype=torch.int32)
    for request in range(local_batch):
        bound = int(bounds[request].max().item())
        windows[request] = torch.tensor((0, bound, 0), dtype=torch.int32)
        for page in range(page_counts[request]):
            entry = page_offsets[request] + page
            page_entries[entry, 0] = (request * 17 + page) % KV_CMP_BLOCK_NUM
            page_entries[entry, 1] = request_epochs[request]

    def init_q():
        rank = torch.arange(TP_SIZE, dtype=torch.float32).reshape(TP_SIZE, 1, 1, 1)
        token = torch.arange(local_t, dtype=torch.float32).reshape(1, local_t, 1, 1)
        head = torch.arange(H, dtype=torch.float32).reshape(1, 1, H, 1)
        q = torch.zeros(TP_SIZE, local_t, H, HEAD_DIM, dtype=torch.bfloat16)
        q[..., 0] = (0.25 + rank * 0.03125 + token * 0.0078125 + head * 0.00390625).squeeze(-1).to(torch.bfloat16)
        return q

    def init_cache(blocks, rank_scale=0.0):
        cache = torch.zeros(TP_SIZE, blocks, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
        cache[..., 0] = (0.25 + rank_scale * torch.arange(TP_SIZE, dtype=torch.float32).reshape(TP_SIZE, 1, 1, 1, 1)).squeeze(-1)
        cache[..., 1] = 0.015625
        return cache

    def init_current_kv():
        current = torch.zeros(TP_SIZE, local_t, HEAD_DIM, dtype=torch.bfloat16)
        current[..., 0] = 0.25
        return current

    def init_topk():
        topk = torch.full((TP_SIZE, local_t, CSA_TOPK), -1, dtype=torch.int32)
        for rank in range(TP_SIZE):
            for token in range(local_t):
                bound = int(flat_bounds[token].item())
                count = min(bound, CSA_TOPK)
                if count:
                    topk[rank, token, :count] = torch.arange(
                        bound - count, bound, dtype=torch.int32
                    )
        return topk

    def init_weights():
        wo_a = torch.zeros(TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16)
        for rank in range(TP_SIZE):
            for group in range(LOCAL_O_GROUPS):
                for head in range(HEADS_PER_GROUP):
                    wo_a[rank, group, head, head * HEAD_DIM] = (rank + 1) * (group + 1) * (head + 1) * 0.03125
        base = torch.arange(D * LOCAL_O_WIDTH, dtype=torch.int32).reshape(D, LOCAL_O_WIDTH)
        wo_b = (base.unsqueeze(0) + torch.arange(TP_SIZE, dtype=torch.int32).reshape(TP_SIZE, 1, 1)).remainder(7).sub(3).to(torch.int8)
        scale = (torch.arange(D, dtype=torch.float32).remainder(4) * 0.25 + 0.5).reshape(1, D).expand(TP_SIZE, -1).clone()
        return wo_a, wo_b, scale

    wo_a, wo_b, wo_b_scale = init_weights()
    freqs_cos = torch.ones(TP_SIZE, local_t, ROPE_HEAD_DIM, dtype=torch.bfloat16)
    freqs_sin = torch.zeros(TP_SIZE, local_t, ROPE_HEAD_DIM, dtype=torch.bfloat16)
    request_ids = torch.arange(local_batch, dtype=torch.int32).repeat_interleave(S)
    idx_topk = init_topk()
    return [
        TensorSpec("q", [TP_SIZE, local_t, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [TP_SIZE, KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=lambda: init_cache(KV_ORI_BLOCK_NUM, 0.03125)),
        TensorSpec("current_kv", [TP_SIZE, local_t, HEAD_DIM], torch.bfloat16, init_value=init_current_kv),
        TensorSpec("swa_sources", [TP_SIZE, local_t, WIN], torch.int32, init_value=lambda: sources.reshape(1, local_t, WIN).expand(TP_SIZE, -1, -1).clone()),
        TensorSpec("cmp_kv", [TP_SIZE, KV_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=lambda: init_cache(KV_CMP_BLOCK_NUM, 0.0625)),
        TensorSpec("query_request_ids", [TP_SIZE, local_t], torch.int32, init_value=lambda: request_ids.reshape(1, -1).expand(TP_SIZE, -1).clone()),
        TensorSpec("csa_pages", [TP_SIZE, page_entries.shape[0], 2], torch.int32, init_value=lambda: page_entries.reshape(1, -1, 2).expand(TP_SIZE, -1, -1).clone()),
        TensorSpec("csa_page_offsets", [TP_SIZE, local_batch + 1], torch.int32, init_value=lambda: torch.tensor(page_offsets, dtype=torch.int32).reshape(1, -1).expand(TP_SIZE, -1).clone()),
        TensorSpec("csa_windows", [TP_SIZE, local_batch, 3], torch.int32, init_value=lambda: windows.reshape(1, local_batch, 3).expand(TP_SIZE, -1, -1).clone()),
        TensorSpec("csa_request_epochs", [TP_SIZE, local_batch], torch.int32, init_value=lambda: request_epochs.reshape(1, -1).expand(TP_SIZE, -1).clone()),
        TensorSpec("idx_topk", [TP_SIZE, local_t, CSA_TOPK], torch.int32, init_value=lambda: idx_topk.clone()),
        TensorSpec("attn_sink", [TP_SIZE, H], torch.float32, init_value=lambda: torch.full((TP_SIZE, H), 4.0)),
        TensorSpec("freqs_cos", [TP_SIZE, local_t, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: freqs_cos.clone()),
        TensorSpec("freqs_sin", [TP_SIZE, local_t, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: freqs_sin.clone()),
        TensorSpec("wo_a", [TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=lambda: wo_a.clone()),
        TensorSpec("wo_b", [TP_SIZE, D, LOCAL_O_WIDTH], torch.int8, init_value=lambda: wo_b.clone()),
        TensorSpec("wo_b_scale", [TP_SIZE, D], torch.float32, init_value=lambda: wo_b_scale.clone()),
        TensorSpec("o_local", [TP_SIZE, LOCAL_T_PAD, D], torch.bfloat16, init_value=TP_FIXTURE_OUTPUT_SENTINEL, is_output=True),
        ScalarSpec("local_t", torch.int32, local_t),
    ]


def golden_decode_csa(tensors):
    """Reference TP all-to-all plus sharded output projection."""
    import torch

    from decode_sparse_attn_csa import golden_sparse_attn

    tp, local_t = tensors["q"].shape[:2]
    global_o_a = []
    global_o_b = []
    for destination in range(tp):
        o_a = torch.zeros(O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16)
        group0 = destination * LOCAL_O_GROUPS
        o_a[group0 : group0 + LOCAL_O_GROUPS] = tensors["wo_a"][destination]
        o_b = torch.zeros(D, O_GROUPS * O_LORA, dtype=torch.int8)
        col0 = group0 * O_LORA
        o_b[:, col0 : col0 + LOCAL_O_WIDTH] = tensors["wo_b"][destination]
        global_o_a.append(o_a)
        global_o_b.append(o_b)

    group_t = tp * local_t
    # Each destination rank owns one O-projection shard, but reduce-scatter
    # first sums every destination's partial over the global token axis and
    # only then returns that rank's local token rows.  Keep the reference in
    # the same order as the device collective rather than exposing one
    # destination partial as the rank result.
    reduced = torch.zeros(group_t, D, dtype=torch.float32)
    for destination in range(tp):
        destination_reduced = torch.zeros(group_t, D, dtype=torch.float32)
        for source in range(tp):
            out = torch.zeros(local_t, D, dtype=torch.bfloat16)
            golden_sparse_attn({
                "q": tensors["q"][source],
                "ori_kv": tensors["ori_kv"][source],
                "current_kv": tensors["current_kv"][source],
                "swa_sources": tensors["swa_sources"][source],
                "cmp_kv": tensors["cmp_kv"][source],
                "query_request_ids": tensors["query_request_ids"][source],
                "csa_pages": tensors["csa_pages"][source],
                "csa_page_offsets": tensors["csa_page_offsets"][source],
                "csa_windows": tensors["csa_windows"][source],
                "request_epochs": tensors["csa_request_epochs"][source],
                "idx_topk": tensors["idx_topk"][source],
                "attn_sink": tensors["attn_sink"][source],
                "freqs_cos": tensors["freqs_cos"][source],
                "freqs_sin": tensors["freqs_sin"][source],
                "wo_a": global_o_a[destination],
                "wo_b": global_o_b[destination],
                "wo_b_scale": tensors["wo_b_scale"][destination],
                "attn_out": out,
            })
            source_start = source * local_t
            destination_reduced[source_start : source_start + local_t] = out.float()
        reduced += destination_reduced
    tensors["o_local"].fill_(TP_FIXTURE_OUTPUT_SENTINEL)
    for destination in range(tp):
        token_start = destination * local_t
        tensors["o_local"][destination, :local_t] = reduced[
            token_start : token_start + local_t
        ].to(torch.bfloat16)


def build_tp_output_compare(local_t):
    import torch
    from golden import ratio_allclose

    # The baseline O-B path performs INT8 activation quantization followed by
    # tiled FP32 accumulation.  The host reference uses a different reduction
    # order, so retain the baseline relative threshold with a BF16/INT8
    # projection floor for this standalone TP fixture.
    prefix = ratio_allclose(atol=2e-2, rtol=1.0 / 128, max_error_ratio=0.005, valid_rows=local_t, valid_axis=1)

    def compare(actual, expected, **kwargs):
        if not torch.equal(actual[:, local_t:], expected[:, local_t:]):
            return False, "    inactive token tail was modified"
        return prefix(actual, expected, **kwargs)

    return compare


def build_tensor_specs(start_pos=None, batch=B):
    """Build deterministic standalone tensors over the 1M ragged ABI.

    The HC, projection, compressor, sparse-value, and HC-post stages see full
    active T; the exact Top-512 forest sees chunk-major descriptors with a
    fixed 16-query task-array bound. Token-local half-width RoPE
    rows are emitted here and interleaved on-core inside ``attention_csa``.
    """
    import torch

    from decode_indexer import gen_shared_weight
    from golden import TensorSpec
    from utils import (
        block_table,
        csa_decode_start_set,
        csa_event_local_rope,
        int8_quant_per_row,
        ori_slot_mapping,
        position_ids_from_starts,
        resolve_start_positions,
        swa_indices_and_lens,
        token_local_rope,
    )

    if batch < 1 or batch > B:
        raise ValueError(f"batch must be in [1, {B}], got {batch}")
    tokens = batch * S

    starts = resolve_start_positions(
        start_pos,
        batch=batch,
        seq=S,
        max_seq_len=M.max_position_embeddings,
        default_fn=lambda: csa_decode_start_set(
            batch=batch,
            seq=S,
            compress_ratio=COMPRESS_RATIO,
            state_block_size=CSA_STATE_BLOCK_SIZE,
            window=WIN,
        ),
    ).to(torch.int64)
    positions = position_ids_from_starts(starts, seq=S).to(torch.int64)
    if int(positions.max().item()) >= M.max_position_embeddings:
        raise ValueError("fixture positions exceed the 1M context ceiling")
    position_ids = positions.reshape(-1).to(torch.int32).contiguous()

    # --- ragged compressor state/event metadata (J1 pattern) ---------------
    state_page_ids = torch.arange(
        batch * CSA_STATE_PAGES_PER_REQUEST - 1, -1, -1, dtype=torch.int32
    ).reshape(batch, CSA_STATE_PAGES_PER_REQUEST)
    inner_state_page_ids = torch.arange(
        batch * CSA_INNER_STATE_PAGES_PER_REQUEST - 1,
        -1,
        -1,
        dtype=torch.int32,
    ).reshape(batch, CSA_INNER_STATE_PAGES_PER_REQUEST)
    state_valid_ranges = torch.zeros(batch, 2, dtype=torch.int32)
    for request, start in enumerate(starts.tolist()):
        state_valid_ranges[request, 0] = max(
            0, start - CSA_STATE_ROWS_PER_REQUEST
        )
        state_valid_ranges[request, 1] = start
    request_epochs = torch.arange(batch, dtype=torch.int32) + 17
    state_page_epochs = request_epochs[:, None].expand(
        batch, CSA_STATE_PAGES_PER_REQUEST
    ).clone()
    inner_state_valid_ranges = state_valid_ranges.clone()
    inner_state_page_epochs = request_epochs[:, None].expand(
        batch, CSA_INNER_STATE_PAGES_PER_REQUEST
    ).clone()
    state_slots = torch.empty(batch, S, dtype=torch.int64)
    for request in range(batch):
        for local_query in range(S):
            position = int(positions[request, local_query].item())
            ring_row = position % CSA_STATE_ROWS_PER_REQUEST
            relative_page = ring_row // CSA_STATE_BLOCK_SIZE
            page = int(state_page_ids[request, relative_page].item())
            state_slots[request, local_query] = (
                page * CSA_STATE_BLOCK_SIZE + ring_row % CSA_STATE_BLOCK_SIZE
            )
    inner_state_slots = torch.empty(batch, S, dtype=torch.int64)
    for request in range(batch):
        for local_query in range(S):
            position = int(positions[request, local_query].item())
            ring_row = position % CSA_STATE_ROWS_PER_REQUEST
            relative_page = ring_row // CSA_INNER_STATE_BLOCK_SIZE
            page = int(inner_state_page_ids[request, relative_page].item())
            inner_state_slots[request, local_query] = (
                page * CSA_INNER_STATE_BLOCK_SIZE
                + ring_row % CSA_INNER_STATE_BLOCK_SIZE
            )

    event_queries: list[int] = []
    for request in range(batch):
        for local_query in range(S):
            query = request * S + local_query
            if (int(positions[request, local_query].item()) + 1) % COMPRESS_RATIO == 0:
                event_queries.append(query)
    has_events = bool(event_queries)
    if not has_events:
        event_queries = [-1]
    event_query_ids = torch.tensor(event_queries, dtype=torch.int32)
    cache_blocks = max(4, len(event_queries) if has_events else 4)
    event_write_slots = torch.full(
        (len(event_queries),), -1, dtype=torch.int64
    )

    # --- forest descriptors (chunk-major, LOCAL query indices per chunk) ----
    candidate_counts = [
        (int(positions[q // S, q % S].item()) + 1) // COMPRESS_RATIO
        for q in range(tokens)
    ]
    logical_begins = [0] * tokens

    def _pack_forest(candidate_counts, logical_begins):
        """Pack per-chunk heterogeneous forests with LOCAL query indices.

        Each 16-query chunk gets its own leaf/pair/singleton/upper/root
        descriptors with LOCAL query indices (0-15) and LOCAL leaf IDs/slots,
        stacked along axis 0 to the production
        ``[CSA_INDEXER_MAX_CHUNKS, *_DYN, FIELDS]`` ABI.
        """
        if len(candidate_counts) != len(logical_begins):
            raise ValueError(
                "candidate_counts and logical_begins must have equal length"
            )

        ready_frontier = 8
        invalid_slot = CSA_TOPK_INVALID_TASK_SLOT
        n_chunks = (len(candidate_counts) + CSA_INDEXER_CHUNK_T - 1) // CSA_INDEXER_CHUNK_T
        if n_chunks > CSA_INDEXER_MAX_CHUNKS:
            raise ValueError(
                f"forest needs {n_chunks} chunks, capacity is "
                f"{CSA_INDEXER_MAX_CHUNKS}"
            )

        chunk_leaf = []
        chunk_pair = []
        chunk_singleton = []
        chunk_upper = []
        chunk_root = []
        chunk_pair_counts = []
        chunk_singleton_counts = []
        chunk_upper_counts = []

        for chunk in range(n_chunks):
            q0 = chunk * CSA_INDEXER_CHUNK_T
            q1 = min(q0 + CSA_INDEXER_CHUNK_T, len(candidate_counts))
            node_kinds = []
            node_leaf_ids = []
            node_left_slots = []
            node_right_slots = []
            node_credit_slots = []
            leaf_rows = []
            root_slots = []

            def append_leaf(local_query, local_leaf, valid, credit_slot):
                leaf_id = len(leaf_rows)
                slot = len(node_kinds)
                global_query = q0 + local_query
                begin = logical_begins[global_query] + local_leaf * CSA_CANDIDATES_PER_LEAF
                leaf_rows.append([local_query, begin, valid, slot, credit_slot])
                node_kinds.append(0)
                node_leaf_ids.append(leaf_id)
                node_left_slots.append(-1)
                node_right_slots.append(-1)
                node_credit_slots.append(credit_slot)
                return slot

            def append_merge(left_slot, right_slot):
                slot = len(node_kinds)
                node_kinds.append(1)
                node_leaf_ids.append(-1)
                node_left_slots.append(left_slot)
                node_right_slots.append(right_slot)
                node_credit_slots.append(invalid_slot)
                return slot

            for local_query in range(q1 - q0):
                global_query = q0 + local_query
                candidates = candidate_counts[global_query]
                leaves = (candidates + CSA_CANDIDATES_PER_LEAF - 1) // CSA_CANDIDATES_PER_LEAF
                if leaves == 0:
                    root_slots.append(-1)
                    continue

                level1_slots = []
                for group in range(leaves // 2):
                    credit = invalid_slot
                    if group >= ready_frontier:
                        credit = level1_slots[group - ready_frontier]
                    left_leaf = group * 2
                    left_valid = min(
                        CSA_CANDIDATES_PER_LEAF,
                        candidates - left_leaf * CSA_CANDIDATES_PER_LEAF,
                    )
                    left_slot = append_leaf(local_query, left_leaf, left_valid, credit)
                    right_leaf = left_leaf + 1
                    right_valid = min(
                        CSA_CANDIDATES_PER_LEAF,
                        candidates - right_leaf * CSA_CANDIDATES_PER_LEAF,
                    )
                    right_slot = append_leaf(local_query, right_leaf, right_valid, credit)
                    level1_slots.append(append_merge(left_slot, right_slot))

                if leaves % 2:
                    local_leaf = leaves - 1
                    credit = invalid_slot
                    if len(level1_slots) >= ready_frontier:
                        credit = level1_slots[-ready_frontier]
                    valid = min(
                        CSA_CANDIDATES_PER_LEAF,
                        candidates - local_leaf * CSA_CANDIDATES_PER_LEAF,
                    )
                    level1_slots.append(append_leaf(local_query, local_leaf, valid, credit))

                current = level1_slots
                while len(current) > 1:
                    next_level = []
                    for pair in range(len(current) // 2):
                        next_level.append(append_merge(current[2 * pair], current[2 * pair + 1]))
                    if len(current) % 2:
                        next_level.append(current[-1])
                    current = next_level
                root_slots.append(current[0])

            pair_rows = []
            upper_rows = []
            paired_leaf_slots = set()
            for output_slot, kind in enumerate(node_kinds):
                if kind != 1:
                    continue
                left_slot = node_left_slots[output_slot]
                right_slot = node_right_slots[output_slot]
                if node_kinds[left_slot] == 0 and node_kinds[right_slot] == 0:
                    pair_rows.append([
                        node_leaf_ids[left_slot],
                        node_leaf_ids[right_slot],
                        left_slot,
                        right_slot,
                        output_slot,
                        node_credit_slots[left_slot],
                    ])
                    paired_leaf_slots.update((left_slot, right_slot))
                else:
                    upper_rows.append([left_slot, right_slot, output_slot])

            singleton_slots = [
                slot
                for slot, kind in enumerate(node_kinds)
                if kind == 0 and slot not in paired_leaf_slots
            ]
            singleton_rows = [
                [node_leaf_ids[slot], slot, node_credit_slots[slot]]
                for slot in singleton_slots
            ]
            root_deps = [root if root >= 0 else invalid_slot for root in root_slots]
            while len(root_deps) < CSA_INDEXER_CHUNK_T:
                root_deps.append(invalid_slot)
            while len(root_slots) < CSA_INDEXER_CHUNK_T:
                root_slots.append(-1)

            chunk_leaf.append(leaf_rows)
            chunk_pair.append(pair_rows)
            chunk_singleton.append(singleton_rows)
            chunk_upper.append(upper_rows)
            chunk_root.append(
                [[r, d] for r, d in zip(root_slots, root_deps)]
            )
            chunk_pair_counts.append(len(pair_rows))
            chunk_singleton_counts.append(len(singleton_rows))
            chunk_upper_counts.append(len(upper_rows))

        invalid_root_chunk = [
            [-1, invalid_slot] for _ in range(CSA_INDEXER_CHUNK_T)
        ]
        while len(chunk_root) < CSA_INDEXER_MAX_CHUNKS:
            chunk_leaf.append([])
            chunk_pair.append([])
            chunk_singleton.append([])
            chunk_upper.append([])
            chunk_root.append([list(row) for row in invalid_root_chunk])
            chunk_pair_counts.append(0)
            chunk_singleton_counts.append(0)
            chunk_upper_counts.append(0)

        # Keep a non-null host buffer for an empty descriptor family.  A
        # short request may have no pair or upper nodes, but a zero-row torch
        # tensor has a null data pointer and cannot become a runtime
        # BufferDescriptor.  The actual-count scalars remain zero, so these
        # sentinel rows are never submitted to the forest.
        max_leaf = max(1, max((len(r) for r in chunk_leaf), default=0))
        max_pair = max(1, max((len(r) for r in chunk_pair), default=0))
        max_singleton = max(1, max((len(r) for r in chunk_singleton), default=0))
        max_upper = max(1, max((len(r) for r in chunk_upper), default=0))

        def pad_chunk(rows, max_rows, width):
            padded = [list(r) for r in rows]
            while len(padded) < max_rows:
                padded.append([-1] * width)
            return padded

        def stack(chunk_rows, max_rows, width):
            if max_rows == 0:
                return torch.empty(
                    CSA_INDEXER_MAX_CHUNKS, 0, width, dtype=torch.int32
                )
            return torch.tensor(
                [pad_chunk(rows, max_rows, width) for rows in chunk_rows],
                dtype=torch.int32,
            )

        return {
            "leaf_descriptors": stack(chunk_leaf, max_leaf, PHASE_D_LEAF_FIELDS),
            "pair_descriptors": stack(chunk_pair, max_pair, PHASE_D_PAIR_FIELDS),
            "singleton_descriptors": stack(chunk_singleton, max_singleton, PHASE_D_SINGLETON_FIELDS),
            "upper_descriptors": stack(chunk_upper, max_upper, PHASE_D_UPPER_FIELDS),
            "root_descriptors": torch.tensor(
                chunk_root, dtype=torch.int32
            ).reshape(
                CSA_INDEXER_MAX_CHUNKS,
                CSA_INDEXER_CHUNK_T,
                PHASE_D_ROOT_FIELDS,
            ),
            "pair_group_chunk_counts": torch.tensor(chunk_pair_counts, dtype=torch.int32),
            "singleton_chunk_counts": torch.tensor(chunk_singleton_counts, dtype=torch.int32),
            "upper_merge_chunk_counts": torch.tensor(chunk_upper_counts, dtype=torch.int32),
        }

    forest = _pack_forest(candidate_counts, logical_begins)

    # --- full-active global descriptors (reused per chunk inside indexer) -
    # The zero-score Top-512 keeps the first 512 candidates, so the value-side
    # window only needs to cover that prefix; the forest-side pages cover the
    # complete candidate range the leaves score.
    request_ends = [
        (int(starts[r].item()) + S) // COMPRESS_RATIO for r in range(batch)
    ]
    # The sparse value side reads at most Top-512 rows per query, but those
    # rows can come from any logical candidate.  Keep the physical page pool
    # wide enough for the full candidate range; Top-K is a selector width,
    # not a cache-capacity limit.
    value_ends = request_ends[:]

    idx_page_counts = [
        (end + BLOCK_SIZE - 1) // BLOCK_SIZE for end in request_ends
    ]
    value_page_counts = [
        (end + BLOCK_SIZE - 1) // BLOCK_SIZE for end in value_ends
    ]
    request_epoch_values = [int(value) for value in request_epochs.tolist()]

    idx_rows = []
    idx_offsets = [0]
    idx_windows = torch.zeros(batch, 3, dtype=torch.int32)
    idx_epochs = torch.zeros(batch, dtype=torch.int32)
    idx_cursor = 0
    for request in range(batch):
        count = idx_page_counts[request]
        idx_rows.extend(
            (idx_cursor + page, request_epoch_values[request])
            for page in range(count)
        )
        idx_cursor += count
        idx_offsets.append(idx_offsets[-1] + count)
        idx_windows[request] = torch.tensor(
            [0, request_ends[request], 0], dtype=torch.int32
        )
        idx_epochs[request] = request_epoch_values[request]
    idx_pages = torch.tensor(idx_rows, dtype=torch.int32).reshape(-1, 2)
    idx_page_offsets = torch.tensor(idx_offsets, dtype=torch.int32)
    idx_rows_total = idx_cursor * BLOCK_SIZE

    value_rows = []
    value_offsets = [0]
    value_windows = torch.zeros(batch, 3, dtype=torch.int32)
    value_cursor = 0
    for request in range(batch):
        count = value_page_counts[request]
        value_rows.extend(
            (value_cursor + page, request_epoch_values[request])
            for page in range(count)
        )
        value_cursor += count
        value_offsets.append(value_offsets[-1] + count)
        value_windows[request] = torch.tensor(
            [0, value_ends[request], 0], dtype=torch.int32
        )
    csa_pages = torch.tensor(value_rows, dtype=torch.int32).reshape(-1, 2)
    csa_page_offsets = torch.tensor(value_offsets, dtype=torch.int32)
    query_request_ids = torch.arange(
        batch, dtype=torch.int32
    ).repeat_interleave(S)
    query_request_ids_padded = torch.full(
        (T_PAD,), -1, dtype=torch.int32
    )
    query_request_ids_padded[:tokens] = query_request_ids

    main_cache_blocks = max(value_offsets[-1], cache_blocks)
    idx_rows_total = max(idx_rows_total, cache_blocks * BLOCK_SIZE)

    # Main and index pools use independent tensors but the same logical page
    # to physical-page mapping.  Bind every current-step event to the page
    # selected by its request's descriptor range; a permuted page list here
    # would make the compressor write one row while sparse/index lookup reads
    # another.
    event_write_slots.fill_(-1)
    if has_events:
        for event_index, query in enumerate(event_queries):
            request = query // S
            candidate = int(positions.reshape(-1)[query].item()) // COMPRESS_RATIO
            page = value_offsets[request] + candidate // BLOCK_SIZE
            event_write_slots[event_index] = (
                page * BLOCK_SIZE + candidate % BLOCK_SIZE
            )
        valid_event_slots = event_write_slots[event_write_slots >= 0]
        if torch.unique(valid_event_slots).numel() != valid_event_slots.numel():
            raise AssertionError("CSA event write slots must not alias")

    # --- sliding-window sources + raw write slots (baseline pattern) -----
    # First call baseline swa_indices_and_lens() to obtain the canonical
    # physical raw-cache rows and visible lengths, then
    # replace only causal, request-local rows with overlay encodings.
    raw_block_table = block_table(
        batch=batch,
        table_blocks=ORI_MAX_BLOCKS,
        physical_blocks=ORI_BLOCK_NUM,
    )
    sources, lens = swa_indices_and_lens(
        positions, raw_block_table, block_size=BLOCK_SIZE, window=WIN
    )
    for request in range(batch):
        for s_idx in range(S):
            query = request * S + s_idx
            overlay_begin = int(lens[query].item()) - s_idx - 1
            for overlay_s in range(s_idx + 1):
                overlay_query = request * S + overlay_s
                sources[query, overlay_begin + overlay_s] = (
                    SWA_SOURCE_OVERLAY_BASE - overlay_query
                )
    swa_sources = sources.contiguous()
    raw_write_slots = (
        ori_slot_mapping(positions, raw_block_table, block_size=BLOCK_SIZE)
        .reshape(-1)
        .contiguous()
    )

    # --- rope tables ---------------------------------------------------------
    # Token-local half-width cos/sin rows: the device interleaves them
    # on-core inside attention_csa. The event-local rows for
    # the compressors are derived the same way on-core.
    freqs_cos, freqs_sin = token_local_rope(
        M, COMPRESS_RATIO, position_ids, dtype=torch.bfloat16
    )
    cos_half = freqs_cos[:, :HALF_ROPE].float().contiguous()
    sin_half = freqs_sin[:, :HALF_ROPE].float().contiguous()

    # Event-local half-width cos/sin at compression starts (4r), not at the
    # event token position (4r+3). Phase D fixes the event's RoPE row to the
    # compression start; using the event token's row is a semantic error that
    # the device and golden previously shared, so the test passed for the
    # wrong reason. Mirrors csa_event_local_rope used by the standalone
    # compressor; the device interleaves these on-core.
    event_rows = torch.zeros(len(event_queries), dtype=torch.int64)
    for event_index, query in enumerate(event_queries):
        if query >= 0:
            event_rows[event_index] = (
                positions.reshape(-1)[query] // COMPRESS_RATIO
            )
    event_cos_full, event_sin_full = csa_event_local_rope(
        M, event_rows, dtype=torch.float32
    )
    event_cos_half = event_cos_full[:, :HALF_ROPE].contiguous()
    event_sin_half = event_sin_full[:, :HALF_ROPE].contiguous()

    # --- weights -------------------------------------------------------------
    def round_half_away_from_zero(x):
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, w.shape[1])
        w_i32 = round_half_away_from_zero(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def quant_w_per_row(w):
        amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.unsqueeze(-1)
        w_i32 = round_half_away_from_zero(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0519

    def init_hc_attn_scale():
        return torch.tensor([0.076099, 0.032597, 0.226994])

    def init_hc_attn_base():
        return torch.tensor([
            5.9166, -3.6223, -2.9324, -3.3124,
            -3.9100, -0.9384, -3.3256, -2.5240,
            2.0706, -2.5728, 0.1424, -3.9453,
            -3.8859, 3.4634, -3.3799, -2.6077,
            -2.7191, -2.4846, 2.0395, -0.5010,
            -3.5992, -2.7520, -3.3493, 3.1587,
        ])

    def init_wq_a():
        return torch.randn(D, Q_LORA) / D ** 0.5

    def init_wq_b_bf16():
        return torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5

    wq_b_i8, wq_b_scale = quant_w_per_output_channel(
        init_wq_b_bf16().to(torch.bfloat16)
    )

    def init_wkv():
        return torch.randn(D, HEAD_DIM) / D ** 0.5

    def init_cmp_wkv():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0240

    def init_cmp_wgate():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0381

    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.1226

    def init_cmp_norm_w():
        return 0.9569 + 0.1916 * torch.randn(HEAD_DIM)

    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0270

    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0513

    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1524

    def init_inner_norm_w():
        return 0.6903 + 0.2663 * torch.randn(IDX_HEAD_DIM)

    def init_hadamard_idx():
        h = torch.ones((1, 1))
        while h.shape[0] < IDX_HEAD_DIM:
            h = torch.cat([
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ], dim=0)
        return h / (IDX_HEAD_DIM ** 0.5)

    def init_idx_weights_proj():
        return torch.randn(D, IDX_N_HEADS) * 0.2218

    def init_state_history(
        page_ids, valid_ranges, dim, out_dim, block_size, rows_per_request
    ):
        state = torch.zeros(
            page_ids.numel(), block_size, dim
        )
        state[:, :, out_dim:] = float("-inf")
        history = torch.randn_like(state) * 0.05
        for request in range(batch):
            valid_begin = int(valid_ranges[request, 0].item())
            valid_end = int(valid_ranges[request, 1].item())
            for position in range(valid_begin, valid_end):
                ring_row = position % rows_per_request
                relative_page = ring_row // block_size
                page = int(page_ids[request, relative_page].item())
                intra = ring_row % block_size
                state[page, intra] = history[page, intra]
        return state

    def init_main_state():
        return init_state_history(
            state_page_ids,
            state_valid_ranges,
            MAIN_STATE_DIM,
            MAIN_OUT_DIM,
            CSA_STATE_BLOCK_SIZE,
            CSA_STATE_ROWS_PER_REQUEST,
        )

    def init_inner_state():
        return init_state_history(
            inner_state_page_ids,
            inner_state_valid_ranges,
            INNER_STATE_DIM,
            INNER_OUT_DIM,
            CSA_INNER_STATE_BLOCK_SIZE,
            CSA_INNER_STATE_ROWS_PER_REQUEST,
        )

    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
        denom = denom.clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    def init_main_cache():
        return init_normalized_cache(
            (main_cache_blocks, BLOCK_SIZE, 1, HEAD_DIM)
        )

    def init_kv_cache():
        return init_normalized_cache(
            (ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        )

    def init_idx_cache():
        rows = init_normalized_cache((idx_rows_total, IDX_HEAD_DIM))
        flat = rows.float().reshape(idx_rows_total, IDX_HEAD_DIM)
        quant, scale = int8_quant_per_row(flat)
        return quant, scale

    idx_kv_i8, idx_kv_sc = init_idx_cache()

    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5

    wo_b_i8, wo_b_scale = quant_w_per_row(
        (torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5)
        .to(torch.bfloat16)
    )

    # idx_wq_b: MXFP8-grid simulated weight, matching the J3 indexer fixture.
    idx_wq_b_i8_T, idx_wq_b_scale = gen_shared_weight(
        (IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56
    )
    idx_wq_b_i8 = idx_wq_b_i8_T.t().contiguous()

    state_pages_main = batch * CSA_STATE_PAGES_PER_REQUEST
    state_pages_inner = batch * CSA_INNER_STATE_PAGES_PER_REQUEST

    return [
        # --- HC + norm + QKV + per-token rope -------------------------------
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32,
                   init_value=lambda: torch.empty(
                       tokens, HC_MULT, D).uniform_(-1, 1)),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32,
                   init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32,
                   init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32,
                   init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.bfloat16,
                   init_value=lambda: torch.ones(D)),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8,
                   init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32,
                   init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16,
                   init_value=lambda: torch.ones(Q_LORA)),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16,
                   init_value=lambda: torch.ones(HEAD_DIM)),
        TensorSpec("freqs_cos", [tokens, ROPE_HEAD_DIM], torch.bfloat16,
                   init_value=lambda: freqs_cos.clone()),
        TensorSpec("freqs_sin", [tokens, ROPE_HEAD_DIM], torch.bfloat16,
                   init_value=lambda: freqs_sin.clone()),
        # --- main compressor ------------------------------------------------
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16,
                   init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16,
                   init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32,
                   init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16,
                   init_value=init_cmp_norm_w),
        TensorSpec("main_state",
                   [state_pages_main, CSA_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
                   torch.float32, init_value=init_main_state),
        TensorSpec("main_state_page_ids",
                   [batch, CSA_STATE_PAGES_PER_REQUEST], torch.int32,
                   init_value=lambda: state_page_ids.clone()),
        TensorSpec("main_state_valid_ranges", [batch, 2], torch.int32,
                   init_value=lambda: state_valid_ranges.clone()),
        TensorSpec("main_state_page_epochs",
                   [batch, CSA_STATE_PAGES_PER_REQUEST], torch.int32,
                   init_value=lambda: state_page_epochs.clone()),
        TensorSpec("compressor_request_epochs", [batch], torch.int32,
                   init_value=lambda: request_epochs.clone()),
        TensorSpec("event_query_ids", [event_query_ids.numel()], torch.int32,
                   init_value=lambda: event_query_ids.clone()),
        TensorSpec("main_event_write_slots", [event_write_slots.numel()],
                   torch.int64,
                   init_value=lambda: event_write_slots.clone()),
        TensorSpec("main_cache",
                   [main_cache_blocks, BLOCK_SIZE, 1, HEAD_DIM],
                   torch.bfloat16, init_value=init_main_cache),
        TensorSpec("main_state_write_slots", [tokens], torch.int64,
                   init_value=lambda: state_slots.reshape(-1).clone()),
        # --- inner compressor -----------------------------------------------
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16,
                   init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16,
                   init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32,
                   init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.bfloat16,
                   init_value=init_inner_norm_w),
        TensorSpec("inner_hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM],
                   torch.bfloat16, init_value=init_hadamard_idx),
        TensorSpec("inner_state",
                   [state_pages_inner, CSA_INNER_STATE_BLOCK_SIZE,
                    INNER_STATE_DIM],
                   torch.float32, init_value=init_inner_state),
        TensorSpec("inner_state_page_ids",
                   [batch, CSA_INNER_STATE_PAGES_PER_REQUEST], torch.int32,
                   init_value=lambda: inner_state_page_ids.clone()),
        TensorSpec("inner_state_valid_ranges", [batch, 2], torch.int32,
                   init_value=lambda: inner_state_valid_ranges.clone()),
        TensorSpec("inner_state_page_epochs",
                   [batch, CSA_INNER_STATE_PAGES_PER_REQUEST], torch.int32,
                   init_value=lambda: inner_state_page_epochs.clone()),
        TensorSpec("inner_event_write_slots", [event_write_slots.numel()],
                   torch.int64,
                   init_value=lambda: event_write_slots.clone()),
        TensorSpec("inner_state_write_slots", [tokens], torch.int64,
                   init_value=lambda: inner_state_slots.reshape(-1).clone()),
        TensorSpec("idx_kv_cache_flat", [idx_rows_total, IDX_HEAD_DIM],
                   torch.int8, init_value=lambda: idx_kv_i8.clone()),
        TensorSpec("idx_kv_scale_flat", [idx_rows_total, 1], torch.float32,
                   init_value=lambda: idx_kv_sc.clone()),
        # --- indexer --------------------------------------------------------
        TensorSpec("idx_wq_b",
                   [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8,
                   init_value=lambda: idx_wq_b_i8),
        TensorSpec("idx_wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM],
                   torch.float32,
                   init_value=lambda: idx_wq_b_scale),
        TensorSpec("idx_weights_proj", [D, IDX_N_HEADS], torch.bfloat16,
                   init_value=init_idx_weights_proj),
        TensorSpec("idx_hadamard", [IDX_HEAD_DIM, IDX_HEAD_DIM],
                   torch.bfloat16, init_value=init_hadamard_idx),
        TensorSpec("query_vectors",
                   [T_PAD, IDX_N_HEADS, IDX_HEAD_DIM], torch.int8),
        TensorSpec("query_scales", [T_PAD, IDX_N_HEADS], torch.float32),
        TensorSpec("query_weights", [T_PAD, IDX_N_HEADS], torch.float32),
        TensorSpec("query_request_ids_padded", [T_PAD], torch.int32,
                   init_value=lambda: query_request_ids_padded.clone()),
        # Token-local half-width RoPE; interleaved on-core inside attention_csa.
        # Event-local rows at compression starts (4r) are appended after the
        # token-local rows so the device can derive event_cos_il from the same
        # cos_il/sin_signed table via a t_dim offset.
        TensorSpec("cos", [tokens + len(event_queries), HALF_ROPE], torch.float32,
                   init_value=lambda: torch.cat([cos_half, event_cos_half], dim=0).clone()),
        TensorSpec("sin", [tokens + len(event_queries), HALF_ROPE], torch.float32,
                   init_value=lambda: torch.cat([sin_half, event_sin_half], dim=0).clone()),
        # Full-active global descriptors (reused per chunk inside indexer).
        TensorSpec("idx_pages", [int(idx_pages.shape[0]), 2], torch.int32,
                   init_value=lambda: idx_pages.clone()),
        TensorSpec("idx_page_offsets", [batch + 1], torch.int32,
                   init_value=lambda: idx_page_offsets.clone()),
        TensorSpec("idx_windows", [batch, 3], torch.int32,
                   init_value=lambda: idx_windows.clone()),
        TensorSpec("idx_request_epochs", [batch], torch.int32,
                   init_value=lambda: idx_epochs.clone()),
        # Chunk-major forest descriptors (LOCAL query indices per chunk).
        TensorSpec("leaf_descriptors",
                   list(forest["leaf_descriptors"].shape), torch.int32,
                   init_value=lambda: forest["leaf_descriptors"]),
        TensorSpec("pair_descriptors",
                   list(forest["pair_descriptors"].shape), torch.int32,
                   init_value=lambda: forest["pair_descriptors"]),
        TensorSpec("singleton_descriptors",
                   list(forest["singleton_descriptors"].shape), torch.int32,
                   init_value=lambda: forest["singleton_descriptors"]),
        TensorSpec("upper_descriptors",
                   list(forest["upper_descriptors"].shape), torch.int32,
                   init_value=lambda: forest["upper_descriptors"]),
        TensorSpec("root_descriptors",
                   list(forest["root_descriptors"].shape), torch.int32,
                   init_value=lambda: forest["root_descriptors"]),
        TensorSpec("pair_group_chunk_counts",
                   list(forest["pair_group_chunk_counts"].shape), torch.int32,
                   init_value=lambda: forest["pair_group_chunk_counts"]),
        TensorSpec("singleton_chunk_counts",
                   list(forest["singleton_chunk_counts"].shape), torch.int32,
                   init_value=lambda: forest["singleton_chunk_counts"]),
        TensorSpec("upper_merge_chunk_counts",
                   list(forest["upper_merge_chunk_counts"].shape), torch.int32,
                   init_value=lambda: forest["upper_merge_chunk_counts"]),
        # --- sparse attention + raw ring ------------------------------------
        TensorSpec("kv_cache",
                   [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
                   torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("raw_write_slots", [tokens], torch.int64,
                   init_value=lambda: raw_write_slots.clone()),
        TensorSpec("swa_sources", [tokens, WIN], torch.int32,
                   init_value=lambda: swa_sources.clone()),
        TensorSpec("query_request_ids", [tokens], torch.int32,
                   init_value=lambda: query_request_ids.clone()),
        TensorSpec("csa_pages", [int(csa_pages.shape[0]), 2], torch.int32,
                   init_value=lambda: csa_pages.clone()),
        TensorSpec("csa_page_offsets", [batch + 1], torch.int32,
                   init_value=lambda: csa_page_offsets.clone()),
        TensorSpec("csa_windows", [batch, 3], torch.int32,
                   init_value=lambda: value_windows.clone()),
        TensorSpec("csa_request_epochs", [batch], torch.int32,
                   init_value=lambda: request_epochs.clone()),
        TensorSpec("attn_sink", [H], torch.float32,
                   init_value=lambda: torch.ones(H) * 4.0),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16,
                   init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8,
                   init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32,
                   init_value=lambda: wo_b_scale),
        # --- output ----------------------------------------------------------
        TensorSpec("x_out", [tokens, HC_MULT, D], torch.float32,
                   is_output=True),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)))
    parser.add_argument("-b", "--batch", type=int, default=LOCAL_T // S,
                        help=f"active requests per rank, from 1 to {LOCAL_T // S}")
    parser.add_argument("--start-pos", type=int, default=None,
                        help="uniform absolute decode start position")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        parser.error(f"--tp must remain {TP_SIZE} after import-time specialization")
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(f"--device must be a comma-separated integer list, got {args.device!r}")
    if len(device_ids) != TP_SIZE or len(set(device_ids)) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} distinct devices, got {device_ids}")
    if args.batch < 1 or args.batch > LOCAL_T // S:
        parser.error(f"--batch must be in [1, {LOCAL_T // S}], got {args.batch}")
    local_t = args.batch * S

    output_compare = build_tp_output_compare(local_t)
    result = run_jit(
        fn=l3_decode_csa,
        specs=build_tp_tensor_specs(local_t, start_pos=args.start_pos),
        golden_fn=golden_decode_csa,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        ),
        runtime_cfg=dict(platform=args.platform),
        compare_fn={"o_local": output_compare},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
