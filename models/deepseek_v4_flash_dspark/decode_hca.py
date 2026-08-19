# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 HCA decode orchestration with configurable TP output."""


import sys

import config


# Sub-kernels freeze TP-derived shapes at import time, so select the standalone
# program's TP world before importing them below.
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
    C128_COMPRESSOR_BLOCK_SIZE,
    KV_ORI_BLOCK_NUM,
    HCA_STATE_PHYSICAL_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    HCA_KV_POOL_BLOCKS,
    HCA_ROWS_PER_SHARD,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    SWA_SOURCE_OVERLAY_BASE,
)
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from rope_interleave import rope_interleave
from decode_compressor_ratio128 import compressor_ratio128
from decode_o_proj import (
    ATTENTION_WINDOW_ROWS,
    GROUP_T_PAD,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    LOCAL_T,
    LOCAL_T_PAD,
    O_WINDOW_ROWS,
    decode_o_proj,
    decode_o_proj_tp1,
    o_group_a2a,
    o_proj_reduce_scatter,
)
from decode_sparse_attn_hca import (
    HALF_ROPE,
    NOPE_DIM,
    ROPE_DIM,
    T_PAD,
    HCA_WORK_DYN,
    HCA_PAGES_DYN,
    HCA_REQUEST_OFFSETS_DYN,
    HCA_QUERY_OFFSETS_DYN,
    sparse_attn_hca,
    sparse_attn_hca_heads,
)

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("HCA_STATE_BLOCK_NUM_DYN")


# model config
B = DECODE_BATCH // TP_SIZE
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local (HCA: ratio-128 main compressor, no indexer)
COMPRESS_RATIO = 128  # HCA
OVERLAP = COMPRESS_RATIO == 4   # always False for HCA
COFF = 1 + int(OVERLAP)         # always 1 for HCA
MAIN_OUT_DIM = COFF * HEAD_DIM
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_BLOCK_NUM = HCA_KV_POOL_BLOCKS
CMP_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
# Main compressor state pool (kv + score channels merged into one paged FP32 buffer).
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_PHYSICAL_BLOCKS = HCA_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
COMPRESS_TOPK = MAX_SEQ_LEN // COMPRESS_RATIO
CMP_TOPK = ((COMPRESS_TOPK + HCA_ROWS_PER_SHARD - 1) // HCA_ROWS_PER_SHARD) * HCA_ROWS_PER_SHARD
# HCA has no learned indexer: its compressed tail is the complete ratio-128
# cache capacity. Keep the baseline aliases used by the TP1 compatibility
# path and its golden helper.
HCA_TOPK_LIMIT = COMPRESS_TOPK
HCA_CMP_TOPK = CMP_TOPK
VALID_TOKEN_TILE = 8

# tiling
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE
HCA_TOPK_TOKEN_TILE = 8   # tokens per cache-window topk SPMD block
HCA_WB_TOKEN_TILE = 8  # tokens per cache-writeback SPMD block

if T != LOCAL_T:
    raise ValueError(f"HCA token capacity {T} must equal TP local token capacity {LOCAL_T}")
if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"HCA token capacity {T_PAD} must equal TP local token capacity {LOCAL_T_PAD}")


@pl.jit
def decode_hca(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    current_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    swa_sources: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    hca_pages: pl.Tensor[[HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[HCA_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[B_DYN], pl.INT32],
    hca_query_work_offsets: pl.Tensor[[HCA_QUERY_OFFSETS_DYN], pl.INT32],
    hca_work_query_ids: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[HCA_WORK_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
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
    """Run rank-local HCA heads, output A2A, sharded O projection, and RS."""
    q.bind_dynamic(0, T_DYN)
    ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    current_kv.bind_dynamic(0, T_DYN)
    swa_sources.bind_dynamic(0, T_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    query_request_ids.bind_dynamic(0, T_DYN)
    hca_pages.bind_dynamic(0, HCA_PAGES_DYN)
    hca_page_offsets.bind_dynamic(0, HCA_REQUEST_OFFSETS_DYN)
    hca_windows.bind_dynamic(0, B_DYN)
    request_epochs.bind_dynamic(0, B_DYN)
    hca_query_work_offsets.bind_dynamic(0, HCA_QUERY_OFFSETS_DYN)
    hca_work_query_ids.bind_dynamic(0, HCA_WORK_DYN)
    hca_work_row_begin.bind_dynamic(0, HCA_WORK_DYN)
    hca_work_valid_rows.bind_dynamic(0, HCA_WORK_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)

    attention_grouped = pl.create_tensor([O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], dtype=pl.BF16)
    cmp_dep = pl.system.task_dummy(deps=[])
    attention_grouped, _ = sparse_attn_hca_heads(
        q, ori_kv, current_kv, swa_sources,
        cmp_kv, query_request_ids,
        hca_pages, hca_page_offsets, hca_windows, request_epochs,
        hca_query_work_offsets, hca_work_query_ids,
        hca_work_row_begin, hca_work_valid_rows,
        attn_sink, freqs_cos, freqs_sin,
        attention_grouped, cmp_dep,
    )

    attention_local_flat = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_local_flat, attention_signal = o_group_a2a(
        attention_grouped, attention_local_flat,
        attention_window, attention_signal,
        group_base, tp_rank, local_t,
    )

    attention_local_groups = pl.reshape(attention_local_flat, [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN])
    o_partial = pl.create_tensor([GROUP_T_PAD, D], dtype=pl.FP32)
    o_partial, projection_tid = decode_o_proj(
        attention_local_groups,
        wo_a, wo_b, wo_b_scale,
        local_t, o_partial,
    )

    o_local, o_signal = o_proj_reduce_scatter(
        o_partial, o_local,
        o_window, o_signal,
        group_base, tp_rank, local_t, projection_tid,
    )
    return o_local, attention_signal, o_signal


@pl.jit.host
def l3_decode_hca(
    q: pl.Tensor[[TP_SIZE, T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    current_kv: pl.Tensor[[TP_SIZE, T_DYN, HEAD_DIM], pl.BF16],
    swa_sources: pl.Tensor[[TP_SIZE, T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[TP_SIZE, CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    query_request_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    hca_pages: pl.Tensor[[TP_SIZE, HCA_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[TP_SIZE, HCA_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[TP_SIZE, B_DYN, 3], pl.INT32],
    request_epochs: pl.Tensor[[TP_SIZE, B_DYN], pl.INT32],
    hca_query_work_offsets: pl.Tensor[[TP_SIZE, HCA_QUERY_OFFSETS_DYN], pl.INT32],
    hca_work_query_ids: pl.Tensor[[TP_SIZE, HCA_WORK_DYN], pl.INT32],
    hca_work_row_begin: pl.Tensor[[TP_SIZE, HCA_WORK_DYN], pl.INT32],
    hca_work_valid_rows: pl.Tensor[[TP_SIZE, HCA_WORK_DYN], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    freqs_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    o_local: pl.InOut[pl.Tensor[[TP_SIZE, LOCAL_T_PAD, D], pl.BF16]],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch the HCA output path on one TP group."""
    q.bind_dynamic(1, T_DYN)
    current_kv.bind_dynamic(1, T_DYN)
    swa_sources.bind_dynamic(1, T_DYN)
    cmp_kv.bind_dynamic(1, CMP_BLOCK_NUM_DYN)
    query_request_ids.bind_dynamic(1, T_DYN)
    hca_pages.bind_dynamic(1, HCA_PAGES_DYN)
    hca_page_offsets.bind_dynamic(1, HCA_REQUEST_OFFSETS_DYN)
    hca_windows.bind_dynamic(1, B_DYN)
    request_epochs.bind_dynamic(1, B_DYN)
    hca_query_work_offsets.bind_dynamic(1, HCA_QUERY_OFFSETS_DYN)
    hca_work_query_ids.bind_dynamic(1, HCA_WORK_DYN)
    hca_work_row_begin.bind_dynamic(1, HCA_WORK_DYN)
    hca_work_valid_rows.bind_dynamic(1, HCA_WORK_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        decode_hca(
            q[rank], ori_kv[rank], current_kv[rank], swa_sources[rank],
            cmp_kv[rank], query_request_ids[rank],
            hca_pages[rank], hca_page_offsets[rank], hca_windows[rank], request_epochs[rank],
            hca_query_work_offsets[rank], hca_work_query_ids[rank],
            hca_work_row_begin[rank], hca_work_valid_rows[rank],
            attn_sink[rank], freqs_cos[rank], freqs_sin[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank], o_local[rank],
            attention_window, attention_signal, o_window, o_signal,
            0, rank, local_t, device=rank,
        )


@pl.jit.inline
def decode_hca_tp1(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # main compressor (head_dim=HEAD_DIM, ratio=128, overlap=False)
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    # KV cache split into ori (sliding window) and cmp (compressed) pools to match sparse_attn's contract.
    # cmp_kv is shared with the compressor: it writes the compressed row directly into this pool.
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj (fused into sparse_attn)
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    """HCA decode orchestration for compress_ratio=128."""
    t_dim = pl.tensor.dim(x_hc, 0)
    b_dim = t_dim // S
    topk_blocks = t_dim // HCA_TOPK_TOKEN_TILE
    wb_blocks = t_dim // HCA_WB_TOKEN_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    rope_cos_t = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    cmp_cos = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    # Interleave-duplicated / sign-folded compressed-position rope rows, built once over B rows.
    cmp_cos_il = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_rope"):
        for b in pl.range(b_dim):
            first_t = b * S
            first_pos_b = pl.read(position_ids, [first_t])
            cmp_offset_b = COMPRESS_RATIO - (first_pos_b % COMPRESS_RATIO)
            cmp_pos_b = pl.cast(first_pos_b + cmp_offset_b - COMPRESS_RATIO, pl.INDEX)
            cmp_cos_row = freqs_cos[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_sin_row = freqs_sin[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_cos[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_cos_row, target_type=pl.FP32)
            cmp_sin[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_sin_row, target_type=pl.FP32)
            for s in pl.range(S):
                t = b * S + s
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                step_cos_row = pl.cast(freqs_cos[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                step_sin_row = pl.cast(freqs_sin[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                rope_cos_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(step_cos_row, target_type=pl.BF16, mode="rint")
                rope_sin_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(step_sin_row, target_type=pl.BF16, mode="rint")

    rope_interleave(cmp_cos, cmp_sin, cmp_cos_il, cmp_sin_signed)

    x_normed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    # Dispatch barrier: kv_proj_matmul resolves one hop after rms_norm.
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)        # unused on HCA path
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, late_dep,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    for wb_blk in pl.spmd(wb_blocks, name_hint="hca_cache_writeback"):
        wb_t0 = wb_blk * HCA_WB_TOKEN_TILE
        for write_dt in pl.range(HCA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            write_row_i64 = pl.read(ori_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]

    cmp_kv_proj = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    compressor_ratio128(
        x_normed, cmp_kv_proj,
        compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_cos_il, cmp_sin_signed, cmp_kv,
        position_ids, cmp_slot_mapping, state_slot_mapping,
        late_dep,
    )

    # Sparse-index build over an SPMD of 8 tokens per block: window column k -> ring
    # slot k, live iff k <= abs_pos, with the compressed-slot ramp fused into the same block.
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    topk_all = pl.create_tensor([t_dim, HCA_CMP_TOPK], dtype=pl.INT32)
    for topk_block in pl.spmd(topk_blocks, name_hint="hca_cache_topk"):
        topk_t0 = topk_block * HCA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(HCA_TOPK_TOKEN_TILE):
            topk_t = topk_t0 + topk_dt
            if topk_t < t_dim:
                topk_b = topk_t // S
                topk_abs_pos = pl.read(position_ids, [topk_t])

                topk_cmp_valid = pl.min(
                    HCA_TOPK_LIMIT,
                    pl.min((topk_abs_pos + 1) // COMPRESS_RATIO, pl.read(kv_seq_lens, [topk_b]) // COMPRESS_RATIO),
                )
                for topk_ck in pl.range(HCA_CMP_TOPK):
                    if topk_ck < topk_cmp_valid:
                        pl.write(topk_all, [topk_t, topk_ck], pl.cast(topk_ck, pl.INT32))
                    else:
                        pl.write(topk_all, [topk_t, topk_ck], pl.cast(-1, pl.INT32))

    o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)
    o_packed_heads, heads_dep = sparse_attn_hca(
        q, kv_cache, window_swa_indices,
        cmp_kv, cmp_block_table, topk_all,
        attn_sink, rope_cos_t, rope_sin_t,
        o_packed_heads,
    )
    attn_out = decode_o_proj_tp1(
        o_packed_heads,
        wo_a, wo_b, wo_b_scale,
        attn_out, heads_dep,
    )

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def decode_hca_tp1_test(
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x_hc.bind_dynamic(0, T_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    x_out.bind_dynamic(0, T_DYN)

    decode_hca_tp1(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, state_slot_mapping,
        position_ids, kv_seq_lens,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
    )
    return x_out


# fixture
FIXTURE_WINDOW_BLOCKS = (WIN + BLOCK_SIZE - 1) // BLOCK_SIZE
FIXTURE_CMP_SLOTS = (0, 31, 32, 63, 64, 95, 96, 127)
FIXTURE_CMP_LOGICAL_BLOCKS = (max(FIXTURE_CMP_SLOTS) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
FIXTURE_CMP_BLOCKS_PER_RANK = CMP_BLOCK_NUM // TP_SIZE
FIXTURE_OUTPUT_SENTINEL = -7.0

if max(FIXTURE_CMP_SLOTS) >= CMP_TOPK:
    raise ValueError("HCA fixture compressed slots exceed the configured top-k capacity")
if FIXTURE_WINDOW_BLOCKS > ORI_BLOCK_NUM:
    raise ValueError("HCA fixture window exceeds the original KV cache capacity")
if FIXTURE_CMP_LOGICAL_BLOCKS > CMP_MAX_BLOCKS:
    raise ValueError("HCA fixture compressed slots exceed the compressed block table")
if CMP_BLOCK_NUM % TP_SIZE != 0:
    raise ValueError("HCA fixture requires equal compressed-cache partitions across TP ranks")
if (LOCAL_T // S) * FIXTURE_CMP_LOGICAL_BLOCKS > FIXTURE_CMP_BLOCKS_PER_RANK:
    raise ValueError("HCA fixture compressed-cache partition is too small for the local requests")
if O_LORA < 3 * HEADS_PER_GROUP:
    raise ValueError("HCA fixture output projection needs three observable rows per local head")


def build_tp_tensor_specs(local_t, start_pos=None):
    """Build rank-major HCA inputs with ragged compressed-page metadata."""
    import torch

    from golden import ScalarSpec, TensorSpec
    from utils import (
        block_table,
        position_ids_from_starts,
        resolve_start_positions,
        swa_indices_and_lens,
        token_local_rope,
    )

    if local_t < VALID_TOKEN_TILE or local_t > LOCAL_T or local_t % VALID_TOKEN_TILE or local_t % S:
        raise ValueError(f"local_t must be a multiple of {VALID_TOKEN_TILE} in [{VALID_TOKEN_TILE}, {LOCAL_T}]")
    local_batch = local_t // S
    starts = resolve_start_positions(
        start_pos, batch=local_batch, seq=S, max_seq_len=MAX_SEQ_LEN,
        default_fn=lambda: torch.full((local_batch,), WIN - 1, dtype=torch.int32),
    )
    positions = position_ids_from_starts(starts, seq=S)
    flat_positions = positions.reshape(-1).contiguous()
    request_ids = torch.arange(local_batch, dtype=torch.int32).repeat_interleave(S)

    raw_table = block_table(
        batch=local_batch,
        table_blocks=ORI_MAX_BLOCKS,
        physical_blocks=ORI_BLOCK_NUM,
    )
    sources, _ = swa_indices_and_lens(
        positions, raw_table, block_size=BLOCK_SIZE, window=WIN,
    )
    for request in range(local_batch):
        for local in range(S):
            query = request * S + local
            overlay_begin = int((sources[query] >= 0).sum().item()) - local - 1
            for overlay_local in range(local + 1):
                sources[query, overlay_begin + overlay_local] = SWA_SOURCE_OVERLAY_BASE - (request * S + overlay_local)

    page_entries = []
    page_offsets = [0]
    windows = []
    work_offsets = [0]
    work_query_ids = []
    work_rows = []
    work_valid = []
    next_page = 0
    for request in range(local_batch):
        request_positions = positions[request]
        valid_end = int((int(request_positions[-1]) + 1) // COMPRESS_RATIO)
        page_count = max(1, (valid_end + BLOCK_SIZE - 1) // BLOCK_SIZE)
        if next_page + page_count > CMP_BLOCK_NUM:
            raise ValueError("HCA fixture exceeds compressed KV pool")
        page_entries.extend((next_page + page, 7) for page in range(page_count))
        page_offsets.append(len(page_entries))
        windows.append((0, valid_end, 0))
        next_page += page_count
        for local in range(S):
            query = request * S + local
            visible = int((int(request_positions[local]) + 1) // COMPRESS_RATIO)
            for row in range(0, visible, HCA_ROWS_PER_SHARD):
                work_query_ids.append(query)
                work_rows.append(row)
                work_valid.append(min(HCA_ROWS_PER_SHARD, visible - row))
            work_offsets.append(len(work_query_ids))

    if not page_entries:
        page_entries = [(0, 7)]
    work_count = len(work_query_ids)
    cos, sin = token_local_rope(M, COMPRESS_RATIO, flat_positions, dtype=torch.bfloat16)
    q = torch.zeros(TP_SIZE, local_t, H, HEAD_DIM, dtype=torch.bfloat16)
    q_rank = torch.arange(TP_SIZE, dtype=torch.float32).reshape(TP_SIZE, 1, 1)
    q_token = torch.arange(local_t, dtype=torch.float32).reshape(1, local_t, 1)
    q_head = torch.arange(H, dtype=torch.float32).reshape(1, 1, H)
    q[..., 0] = (q_rank + q_token * 0.25 + q_head * 0.0625).to(torch.bfloat16)
    ori = torch.zeros(TP_SIZE, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    ori[:, :, :, 0, 0] = 0.25
    current = torch.zeros(TP_SIZE, local_t, HEAD_DIM, dtype=torch.bfloat16)
    current[:, :, 0] = 0.25
    cmp = torch.zeros(TP_SIZE, CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    cmp[:, :, :, 0, 0] = 0.25
    sink = (4.0 + torch.arange(H, dtype=torch.float32).remainder(HEADS_PER_GROUP) * 0.125)
    sink = sink.reshape(1, H).expand(TP_SIZE, H).clone()
    wo_a = torch.zeros(TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16)
    for rank in range(TP_SIZE):
        for group in range(LOCAL_O_GROUPS):
            scale = (rank + 1) * (group + 1) * 0.125
            for head in range(HEADS_PER_GROUP):
                col = head * HEAD_DIM
                wo_a[rank, group, head, col] = scale * (head + 1)
                wo_a[rank, group, HEADS_PER_GROUP + head, col + NOPE_DIM] = scale * 0.75
                wo_a[rank, group, 2 * HEADS_PER_GROUP + head, col + NOPE_DIM + 1] = -scale * 0.5
    base = torch.arange(D * LOCAL_O_WIDTH, dtype=torch.int32).reshape(D, LOCAL_O_WIDTH)
    wo_b = (base.unsqueeze(0) + torch.arange(TP_SIZE, dtype=torch.int32).reshape(TP_SIZE, 1, 1)).remainder(7).sub(3).to(torch.int8)
    wo_b_scale = (torch.arange(D, dtype=torch.float32).remainder(4) * 0.25 + 0.5).reshape(1, D).expand(TP_SIZE, D).clone()
    return [
        TensorSpec("q", list(q.shape), torch.bfloat16, init_value=q),
        TensorSpec("ori_kv", list(ori.shape), torch.bfloat16, init_value=ori),
        TensorSpec("current_kv", list(current.shape), torch.bfloat16, init_value=current),
        TensorSpec("swa_sources", [TP_SIZE, local_t, WIN], torch.int32, init_value=sources.reshape(1, local_t, WIN).expand(TP_SIZE, -1, -1).clone()),
        TensorSpec("cmp_kv", list(cmp.shape), torch.bfloat16, init_value=cmp),
        TensorSpec("query_request_ids", [TP_SIZE, local_t], torch.int32, init_value=request_ids.reshape(1, -1).expand(TP_SIZE, -1).clone()),
        TensorSpec("hca_pages", [TP_SIZE, len(page_entries), 2], torch.int32, init_value=torch.tensor(page_entries, dtype=torch.int32).reshape(1, -1, 2).expand(TP_SIZE, -1, -1).clone()),
        TensorSpec("hca_page_offsets", [TP_SIZE, len(page_offsets)], torch.int32, init_value=torch.tensor(page_offsets, dtype=torch.int32).reshape(1, -1).expand(TP_SIZE, -1).clone()),
        TensorSpec("hca_windows", [TP_SIZE, local_batch, 3], torch.int32, init_value=torch.tensor(windows, dtype=torch.int32).reshape(1, local_batch, 3).expand(TP_SIZE, -1, -1).clone()),
        TensorSpec("request_epochs", [TP_SIZE, local_batch], torch.int32, init_value=torch.full((TP_SIZE, local_batch), 7, dtype=torch.int32)),
        TensorSpec("hca_query_work_offsets", [TP_SIZE, len(work_offsets)], torch.int32, init_value=torch.tensor(work_offsets, dtype=torch.int32).reshape(1, -1).expand(TP_SIZE, -1).clone()),
        TensorSpec("hca_work_query_ids", [TP_SIZE, work_count], torch.int32, init_value=torch.tensor(work_query_ids, dtype=torch.int32).reshape(1, -1).expand(TP_SIZE, -1).clone()),
        TensorSpec("hca_work_row_begin", [TP_SIZE, work_count], torch.int32, init_value=torch.tensor(work_rows, dtype=torch.int32).reshape(1, -1).expand(TP_SIZE, -1).clone()),
        TensorSpec("hca_work_valid_rows", [TP_SIZE, work_count], torch.int32, init_value=torch.tensor(work_valid, dtype=torch.int32).reshape(1, -1).expand(TP_SIZE, -1).clone()),
        TensorSpec("attn_sink", [TP_SIZE, H], torch.float32, init_value=sink),
        TensorSpec("freqs_cos", [TP_SIZE, local_t, ROPE_DIM], torch.bfloat16, init_value=cos.reshape(1, local_t, ROPE_DIM).expand(TP_SIZE, -1, -1).clone()),
        TensorSpec("freqs_sin", [TP_SIZE, local_t, ROPE_DIM], torch.bfloat16, init_value=sin.reshape(1, local_t, ROPE_DIM).expand(TP_SIZE, -1, -1).clone()),
        TensorSpec("wo_a", list(wo_a.shape), torch.bfloat16, init_value=wo_a),
        TensorSpec("wo_b", list(wo_b.shape), torch.int8, init_value=wo_b),
        TensorSpec("wo_b_scale", list(wo_b_scale.shape), torch.float32, init_value=wo_b_scale),
        TensorSpec("o_local", [TP_SIZE, LOCAL_T_PAD, D], torch.bfloat16, init_value=FIXTURE_OUTPUT_SENTINEL, is_output=True),
        ScalarSpec("local_t", torch.int32, local_t),
    ]


def golden_decode_hca(tensors):
    """Compute the controlled HCA heads, sharded O projection, and reduced rows."""
    import torch

    local_t = int(tensors["local_t"])
    group_t = TP_SIZE * local_t
    q = tensors["q"].float()
    window_kv = tensors["ori_kv"][:, 0, 0, 0].float()

    window_mi = torch.einsum("rthd,rd->rth", q, window_kv)
    window_mi = (window_mi * M.softmax_scale).unsqueeze(-1)
    window_count = torch.zeros(TP_SIZE, local_t, dtype=torch.float32)
    cmp_count = torch.zeros(TP_SIZE, local_t, dtype=torch.float32)
    sources = tensors["swa_sources"]
    request_ids = tensors["query_request_ids"]
    offsets = tensors["hca_query_work_offsets"]
    valid_rows = tensors["hca_work_valid_rows"]
    for rank in range(TP_SIZE):
        for token in range(local_t):
            request = int(request_ids[rank, token])
            for source in sources[rank, token].tolist():
                if source >= 0:
                    window_count[rank, token] += 1
                elif source <= SWA_SOURCE_OVERLAY_BASE:
                    overlay = SWA_SOURCE_OVERLAY_BASE - int(source)
                    if 0 <= overlay < local_t and int(request_ids[rank, overlay]) == request and overlay <= token:
                        window_count[rank, token] += 1
            begin = int(offsets[rank, token])
            end = int(offsets[rank, token + 1])
            if end > begin:
                cmp_count[rank, token] = valid_rows[rank, begin:end].to(torch.float32).sum()
    window_li = window_count.unsqueeze(-1).unsqueeze(-1).expand_as(window_mi)
    window_oi = window_kv.reshape(TP_SIZE, 1, 1, HEAD_DIM) * window_count.unsqueeze(-1).unsqueeze(-1)

    cmp_mi = window_mi
    cmp_li = cmp_count.unsqueeze(-1).unsqueeze(-1).expand_as(cmp_mi)
    cmp_oi = torch.zeros_like(q)
    cmp_oi[..., 0] = 0.25 * cmp_count.unsqueeze(-1)

    score_max = torch.maximum(window_mi, cmp_mi)
    window_alpha = torch.exp(window_mi - score_max)
    cmp_beta = torch.exp(cmp_mi - score_max)
    li = window_alpha * window_li + cmp_beta * cmp_li
    oi = window_alpha * window_oi + cmp_beta * cmp_oi
    sink = tensors["attn_sink"].float().reshape(TP_SIZE, 1, H, 1)
    head_value = oi / (li + torch.exp(sink - score_max))

    rope_pair = head_value[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = tensors["freqs_cos"][..., :HALF_ROPE].float().unsqueeze(2)
    sin_half = tensors["freqs_sin"][..., :HALF_ROPE].float().unsqueeze(2)
    inverse_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inverse_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    inverse_rope = torch.stack((inverse_even, inverse_odd), dim=-1).flatten(-2)
    head_value = torch.cat((head_value[..., :NOPE_DIM], inverse_rope), dim=-1).to(torch.bfloat16)

    attention_grouped = head_value.reshape(
        TP_SIZE, local_t, O_GROUPS, HEADS_PER_GROUP, HEAD_DIM,
    )
    attention_grouped = attention_grouped.permute(0, 2, 1, 3, 4)
    attention_grouped = attention_grouped.reshape(
        TP_SIZE, O_GROUPS, local_t, O_GROUP_IN,
    )

    attention_local = torch.zeros(TP_SIZE, LOCAL_O_GROUPS, group_t, O_GROUP_IN, dtype=torch.bfloat16)
    for destination_rank in range(TP_SIZE):
        for local_group in range(LOCAL_O_GROUPS):
            global_group = destination_rank * LOCAL_O_GROUPS + local_group
            group_rows = attention_grouped[:, global_group].reshape(group_t, O_GROUP_IN)
            attention_local[destination_rank, local_group] = group_rows

    partials = torch.zeros(TP_SIZE, group_t, D, dtype=torch.float32)
    for rank in range(TP_SIZE):
        attention = attention_local[rank].float()
        o_a = torch.einsum("gti,gri->gtr", attention, tensors["wo_a"][rank].float())
        row_amax = o_a.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_q = INT8_SCALE_MAX / row_amax
        o_a_i8 = torch.round(o_a * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
        scale_dq = 1.0 / scale_q
        wo_b = tensors["wo_b"][rank].reshape(D, LOCAL_O_GROUPS, O_LORA)
        for local_group in range(LOCAL_O_GROUPS):
            group_i32 = o_a_i8[local_group].to(torch.int32)
            weight_i32 = wo_b[:, local_group].to(torch.int32)
            group_partial = group_i32 @ weight_i32.T
            partials[rank] += group_partial.float() * scale_dq[local_group]
        partials[rank] *= tensors["wo_b_scale"][rank].float().reshape(1, D)

    reduced = partials.sum(dim=0)
    tensors["o_local"].fill_(FIXTURE_OUTPUT_SENTINEL)
    for rank in range(TP_SIZE):
        row_start = rank * local_t
        tensors["o_local"][rank, :local_t] = reduced[row_start : row_start + local_t].to(torch.bfloat16)


def build_o_local_compare(local_t):
    """Compare valid token rows and require the poisoned capacity tail to survive."""
    import torch

    from golden import ratio_allclose

    prefix_compare = ratio_allclose(
        atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005,
        valid_rows=local_t, valid_axis=1,
    )

    def compare(actual, expected, **kwargs):
        actual_tail = actual[:, local_t:]
        expected_tail = expected[:, local_t:]
        if not torch.equal(actual_tail, expected_tail):
            mismatch_count = int((actual_tail != expected_tail).sum().item())
            return False, f"    inactive token tail mismatch count={mismatch_count}"
        return prefix_compare(actual, expected, **kwargs)

    compare.__name__ = f"hca_output_prefix_and_tail(local_t={local_t})"
    return compare


def golden_decode_hca_tp1(tensors):
    """End-to-end orchestration for the ratio=128 (HCA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==128 path: main compressor only,
    no indexer, compress_topk_idxs computed deterministically) + Block.hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm
    from decode_compressor_ratio128 import golden_compressor
    from decode_sparse_attn_hca import golden_sparse_attn

    tokens = tensors["x_hc"].shape[0]
    batch = tokens // S
    from hc_post import golden_hc_post

    # ---- Block.hc_pre ----
    x_mixed = torch.zeros(tokens, D, dtype=torch.bfloat16)
    post_t = torch.zeros(tokens, HC_MULT)
    comb_t = torch.zeros(tokens, HC_MULT * HC_MULT)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    # Attention.forward, ratio==128 branch
    position_ids = tensors["position_ids"].to(torch.int64)
    kv_seq_lens = tensors["kv_seq_lens"].to(torch.int64)
    ratio = COMPRESS_RATIO
    rd = ROPE_HEAD_DIM

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_T = torch.empty(tokens, rd, dtype=freqs_cos.dtype)
    rope_sin_T = torch.empty(tokens, rd, dtype=freqs_sin.dtype)
    for t in range(tokens):
        pos = int(position_ids[t].item())
        rope_cos_T[t] = freqs_cos[pos]
        rope_sin_T[t] = freqs_sin[pos]

    # q + win kv (W8A8 q_proj)
    q = torch.zeros(tokens, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(tokens, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(tokens, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    golden_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_T,
        "rope_sin": rope_sin_T,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,                                                              # qr unused on HCA path
        "qr_scale": qr_scale,
    })

    kv_cache = tensors["kv_cache"]
    window_swa_indices = tensors["window_swa_indices"]
    cmp_kv = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]
    attn_out = torch.zeros(tokens, D, dtype=torch.bfloat16)

    half_rd = rd // 2
    cmp_cos = torch.empty(batch, half_rd, dtype=torch.float32)
    cmp_sin = torch.empty(batch, half_rd, dtype=torch.float32)
    for b in range(batch):
        first_pos_b = int(position_ids[b * S].item())
        cmp_offset_b = ratio - (first_pos_b % ratio)
        cmp_pos_b = first_pos_b + cmp_offset_b - ratio
        cmp_cos[b] = freqs_cos[cmp_pos_b, :half_rd].float()
        cmp_sin[b] = freqs_sin[cmp_pos_b, :half_rd].float()

    # The compressor ABI is token-major flat.
    cmp_kv_proj = torch.zeros(tokens, HEAD_DIM, dtype=torch.float32)
    position_ids_flat = position_ids.reshape(-1).to(torch.int32).contiguous()
    cmp_slot_mapping_flat = tensors["cmp_slot_mapping"].reshape(-1).to(torch.int64).contiguous()
    state_slot_mapping_flat = tensors["state_slot_mapping"].reshape(-1).to(torch.int64).contiguous()
    golden_compressor({
        "x": x_normed,
        "kv": cmp_kv_proj,
        "compress_state": tensors["compress_state"],
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "cos": cmp_cos,
        "sin": cmp_sin,
        "cmp_kv_cache": cmp_kv,
        "position_ids": position_ids_flat,
        "cmp_slot_mapping": cmp_slot_mapping_flat,
        "state_slot_mapping": state_slot_mapping_flat,
    })

    ori_slot_mapping = tensors["ori_slot_mapping"].to(torch.int64)
    for t in range(tokens):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]

    topk_all = torch.full((tokens, HCA_CMP_TOPK), -1, dtype=torch.int32)
    for t in range(tokens):
        b = t // S
        abs_pos = int(position_ids[t].item())
        cmp_valid = min(HCA_TOPK_LIMIT, (abs_pos + 1) // ratio, int(kv_seq_lens[b].item()) // ratio)
        if cmp_valid:
            topk_all[t, :cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32)

    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "window_swa_indices": window_swa_indices,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "cmp_sparse_indices": topk_all,
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # ===== Block.hc_post =====
    y = torch.zeros(tokens, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })

    tensors["x_out"][:] = y


def build_tensor_specs(start_pos=None, batch=B):
    tokens = batch * S
    import torch
    from utils import (
        block_table,
        compressed_slot_mapping,
        hca_decode_start_set,
        kv_seq_lens_from_starts,
        ori_slot_mapping,
        position_ids_from_starts,
        resolve_start_positions,
        state_slot_mapping,
        swa_indices_and_lens,
    )
    from golden import TensorSpec
    from utils import build_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def quant_w_per_row(w):
        amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.unsqueeze(-1)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x_hc():
        return torch.empty(tokens, HC_MULT, D).uniform_(-1, 1)
    # Real layer-9 (HCA, ratio-128) hc_attn scale/base; fn is synthetic at the real magnitude.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0495
    def init_hc_attn_scale():
        return torch.tensor([0.079046, 0.04213, 0.121901])
    def init_hc_attn_base():
        return torch.tensor([
            -3.3004, 2.5553, -2.2787, -3.4925,
            -3.8197, -3.4161, -2.7144, -2.9181,
            2.362, -2.4746, -2.1352, -3.2216,
            -4.474, 2.2488, -2.1053, -3.1675,
            -2.8362, -1.9042, 2.0432, -3.062,
            -2.7902, -3.0908, -3.002, 3.1161,
        ])
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return torch.randn(D, Q_LORA) / D ** 0.5
    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / D ** 0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    # BF16 weight std and RMSNorm gamma mean/std, averaged over DeepSeek-V4-Flash-0731
    # layers 7/9 (the ratio-128 HCA main compressor).
    def init_cmp_wkv():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0240
    def init_cmp_wgate():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0309
    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.0332
    def init_cmp_norm_w():
        return 0.0982 + 0.0539 * torch.randn(HEAD_DIM)
    def init_compress_state():
        return torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
    def init_compress_state_block_table():
        return block_table(
            batch=batch,
            table_blocks=COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=COMPRESS_STATE_PHYSICAL_BLOCKS,
        )
    def init_kv_cache():
        return init_normalized_cache((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))
    def init_cmp_kv():
        return init_normalized_cache((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_window_block_table():
        return block_table(batch=batch, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_cmp_block_table():
        return block_table(
            batch=batch,
            table_blocks=CMP_MAX_BLOCKS,
            physical_blocks=CMP_BLOCK_NUM,
        )

    def init_attn_sink():
        return torch.zeros(H)
    def init_default_start_pos():
        # Canonical HCA start-position set (ratio-128 compressor branches + 8k long-context).
        return hca_decode_start_set(
            batch=batch, compress_ratio=COMPRESS_RATIO, state_block_size=COMPRESS_STATE_BLOCK_SIZE)
    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=batch,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )
    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S).reshape(-1).contiguous()
    def init_kv_seq_lens():
        return kv_seq_lens_from_starts(init_start_pos(), seq=S)
    def init_ori_slot_mapping():
        return ori_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_window_block_table(),
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()
    def init_window_swa_metadata():
        return swa_indices_and_lens(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_window_block_table(),
            block_size=BLOCK_SIZE,
            window=WIN,
        )
    def init_window_swa_indices():
        return init_window_swa_metadata()[0].contiguous()
    def init_window_swa_lens():
        return init_window_swa_metadata()[1].contiguous()
    def init_cmp_slot_mapping():
        positions = position_ids_from_starts(init_start_pos(), seq=S)
        return compressed_slot_mapping(
            positions,
            init_cmp_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()
    def init_state_slot_mapping():
        return state_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_compress_state_block_table(),
            state_block_size=COMPRESS_STATE_BLOCK_SIZE,
        ).reshape(-1).contiguous()
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.bfloat16, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("compress_state_block_table", [batch, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [batch, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("ori_slot_mapping", [tokens], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("window_swa_indices", [tokens, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("window_swa_lens", [tokens], torch.int32, init_value=init_window_swa_lens),
        TensorSpec("cmp_slot_mapping", [tokens], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [tokens], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=init_position_ids),
        TensorSpec("kv_seq_lens", [batch], torch.int32, init_value=init_kv_seq_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [tokens, HC_MULT, D], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    default_devices = ",".join(str(rank) for rank in range(TP_SIZE))
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES), help="tensor-parallel world size")
    parser.add_argument(
        "-d", "--device", type=str, default=default_devices,
        help=f"comma-separated device ids; need exactly {TP_SIZE}",
    )
    parser.add_argument("--case", choices=("all", "max", "subcapacity"), default="all")
    parser.add_argument(
        "--batch", type=int, default=None,
        help="local request count for a single length fixture; 1M requires batch=1",
    )
    parser.add_argument("--start-pos", type=int, default=None,
                        help="absolute decode start position for the generated HCA metadata")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.tp != TP_SIZE:
        parser.error(f"--tp must remain {TP_SIZE} after import-time specialization")
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(f"--device must be a comma-separated integer list, got {args.device!r}")
    if len(device_ids) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {device_ids}")
    if any(device < 0 for device in device_ids):
        parser.error(f"--device IDs must be non-negative, got {device_ids}")
    if len(set(device_ids)) != TP_SIZE:
        parser.error(f"need {TP_SIZE} distinct devices, got {device_ids}")

    if args.batch is not None:
        if args.batch < 1 or args.batch > LOCAL_T // S:
            parser.error(f"--batch must be in [1, {LOCAL_T // S}], got {args.batch}")
        case_local_t = {"max": args.batch * S}
        if args.case == "all":
            args.case = "max"
    else:
        case_local_t = {"max": LOCAL_T, "subcapacity": LOCAL_T - VALID_TOKEN_TILE}
    selected_cases = tuple(case_local_t) if args.case == "all" else (args.case,)
    for case in selected_cases:
        local_t = case_local_t[case]
        result = run_jit(
            fn=l3_decode_hca,
            specs=build_tp_tensor_specs(local_t, start_pos=args.start_pos),
            golden_fn=golden_decode_hca,
            compile_only=args.compile_only,
            compile_cfg=dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            ),
            runtime_cfg=dict(platform=args.platform),
            compare_fn={"o_local": build_o_local_compare(local_t)},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
