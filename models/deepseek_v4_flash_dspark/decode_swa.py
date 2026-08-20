# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 SWA full-layer TP, TP1, and tensor-parallel output entries."""


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
    KV_ORI_BLOCK_NUM,
    DECODE_SEQ,
    BLOCK_SIZE,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
)
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
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
from decode_sparse_attn_swa import ATTN_K_TILE, PADDED_TOPK, T_PAD, sparse_attn_swa

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")


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
# SWA-local context ceiling. The global model ceiling remains unchanged.
MAX_SEQ_LEN = 1_048_576
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local (SWA: ratio-0, no compressor/indexer)
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
TOPK = WIN                          # SWA: sparse_attn topk = window only
SPARSE_IDX_TOPK = M.index_topk      # sparse_attn module's IDX_TOPK (static shape contract)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK
SPARSE_CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS

# tiling
BIAS_T_TILE = 8  # sparse_bias row block; T is a multiple of 8 by the batch contract
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE
NEG_INF = -1.0e20

if T != LOCAL_T:
    raise ValueError(f"SWA token capacity {T} must equal TP-local token capacity {LOCAL_T}")
if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"SWA padded token capacity {T_PAD} must equal TP capacity {LOCAL_T_PAD}")


@pl.jit.inline
def decode_swa_output(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    sparse_bias: pl.Tensor[[T_DYN, PADDED_TOPK], pl.FP32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    o_local: pl.Tensor[[LOCAL_T_PAD, D], pl.BF16],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Run sparse attention and the distributed sharded O projection."""
    o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    sparse_attn_swa(
        q, ori_kv, swa_indices, sparse_bias,
        attn_sink, freqs_cos, freqs_sin,
        o_packed_heads,
    )

    attention_grouped = pl.reshape(o_packed_heads, [O_GROUPS * LOCAL_T_PAD, O_GROUP_IN])
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


@pl.jit
def decode_swa_output_test(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
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
    """Bind dynamic inputs for standalone SWA output-half validation."""
    q.bind_dynamic(0, T_DYN)
    ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    swa_indices.bind_dynamic(0, T_DYN)
    swa_lens.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)

    t_dim = pl.tensor.dim(q, 0)
    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_valid_bias"):
        valid_col = pl.cast(pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32), target_type=pl.FP32)
        for bias_block in pl.range(t_dim // BIAS_T_TILE):
            token_start = bias_block * BIAS_T_TILE
            zero_rows = pl.full([BIAS_T_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0)
            valid_cols = pl.col_expand(zero_rows, valid_col)
            lens_slice = swa_lens[token_start : token_start + BIAS_T_TILE]
            lens_col = pl.reshape(lens_slice, [BIAS_T_TILE, 1])
            lens_fp32 = pl.cast(lens_col, target_type=pl.FP32)
            valid = pl.neg(pl.row_expand_sub(valid_cols, lens_fp32))
            valid = pl.maximum(valid, 0.0)
            valid = pl.minimum(valid, 1.0)
            invalid = pl.sub(valid, 1.0)
            sparse_bias[token_start : token_start + BIAS_T_TILE, 0:ATTN_K_TILE] = pl.mul(invalid, -NEG_INF)

    return decode_swa_output(
        q, ori_kv, swa_indices, sparse_bias,
        attn_sink, freqs_cos, freqs_sin,
        wo_a, wo_b, wo_b_scale, o_local,
        attention_window, attention_signal, o_window, o_signal,
        group_base, tp_rank, local_t,
    )


@pl.jit.host
def l3_decode_swa_output_test(
    q: pl.Tensor[[TP_SIZE, T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[TP_SIZE, T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    freqs_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    o_local: pl.InOut[pl.Tensor[[TP_SIZE, LOCAL_T_PAD, D], pl.BF16]],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch the SWA output half on one physical TP group."""
    q.bind_dynamic(1, T_DYN)
    ori_kv.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    swa_indices.bind_dynamic(1, T_DYN)
    swa_lens.bind_dynamic(1, T_DYN)
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
        decode_swa_output_test(
            q[rank], ori_kv[rank], swa_indices[rank], swa_lens[rank],
            attn_sink[rank], freqs_cos[rank], freqs_sin[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank], o_local[rank],
            attention_window, attention_signal, o_window, o_signal,
            0, rank, local_t, device=rank,
        )


@pl.jit.inline
def decode_swa(
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # sharded o_proj
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    # TP communication
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Run the complete SWA layer with tensor-parallel output."""
    t_dim = pl.tensor.dim(x_hc, 0)
    bias_blocks = t_dim // BIAS_T_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_t)
    # Dispatch barrier: kv_proj_matmul resolves one hop after rms_norm.
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed_t, wq_a, wq_b, wq_b_scale, wkv,
        freqs_cos, freqs_sin, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, late_dep,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    cache_rows = ori_block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [cache_rows, HEAD_DIM])
    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_cache_insert_valid_bias"):
        for write_t in pl.range(t_dim):
            write_row_i64 = pl.read(swa_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]
        valid_col = pl.cast(pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32), target_type=pl.FP32)
        for bias_block in pl.range(bias_blocks):
            token_start = bias_block * BIAS_T_TILE
            zero_rows = pl.full([BIAS_T_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0)
            valid_cols = pl.col_expand(zero_rows, valid_col)
            lens_slice = swa_lens[token_start : token_start + BIAS_T_TILE]
            lens_col = pl.reshape(lens_slice, [BIAS_T_TILE, 1])
            lens_fp32 = pl.cast(lens_col, target_type=pl.FP32)
            valid = pl.neg(pl.row_expand_sub(valid_cols, lens_fp32))
            valid = pl.maximum(valid, 0.0)
            valid = pl.minimum(valid, 1.0)
            invalid = pl.sub(valid, 1.0)
            sparse_bias[token_start : token_start + BIAS_T_TILE, 0 : ATTN_K_TILE] = pl.mul(invalid, -NEG_INF)

    o_local = pl.create_tensor([LOCAL_T_PAD, D], dtype=pl.BF16)
    o_local, attention_signal, o_signal = decode_swa_output(
        q, kv_cache, swa_indices, sparse_bias,
        attn_sink, freqs_cos, freqs_sin,
        wo_a, wo_b, wo_b_scale, o_local,
        attention_window, attention_signal,
        o_window, o_signal,
        group_base, tp_rank, local_t,
    )

    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_o_local"):
        for token_start in pl.range(0, t_dim, BIAS_T_TILE):
            o_local_rows = o_local[token_start : token_start + BIAS_T_TILE, 0 : D]
            attn_out[token_start : token_start + BIAS_T_TILE, 0 : D] = o_local_rows
    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def decode_swa_test(
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
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Bind dynamic inputs for the complete tensor-parallel SWA layer."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(0, T_DYN)
    swa_indices.bind_dynamic(0, T_DYN)
    swa_lens.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)

    decode_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
        attention_window, attention_signal, o_window, o_signal,
        group_base, tp_rank, local_t,
    )
    return x_out


@pl.jit.host
def l3_decode_swa(
    x_hc: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[TP_SIZE, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[TP_SIZE, T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch the complete SWA layer on one tensor-parallel group."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(1, T_DYN)
    swa_indices.bind_dynamic(1, T_DYN)
    swa_lens.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    x_out.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        decode_swa_test(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank], wkv[rank],
            gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            kv_cache[rank], swa_slot_mapping[rank], swa_indices[rank], swa_lens[rank], position_ids[rank],
            attn_sink[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            x_out[rank],
            attention_window, attention_signal, o_window, o_signal,
            0, rank, local_t,
            device=rank,
        )


@pl.jit.inline
def decode_swa_tp1(
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    # Token-local RoPE: the host already materialized the active rows into
    # freqs_cos/freqs_sin ([T_DYN, ROPE_HEAD_DIM]), so the device no longer
    # gathers a full-context table through position_ids. position_ids stays in
    # the ABI for host admission and golden semantics only.
    t_dim = pl.tensor.dim(x_hc, 0)
    bias_blocks = t_dim // BIAS_T_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_t)
    # Dispatch barrier: kv_proj_matmul resolves one hop after rms_norm.
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed_t, wq_a, wq_b, wq_b_scale, wkv,
        freqs_cos, freqs_sin, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, late_dep,
    )

    # Commit current decode KV and build its additive padding mask in one task.
    # The SWA attention kernel reads every visible row through metadata-expanded
    # physical cache indices, so all cache writes must complete before it starts.
    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([t_dim, WIN], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_cache_insert_valid_bias"):
        for write_t in pl.range(t_dim):
            write_row_i64 = pl.read(swa_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]
        v_col = pl.cast(pl.arange(0, [1, WIN], dtype=pl.INT32), target_type=pl.FP32)
        for v_blk in pl.range(bias_blocks):
            v_t0 = v_blk * BIAS_T_TILE
            v_col_m = pl.col_expand(pl.full([BIAS_T_TILE, WIN], dtype=pl.FP32, value=0.0), v_col)
            v_lens = pl.cast(pl.reshape(swa_lens[v_t0 : v_t0 + BIAS_T_TILE], [BIAS_T_TILE, 1]), target_type=pl.FP32)
            v_valid = pl.minimum(
                pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0),
                1.0,
            )
            sparse_bias[v_t0 : v_t0 + BIAS_T_TILE, 0:WIN] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    o_packed_heads, heads_dep = sparse_attn_swa(
        q, kv_cache, swa_indices, sparse_bias,
        attn_sink, freqs_cos, freqs_sin,
        o_packed_heads,
    )
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T_PAD, O_GROUP_IN])
    attn_out = decode_o_proj_tp1(
        o_packed,
        wo_a, wo_b, wo_b_scale,
        attn_out, heads_dep,
    )

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def decode_swa_tp1_test(
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(0, T_DYN)
    swa_indices.bind_dynamic(0, T_DYN)
    swa_lens.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)

    decode_swa_tp1(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
    )
    return x_out


# fixture
TP_FIXTURE_WINDOW_BLOCKS = (WIN + BLOCK_SIZE - 1) // BLOCK_SIZE
TP_FIXTURE_OUTPUT_SENTINEL = -7.0

if TP_FIXTURE_WINDOW_BLOCKS > ORI_BLOCK_NUM:
    raise ValueError("SWA fixture window exceeds the original KV cache capacity")


def build_output_tensor_specs(local_t):
    """Build deterministic tensor-parallel SWA output-half inputs."""
    import torch

    from golden import ScalarSpec, TensorSpec

    if local_t < BIAS_T_TILE or local_t > LOCAL_T or local_t % BIAS_T_TILE != 0 or local_t % S != 0:
        raise ValueError(f"local_t must be a multiple of {BIAS_T_TILE} in [{BIAS_T_TILE}, {LOCAL_T}], got {local_t}")

    def init_q():
        q = torch.zeros(TP_SIZE, local_t, H, HEAD_DIM, dtype=torch.bfloat16)
        rank = torch.arange(TP_SIZE, dtype=torch.float32).reshape(TP_SIZE, 1, 1)
        token = torch.arange(local_t, dtype=torch.float32).reshape(1, local_t, 1)
        head = torch.arange(H, dtype=torch.float32).reshape(1, 1, H)
        q[..., 0] = (rank + token * 0.25 + head * 0.0625).to(torch.bfloat16)
        return q

    def init_ori_kv():
        shape = (ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache = torch.full(shape, float("nan"), dtype=torch.bfloat16)
        cache[:TP_FIXTURE_WINDOW_BLOCKS, :, 0, :] = 0.0
        cache[:TP_FIXTURE_WINDOW_BLOCKS, :, 0, 0] = 0.25
        return cache.unsqueeze(0).expand(TP_SIZE, *shape).clone()

    def init_swa_indices():
        window_row = torch.arange(WIN, dtype=torch.int32)
        return window_row.reshape(1, 1, WIN).expand(TP_SIZE, local_t, WIN).clone()

    def init_swa_lens():
        return torch.full((TP_SIZE, local_t), WIN, dtype=torch.int32)

    def init_attn_sink():
        head = torch.arange(H, dtype=torch.int32).remainder(HEADS_PER_GROUP).to(torch.float32)
        sink = 4.0 + head * 0.125
        return sink.reshape(1, H).expand(TP_SIZE, H).clone()

    def init_freqs_cos():
        cos = torch.ones(local_t, ROPE_HEAD_DIM, dtype=torch.bfloat16)
        return cos.unsqueeze(0).expand(TP_SIZE, local_t, ROPE_HEAD_DIM).clone()

    def init_freqs_sin():
        sin = torch.zeros(local_t, ROPE_HEAD_DIM, dtype=torch.bfloat16)
        return sin.unsqueeze(0).expand(TP_SIZE, local_t, ROPE_HEAD_DIM).clone()

    def init_wo_a():
        wo_a = torch.zeros(TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16)
        for rank in range(TP_SIZE):
            for local_group in range(LOCAL_O_GROUPS):
                shard_scale = (rank + 1) * (local_group + 1) * 0.125
                for head in range(HEADS_PER_GROUP):
                    wo_a[rank, local_group, head, head * HEAD_DIM] = shard_scale * (head + 1)
        return wo_a

    def init_wo_b():
        base = torch.arange(D * LOCAL_O_WIDTH, dtype=torch.int32).reshape(D, LOCAL_O_WIDTH)
        rank = torch.arange(TP_SIZE, dtype=torch.int32).reshape(TP_SIZE, 1, 1)
        return (base.unsqueeze(0) + rank).remainder(7).sub(3).to(torch.int8)

    def init_wo_b_scale():
        channel_scale = torch.arange(D, dtype=torch.int32).remainder(4).to(torch.float32) * 0.25 + 0.5
        return channel_scale.reshape(1, D).expand(TP_SIZE, D).clone()

    specs = [
        TensorSpec("q", [TP_SIZE, local_t, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [TP_SIZE, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("swa_indices", [TP_SIZE, local_t, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [TP_SIZE, local_t], torch.int32, init_value=init_swa_lens),
        TensorSpec("attn_sink", [TP_SIZE, H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [TP_SIZE, local_t, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [TP_SIZE, local_t, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("wo_a", [TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [TP_SIZE, D, LOCAL_O_WIDTH], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [TP_SIZE, D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec(
            "o_local", [TP_SIZE, LOCAL_T_PAD, D], torch.bfloat16,
            init_value=TP_FIXTURE_OUTPUT_SENTINEL, is_output=True,
        ),
        ScalarSpec("local_t", torch.int32, local_t),
    ]
    for spec in specs:
        if isinstance(spec, TensorSpec):
            spec.resident = "stacked"
    return specs


def golden_decode_swa_output(tensors):
    """Compute the controlled SWA heads, sharded O projection, and reduced rows."""
    import torch

    local_t = tensors["q"].shape[1]
    group_t = TP_SIZE * local_t
    q0 = tensors["q"][..., 0].float()
    kv0 = tensors["ori_kv"][:, 0, 0, 0, 0].float()
    sink = tensors["attn_sink"].float()
    score = q0 * kv0.reshape(TP_SIZE, 1, 1) * M.softmax_scale
    numerator = WIN * kv0.reshape(TP_SIZE, 1, 1)
    head_value = numerator / (WIN + torch.exp(sink.reshape(TP_SIZE, 1, H) - score))
    head_value = head_value.to(torch.bfloat16)

    attention_grouped = torch.zeros(TP_SIZE, O_GROUPS, local_t, O_GROUP_IN, dtype=torch.bfloat16)
    for group in range(O_GROUPS):
        for head in range(HEADS_PER_GROUP):
            global_head = group * HEADS_PER_GROUP + head
            group_col = head * HEAD_DIM
            attention_grouped[:, group, :, group_col] = head_value[:, :, global_head]

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
    tensors["o_local"].fill_(TP_FIXTURE_OUTPUT_SENTINEL)
    for rank in range(TP_SIZE):
        row_start = rank * local_t
        tensors["o_local"][rank, :local_t] = reduced[row_start : row_start + local_t].to(torch.bfloat16)


def build_o_local_compare(local_t):
    """Compare valid token rows and require the poisoned capacity tail to survive."""
    import torch

    from golden import ratio_allclose

    prefix_compare = ratio_allclose(
        atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0,
        valid_rows=local_t, valid_axis=1,
    )

    def compare(actual, expected, **kwargs):
        actual_tail = actual[:, local_t:]
        expected_tail = expected[:, local_t:]
        if not torch.equal(actual_tail, expected_tail):
            mismatch_count = int((actual_tail != expected_tail).sum().item())
            return False, f"    inactive token tail mismatch count={mismatch_count}"
        return prefix_compare(actual, expected, **kwargs)

    compare.__name__ = f"swa_output_prefix_and_tail(local_t={local_t})"
    return compare


def golden_decode_swa_tp1(tensors):
    """End-to-end orchestration for the ratio=0 (SWA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==0 path: no compressor,
    no indexer, no cmp_kv) + Block.hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm
    from decode_o_proj import golden_decode_o_proj_tp1
    from decode_sparse_attn_swa import golden_sparse_attn

    tokens = tensors["x_hc"].shape[0]
    from hc_post import golden_hc_post

    # Block.hc_pre
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

    # Attention.forward, ratio==0 branch. RoPE inputs are token-local.

    # q + win kv
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
        "rope_cos": tensors["freqs_cos"],
        "rope_sin": tensors["freqs_sin"],
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,                                                              # qr unused on SWA path
        "qr_scale": qr_scale,
    })

    kv_cache = tensors["kv_cache"]

    # Current decode KV is visible to SWA through the same physical cache slots
    # that metadata points at.
    swa_slot_mapping = tensors["swa_slot_mapping"].to(torch.int64)
    for t in range(tokens):
        write_row = int(swa_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]

    o_packed_heads = torch.zeros(O_GROUPS, T_PAD * HEADS_PER_GROUP, HEAD_DIM, dtype=torch.bfloat16)
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "swa_indices": tensors["swa_indices"],
        "swa_lens": tensors["swa_lens"],
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "o_packed_heads": o_packed_heads,
    })
    attn_out = golden_decode_o_proj_tp1(
        o_packed_heads, tensors["wo_a"], tensors["wo_b"], tensors["wo_b_scale"], tokens,
    )

    # Block.hc_post
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
    if batch <= 0 or tokens > LOCAL_T:
        raise ValueError(f"batch must produce between {S} and {LOCAL_T} tokens, got {tokens}")
    import torch
    from utils import (
        block_table,
        paged_slot_mapping,
        position_ids_from_starts,
        resolve_start_positions,
        swa_indices_and_lens,
        swa_decode_start_set,
    )
    from golden import TensorSpec

    # Token-local RoPE: compute only the active-position rows the device needs,
    # never a full-context table. SWA uses the uncompressed RoPE profile.
    _inv_freq = 1.0 / (
        float(M.rope_theta)
        ** (torch.arange(0, ROPE_HEAD_DIM, 2, dtype=torch.float32) / ROPE_HEAD_DIM)
    )

    def init_rope_rows():
        positions = init_position_ids().to(torch.float32)
        angles = torch.outer(positions, _inv_freq)
        cos_half = torch.cos(angles)
        sin_half = torch.sin(angles)
        return (
            torch.cat([cos_half, cos_half], dim=-1).to(torch.bfloat16),
            torch.cat([sin_half, sin_half], dim=-1).to(torch.bfloat16),
        )

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
    # Real layer-0 (SWA) hc_attn scale/base; fn is synthetic at the real magnitude.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.039
    def init_hc_attn_scale():
        return torch.tensor([2.076026, 0.018729, 0.245936])
    def init_hc_attn_base():
        return torch.tensor([
            3.9083, -2.0399, -2.2033, -2.017,
            -2.4443, -10.3158, -8.9943, -6.3581,
            9.8577, -9.5177, -24.8724, -22.8929,
            -21.545, 0.7791, -3.386, 1.1948,
            -20.9605, -0.7702, 1.4218, -4.8994,
            1.5177, -29.7663, -30.1413, -1.2413,
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
    _rope_rows_cache = {}
    def init_freqs_cos():
        if "cos" not in _rope_rows_cache:
            _rope_rows_cache["cos"], _rope_rows_cache["sin"] = init_rope_rows()
        return _rope_rows_cache["cos"]
    def init_freqs_sin():
        if "sin" not in _rope_rows_cache:
            _rope_rows_cache["cos"], _rope_rows_cache["sin"] = init_rope_rows()
        return _rope_rows_cache["sin"]
    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    def init_kv_cache():
        return init_normalized_cache((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_block_table():
        # Logical block-table cols cover the full SWA ceiling so 1M positions
        # map into the fixed physical pool (ORI_BLOCK_NUM) via % wrapping.
        table_blocks = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
        return block_table(batch=batch, table_blocks=table_blocks, physical_blocks=ORI_BLOCK_NUM)

    def init_attn_sink():
        return torch.zeros(H)
    def init_default_start_pos():
        # Canonical SWA start-position set (sliding-window regimes + 8k long-context).
        return swa_decode_start_set(batch=batch, window=WIN)
    def init_start_pos():
        # A list/tuple mixes per-request start positions within one batch
        # (e.g. 16K + 512K). A scalar broadcasts to the whole batch; None
        # falls back to the canonical SWA position set.
        if isinstance(start_pos, (list, tuple)):
            starts = torch.tensor(start_pos, dtype=torch.int32)
            if starts.shape != (batch,):
                raise ValueError(
                    f"mixed start_pos needs {batch} entries, got {starts.numel()}"
                )
            if bool((starts < 0).any()):
                raise ValueError("decode start positions must be non-negative")
            if bool((starts.to(torch.int64) + S > MAX_SEQ_LEN).any()):
                raise ValueError(
                    "decode start positions plus seq length must fit "
                    f"MAX_SEQ_LEN={MAX_SEQ_LEN}"
                )
            return starts
        return resolve_start_positions(
            start_pos,
            batch=batch,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )
    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S).reshape(-1).contiguous()
    def init_swa_slot_mapping():
        return paged_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_block_table(),
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()
    def init_swa_metadata():
        return swa_indices_and_lens(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_block_table(),
            block_size=BLOCK_SIZE,
            window=WIN,
        )
    def init_swa_indices():
        return init_swa_metadata()[0].contiguous()
    def init_swa_lens():
        return init_swa_metadata()[1].contiguous()
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
        TensorSpec("freqs_cos", [tokens, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [tokens, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("swa_slot_mapping", [tokens], torch.int64, init_value=init_swa_slot_mapping),
        TensorSpec("swa_indices", [tokens, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [tokens], torch.int32, init_value=init_swa_lens),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [tokens, HC_MULT, D], torch.float32, is_output=True),
    ]


def build_distributed_tensor_specs(local_t, start_pos=None):
    """Build full-layer SWA inputs with rank-local requests and sharded O weights."""
    import torch

    from golden import ScalarSpec, TensorSpec

    if local_t < BIAS_T_TILE or local_t > LOCAL_T or local_t % BIAS_T_TILE != 0 or local_t % S != 0:
        raise ValueError(f"local_t must be a multiple of {BIAS_T_TILE} in [{BIAS_T_TILE}, {LOCAL_T}], got {local_t}")

    rank_tensors = []
    with torch.random.fork_rng():
        for rank in range(TP_SIZE):
            torch.manual_seed(20260819 + rank)
            local_specs = build_tensor_specs(start_pos=start_pos, batch=local_t // S)
            rank_tensors.append({
                spec.name: spec.create_tensor()
                for spec in local_specs
                if spec.name != "x_out"
            })

    replicated_names = (
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin", "attn_sink", "wo_b_scale",
    )
    for rank in range(1, TP_SIZE):
        for name in replicated_names:
            rank_tensors[rank][name] = rank_tensors[0][name].clone()

    full_wo_a = rank_tensors[0]["wo_a"]
    full_wo_b = rank_tensors[0]["wo_b"]
    wo_a = torch.stack([
        full_wo_a[rank * LOCAL_O_GROUPS : (rank + 1) * LOCAL_O_GROUPS]
        for rank in range(TP_SIZE)
    ])
    wo_b = torch.stack([
        full_wo_b[:, rank * LOCAL_O_WIDTH : (rank + 1) * LOCAL_O_WIDTH]
        for rank in range(TP_SIZE)
    ])

    def stacked(name):
        return torch.stack([rank_tensors[rank][name] for rank in range(TP_SIZE)])

    specs = [
        TensorSpec("x_hc", [TP_SIZE, local_t, HC_MULT, D], torch.float32, init_value=stacked("x_hc")),
        TensorSpec("hc_attn_fn", [TP_SIZE, MIX_HC, HC_DIM], torch.float32, init_value=stacked("hc_attn_fn")),
        TensorSpec("hc_attn_scale", [TP_SIZE, 3], torch.float32, init_value=stacked("hc_attn_scale")),
        TensorSpec("hc_attn_base", [TP_SIZE, MIX_HC], torch.float32, init_value=stacked("hc_attn_base")),
        TensorSpec("attn_norm_w", [TP_SIZE, D], torch.bfloat16, init_value=stacked("attn_norm_w")),
        TensorSpec("wq_a", [TP_SIZE, D, Q_LORA], torch.bfloat16, init_value=stacked("wq_a")),
        TensorSpec("wq_b", [TP_SIZE, Q_LORA, H * HEAD_DIM], torch.int8, init_value=stacked("wq_b")),
        TensorSpec("wq_b_scale", [TP_SIZE, H * HEAD_DIM], torch.float32, init_value=stacked("wq_b_scale")),
        TensorSpec("wkv", [TP_SIZE, D, HEAD_DIM], torch.bfloat16, init_value=stacked("wkv")),
        TensorSpec("gamma_cq", [TP_SIZE, Q_LORA], torch.bfloat16, init_value=stacked("gamma_cq")),
        TensorSpec("gamma_ckv", [TP_SIZE, HEAD_DIM], torch.bfloat16, init_value=stacked("gamma_ckv")),
        TensorSpec("freqs_cos", [TP_SIZE, local_t, ROPE_HEAD_DIM], torch.bfloat16, init_value=stacked("freqs_cos")),
        TensorSpec("freqs_sin", [TP_SIZE, local_t, ROPE_HEAD_DIM], torch.bfloat16, init_value=stacked("freqs_sin")),
        TensorSpec(
            "kv_cache", [TP_SIZE, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
            init_value=stacked("kv_cache"), is_output=True,
        ),
        TensorSpec("swa_slot_mapping", [TP_SIZE, local_t], torch.int64, init_value=stacked("swa_slot_mapping")),
        TensorSpec("swa_indices", [TP_SIZE, local_t, WIN], torch.int32, init_value=stacked("swa_indices")),
        TensorSpec("swa_lens", [TP_SIZE, local_t], torch.int32, init_value=stacked("swa_lens")),
        TensorSpec("position_ids", [TP_SIZE, local_t], torch.int32, init_value=stacked("position_ids")),
        TensorSpec("attn_sink", [TP_SIZE, H], torch.float32, init_value=stacked("attn_sink")),
        TensorSpec("wo_a", [TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=wo_a),
        TensorSpec("wo_b", [TP_SIZE, D, LOCAL_O_WIDTH], torch.int8, init_value=wo_b),
        TensorSpec("wo_b_scale", [TP_SIZE, D], torch.float32, init_value=stacked("wo_b_scale")),
        TensorSpec("x_out", [TP_SIZE, local_t, HC_MULT, D], torch.float32, is_output=True),
        ScalarSpec("local_t", torch.int32, local_t),
    ]
    resident_names = frozenset((*replicated_names, "kv_cache", "wo_a", "wo_b"))
    for spec in specs:
        if isinstance(spec, TensorSpec) and spec.name in resident_names:
            spec.resident = "stacked"
    return specs


def golden_decode_swa(tensors):
    """Run the complete TP1 SWA reference independently for each request shard."""
    full_wo_a = tensors["wo_a"].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    full_wo_b = tensors["wo_b"].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)
    for rank in range(TP_SIZE):
        rank_tensors = {
            name: value[rank]
            for name, value in tensors.items()
            if name != "local_t"
        }
        rank_tensors["wo_a"] = full_wo_a
        rank_tensors["wo_b"] = full_wo_b
        rank_tensors["wo_b_scale"] = tensors["wo_b_scale"][0]
        golden_decode_swa_tp1(rank_tensors)


if __name__ == "__main__":
    import argparse

    from golden import mapped_pool_ratio_allclose, ratio_reldiff, run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES), help="tensor-parallel world size")
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help=f"comma-separated device ids; full/output need {TP_SIZE}, tp1 needs one",
    )
    parser.add_argument("--entry", choices=("full", "output", "tp1"), default="full")
    parser.add_argument("--case", choices=("all", "max", "subcapacity"), default="all")
    parser.add_argument(
        "--start-pos", type=str, default=None,
        help="absolute decode start position for full or tp1; a scalar sets batch=1, "
             "and a comma-separated list sets batch to its length",
    )
    parser.add_argument("--enable-l2-swimlane", type=int, choices=(0, 1, 2, 4), default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.start_pos is not None:
        parts = [p.strip() for p in args.start_pos.split(",") if p.strip() != ""]
        if len(parts) == 1:
            args.start_pos = int(parts[0])
        else:
            args.start_pos = [int(p) for p in parts]

    if args.tp != TP_SIZE:
        parser.error(f"--tp must remain {TP_SIZE} after import-time specialization")
    if args.device is None:
        args.device = ",".join(str(rank) for rank in range(TP_SIZE)) if args.entry in ("full", "output") else "0"
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(f"--device must be a comma-separated integer list, got {args.device!r}")
    if any(device < 0 for device in device_ids):
        parser.error(f"--device IDs must be non-negative, got {device_ids}")
    if len(set(device_ids)) != len(device_ids):
        parser.error(f"--device IDs must be distinct, got {device_ids}")

    expected_devices = TP_SIZE if args.entry in ("full", "output") else 1
    if len(device_ids) != expected_devices:
        parser.error(f"{args.entry} entry needs exactly {expected_devices} device(s), got {device_ids}")

    if args.entry == "output" and args.start_pos is not None:
        parser.error("--start-pos does not apply to the output-only entry")

    if args.entry == "tp1":
        if args.case == "subcapacity":
            parser.error("--case subcapacity only applies to the full and output entries")
        batch = len(args.start_pos) if isinstance(args.start_pos, list) else (1 if args.start_pos is not None else B)
        tokens = batch * S
        result = run_jit(
            fn=decode_swa_tp1_test,
            specs=build_tensor_specs(start_pos=args.start_pos, batch=batch),
            golden_fn=golden_decode_swa_tp1,
            compile_only=args.compile_only,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=device_ids[0],
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
                "kv_cache": mapped_pool_ratio_allclose(
                    "swa_slot_mapping",
                    mapping_shape=(tokens,),
                    block_size=BLOCK_SIZE,
                    pool_name="KV cache",
                    atol=1e-4,
                    rtol=1.0 / 128,
                    max_error_ratio=0.005,
                ),
            },
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
        raise SystemExit(0)

    case_local_t = {"max": LOCAL_T, "subcapacity": LOCAL_T - BIAS_T_TILE}
    if args.start_pos is not None:
        batch = len(args.start_pos) if isinstance(args.start_pos, list) else 1
        selected_cases = (("positions", batch * S),)
    else:
        case_names = tuple(case_local_t) if args.case == "all" else (args.case,)
        selected_cases = tuple((case, case_local_t[case]) for case in case_names)
    for _case, local_t in selected_cases:
        compile_cfg = dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
        )
        if args.entry == "full":
            result = run_jit(
                fn=l3_decode_swa,
                specs=build_distributed_tensor_specs(local_t, start_pos=args.start_pos),
                golden_fn=golden_decode_swa,
                compile_only=args.compile_only,
                compile_cfg=compile_cfg,
                runtime_cfg=dict(
                    platform=args.platform,
                    enable_l2_swimlane=args.enable_l2_swimlane,
                ),
                rtol=1e-2,
                atol=1e-2,
                compare_fn={
                    "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
                    "kv_cache": mapped_pool_ratio_allclose(
                        "swa_slot_mapping",
                        mapping_shape=(TP_SIZE, local_t),
                        block_size=BLOCK_SIZE,
                        leading_rank_axis=True,
                        pool_name="KV cache",
                        atol=1e-4,
                        rtol=1.0 / 128,
                        max_error_ratio=0.005,
                    ),
                },
            )
        else:
            result = run_jit(
                fn=l3_decode_swa_output_test,
                specs=build_output_tensor_specs(local_t),
                golden_fn=golden_decode_swa_output,
                compile_only=args.compile_only,
                compile_cfg=compile_cfg,
                runtime_cfg=dict(
                    platform=args.platform,
                    enable_l2_swimlane=args.enable_l2_swimlane,
                ),
                compare_fn={"o_local": build_o_local_compare(local_t)},
            )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
