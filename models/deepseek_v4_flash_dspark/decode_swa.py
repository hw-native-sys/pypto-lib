# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 SWA full-layer TP and TP1 entries."""


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
)
from hc_pre import hc_pre
from hc_post import hc_post
from decode_cp_token_allgather import (
    KV_T_DYN,
    DECODE_GROUP_CAP,
    decode_cp_token_allgather_step,
)
from qkv_proj_rope import kv_proj_rope, q_proj_rope, qkv_proj_rope, rope_prepare
from rmsnorm import rms_norm
from decode_o_proj import (
    ATTENTION_WINDOW_ROWS,
    GROUP_T_PAD,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    LOCAL_T,
    LOCAL_T_PAD,
    O_WINDOW_ROWS,
    decode_o_proj_tp1,
    o_group_a2a,
    o_proj_reduce_scatter,
)
from decode_sparse_attn_swa import (
    ATTENTION_PUBLISH_T_TILE,
    ATTENTION_PUBLISH_WORKERS,
    ATTN_K_TILE,
    H_TILE,
    PADDED_TOPK,
    PUBLISH_GROUPS,
    T_PAD,
    sparse_attn_swa,
    sparse_attn_swa_tp1,
)

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
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local (SWA: ratio-0, no compressor/indexer)
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
TOPK = WIN                          # SWA: sparse_attn topk = window only
SPARSE_IDX_TOPK = M.index_topk      # sparse_attn module's IDX_TOPK (static shape contract)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK

# tiling
BIAS_T_TILE = 8  # sparse_bias row block; T is a multiple of 8 by the batch contract
SWA_WB_TOKEN_TILE = 32  # tokens per cache-writeback SPMD block
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE
NEG_INF = -1.0e20

if T != LOCAL_T:
    raise ValueError(f"SWA token capacity {T} must equal TP-local token capacity {LOCAL_T}")
if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"SWA padded token capacity {T_PAD} must equal TP capacity {LOCAL_T_PAD}")


@pl.jit.inline(auto_scope=False)
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
    freqs_cos_local: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_local: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
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
    gather_window: pld.DistributedTensor[[DECODE_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Run one rank of the context-parallel SWA layer."""
    t_dim = pl.tensor.dim(x_hc, 0)
    kv_dim = pl.tensor.dim(swa_slot_mapping, 0)
    bias_blocks = t_dim // BIAS_T_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    # split_pre_post -> hc_post is already covered by x_mixed -> attn_out.
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32, manual_dep=True)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_t)
    # Dispatch barrier: kv_proj_matmul resolves one hop after rms_norm.
    late_dep = pl.system.task_dummy(deps=[rms_tid])

    # All-gather the local post-norm rows into the TP group's token stream, which
    # the KV branch and its cache write consume.
    x_normed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    with pl.scope():
        # decode_cp_token_allgather_step writes x_normed_full in place; keep the
        # original handle, since a returned inline handle cannot cross into
        # kv_proj_rope.
        _gathered_normed, gather_signal = decode_cp_token_allgather_step(
            x_normed_t, x_normed_full,
            gather_window, gather_signal,
            group_base, tp_rank,
        )

    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv_full = pl.create_tensor([kv_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    kv_cos_il = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    kv_sin_signed = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    kv_swap_idx = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_prepare(freqs_cos, freqs_sin, kv_cos_il, kv_sin_signed, kv_swap_idx)

    q_cos_il = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    q_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    q_swap_idx = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_prepare(freqs_cos_local, freqs_sin_local, q_cos_il, q_sin_signed, q_swap_idx)

    q_proj_rope(
        x_normed_t, wq_a, wq_b, wq_b_scale, gamma_cq,
        q_cos_il, q_sin_signed, q_swap_idx,
        q, qr, qr_scale,
    )

    kv_proj_rope(
        x_normed_full, wkv, gamma_ckv,
        kv_cos_il, kv_sin_signed, kv_swap_idx,
        kv_full, late_dep,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    cache_rows = ori_block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [cache_rows, HEAD_DIM])
    sparse_bias = pl.create_tensor([t_dim, PADDED_TOPK], dtype=pl.FP32)
    wb_blocks = (kv_dim + SWA_WB_TOKEN_TILE - 1) // SWA_WB_TOKEN_TILE
    with pl.spmd(wb_blocks, name_hint="swa_cache_writeback"):
        wb_blk = pl.tile.get_block_idx()
        wb_t0 = wb_blk * SWA_WB_TOKEN_TILE
        wb_rows = pl.min(SWA_WB_TOKEN_TILE, kv_dim - wb_t0)
        for write_dt in pl.range(wb_rows):
            write_t = wb_t0 + write_dt
            write_row_i64 = pl.read(swa_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = (
                    kv_full[write_t : write_t + 1, 0 : HEAD_DIM]
                )

    with pl.spmd(bias_blocks, name_hint="swa_valid_bias"):
        bias_block = pl.tile.get_block_idx()
        valid_col = pl.cast(pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32), target_type=pl.FP32)
        token_start = bias_block * BIAS_T_TILE
        zero_rows = pl.full([BIAS_T_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0)
        valid_cols = pl.col_expand(zero_rows, valid_col)
        lens_slice = swa_lens[token_start : token_start + BIAS_T_TILE]
        lens_col = pl.reshape(lens_slice, [BIAS_T_TILE, 1])
        lens_fp32 = pl.cast(lens_col, target_type=pl.FP32)
        valid = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(valid_cols, lens_fp32)), 0.0), 1.0)
        sparse_bias[token_start : token_start + BIAS_T_TILE, 0 : ATTN_K_TILE] = pl.mul(
            pl.sub(valid, 1.0), -NEG_INF,
        )

    attention_local_flat = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.scope():
        (
            sparse_blk_mi, sparse_blk_li, sparse_blk_oi,
            rope_cos_il, rope_sin_signed, rope_swap_idx,
            qk_tid, rope_tid,
        ) = sparse_attn_swa(
            q, kv_cache, swa_indices, swa_lens, sparse_bias,
            freqs_cos_local, freqs_sin_local,
        )

        attention_grouped = pl.create_tensor([O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], dtype=pl.BF16)
        pack_work_count = (t_dim // ATTENTION_PUBLISH_T_TILE) * (H // H_TILE)
        o_packed_heads = pl.reshape(attention_grouped, [O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM])
        with pl.spmd(
            ATTENTION_PUBLISH_WORKERS,
            name_hint="swa_merge_pack_publish",
            deps=[qk_tid, rope_tid],
        ) as publish_tid:
            worker = pl.tile.get_block_idx()
            for pack_work in pl.range(worker, pack_work_count, ATTENTION_PUBLISH_WORKERS):
                token_block = pack_work // (H // H_TILE)
                head_tile = pack_work - token_block * (H // H_TILE)
                pack_t0 = token_block * ATTENTION_PUBLISH_T_TILE
                head_start = head_tile * H_TILE
                global_group_start = head_start // HEADS_PER_GROUP

                for token_delta in pl.range(ATTENTION_PUBLISH_T_TILE):
                    token = pack_t0 + token_delta
                    block_base = token * H + head_start
                    block_m = sparse_blk_mi[block_base : block_base + H_TILE, 0:1]
                    block_l = sparse_blk_li[block_base : block_base + H_TILE, 0:1]
                    block_o = sparse_blk_oi[block_base : block_base + H_TILE, 0:HEAD_DIM]

                    sink = pl.reshape(attn_sink[head_start : head_start + H_TILE], [H_TILE, 1])
                    sink_delta = pl.sub(sink, block_m)
                    sink_exp = pl.exp(sink_delta)
                    denom = pl.add(block_l, sink_exp)
                    normalized = pl.row_expand_div(block_o, denom)
                    full = normalized[0:H_TILE, 0:HEAD_DIM]
                    full_bf16 = pl.cast(full, target_type=pl.BF16, mode="rint")

                    rope = full[:, NOPE_HEAD_DIM:HEAD_DIM]
                    swapped = pl.gather(rope, dim=-1, index=rope_swap_idx[:, :])
                    cos_il = rope_cos_il[token : token + 1, 0:ROPE_HEAD_DIM]
                    sin_signed = rope_sin_signed[token : token + 1, 0:ROPE_HEAD_DIM]
                    rope_cos = pl.col_expand_mul(rope, cos_il)
                    swap_sin = pl.col_expand_mul(swapped, sin_signed)
                    rotated = pl.add(rope_cos, swap_sin)
                    rope_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")

                    for group_slot in pl.unroll(PUBLISH_GROUPS):
                        source_head = group_slot * HEADS_PER_GROUP
                        pack_row = (global_group_start + group_slot) * T_PAD + token
                        destination_head = pack_row * HEADS_PER_GROUP
                        o_packed_heads[
                            destination_head : destination_head + HEADS_PER_GROUP,
                            0:NOPE_HEAD_DIM,
                        ] = full_bf16[
                            source_head : source_head + HEADS_PER_GROUP,
                            0:NOPE_HEAD_DIM,
                        ]
                        o_packed_heads[
                            destination_head : destination_head + HEADS_PER_GROUP,
                            NOPE_HEAD_DIM:HEAD_DIM,
                        ] = rope_bf16[
                            source_head : source_head + HEADS_PER_GROUP,
                            0:ROPE_HEAD_DIM,
                        ]

                for group_slot in pl.unroll(PUBLISH_GROUPS):
                    global_group = global_group_start + group_slot
                    destination_rank = global_group // LOCAL_O_GROUPS
                    local_group = global_group - destination_rank * LOCAL_O_GROUPS
                    source_row = global_group * T_PAD + pack_t0
                    target_row = local_group * GROUP_T_PAD + tp_rank * local_t + pack_t0
                    pld.tensor.put(
                        dst=attention_window,
                        peer=group_base + destination_rank,
                        src=attention_grouped,
                        dst_offsets=[target_row, 0],
                        src_offsets=[source_row, 0],
                        shape=[ATTENTION_PUBLISH_T_TILE, O_GROUP_IN],
                        chunk_rows=ATTENTION_PUBLISH_T_TILE,
                        chunk_cols=O_GROUP_IN,
                    )

            for peer_tp in pl.range(TP_SIZE):
                if peer_tp != tp_rank:
                    pld.system.notify(
                        target=attention_signal,
                        peer=group_base + peer_tp,
                        offsets=[tp_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

        attention_local_flat, attention_signal = o_group_a2a(
            attention_local_flat,
            attention_window, attention_signal,
            group_base, tp_rank, local_t,
            publish_tid, ATTENTION_PUBLISH_WORKERS,
        )

        attention_local_groups = pl.reshape(attention_local_flat, [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN])
        # o_proj_reduce_scatter writes attn_out in place; keep the original
        # handle, since a returned inline handle cannot cross into hc_post.
        _o_reduced, o_signal = o_proj_reduce_scatter(
            attention_local_groups,
            wo_a, wo_b, wo_b_scale,
            local_t, attn_out,
            o_window, o_signal,
            group_base, tp_rank,
        )

    with pl.scope():
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
    freqs_cos_local: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_local: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[DECODE_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Bind dynamic inputs for the complete tensor-parallel SWA layer."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos_local.bind_dynamic(0, T_DYN)
    freqs_sin_local.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, KV_T_DYN)
    freqs_sin.bind_dynamic(0, KV_T_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(0, KV_T_DYN)
    swa_indices.bind_dynamic(0, T_DYN)
    swa_lens.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)

    decode_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos_local, freqs_sin_local, freqs_cos, freqs_sin,
        kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
        gather_window, gather_signal,
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
    freqs_cos_local: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[TP_SIZE, KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_local: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[TP_SIZE, KV_T_DYN], pl.INT64],
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
    freqs_cos_local.bind_dynamic(1, T_DYN)
    freqs_sin_local.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, KV_T_DYN)
    freqs_sin.bind_dynamic(1, KV_T_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(1, KV_T_DYN)
    swa_indices.bind_dynamic(1, T_DYN)
    swa_lens.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    x_out.bind_dynamic(1, T_DYN)

    gather_window_buf = pld.alloc_window_buffer([DECODE_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.BF16)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [DECODE_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.BF16)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        decode_swa_test(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank], wkv[rank],
            gamma_cq[rank], gamma_ckv[rank],
            freqs_cos_local[rank], freqs_sin_local[rank],
            freqs_cos[rank], freqs_sin[rank],
            kv_cache[rank], swa_slot_mapping[rank], swa_indices[rank], swa_lens[rank], position_ids[rank],
            attn_sink[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            x_out[rank],
            gather_window, gather_signal,
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
            v_valid = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0), 1.0)
            sparse_bias[v_t0 : v_t0 + BIAS_T_TILE, 0:WIN] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD * HEADS_PER_GROUP, HEAD_DIM], dtype=pl.BF16)
    o_packed_heads, heads_dep = sparse_attn_swa_tp1(
        q, kv_cache, swa_indices, swa_lens, sparse_bias,
        attn_sink, freqs_cos, freqs_sin,
        o_packed_heads,
    )
    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T_PAD, O_GROUP_IN])
    attn_out = decode_o_proj_tp1(o_packed, wo_a, wo_b, wo_b_scale, attn_out, heads_dep)

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


def golden_decode_swa_tp1(tensors, cp_full=None):
    """End-to-end orchestration for the ratio=0 (SWA) layers.

    ``cp_full`` carries the CP group's gathered KV rows and their slot mapping,
    which every rank writes into its replicated cache.
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
    if cp_full is None:
        kv_rows, swa_slot_mapping = kv, tensors["swa_slot_mapping"].to(torch.int64)
    else:
        kv_rows, swa_slot_mapping = cp_full["kv"], cp_full["swa_slot_mapping"]
    for t in range(kv_rows.shape[0]):
        write_row = int(swa_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv_rows[t]

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
    attn_out = golden_decode_o_proj_tp1(o_packed_heads, tensors["wo_a"], tensors["wo_b"], tensors["wo_b_scale"], tokens)

    # Block.hc_post
    y = torch.zeros(tokens, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({ "x": attn_out, "residual": tensors["x_hc"], "post": post_t, "comb": comb_t, "y": y, })

    tensors["x_out"][:] = y


def build_tensor_specs(start_pos=None, batch=B):
    tokens = batch * S
    group_cap = TP_SIZE * LOCAL_T
    if batch <= 0 or tokens > group_cap:
        raise ValueError(f"batch must produce between {S} and {group_cap} tokens, got {tokens}")
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
    _inv_freq = 1.0 / (float(M.rope_theta) ** (torch.arange(0, ROPE_HEAD_DIM, 2, dtype=torch.float32) / ROPE_HEAD_DIM))

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
                    f"mixed start_pos needs {batch} entries, got {starts.numel()}",
                )
            if bool((starts < 0).any()):
                raise ValueError("decode start positions must be non-negative")
            if bool((starts.to(torch.int64) + S > MAX_SEQ_LEN).any()):
                raise ValueError(
                    "decode start positions plus seq length must fit "
                    f"MAX_SEQ_LEN={MAX_SEQ_LEN}",
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
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("swa_slot_mapping", [tokens], torch.int64, init_value=init_swa_slot_mapping),
        TensorSpec("swa_indices", [tokens, WIN], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [tokens], torch.int32, init_value=init_swa_lens),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [tokens, HC_MULT, D], torch.float32),
    ]


def build_distributed_tensor_specs(local_t, start_pos=None):
    """Build one logical SWA layer fixture split over the CP token ranks."""
    import torch

    from golden import ScalarSpec, TensorSpec
    from decode_cp_token_allgather import cp_split, cp_stack, materialize_spec

    if local_t < BIAS_T_TILE or local_t > LOCAL_T or local_t % BIAS_T_TILE != 0 or local_t % S != 0:
        raise ValueError(f"local_t must be a multiple of {BIAS_T_TILE} in [{BIAS_T_TILE}, {LOCAL_T}], got {local_t}")

    local_batch = local_t // S
    group_batch = TP_SIZE * local_batch
    if isinstance(start_pos, (list, tuple)):
        start_pos = list(start_pos)
        if len(start_pos) == local_batch:
            start_pos *= TP_SIZE
        elif len(start_pos) != group_batch:
            raise ValueError(
                f"distributed SWA start positions need {local_batch} local or "
                f"{group_batch} group rows, got {len(start_pos)}",
            )

    # Token rows the rank owns; the KV cache write runs on the gathered stream.
    local_token_names = frozenset({
        "x_hc", "freqs_cos", "freqs_sin", "swa_indices", "swa_lens", "position_ids",
    })
    full_only_names = frozenset({"swa_slot_mapping"})
    dual_names = ("freqs_cos", "freqs_sin")
    resident_names = frozenset({
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "kv_cache", "attn_sink",
    })

    specs = []
    for spec in build_tensor_specs(start_pos=start_pos, batch=group_batch):
        if spec.name == "x_out":
            specs.append(TensorSpec(
                "x_out", [TP_SIZE, local_t, HC_MULT, D], torch.float32, 
            ))
            continue

        value = materialize_spec(spec)
        if spec.name in full_only_names:
            specs.append(TensorSpec(
                spec.name, [TP_SIZE, *spec.shape], spec.dtype,
                init_value=cp_stack(value, TP_SIZE),
            ))
            continue
        if spec.name == "wo_a":
            shards = value.reshape(TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN).contiguous()
            wo_a_spec = TensorSpec(
                "wo_a", [TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], spec.dtype, init_value=shards,
            )
            wo_a_spec.resident = "stacked"
            specs.append(wo_a_spec)
            continue
        if spec.name == "wo_b":
            shards = value.reshape(D, TP_SIZE, LOCAL_O_WIDTH).permute(1, 0, 2).contiguous()
            wo_b_spec = TensorSpec("wo_b", [TP_SIZE, D, LOCAL_O_WIDTH], spec.dtype, init_value=shards)
            wo_b_spec.resident = "stacked"
            specs.append(wo_b_spec)
            continue
        if spec.name == "wo_b_scale":
            wo_b_scale_spec = TensorSpec(
                "wo_b_scale", [TP_SIZE, *spec.shape], spec.dtype, init_value=cp_stack(value, TP_SIZE),
            )
            wo_b_scale_spec.resident = "stacked"
            specs.append(wo_b_scale_spec)
            continue

        if spec.name in local_token_names:
            rank_value = cp_split(value, TP_SIZE)
        else:
            rank_value = cp_stack(value, TP_SIZE)
        # A dual name carries a replicated full-stream twin, so the rank's rows
        # take the _local suffix that names the half they are.
        local_name = f"{spec.name}_local" if spec.name in dual_names else spec.name
        distributed_spec = TensorSpec(
            local_name, list(rank_value.shape), spec.dtype,
            init_value=rank_value, 
        )
        if spec.name in resident_names:
            distributed_spec.resident = "stacked"
        specs.append(distributed_spec)

        if spec.name in dual_names:
            specs.append(TensorSpec(
                spec.name, [TP_SIZE, *spec.shape], spec.dtype,
                init_value=cp_stack(value, TP_SIZE),
            ))
    specs.append(ScalarSpec("local_t", torch.int32, local_t))
    return specs


def golden_decode_swa(tensors):
    """Build KV from the gathered stream, then run the per-rank reference."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm

    tp_size, local_t = tensors["x_hc"].shape[:2]
    full_wo_a = tensors["wo_a"].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    full_wo_b = tensors["wo_b"].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)

    # Post-norm rows the all-gather publishes, rank-major.
    kv_chunks = []
    for rank in range(tp_size):
        x_mixed = torch.zeros(local_t, D, dtype=torch.bfloat16)
        post_t = torch.zeros(local_t, HC_MULT, dtype=torch.float32)
        comb_t = torch.zeros(local_t, HC_MULT * HC_MULT, dtype=torch.float32)
        golden_hc_pre({
            "x": tensors["x_hc"][rank],
            "hc_fn": tensors["hc_attn_fn"][rank],
            "hc_scale": tensors["hc_attn_scale"][rank],
            "hc_base": tensors["hc_attn_base"][rank],
            "x_mixed": x_mixed,
            "post": post_t,
            "comb": comb_t,
        })
        x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"][rank])

        rows = slice(rank * local_t, (rank + 1) * local_t)
        kv_chunk = torch.zeros(local_t, HEAD_DIM, dtype=torch.bfloat16)
        golden_qkv_proj_rope({
            "x": x_normed,
            "wq_a": tensors["wq_a"][0],
            "wq_b": tensors["wq_b"][0],
            "wq_b_scale": tensors["wq_b_scale"][0],
            "wkv": tensors["wkv"][0],
            "rope_cos": tensors["freqs_cos"][0][rows],
            "rope_sin": tensors["freqs_sin"][0][rows],
            "gamma_cq": tensors["gamma_cq"][0],
            "gamma_ckv": tensors["gamma_ckv"][0],
            "q": torch.zeros(local_t, H, HEAD_DIM, dtype=torch.bfloat16),
            "kv": kv_chunk,
            "qr": torch.zeros(local_t, Q_LORA, dtype=torch.int8),
            "qr_scale": torch.zeros(local_t, 1, dtype=torch.float32),
        })
        kv_chunks.append(kv_chunk)

    cp_full = {
        "kv": torch.cat(kv_chunks, dim=0),
        "swa_slot_mapping": tensors["swa_slot_mapping"][0].to(torch.int64),
    }

    for rank in range(tp_size):
        rank_tensors = { name: value[rank] for name, value in tensors.items() if name != "local_t" }
        # The TP1 reference names its token-local rows without a suffix, so the
        # _local halves replace the gathered stream that now holds the bare name.
        rank_tensors["freqs_cos"] = tensors["freqs_cos_local"][rank]
        rank_tensors["freqs_sin"] = tensors["freqs_sin_local"][rank]
        rank_tensors["wo_a"] = full_wo_a
        rank_tensors["wo_b"] = full_wo_b
        rank_tensors["wo_b_scale"] = tensors["wo_b_scale"][0]
        golden_decode_swa_tp1(rank_tensors, cp_full=cp_full)


if __name__ == "__main__":
    import argparse

    from golden import mapped_pool_ratio_allclose, ratio_reldiff, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES), help="tensor-parallel world size")
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help=f"comma-separated device ids; --tp {TP_SIZE} needs {TP_SIZE}",
    )
    parser.add_argument(
        "--start-pos", type=str, default=None,
        help="absolute decode start position; a scalar sets batch=1, "
             "and a comma-separated list sets batch to its length",
    )
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, choices=(0, 1, 2, 4), default=0)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
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
        args.device = ",".join(str(rank) for rank in range(TP_SIZE))
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(f"--device must be a comma-separated integer list, got {args.device!r}")
    if any(device < 0 for device in device_ids):
        parser.error(f"--device IDs must be non-negative, got {device_ids}")
    if len(set(device_ids)) != len(device_ids):
        parser.error(f"--device IDs must be distinct, got {device_ids}")
    if len(device_ids) != TP_SIZE:
        parser.error(f"--tp {TP_SIZE} needs exactly {TP_SIZE} device(s), got {device_ids}")

    if args.start_pos is not None:
        batch = len(args.start_pos) if isinstance(args.start_pos, list) else 1
        token_counts = (batch * S,)
    else:
        token_counts = (LOCAL_T,)

    for local_t in token_counts:
        if TP_SIZE == 1:
            result = run(
                fn=decode_swa_tp1_test,
                specs=build_tensor_specs(start_pos=args.start_pos, batch=local_t // S),
                golden_fn=golden_decode_swa_tp1,
                golden_data=args.golden_data,
                save_data=args.save_data,
                compile_only=args.compile_only,
                config=dict(
                    dump_passes=args.dump_passes,
                    platform=args.platform,
                    device_id=device_ids[0],
                    enable_chip_swimlane=args.enable_chip_swimlane,
                    enable_dep_gen=args.enable_dep_gen,
                ),
                rtol=1e-2,
                atol=1e-2,
                compare_fn={
                    "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
                    "kv_cache": mapped_pool_ratio_allclose(
                        "swa_slot_mapping",
                        mapping_shape=(local_t,),
                        block_size=BLOCK_SIZE,
                        pool_name="KV cache",
                        atol=1e-4,
                        rtol=1.0 / 128,
                        max_error_ratio=0.005,
                    ),
                },
            )
        else:
            result = run(
                fn=l3_decode_swa,
                specs=build_distributed_tensor_specs(local_t, start_pos=args.start_pos),
                golden_fn=golden_decode_swa,
                golden_data=args.golden_data,
                save_data=args.save_data,
                compile_only=args.compile_only,
                config=dict(
                    dump_passes=args.dump_passes,
                    distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
                    platform=args.platform,
                    enable_chip_swimlane=args.enable_chip_swimlane,
                    enable_dep_gen=args.enable_dep_gen,
                ),
                rtol=1e-2,
                atol=1e-2,
                compare_fn={
                    "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
                    "kv_cache": mapped_pool_ratio_allclose(
                        "swa_slot_mapping",
                        mapping_shape=(TP_SIZE, TP_SIZE * local_t),
                        block_size=BLOCK_SIZE,
                        leading_rank_axis=True,
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
