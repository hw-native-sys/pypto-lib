# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 HCA full-layer TP and TP1 entries."""


import math
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
    KV_CMP_BLOCK_NUM,
    KV_ORI_BLOCK_NUM,
    HCA_STATE_PHYSICAL_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import kv_proj_rope, q_proj_rope, qkv_proj_rope, rope_prepare
from rmsnorm import rms_norm
from decode_cp_token_allgather import (
    KV_B_DYN,
    KV_T_DYN,
    DECODE_GROUP_CAP,
    decode_cp_token_allgather_step,
)
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
    decode_o_proj_tp1,
    o_group_a2a,
    o_proj_reduce_scatter,
)
from decode_sparse_attn_hca import (
    ATTENTION_PUBLISH_T_TILE,
    ATTENTION_PUBLISH_WORKERS,
    H_TILE,
    HCA_MAX_COMPRESSED_ROWS,
    PUBLISH_GROUPS,
    T_PAD,
    VALID_TOKEN_TILE,
    sparse_attn_hca,
    sparse_attn_hca_tp1,
)

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
CMP_TABLE_BLOCKS_DYN = pl.dynamic("CMP_TABLE_BLOCKS_DYN")
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
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM
ORI_TABLE_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
# Main compressor state pool (kv + score channels merged into one paged FP32 buffer).
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_PHYSICAL_BLOCKS = HCA_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
# tiling
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE
HCA_WB_TOKEN_TILE = 8  # tokens per cache-writeback SPMD block

if T != LOCAL_T:
    raise ValueError(f"HCA token capacity {T} must equal TP local token capacity {LOCAL_T}")
if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"HCA token capacity {T_PAD} must equal TP local token capacity {LOCAL_T_PAD}")


@pl.jit.inline(auto_scope=False)
def decode_hca(
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
    cmp_freqs_cos: pl.Tensor[[KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[KV_B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[KV_T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    """Run one rank of the context-parallel HCA layer."""
    t_dim = pl.tensor.dim(x_hc, 0)
    kv_dim = pl.tensor.dim(ori_slot_mapping, 0)
    kv_b_dim = pl.tensor.dim(compress_state_block_table, 0)
    kv_wb_blocks = kv_dim // HCA_WB_TOKEN_TILE

    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    cmp_cos_il = pl.create_tensor([kv_b_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([kv_b_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_interleave(cmp_freqs_cos, cmp_freqs_sin, cmp_cos_il, cmp_sin_signed)

    x_normed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    late_dep = pl.system.task_dummy(deps=[rms_tid])

    # All-gather the local post-norm rows into the TP group's token stream, which
    # the KV branch, its cache write and the compressor consume.
    x_normed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    with pl.scope():
        # decode_cp_token_allgather_step writes x_normed_full in place; keep the
        # original handle, since a returned inline handle cannot cross into
        # kv_proj_rope.
        _gathered_normed, gather_signal = decode_cp_token_allgather_step(
            x_normed, x_normed_full,
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
        x_normed, wq_a, wq_b, wq_b_scale, gamma_cq,
        q_cos_il, q_sin_signed, q_swap_idx,
        q, qr, qr_scale,
    )

    kv_proj_rope(
        x_normed_full, wkv, gamma_ckv,
        kv_cos_il, kv_sin_signed, kv_swap_idx,
        kv_full, late_dep,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.spmd(kv_wb_blocks, name_hint="hca_cache_writeback") as ori_cache_write_tid:
        wb_blk = pl.tile.get_block_idx()
        wb_t0 = wb_blk * HCA_WB_TOKEN_TILE
        for write_dt in pl.range(HCA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            write_row_i64 = pl.read(ori_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = (
                    kv_full[write_t : write_t + 1, 0 : HEAD_DIM]
                )

    # Hand the compressor scalar-extent views: its token and request axes bind to
    # one row count per call, and mixing them with the gathered stream's symbols
    # leaves the two not provably equal across the call.
    cmp_positions = pl.reshape(position_ids, [kv_dim])
    cmp_slots = pl.reshape(cmp_slot_mapping, [kv_dim])
    cmp_state_slots = pl.reshape(state_slot_mapping, [kv_dim])
    cmp_state_table = pl.reshape(
        compress_state_block_table, [kv_b_dim, COMPRESS_STATE_MAX_BLOCKS],
    )
    cmp_kv_proj = pl.create_tensor([kv_dim, HEAD_DIM], dtype=pl.FP32)
    cmp_kv_proj, cmp_cache_write_tid = compressor_ratio128(
        x_normed_full, cmp_kv_proj,
        compress_state, cmp_state_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_cos_il, cmp_sin_signed, cmp_kv,
        cmp_positions, cmp_slots, cmp_state_slots,
        late_dep,
    )
    cache_ready_dep = pl.system.task_dummy(deps=[ori_cache_write_tid, cmp_cache_write_tid])

    attention_local_flat = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.scope():
        (
            stream_heads,
            rope_cos_il, rope_sin_signed, rope_swap_idx,
            stream_heads_tid, rope_swap_tid, rope_cs_tid,
        ) = sparse_attn_hca(
            q, kv_cache, window_swa_indices, window_swa_lens,
            cmp_kv, cmp_block_table,
            position_ids_local, kv_seq_lens,
            attn_sink, freqs_cos_local, freqs_sin_local,
            cache_ready_dep,
        )

        attention_grouped = pl.create_tensor([O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], dtype=pl.BF16)
        pack_work_count = (t_dim // ATTENTION_PUBLISH_T_TILE) * (H // H_TILE)
        with pl.spmd(
            ATTENTION_PUBLISH_WORKERS,
            name_hint="hca_stream_pack_publish",
            deps=[stream_heads_tid, rope_swap_tid, rope_cs_tid],
        ) as publish_tid:
            worker = pl.tile.get_block_idx()
            for pack_work in pl.range(worker, pack_work_count, ATTENTION_PUBLISH_WORKERS):
                token_block = pack_work // (H // H_TILE)
                stream_h_tile = pack_work - token_block * (H // H_TILE)
                stream_t0 = token_block * ATTENTION_PUBLISH_T_TILE
                stream_h0 = stream_h_tile * H_TILE
                global_group0 = stream_h0 // HEADS_PER_GROUP
                destination_rank = global_group0 // LOCAL_O_GROUPS
                local_group0 = global_group0 - destination_rank * LOCAL_O_GROUPS

                for stream_dt in pl.unroll(ATTENTION_PUBLISH_T_TILE):
                    stream_t = stream_t0 + stream_dt
                    stream_state_row = stream_t * H + stream_h0
                    stream_output = stream_heads[stream_state_row : stream_state_row + H_TILE, 0:HEAD_DIM]
                    stream_bf16 = pl.cast(stream_output, target_type=pl.BF16, mode="rint")
                    stream_rope = stream_output[0:H_TILE, NOPE_HEAD_DIM:HEAD_DIM]
                    stream_cos_il = rope_cos_il[stream_t : stream_t + 1, 0:ROPE_HEAD_DIM]
                    stream_sin_signed = rope_sin_signed[stream_t : stream_t + 1, 0:ROPE_HEAD_DIM]
                    stream_swap_zero = pl.full([H_TILE, ROPE_HEAD_DIM], dtype=pl.INT32, value=0)
                    stream_swap_idx = pl.col_expand_add(stream_swap_zero, rope_swap_idx[0:1, 0:ROPE_HEAD_DIM])
                    stream_swapped = pl.gather(stream_rope, dim=-1, index=stream_swap_idx)
                    stream_rot = pl.add(
                        pl.col_expand_mul(stream_rope, stream_cos_il),
                        pl.col_expand_mul(stream_swapped, stream_sin_signed),
                    )
                    n_rope_bf16 = pl.cast(stream_rot, target_type=pl.BF16, mode="rint")
                    n_full_bf16 = pl.concat(stream_bf16[0:H_TILE, 0:NOPE_HEAD_DIM], n_rope_bf16)
                    for n_hi in pl.unroll(H_TILE):
                        n_head = stream_h0 + n_hi
                        source_row = (n_head // HEADS_PER_GROUP) * T_PAD + stream_t
                        source_col = (n_head % HEADS_PER_GROUP) * HEAD_DIM
                        attention_grouped[
                            source_row : source_row + 1,
                            source_col : source_col + HEAD_DIM,
                        ] = n_full_bf16[n_hi : n_hi + 1, 0:HEAD_DIM]

                for group_slot in pl.unroll(PUBLISH_GROUPS):
                    source_row = (global_group0 + group_slot) * T_PAD + stream_t0
                    target_row = ((local_group0 + group_slot) * GROUP_T_PAD + tp_rank * local_t + stream_t0)
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
def decode_hca_test(
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
    cmp_freqs_cos: pl.Tensor[[KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[KV_B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[KV_T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
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
    """Test one rank of the complete HCA tensor-parallel layer."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos_local.bind_dynamic(0, T_DYN)
    freqs_sin_local.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, KV_T_DYN)
    freqs_sin.bind_dynamic(0, KV_T_DYN)
    cmp_freqs_cos.bind_dynamic(0, KV_B_DYN)
    cmp_freqs_sin.bind_dynamic(0, KV_B_DYN)
    compress_state.bind_dynamic(0, COMPRESS_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, KV_B_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(1, CMP_TABLE_BLOCKS_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    ori_slot_mapping.bind_dynamic(0, KV_T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, KV_T_DYN)
    state_slot_mapping.bind_dynamic(0, KV_T_DYN)
    position_ids_local.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, KV_T_DYN)
    x_out.bind_dynamic(0, T_DYN)

    return decode_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos_local, freqs_sin_local, freqs_cos, freqs_sin,
        cmp_freqs_cos, cmp_freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, state_slot_mapping,
        position_ids_local, position_ids, kv_seq_lens,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
        gather_window, gather_signal,
        attention_window, attention_signal, o_window, o_signal,
        group_base, tp_rank, local_t,
    )


@pl.jit.host
def l3_decode_hca(
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
    freqs_sin_local: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[TP_SIZE, KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[TP_SIZE, KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[TP_SIZE, KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[TP_SIZE, COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[TP_SIZE, COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[TP_SIZE, KV_B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[TP_SIZE, B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[TP_SIZE, KV_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[TP_SIZE, T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[TP_SIZE, KV_T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[TP_SIZE, KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[TP_SIZE, KV_T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[TP_SIZE, B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch the complete HCA layer on one tensor-parallel group."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos_local.bind_dynamic(1, T_DYN)
    freqs_sin_local.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, KV_T_DYN)
    freqs_sin.bind_dynamic(1, KV_T_DYN)
    cmp_freqs_cos.bind_dynamic(1, KV_B_DYN)
    cmp_freqs_sin.bind_dynamic(1, KV_B_DYN)
    compress_state_block_table.bind_dynamic(1, KV_B_DYN)
    cmp_block_table.bind_dynamic(1, B_DYN)
    cmp_block_table.bind_dynamic(2, CMP_TABLE_BLOCKS_DYN)
    kv_seq_lens.bind_dynamic(1, B_DYN)
    ori_slot_mapping.bind_dynamic(1, KV_T_DYN)
    window_swa_indices.bind_dynamic(1, T_DYN)
    window_swa_lens.bind_dynamic(1, T_DYN)
    cmp_slot_mapping.bind_dynamic(1, KV_T_DYN)
    state_slot_mapping.bind_dynamic(1, KV_T_DYN)
    position_ids_local.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, KV_T_DYN)
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
        decode_hca_test(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos_local[rank], freqs_sin_local[rank],
            freqs_cos[rank], freqs_sin[rank],
            cmp_freqs_cos[rank], cmp_freqs_sin[rank],
            cmp_wkv[rank], cmp_wgate[rank], cmp_ape[rank], cmp_norm_w[rank],
            compress_state[rank], compress_state_block_table[rank],
            kv_cache[rank], cmp_kv[rank], cmp_block_table[rank],
            ori_slot_mapping[rank], window_swa_indices[rank], window_swa_lens[rank],
            cmp_slot_mapping[rank], state_slot_mapping[rank],
            position_ids_local[rank], position_ids[rank], kv_seq_lens[rank],
            attn_sink[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            x_out[rank],
            gather_window, gather_signal,
            attention_window, attention_signal, o_window, o_signal,
            0, rank, local_t, device=rank,
        )


@pl.jit.inline(auto_scope=False)
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
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
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
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
    wb_blocks = t_dim // HCA_WB_TOKEN_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    # Interleave-duplicated / sign-folded compressed-position rope rows, built once over B rows.
    cmp_cos_il = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_interleave(cmp_freqs_cos, cmp_freqs_sin, cmp_cos_il, cmp_sin_signed)

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
        freqs_cos, freqs_sin, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, late_dep,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.spmd(wb_blocks, name_hint="hca_cache_writeback") as ori_cache_write_tid:
        wb_blk = pl.tile.get_block_idx()
        wb_t0 = wb_blk * HCA_WB_TOKEN_TILE
        for write_dt in pl.range(HCA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            write_row_i64 = pl.read(ori_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]

    cmp_kv_proj = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    cmp_kv_proj, cmp_cache_write_tid = compressor_ratio128(
        x_normed, cmp_kv_proj,
        compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_cos_il, cmp_sin_signed, cmp_kv,
        position_ids, cmp_slot_mapping, state_slot_mapping,
        late_dep,
    )
    cache_ready_dep = pl.system.task_dummy(deps=[ori_cache_write_tid, cmp_cache_write_tid])

    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.scope():
        o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)
        o_packed_heads, heads_dep = sparse_attn_hca_tp1(
            q, kv_cache, window_swa_indices, window_swa_lens,
            cmp_kv, cmp_block_table,
            position_ids, kv_seq_lens,
            attn_sink, freqs_cos, freqs_sin,
            o_packed_heads, cache_ready_dep,
        )
        with pl.scope():
            decode_o_proj_tp1(o_packed_heads, wo_a, wo_b, wo_b_scale, attn_out, heads_dep)

    with pl.scope():
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[B, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
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
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, COMPRESS_STATE_BLOCK_NUM_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(1, CMP_TABLE_BLOCKS_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    x_out.bind_dynamic(0, T_DYN)

    decode_hca_tp1(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin,
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


def golden_decode_hca_tp1(tensors, cp_full=None):
    """End-to-end orchestration for the ratio=128 (HCA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==128 path: main compressor only,
    no indexer) + Block.hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm
    from decode_compressor_ratio128 import golden_compressor
    from decode_o_proj import golden_decode_o_proj_tp1
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

    # Attention.forward, ratio==128 branch. The wrapper receives token-local
    # QKV/attention RoPE rows and static padded boundary rows for the compressor.
    position_ids = tensors["position_ids"].to(torch.int64)
    rope_cos_T = tensors["freqs_cos"]
    rope_sin_T = tensors["freqs_sin"]

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
    position_ids_flat = position_ids.reshape(-1).to(torch.int32).contiguous()
    if cp_full is None:
        cmp_x = x_normed
        cmp_cos = tensors["cmp_freqs_cos"][:batch]
        cmp_sin = tensors["cmp_freqs_sin"][:batch]
        cmp_state_table = tensors["compress_state_block_table"]
        cmp_positions = position_ids_flat
        cmp_slots = tensors["cmp_slot_mapping"].reshape(-1).to(torch.int64).contiguous()
        cmp_state_slots = tensors["state_slot_mapping"].reshape(-1).to(torch.int64).contiguous()
        kv_rows = kv
        ori_slot_mapping = tensors["ori_slot_mapping"].to(torch.int64)
    else:
        cmp_x = cp_full["x_normed"]
        cmp_cos = cp_full["cmp_freqs_cos"]
        cmp_sin = cp_full["cmp_freqs_sin"]
        cmp_state_table = cp_full["compress_state_block_table"]
        cmp_positions = cp_full["position_ids"]
        cmp_slots = cp_full["cmp_slot_mapping"]
        cmp_state_slots = cp_full["state_slot_mapping"]
        kv_rows = cp_full["kv"]
        ori_slot_mapping = cp_full["ori_slot_mapping"]

    # The compressor ABI is token-major flat.
    cmp_kv_proj = torch.zeros(cmp_x.shape[0], HEAD_DIM, dtype=torch.float32)
    golden_compressor({
        "x": cmp_x,
        "kv": cmp_kv_proj,
        "compress_state": tensors["compress_state"],
        "compress_state_block_table": cmp_state_table,
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "cos": cmp_cos,
        "sin": cmp_sin,
        "cmp_kv_cache": cmp_kv,
        "position_ids": cmp_positions,
        "cmp_slot_mapping": cmp_slots,
        "state_slot_mapping": cmp_state_slots,
    })

    for t in range(kv_rows.shape[0]):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv_rows[t]

    o_packed_heads = torch.zeros(O_GROUPS, T_PAD, O_GROUP_IN, dtype=torch.bfloat16)
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "window_swa_indices": window_swa_indices,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "position_ids": position_ids_flat,
        "kv_seq_lens": tensors["kv_seq_lens"],
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "o_packed_heads": o_packed_heads,
    })
    attn_out = golden_decode_o_proj_tp1(o_packed_heads, tensors["wo_a"], tensors["wo_b"], tensors["wo_b_scale"], tokens)

    # ===== Block.hc_post =====
    y = torch.zeros(tokens, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({ "x": attn_out, "residual": tensors["x_hc"], "post": post_t, "comb": comb_t, "y": y, })

    tensors["x_out"][:] = y


def golden_decode_hca(tensors):
    """Build KV and the compressor from the gathered stream, then run each rank."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm

    tp_size, local_t = tensors["x_hc"].shape[:2]
    full_wo_a = tensors["wo_a"].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    full_wo_b = tensors["wo_b"].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)

    # Post-norm rows the all-gather publishes, rank-major.
    normed_chunks = []
    kv_chunks = []
    for rank in range(tp_size):
        x_mixed = torch.zeros(local_t, D, dtype=torch.bfloat16)
        post_t = torch.zeros(local_t, HC_MULT)
        comb_t = torch.zeros(local_t, HC_MULT * HC_MULT)
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
        normed_chunks.append(x_normed)

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
        "x_normed": torch.cat(normed_chunks, dim=0),
        "kv": torch.cat(kv_chunks, dim=0),
        "ori_slot_mapping": tensors["ori_slot_mapping"][0].to(torch.int64),
        "position_ids": tensors["position_ids"][0].to(torch.int32),
        "cmp_slot_mapping": tensors["cmp_slot_mapping"][0].to(torch.int64),
        "state_slot_mapping": tensors["state_slot_mapping"][0].to(torch.int64),
        "cmp_freqs_cos": tensors["cmp_freqs_cos"][0],
        "cmp_freqs_sin": tensors["cmp_freqs_sin"][0],
        "compress_state_block_table": tensors["compress_state_block_table"][0],
    }

    for rank in range(tp_size):
        rank_tensors = { name: tensor[rank] for name, tensor in tensors.items() if name != "local_t" }
        # The TP1 reference names its token-local rows without a suffix, so the
        # _local halves replace the gathered stream that now holds the bare name.
        rank_tensors["freqs_cos"] = tensors["freqs_cos_local"][rank]
        rank_tensors["freqs_sin"] = tensors["freqs_sin_local"][rank]
        rank_tensors["position_ids"] = tensors["position_ids_local"][rank]
        rank_tensors["wo_a"] = full_wo_a
        rank_tensors["wo_b"] = full_wo_b
        golden_decode_hca_tp1(rank_tensors, cp_full=cp_full)


def _hca_start_positions(start_pos, *, batch):
    """Resolve scalar/list HCA start positions against the local 1M ceiling."""
    import torch

    from utils import hca_decode_start_set

    group_b = TP_SIZE * B
    if batch < 1 or batch > group_b:
        raise ValueError(f"HCA batch must be in [1, {group_b}], got {batch}")
    if start_pos is None:
        starts = hca_decode_start_set(
            batch=batch,
            compress_ratio=COMPRESS_RATIO,
            state_block_size=COMPRESS_STATE_BLOCK_SIZE,
        )
    elif isinstance(start_pos, (list, tuple)):
        if len(start_pos) != batch:
            raise ValueError(f"HCA start-position list has {len(start_pos)} rows, expected batch={batch}")
        starts = torch.tensor([int(value) for value in start_pos], dtype=torch.int32)
    else:
        starts = torch.full((batch,), int(start_pos), dtype=torch.int32)
    if bool((starts < 0).any()) or bool((starts.to(torch.int64) + S > MAX_SEQ_LEN).any()):
        raise ValueError(f"HCA start positions plus S={S} must fit MAX_SEQ_LEN={MAX_SEQ_LEN}")
    return starts.contiguous()


def _validate_hca_token_count(token_count):
    """Validate the dynamic token extent shared by TP1 and distributed HCA."""
    group_cap = TP_SIZE * LOCAL_T
    if (token_count < VALID_TOKEN_TILE or token_count > group_cap
            or token_count % VALID_TOKEN_TILE != 0 or token_count % S != 0):
        raise ValueError(
            f"HCA token count must be a multiple of {VALID_TOKEN_TILE} and S={S} "
            f"in [{VALID_TOKEN_TILE}, {group_cap}], got {token_count}",
        )


def _hca_token_rope_tables(positions):
    """Materialize only requested HCA YaRN RoPE rows, never a 1M host table."""
    import torch

    dim = ROPE_HEAD_DIM
    half_dim = dim // 2
    base = float(M.compress_rope_theta)
    original_seq_len = int(M.original_max_position_embeddings)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    low = math.floor(dim * math.log(original_seq_len / (int(M.beta_fast) * 2 * math.pi)) / (2 * math.log(base)))
    high = math.ceil(dim * math.log(original_seq_len / (int(M.beta_slow) * 2 * math.pi)) / (2 * math.log(base)))
    low = max(low, 0)
    high = min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(half_dim, dtype=torch.float32) - low) / (high - low), 0, 1)
    smooth = 1 - ramp
    inv_freq = inv_freq / float(M.rope_factor) * (1 - smooth) + inv_freq * smooth
    angles = torch.outer(positions.to(torch.float32).reshape(-1), inv_freq)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    return (
        torch.cat((cos_half, cos_half), dim=-1).to(torch.bfloat16).contiguous(),
        torch.cat((sin_half, sin_half), dim=-1).to(torch.bfloat16).contiguous(),
    )


def _hca_cmp_block_table(starts):
    """Allocate rank-local ratio-128 pages without request-to-request aliasing."""
    import torch

    batch = starts.numel()
    page_counts = []
    for start in starts.tolist():
        visible_rows = min((int(start) + S) // COMPRESS_RATIO, HCA_MAX_COMPRESSED_ROWS)
        page_counts.append((visible_rows + BLOCK_SIZE - 1) // BLOCK_SIZE)

    table_blocks = max(max(page_counts, default=0), 1)
    table = torch.full((batch, table_blocks), -1, dtype=torch.int32)
    cursor = 0
    for request, page_count in enumerate(page_counts):
        if cursor + page_count > CMP_BLOCK_NUM:
            raise ValueError(
                f"HCA compressed pool needs {cursor + page_count} pages for batch={batch}, "
                f"capacity is {CMP_BLOCK_NUM}",
            )
        if page_count:
            table[request, :page_count] = torch.arange(cursor, cursor + page_count, dtype=torch.int32)
        cursor += page_count
    return table


def _hca_raw_block_table(positions):
    """Map only the raw pages visible to this decode step into the fixed ring pool."""
    import torch

    batch = positions.shape[0]
    table = torch.full((batch, ORI_TABLE_BLOCKS), -1, dtype=torch.int32)
    cursor = 0
    for request in range(batch):
        first = max(0, int(positions[request, 0].item()) - WIN + 1)
        last = int(positions[request, -1].item())
        first_page = first // BLOCK_SIZE
        last_page = last // BLOCK_SIZE
        page_count = last_page - first_page + 1
        if cursor + page_count > ORI_BLOCK_NUM:
            raise ValueError(
                f"HCA raw pool needs {cursor + page_count} pages for batch={batch}, "
                f"capacity is {ORI_BLOCK_NUM}",
            )
        table[request, first_page:last_page + 1] = torch.arange(cursor, cursor + page_count, dtype=torch.int32)
        cursor += page_count
    return table


def build_tensor_specs(start_pos=None, batch=B):
    tokens = batch * S
    import torch
    from utils import (
        block_table,
        compressed_slot_mapping,
        ori_slot_mapping,
        position_ids_from_starts,
        state_slot_mapping,
        swa_indices_and_lens,
    )
    from golden import TensorSpec

    starts = _hca_start_positions(start_pos, batch=batch)
    _validate_hca_token_count(tokens)
    positions = position_ids_from_starts(starts, seq=S).contiguous()
    token_freqs_cos, token_freqs_sin = _hca_token_rope_tables(positions)
    boundary_positions = starts.to(torch.int64) - starts.to(torch.int64).remainder(COMPRESS_RATIO)
    boundary_cos, boundary_sin = _hca_token_rope_tables(boundary_positions)
    cmp_rows = max(B, batch)
    cmp_freqs_cos = torch.zeros(cmp_rows, ROPE_HEAD_DIM // 2, dtype=torch.float32)
    cmp_freqs_sin = torch.zeros(cmp_rows, ROPE_HEAD_DIM // 2, dtype=torch.float32)
    cmp_freqs_cos[:batch] = boundary_cos[:, :ROPE_HEAD_DIM // 2].float()
    cmp_freqs_sin[:batch] = boundary_sin[:, :ROPE_HEAD_DIM // 2].float()
    window_block_table = _hca_raw_block_table(positions)
    cmp_block_table = _hca_cmp_block_table(starts)

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
        return token_freqs_cos.clone()
    def init_freqs_sin():
        return token_freqs_sin.clone()
    def init_cmp_freqs_cos():
        return cmp_freqs_cos.clone()
    def init_cmp_freqs_sin():
        return cmp_freqs_sin.clone()
    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    def init_injective_state_block_table():
        table = block_table(
            batch=batch,
            table_blocks=COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=COMPRESS_STATE_PHYSICAL_BLOCKS,
        )
        state_positions = positions.to(torch.int64)
        mapping = state_slot_mapping(state_positions, table, state_block_size=COMPRESS_STATE_BLOCK_SIZE)
        valid_rows = mapping[mapping >= 0]
        if torch.unique(valid_rows).numel() == valid_rows.numel():
            return table

        occupancy = [0] * COMPRESS_STATE_PHYSICAL_BLOCKS
        for request in range(batch):
            logical_masks = {}
            for position in state_positions[request].tolist():
                logical_block = position // COMPRESS_STATE_BLOCK_SIZE
                intra = position % COMPRESS_STATE_BLOCK_SIZE
                logical_masks[logical_block] = logical_masks.get(logical_block, 0) | (1 << intra)
            for logical_block, row_mask in logical_masks.items():
                physical_block = next(
                    (
                        block
                        for block, used_mask in enumerate(occupancy)
                        if used_mask != 0 and used_mask & row_mask == 0
                    ),
                    None,
                )
                if physical_block is None:
                    physical_block = next((block for block, used_mask in enumerate(occupancy) if used_mask == 0), None)
                if physical_block is None:
                    raise ValueError(
                        f"HCA fixture cannot place {batch * S} active state rows "
                        f"in {COMPRESS_STATE_BLOCK_NUM * COMPRESS_STATE_BLOCK_SIZE} physical rows",
                    )
                table[request, logical_block] = physical_block
                occupancy[physical_block] |= row_mask

        mapping = state_slot_mapping(state_positions, table, state_block_size=COMPRESS_STATE_BLOCK_SIZE)
        valid_rows = mapping[mapping >= 0]
        if torch.unique(valid_rows).numel() != valid_rows.numel():
            raise ValueError("HCA fixture active state rows remain aliased")
        return table

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
        return init_injective_state_block_table()
    def init_kv_cache():
        return init_normalized_cache((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))
    def init_cmp_kv():
        return init_normalized_cache((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_window_block_table():
        return window_block_table.clone()

    def init_cmp_block_table():
        return cmp_block_table.clone()

    def init_attn_sink():
        return torch.zeros(H)
    def init_position_ids():
        return positions.reshape(-1).clone()
    def init_kv_seq_lens():
        return positions[:, -1].to(torch.int64).add(1).to(torch.int32).contiguous()
    def init_ori_slot_mapping():
        return ori_slot_mapping(
            positions,
            init_window_block_table(),
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()
    def init_window_swa_metadata():
        return swa_indices_and_lens(
            positions,
            init_window_block_table(),
            block_size=BLOCK_SIZE,
            window=WIN,
        )
    def init_window_swa_indices():
        return init_window_swa_metadata()[0].contiguous()
    def init_window_swa_lens():
        return init_window_swa_metadata()[1].contiguous()
    def init_cmp_slot_mapping():
        return compressed_slot_mapping(
            positions,
            init_cmp_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()
    def init_state_slot_mapping():
        return state_slot_mapping(
            positions,
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
        TensorSpec("freqs_cos", [tokens, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [tokens, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_freqs_cos", [cmp_rows, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cmp_freqs_cos),
        TensorSpec("cmp_freqs_sin", [cmp_rows, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cmp_freqs_sin),
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state, is_output=True),
        TensorSpec("compress_state_block_table", [batch, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv, is_output=True),
        TensorSpec("cmp_block_table", list(cmp_block_table.shape), torch.int32, init_value=init_cmp_block_table),
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


def build_distributed_tensor_specs(local_t, start_pos=None):
    """Build one logical HCA layer fixture split over the CP token ranks."""
    import torch

    from golden import ScalarSpec, TensorSpec
    from decode_cp_token_allgather import cp_split, cp_stack, materialize_spec

    _validate_hca_token_count(local_t)
    local_batch = local_t // S
    group_batch = TP_SIZE * local_batch
    if isinstance(start_pos, (list, tuple)):
        start_pos = list(start_pos)
        if len(start_pos) == local_batch:
            start_pos *= TP_SIZE
        elif len(start_pos) != group_batch:
            raise ValueError(
                f"distributed HCA start positions need {local_batch} local or "
                f"{group_batch} group rows, got {len(start_pos)}",
            )

    # Token rows and requests the rank owns. Everything else is either a
    # replicated weight or the group's full stream, which every rank holds.
    local_token_names = frozenset({
        "x_hc", "freqs_cos", "freqs_sin",
        "window_swa_indices", "window_swa_lens", "position_ids",
    })
    local_request_names = frozenset({"cmp_block_table", "kv_seq_lens"})
    # Consumed only on the gathered stream: replicated, and renamed to say so.
    full_only_names = frozenset({
        "ori_slot_mapping", "cmp_slot_mapping", "state_slot_mapping",
        "compress_state_block_table", "cmp_freqs_cos", "cmp_freqs_sin",
    })
    # Consumed on both sides: the rank's rows plus a replicated full-stream twin.
    dual_names = ("freqs_cos", "freqs_sin", "position_ids")
    resident_names = frozenset({
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
        "compress_state",
        "kv_cache", "cmp_kv", "attn_sink",
    })

    specs = []
    for spec in build_tensor_specs(start_pos=start_pos, batch=group_batch):
        if spec.name == "x_out":
            specs.append(TensorSpec(
                "x_out", [TP_SIZE, local_t, HC_MULT, D], torch.float32, is_output=True,
            ))
            continue

        value = materialize_spec(spec)
        if spec.name in full_only_names:
            full_spec = TensorSpec(
                spec.name, [TP_SIZE, *spec.shape], spec.dtype,
                init_value=cp_stack(value, TP_SIZE),
            )
            if spec.name in resident_names:
                full_spec.resident = "stacked"
            specs.append(full_spec)
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

        if spec.name in local_token_names or spec.name in local_request_names:
            rank_value = cp_split(value, TP_SIZE)
        else:
            rank_value = cp_stack(value, TP_SIZE)
        # A dual name carries a replicated full-stream twin, so the rank's rows
        # take the _local suffix that names the half they are.
        local_name = f"{spec.name}_local" if spec.name in dual_names else spec.name
        distributed_spec = TensorSpec(
            local_name, list(rank_value.shape), spec.dtype,
            init_value=rank_value, is_output=spec.is_output,
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


if __name__ == "__main__":
    import argparse

    from golden import mapped_pool_ratio_allclose, ratio_reldiff, run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

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
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.start_pos is not None:
        parts = [part.strip() for part in args.start_pos.split(",") if part.strip()]
        if not parts:
            parser.error("--start-pos must contain at least one integer")
        try:
            args.start_pos = int(parts[0]) if len(parts) == 1 else [int(part) for part in parts]
        except ValueError:
            parser.error(f"--start-pos must be an integer or comma-separated integer list, got {args.start_pos!r}")

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
    if args.golden_data is not None and args.start_pos is None and TP_SIZE != 1:
        parser.error("distributed --golden-data requires --start-pos to select one replay shape")

    if args.start_pos is not None:
        batch = len(args.start_pos) if isinstance(args.start_pos, list) else 1
        token_counts = (batch * S,)
    else:
        token_counts = (LOCAL_T,)

    for local_t in token_counts:
        if TP_SIZE == 1:
            result = run_jit(
                fn=decode_hca_tp1_test,
                specs=build_tensor_specs(start_pos=args.start_pos, batch=local_t // S),
                golden_fn=golden_decode_hca_tp1,
                golden_data=args.golden_data,
                save_data=args.save_data,
                compile_only=args.compile_only,
                compile_cfg=dict(dump_passes=args.dump_passes),
                runtime_cfg=dict(
                    platform=args.platform,
                    device_id=device_ids[0],
                    enable_chip_swimlane=args.enable_chip_swimlane,
                ),
                rtol=1e-3,
                atol=1e-3,
                compare_fn={
                    "compress_state": mapped_pool_ratio_allclose(
                        "state_slot_mapping", mapping_shape=(local_t,),
                        block_size=COMPRESS_STATE_BLOCK_SIZE,
                        pool_name="compressor state", atol=1e-3, rtol=1.0 / 128,
                    ),
                    "kv_cache": mapped_pool_ratio_allclose(
                        "ori_slot_mapping", mapping_shape=(local_t,),
                        block_size=BLOCK_SIZE,
                        pool_name="original KV cache", atol=1e-3, rtol=1.0 / 128,
                    ),
                    "cmp_kv": mapped_pool_ratio_allclose(
                        "cmp_slot_mapping", mapping_shape=(local_t,),
                        block_size=BLOCK_SIZE,
                        pool_name="compressed KV cache", atol=1e-3, rtol=1.0 / 128,
                    ),
                    "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
                },
            )
        else:
            # The A2A3 simulator's distributed O path uses the CSA-level threshold and near-zero cap.
            full_x_out_diff_thd = 4e-3 if args.platform == "a2a3sim" else 3e-3
            full_x_out_max_diff = 2 if args.platform == "a2a3sim" else 1
            mapping_shape = (TP_SIZE, local_t)
            full_mapping_shape = (TP_SIZE, TP_SIZE * local_t)
            result = run_jit(
                fn=l3_decode_hca,
                specs=build_distributed_tensor_specs(local_t, start_pos=args.start_pos),
                golden_fn=golden_decode_hca,
                golden_data=args.golden_data,
                save_data=args.save_data,
                compile_only=args.compile_only,
                compile_cfg=dict(
                    dump_passes=args.dump_passes,
                    distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
                ),
                runtime_cfg=dict(
                    platform=args.platform,
                    enable_chip_swimlane=args.enable_chip_swimlane,
                ),
                rtol=1e-3,
                atol=1e-3,
                compare_fn={
                    "compress_state": mapped_pool_ratio_allclose(
                        "state_slot_mapping", mapping_shape=full_mapping_shape,
                        block_size=COMPRESS_STATE_BLOCK_SIZE, leading_rank_axis=True,
                        pool_name="compressor state", atol=1e-3, rtol=1.0 / 128,
                    ),
                    "kv_cache": mapped_pool_ratio_allclose(
                        "ori_slot_mapping", mapping_shape=full_mapping_shape,
                        block_size=BLOCK_SIZE, leading_rank_axis=True,
                        pool_name="original KV cache", atol=1e-3, rtol=1.0 / 128,
                    ),
                    "cmp_kv": mapped_pool_ratio_allclose(
                        "cmp_slot_mapping", mapping_shape=full_mapping_shape,
                        block_size=BLOCK_SIZE, leading_rank_axis=True,
                        pool_name="compressed KV cache", atol=1e-3, rtol=1.0 / 128,
                    ),
                    "x_out": ratio_reldiff(
                        diff_thd=full_x_out_diff_thd,
                        pct_thd=0.008,
                        max_diff_hd=full_x_out_max_diff,
                    ),
                },
            )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
