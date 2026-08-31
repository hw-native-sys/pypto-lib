# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 packed prefill SWA attention: single-die at --tp 1, context-parallel above it."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    FLASH as M,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    KV_CMP_BLOCK_NUM,
    KV_ORI_BLOCK_NUM,
    PREFILL_SEQ,
)

from hc_post import golden_hc_post, hc_post
from hc_pre import golden_hc_pre, hc_pre
from prefill_cp_token_allgather import (
    PREFILL_GROUP_CAP,
    TP_SIZE,
    cp_stack,
    materialize_spec,
    prefill_cp_token_allgather_step,
)
from prefill_o_proj import (
    O_PROJ_LOCAL_COLS,
    O_PROJ_LOCAL_GROUPS,
    O_PROJ_SCRATCH_COLS,
    O_PROJ_SCRATCH_D,
    O_PROJ_SCRATCH_GROUPS,
    O_PROJ_SCRATCH_INPUT,
    O_PROJ_SCRATCH_RANK,
    O_PROJ_WO_A_WINDOW_COLS,
    O_PROJ_WO_A_WINDOW_ROWS,
    O_PROJ_WO_B_WINDOW_COLS,
    O_PROJ_WO_B_WINDOW_ROWS,
    gather_o_proj_full_weights,
)
from prefill_sparse_attn import (
    PREFILL_ATTN_TILE,
    SPARSE_BIAS_COLS,
    VALID_BLOCK_MASK_COLS,
    golden_prefill_sparse_attn,
    sparse_attn_physical,
)
from qkv_proj_rope import (
    golden_qkv_proj_rope,
    kv_proj_rope,
    q_proj_rope,
    qkv_proj_rope,
    rope_prepare,
)
from rmsnorm import golden_rms_norm, rms_norm


# Dynamic shape variables. The single-die path runs on one token axis; the
# context-parallel path needs two, because the query slice and the gathered KV
# stream carry different extents in one program.
T_DYN = pl.dynamic("PREFILL_SWA_T_DYN")
CP_Q_T_DYN = pl.dynamic("PREFILL_SWA_CP_Q_T_DYN")
CP_KV_T_DYN = pl.dynamic("PREFILL_SWA_CP_KV_T_DYN")
BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")

# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HEAD_DIM = ROPE_DIM
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
IDX_TOPK = M.index_topk
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# paged KV cache. The ratio-0 path carries only the sliding-window cache.
BLOCK_NUM = KV_ORI_BLOCK_NUM
BLOCK_TABLE_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM
SPARSE_CMP_MAX_BLOCKS = (MAX_SEQ_LEN // 128 + BLOCK_SIZE - 1) // BLOCK_SIZE
START_POS = 0


@pl.jit.inline
def prefill_attention_swa(
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
    kv_cache: pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[BLOCK_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    t_dim = pl.tensor.dim(x_hc, 0)
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    # hc_pre -> qkv/rope -> KV writeback -> SWA attention/o_proj -> hc_post.
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)

    x_normed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    # Defers kv_proj_matmul one hop behind rms_norm so qr_proj_matmul dispatches first.
    late_dep = pl.system.task_dummy(deps=[rms_tid])

    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv,
        freqs_cos, freqs_sin, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, late_dep,
    )

    block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cache_write"):
        for write_t in pl.range(t_dim):
            write_row_raw = pl.read(ori_slot_mapping, [write_t])
            if write_row_raw >= 0:
                write_row = pl.cast(write_row_raw, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, :] = kv[write_t : write_t + 1, :]

    swa_indices = pl.create_tensor([t_dim, WIN], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([t_dim, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_window_indices"):
        for idx_t in pl.range(t_dim):
            idx_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            mask_row = pl.full([1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
            abs_pos = pl.read(position_ids, [idx_t])
            window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for win_col in pl.range(WIN):
                win_col_i32 = pl.cast(win_col, pl.INT32)
                if win_col_i32 < window_valid:
                    key_abs = key_start_abs + win_col_i32
                    blk_slot = key_abs // BLOCK_SIZE
                    blk = pl.read(block_table, [pl.cast(blk_slot, pl.INDEX)])
                    if blk >= 0:
                        row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                        pl.write(idx_row, [0, win_col], row)
                        if win_col < SPARSE_BIAS_COLS:
                            block_col = win_col // PREFILL_ATTN_TILE
                            pl.write(mask_row, [0, block_col], pl.cast(1, pl.INT32))
            swa_indices[idx_t : idx_t + 1, 0:WIN] = idx_row
            valid_block_mask[idx_t : idx_t + 1, 0:VALID_BLOCK_MASK_COLS] = mask_row

    cmp_block_table_dummy = pl.create_tensor([SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)
    cmp_kv_dummy = pl.create_tensor([CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_indices_dummy = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.INT32)
    cmp_block_table_dummy_2d = pl.reshape(cmp_block_table_dummy, [1, SPARSE_CMP_MAX_BLOCKS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cmp_dummy_init"):
        cmp_block_table_dummy_2d[:, :] = pl.full([1, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32, value=0)
    for dummy_t in pl.spmd(t_dim, name_hint="prefill_swa_cmp_indices_dummy_init"):
        cmp_indices_dummy[dummy_t : dummy_t + 1, :] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    o_proj_weight_dep = pl.system.task_dummy(deps=[])
    attn_out = sparse_attn_physical(
        q, kv_cache, swa_indices,
        cmp_kv_dummy, cmp_block_table_dummy,
        cmp_indices_dummy,
        valid_block_mask,
        attn_sink,
        freqs_cos, freqs_sin,
        wo_a, wo_b, wo_b_scale, attn_out, o_proj_weight_dep,
    )

    hc_post(attn_out, x_hc, post, comb, x_out)
    return kv_cache, x_out


@pl.jit
def prefill_attention_swa_test(
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
    kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[BLOCK_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    kv_cache.bind_dynamic(0, BLOCK_NUM_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)

    prefill_attention_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, block_table, ori_slot_mapping,
        position_ids,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_out,
    )
    return kv_cache, x_out


@pl.jit.inline
def prefill_attention_swa_cp_core(
    x_normed_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    x_normed_full: pl.Tensor[[CP_KV_T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos_local: pl.Tensor[[CP_Q_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_local: pl.Tensor[[CP_Q_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_cos_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[BLOCK_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
    o_proj_weight_dep: pl.Scalar[pl.TASK_ID],
):
    """SWA attention body: local queries/output projection and replicated full KV."""
    q_dim = pl.tensor.dim(x_normed_local, 0)
    kv_dim = pl.tensor.dim(x_normed_full, 0)

    q_cos_il = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    q_sin_signed = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    q_swap_idx = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_prepare(freqs_cos_local, freqs_sin_local, q_cos_il, q_sin_signed, q_swap_idx)

    q = pl.create_tensor([q_dim, H, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([q_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([q_dim, 1], dtype=pl.FP32)
    q_proj_rope(x_normed_local, wq_a, wq_b, wq_b_scale, gamma_cq, q_cos_il, q_sin_signed, q_swap_idx, q, qr, qr_scale)

    kv_cos_il = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    kv_sin_signed = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    kv_swap_idx = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_prepare(freqs_cos_full, freqs_sin_full, kv_cos_il, kv_sin_signed, kv_swap_idx)

    kv_full = pl.create_tensor([kv_dim, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x_normed_full, wkv, gamma_ckv, kv_cos_il, kv_sin_signed, kv_swap_idx, kv_full, late_dep)

    block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cp_cache_write"):
        for write_t in pl.range(kv_dim):
            write_row_raw = pl.read(ori_slot_mapping_full, [write_t])
            if write_row_raw >= 0:
                write_row = pl.cast(write_row_raw, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, :] = kv_full[write_t : write_t + 1, :]

    swa_indices = pl.create_tensor([q_dim, WIN], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([q_dim, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cp_window_indices"):
        for idx_t in pl.range(q_dim):
            idx_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            mask_row = pl.full([1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
            abs_pos = pl.read(position_ids_local, [idx_t])
            window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for win_col in pl.range(WIN):
                win_col_i32 = pl.cast(win_col, pl.INT32)
                if win_col_i32 < window_valid:
                    key_abs = key_start_abs + win_col_i32
                    blk_slot = key_abs // BLOCK_SIZE
                    blk = pl.read(block_table, [pl.cast(blk_slot, pl.INDEX)])
                    if blk >= 0:
                        block_row = key_abs - blk_slot * BLOCK_SIZE
                        row = pl.cast(blk * BLOCK_SIZE + block_row, pl.INT32)
                        pl.write(idx_row, [0, win_col], row)
                        if win_col < SPARSE_BIAS_COLS:
                            block_col = win_col // PREFILL_ATTN_TILE
                            pl.write(mask_row, [0, block_col], pl.cast(1, pl.INT32))
            swa_indices[idx_t : idx_t + 1, 0:WIN] = idx_row
            valid_block_mask[idx_t : idx_t + 1, 0:VALID_BLOCK_MASK_COLS] = mask_row

    cmp_block_table_dummy = pl.create_tensor([SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)
    cmp_kv_dummy = pl.create_tensor([CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_indices_dummy = pl.create_tensor([q_dim, IDX_TOPK], dtype=pl.INT32)
    cmp_block_table_dummy_2d = pl.reshape(cmp_block_table_dummy, [1, SPARSE_CMP_MAX_BLOCKS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cp_cmp_dummy_init"):
        cmp_block_table_dummy_2d[:, :] = pl.full([1, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32, value=0)
    for dummy_t in pl.spmd(q_dim, name_hint="prefill_swa_cp_cmp_indices_dummy_init"):
        cmp_indices_dummy[dummy_t : dummy_t + 1, :] = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
    attn_out_local = sparse_attn_physical(
        q, kv_cache, swa_indices,
        cmp_kv_dummy, cmp_block_table_dummy, cmp_indices_dummy,
        valid_block_mask, attn_sink,
        freqs_cos_local, freqs_sin_local,
        wo_a, wo_b, wo_b_scale,
        attn_out_local, o_proj_weight_dep,
    )
    return kv_cache, attn_out_local


@pl.jit.inline
def prefill_attention_swa_cp(
    x_hc_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[BLOCK_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_local: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    x_out_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_order_fence: pl.Tensor[[1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    weight_epoch: pl.Scalar[pl.INT32],
):
    """DSA-CP SWA with replicated HC state at both layer boundaries."""
    o_proj_weight_dep = gather_o_proj_full_weights(
        wo_a_local, wo_b_local,
        wo_a_full, wo_b_full,
        o_proj_wo_a_window, o_proj_wo_b_window,
        o_proj_weight_ready, o_proj_weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, weight_epoch,
    )
    q_dim = pl.tensor.dim(position_ids_local, 0)
    kv_dim = pl.tensor.dim(x_hc_full, 0)

    x_mixed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    post_full = pl.create_tensor([kv_dim, HC_MULT], dtype=pl.FP32)
    comb_full = pl.create_tensor([kv_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc_full, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed_full, post_full, comb_full)

    x_normed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed_full, attn_norm_w, x_normed_full)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    local_base = tp_rank * q_dim
    x_normed_local = pl.create_tensor([q_dim, D], dtype=pl.BF16)
    freqs_cos_local = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    freqs_sin_local = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    for local_row in pl.spmd(q_dim, name_hint="prefill_swa_cp_query_slice"):
        full_row = local_base + local_row
        query_row = pl.load(x_normed_full, [full_row, 0], [1, D], target_memory=pl.MemorySpace.Vec)
        pl.store(query_row, [local_row, 0], x_normed_local)
        cos_row = pl.load(freqs_cos, [full_row, 0], [1, ROPE_HEAD_DIM], target_memory=pl.MemorySpace.Vec)
        sin_row = pl.load(freqs_sin, [full_row, 0], [1, ROPE_HEAD_DIM], target_memory=pl.MemorySpace.Vec)
        pl.store(cos_row, [local_row, 0], freqs_cos_local)
        pl.store(sin_row, [local_row, 0], freqs_sin_local)

    attn_out_local = pl.create_tensor([q_dim, D], dtype=pl.BF16)
    kv_cache, attn_out_local = prefill_attention_swa_cp_core(
        x_normed_local, x_normed_full,
        wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv,
        freqs_cos_local, freqs_sin_local,
        freqs_cos, freqs_sin,
        kv_cache, block_table,
        ori_slot_mapping_full,
        position_ids_local, position_ids_full,
        attn_sink,
        wo_a_full, wo_b_full, wo_b_scale,
        attn_out_local,
        late_dep, o_proj_weight_dep,
    )

    attn_out_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    attn_out_full, gather_signal = prefill_cp_token_allgather_step(
        attn_out_local, attn_out_full,
        gather_window, gather_signal,
        group_base, tp_rank,
    )

    hc_post(attn_out_full, x_hc_full, post_full, comb_full, x_out_full)
    return kv_cache, x_out_full, gather_signal


@pl.jit
def prefill_attention_swa_cp_test(
    x_hc_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[BLOCK_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out_full: pl.Out[pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Run one DSA-CP rank's SWA block with replicated layer-boundary state."""
    x_hc_full.bind_dynamic(0, CP_KV_T_DYN)
    freqs_cos.bind_dynamic(0, CP_KV_T_DYN)
    freqs_sin.bind_dynamic(0, CP_KV_T_DYN)
    kv_cache.bind_dynamic(0, BLOCK_NUM_DYN)
    ori_slot_mapping_full.bind_dynamic(0, CP_KV_T_DYN)
    position_ids_local.bind_dynamic(0, CP_Q_T_DYN)
    position_ids_full.bind_dynamic(0, CP_KV_T_DYN)
    x_out_full.bind_dynamic(0, CP_KV_T_DYN)

    wo_a_full = pl.create_tensor([O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], dtype=pl.BF16)
    wo_b_full = pl.create_tensor([O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], dtype=pl.INT8)
    o_proj_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_o_proj_order_init"):
        pl.write(o_proj_order_fence, [0], pl.cast(0, pl.INT32))
    kv_cache, x_out_full, gather_signal = prefill_attention_swa_cp(
        x_hc_full,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, block_table, ori_slot_mapping_full,
        position_ids_local, position_ids_full,
        attn_sink, wo_a, wo_b, wo_b_scale,
        wo_a_full, wo_b_full,
        x_out_full,
        gather_window, gather_signal,
        o_proj_wo_a_window, o_proj_wo_b_window,
        o_proj_weight_ready, o_proj_weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, pl.const(1, pl.INT32),
    )
    return kv_cache, x_out_full


@pl.jit.host
def l3_prefill_attention_swa_cp(
    x_hc_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[TP_SIZE, BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[TP_SIZE, BLOCK_TABLE_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[TP_SIZE, CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    x_out_full: pl.Out[pl.Tensor[[TP_SIZE, CP_KV_T_DYN, HC_MULT, D], pl.FP32]],
):
    """Launch one CP group's SWA block, one child per rank."""
    x_hc_full.bind_dynamic(1, CP_KV_T_DYN)
    freqs_cos.bind_dynamic(1, CP_KV_T_DYN)
    freqs_sin.bind_dynamic(1, CP_KV_T_DYN)
    kv_cache.bind_dynamic(1, BLOCK_NUM_DYN)
    ori_slot_mapping_full.bind_dynamic(1, CP_KV_T_DYN)
    position_ids_local.bind_dynamic(1, CP_Q_T_DYN)
    position_ids_full.bind_dynamic(1, CP_KV_T_DYN)
    x_out_full.bind_dynamic(1, CP_KV_T_DYN)

    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_wo_a_window_buf = pld.alloc_window_buffer([O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16)
    o_proj_wo_b_window_buf = pld.alloc_window_buffer([O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8)
    o_proj_weight_ready_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_weight_consumed_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_wo_a_window = pld.window(
            o_proj_wo_a_window_buf, [O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16
        )
        o_proj_wo_b_window = pld.window(
            o_proj_wo_b_window_buf, [O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8
        )
        o_proj_weight_ready = pld.window(o_proj_weight_ready_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_weight_consumed = pld.window(o_proj_weight_consumed_buf, [TP_SIZE, 1], dtype=pl.INT32)
        prefill_attention_swa_cp_test(
            x_hc_full[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            kv_cache[rank], block_table[rank], ori_slot_mapping_full[rank],
            position_ids_local[rank], position_ids_full[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            x_out_full[rank],
            gather_window, gather_signal,
            o_proj_wo_a_window, o_proj_wo_b_window,
            o_proj_weight_ready, o_proj_weight_consumed,
            0, rank,
            device=rank,
        )



def _quant_w_per_output_channel(w):
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_prefill_attention_swa(tensors):
    """Torch reference for token-major packed SWA prefill."""
    import torch

    from utils import cache_row_from_table

    token_count = tensors["x_hc"].shape[0]
    x_hc_flat = tensors["x_hc"].view(token_count, HC_MULT, D)
    x_mixed = torch.zeros(token_count, D, dtype=torch.bfloat16)
    post = torch.zeros(token_count, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(token_count, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": x_hc_flat,
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })

    q = torch.zeros(token_count, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(token_count, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(token_count, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(token_count, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    rope_cos_t = tensors["freqs_cos"].contiguous()
    rope_sin_t = tensors["freqs_sin"].contiguous()
    golden_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_t,
        "rope_sin": rope_sin_t,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
    })

    kv_cache_in = tensors["kv_cache"].clone()
    kv_cache_flat = kv_cache_in.view(kv_cache_in.shape[0] * BLOCK_SIZE, HEAD_DIM)
    for t in range(token_count):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]

    def build_swa_metadata():
        idx = torch.full((token_count, WIN), -1, dtype=torch.int32)
        pos = tensors["position_ids"]
        table = tensors["block_table"]
        for t in range(token_count):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                row = cache_row_from_table(table, key_abs)
                if row >= 0:
                    idx[t, k] = row
        return idx

    attn_out = torch.zeros(token_count, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": kv_cache_in,
        "swa_indices": build_swa_metadata(),
        "cmp_kv": torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16),
        "cmp_block_table": torch.zeros(SPARSE_CMP_MAX_BLOCKS, dtype=torch.int32),
        "cmp_indices": torch.full((token_count, IDX_TOPK), -1, dtype=torch.int32),
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    tensors["kv_cache"][:] = kv_cache_in

    y = torch.zeros(token_count, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({
        "x": attn_out.view(token_count, D),
        "residual": x_hc_flat,
        "post": post,
        "comb": comb,
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tensor_specs(
    start_pos: int = START_POS,
    token_count: int = PREFILL_SEQ,
):
    import torch
    from golden import TensorSpec
    from utils import cache_row_from_table, quant_w_per_channel, token_local_rope

    # Single-request geometry: the physical token dimension is q_len.
    context_len = start_pos
    q_len = token_count

    if token_count <= 0 or token_count > MAX_SEQ_LEN:
        raise ValueError(f"token_count must be in [1, {MAX_SEQ_LEN}], got {token_count}")
    max_position = context_len + q_len
    if context_len < 0:
        raise ValueError(f"context_len must be non-negative, got {context_len}")
    if max_position > MAX_SEQ_LEN:
        raise ValueError(f"position_ids exceed MAX_SEQ_LEN={MAX_SEQ_LEN}: got {max_position}")

    def token_pos():
        return torch.arange(context_len, context_len + q_len, dtype=torch.int32)

    shared_freqs_cos, shared_freqs_sin = token_local_rope(
        M, 0, token_pos(),
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )

    def init_x_hc():
        return torch.empty(token_count, HC_MULT, D).uniform_(-1, 1)
    # Real layer-0 (SWA) hc_attn scale/base, fn synthetic at real magnitude. A synthetic
    # scale=0.5/base=0 cancels attn_out and the hc residual to near-zero in x_out, where
    # quant noise blows up the relative tail.
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
        return (torch.rand(D, Q_LORA) - 0.5) * D ** -0.5
    def init_wq_b():
        return (torch.rand(Q_LORA, H * HEAD_DIM) - 0.5) * Q_LORA ** -0.5
    def init_wkv():
        return (torch.rand(D, HEAD_DIM) - 0.5) * D ** -0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_block_table():
        tbl = torch.full((BLOCK_TABLE_BLOCKS,), -1, dtype=torch.int32)
        for block in range(BLOCK_TABLE_BLOCKS):
            tbl[block] = block % BLOCK_NUM
        return tbl
    def init_kv_cache():
        cache = torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_block_table()
        start = max(0, context_len - WIN)
        for abs_pos in range(start, context_len):
            row = cache_row_from_table(table, abs_pos)
            value = (torch.rand(HEAD_DIM,) - 0.5) * 0.1
            if row >= 0:
                cache_flat[row] = value.to(torch.bfloat16)
        return cache
    def init_ori_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        pos = token_pos()
        table = init_block_table()
        for t in range(token_count):
            mapping[t] = cache_row_from_table(table, int(pos[t].item()))
        return mapping
    def init_position_ids():
        return token_pos()
    def init_attn_sink():
        return torch.zeros(H)
    def init_wo_a():
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN ** -0.5
    def init_wo_b():
        return (torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

    return [
        TensorSpec("x_hc", [token_count, HC_MULT, D], torch.float32, init_value=init_x_hc),
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
        TensorSpec("freqs_cos", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_kv_cache, is_output=True),
        TensorSpec("block_table", [BLOCK_TABLE_BLOCKS], torch.int32, init_value=init_block_table),
        TensorSpec("ori_slot_mapping", [token_count], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("position_ids", [token_count], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [token_count, HC_MULT, D], torch.float32, is_output=True),
    ]



def build_cp_tensor_specs(
    start_pos: int = START_POS,
    token_count: int = PREFILL_SEQ,
    tp_size: int = TP_SIZE,
):
    """Replicate layer-boundary state and split query metadata across the CP group."""
    import torch

    from golden import TensorSpec

    if tp_size != TP_SIZE:
        raise ValueError(f"tp_size={tp_size} must match import-time TP_SIZE={TP_SIZE}")
    if token_count > PREFILL_GROUP_CAP:
        raise ValueError(f"token_count must be <= {PREFILL_GROUP_CAP}, got {token_count}")
    if token_count % tp_size != 0:
        raise ValueError(f"token_count={token_count} must be a multiple of tp_size={tp_size}")
    local_t = token_count // tp_size

    specs = []
    for spec in build_tensor_specs(start_pos, token_count):
        value = materialize_spec(spec)
        if spec.name == "x_hc":
            specs.append(TensorSpec(
                "x_hc_full", [tp_size, token_count, HC_MULT, D], spec.dtype,
                init_value=cp_stack(value, tp_size),
            ))
        elif spec.name == "ori_slot_mapping":
            specs.append(TensorSpec(
                "ori_slot_mapping_full", [tp_size, token_count], spec.dtype, init_value=cp_stack(value, tp_size),
            ))
        elif spec.name == "position_ids":
            specs.append(TensorSpec(
                "position_ids_local", [tp_size, local_t], spec.dtype,
                init_value=value.reshape(tp_size, local_t).contiguous(),
            ))
            specs.append(TensorSpec(
                "position_ids_full", [tp_size, token_count], spec.dtype, init_value=cp_stack(value, tp_size),
            ))
        elif spec.name == "wo_a":
            shards = [value[rank * O_PROJ_LOCAL_GROUPS : (rank + 1) * O_PROJ_LOCAL_GROUPS] for rank in range(tp_size)]
            specs.append(TensorSpec(
                "wo_a", [tp_size, O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], spec.dtype,
                init_value=torch.stack(shards).contiguous(),
            ))
        elif spec.name == "wo_b":
            shards = [value[:, rank * O_PROJ_LOCAL_COLS : (rank + 1) * O_PROJ_LOCAL_COLS] for rank in range(tp_size)]
            specs.append(TensorSpec(
                "wo_b", [tp_size, D, O_PROJ_LOCAL_COLS], spec.dtype,
                init_value=torch.stack(shards).contiguous(),
            ))
        elif spec.name == "x_out":
            specs.append(TensorSpec("x_out_full", [tp_size, token_count, HC_MULT, D], spec.dtype, is_output=True))
        else:
            specs.append(TensorSpec(
                spec.name, [tp_size, *spec.shape], spec.dtype,
                init_value=cp_stack(value, tp_size), is_output=spec.is_output,
            ))
    return specs


def golden_prefill_attention_swa_cp(tensors):
    """Single-die reference replicated across DSA-CP ranks."""
    import torch

    tp_size, token_count = tensors["x_hc_full"].shape[0], tensors["x_hc_full"].shape[1]

    full = {
        "x_hc": tensors["x_hc_full"][0],
        "hc_attn_fn": tensors["hc_attn_fn"][0],
        "hc_attn_scale": tensors["hc_attn_scale"][0],
        "hc_attn_base": tensors["hc_attn_base"][0],
        "attn_norm_w": tensors["attn_norm_w"][0],
        "wq_a": tensors["wq_a"][0],
        "wq_b": tensors["wq_b"][0],
        "wq_b_scale": tensors["wq_b_scale"][0],
        "wkv": tensors["wkv"][0],
        "gamma_cq": tensors["gamma_cq"][0],
        "gamma_ckv": tensors["gamma_ckv"][0],
        "freqs_cos": tensors["freqs_cos"][0],
        "freqs_sin": tensors["freqs_sin"][0],
        "kv_cache": tensors["kv_cache"][0].clone(),
        "block_table": tensors["block_table"][0],
        "ori_slot_mapping": tensors["ori_slot_mapping_full"][0],
        "position_ids": tensors["position_ids_full"][0],
        "attn_sink": tensors["attn_sink"][0],
        "wo_a": torch.cat([tensors["wo_a"][rank] for rank in range(tp_size)], dim=0),
        "wo_b": torch.cat([tensors["wo_b"][rank] for rank in range(tp_size)], dim=1),
        "wo_b_scale": tensors["wo_b_scale"][0],
        "x_out": torch.zeros(token_count, HC_MULT, D, dtype=torch.float32),
    }
    golden_prefill_attention_swa(full)

    tensors["x_out_full"][:] = full["x_out"].unsqueeze(0).expand(tp_size, *full["x_out"].shape)
    tensors["kv_cache"][:] = full["kv_cache"].unsqueeze(0).expand(tp_size, *full["kv_cache"].shape)

if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser(
        description="Standalone DeepSeek V4 packed prefill SWA correctness test."
    )
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)),
                        help="comma-separated rank group; one ID at --tp 1")
    parser.add_argument("--tp", type=int, default=TP_SIZE,
                        help="rank-group size; must match the import-time --tp")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Absolute position of the first physical query token.")
    parser.add_argument("--token-count", "--num-tokens", dest="token_count", type=int, default=PREFILL_SEQ,
                        help="Physical query-token extent across the group; must divide by --tp.")
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.tp != TP_SIZE:
        raise SystemExit(f"--tp={args.tp} does not match import-time TP_SIZE={TP_SIZE}")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {device_ids}")
    if args.token_count % TP_SIZE != 0:
        parser.error(f"--token-count must be a multiple of --tp={TP_SIZE}, got {args.token_count}")

    if TP_SIZE == 1:
        result = run_jit(
            fn=prefill_attention_swa_test,
            specs=build_tensor_specs(args.start_pos, args.token_count),
            golden_fn=golden_prefill_attention_swa,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=device_ids[0],
                enable_chip_swimlane=args.enable_chip_swimlane,
                enable_dep_gen=args.enable_dep_gen,
            ),
            compile_only=args.compile_only,
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.005, max_diff_hd=1),
                "kv_cache": ratio_allclose(atol=1e-4, rtol=1e-2),
            },
        )
    else:
        from pypto.ir.distributed_compiled_program import DistributedConfig

        result = run_jit(
            fn=l3_prefill_attention_swa_cp,
            specs=build_cp_tensor_specs(args.start_pos, args.token_count, TP_SIZE),
            golden_fn=golden_prefill_attention_swa_cp,
            compile_cfg=dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            ),
            runtime_cfg=dict(platform=args.platform),
            compile_only=args.compile_only,
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                "x_out_full": ratio_reldiff(diff_thd=3e-3, pct_thd=0.005, max_diff_hd=1),
                "kv_cache": ratio_allclose(atol=1e-4, rtol=1e-2),
            },
        )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
