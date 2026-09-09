# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 HCA sparse attention over the sliding window and ratio-128 compressed cache."""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    TP,
    DECODE_SEQ,
    BLOCK_SIZE,
    KV_CMP_BLOCK_NUM,
    KV_ORI_BLOCK_NUM,
)


# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis (block tables)
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
CMP_TABLE_BLOCKS_DYN = pl.dynamic("CMP_TABLE_BLOCKS_DYN")

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
SOFTMAX_SCALE = M.softmax_scale
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

COMPRESS_RATIO = 128
NEG_INF = -1.0e20

# paged KV cache
ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM
# The logical limit is per request; the physical pool is shared by the batch.
# Host metadata builders below admit requests only while their summed page count
# fits HCA_COMPRESSED_POOL_ROWS.
HCA_MAX_COMPRESSED_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
HCA_COMPRESSED_POOL_ROWS = CMP_BLOCK_NUM * BLOCK_SIZE

# tiling
VALID_TOKEN_TILE = 8
GATHER_RUN_TILE = 16
REQUEST_KV_ROWS = WIN + S - 1
ATTN_K_TILE = 128
RAW_K_TILE = ATTN_K_TILE
RAW_WORKERS = 20
H_TILE = 16
NUM_QK_CORES = 24
QK_PRE_LAUNCH = 2
QK_TRANSFER_SLOTS = QK_PRE_LAUNCH + 1
QK_SCORE_READY_EVENT = 0
QK_PROB_READY_EVENT = 1
QK_PV_READY_EVENT = 2
CMP_PAGES_PER_WORK = ATTN_K_TILE // BLOCK_SIZE
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
ROPE_CS_T_TILE = 8
T_PAD = ((T + 16 - 1) // 16) * 16
MERGE_WORKERS = 48
ATTENTION_PUBLISH_WORKERS = 48
ATTENTION_PUBLISH_T_TILE = 4
LOCAL_O_GROUPS = O_GROUPS // TP
GROUP_T_PAD = TP * T_PAD
ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD
PUBLISH_GROUPS = H_TILE // HEADS_PER_GROUP

if WIN != RAW_K_TILE:
    raise ValueError("HCA raw attention evaluates the window in one tile; WIN must equal RAW_K_TILE")
if HCA_MAX_COMPRESSED_ROWS > HCA_COMPRESSED_POOL_ROWS:
    raise ValueError("HCA compressed rows exceed the configured pool")
if ATTN_K_TILE % BLOCK_SIZE != 0:
    raise ValueError("HCA work must contain complete cache pages")
if BLOCK_SIZE % GATHER_RUN_TILE != 0:
    raise ValueError("a contiguous gather run must stay inside one cache block")
if S % ROPE_CS_T_TILE != 0:
    raise ValueError("each request must contain complete inverse-RoPE token tiles")
if H_TILE % HEADS_PER_GROUP != 0:
    raise ValueError(f"HCA head tile {H_TILE} must contain complete output groups")
if PUBLISH_GROUPS != 2:
    raise ValueError("HCA merge pack expects two output groups per head tile")
if O_GROUPS % TP != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {TP}")
if LOCAL_O_GROUPS % PUBLISH_GROUPS != 0:
    raise ValueError("local output groups must contain complete HCA publish tiles")
if T % ATTENTION_PUBLISH_T_TILE != 0:
    raise ValueError("local token capacity must contain complete attention publish tiles")


@pl.jit.inline(auto_scope=False)
def sparse_attn_hca(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    ori_cache_ready_dep: pl.Scalar[pl.TASK_ID],
    cmp_cache_ready_dep: pl.Scalar[pl.TASK_ID],
):
    """Compute raw and compressed HCA states and inverse-RoPE metadata."""
    t_dim = pl.tensor.dim(q, 0)
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_table_blocks = pl.tensor.dim(cmp_block_table, 1)
    cmp_work_count = (cmp_table_blocks + CMP_PAGES_PER_WORK - 1) // CMP_PAGES_PER_WORK
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_dim * H, HEAD_DIM])
    request_count = pl.tensor.dim(cmp_block_table, 0)
    raw_gather_count = request_count
    cmp_gather_count = request_count * cmp_work_count
    cmp_partial_rows = t_dim * H

    stream_state_m = pl.create_tensor([t_dim * H, 1], dtype=pl.FP32)
    stream_state_l = pl.create_tensor([t_dim * H, 1], dtype=pl.FP32)
    stream_heads = pl.create_tensor([t_dim * H, HEAD_DIM], dtype=pl.FP32)
    cmp_partial_m = pl.create_tensor([cmp_partial_rows, 1], dtype=pl.FP32)
    cmp_partial_l = pl.create_tensor([cmp_partial_rows, 1], dtype=pl.FP32)
    cmp_partial_o = pl.create_tensor([cmp_partial_rows, HEAD_DIM], dtype=pl.FP32)
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    raw_branch_tids = pl.array.create(1, pl.TASK_ID)
    cmp_branch_tids = pl.array.create(1, pl.TASK_ID)
    rope_cs_tids = pl.array.create(1, pl.TASK_ID)

    with pl.scope():
        raw_kv = pl.create_tensor([request_count * REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16)
        raw_valid = pl.create_tensor([t_dim, WIN], dtype=pl.FP32)
        with pl.spmd(raw_gather_count, name_hint="hca_gather_kv", deps=[ori_cache_ready_dep]) as raw_gather_tid:
            g_req = pl.tile.get_block_idx()
            g_t0 = g_req * S
            g_base = g_req * REQUEST_KV_ROWS
            g_first_len = pl.read(window_swa_lens, [g_t0])
            g_zero_rows = pl.full([REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16, value=0.0)
            raw_kv[g_base : g_base + REQUEST_KV_ROWS, 0:HEAD_DIM] = g_zero_rows

            g_bulk_matches = pl.cast(g_first_len == WIN, pl.INT32)
            g_bulk_first = pl.read(window_swa_indices, [g_t0, 0])
            g_bulk_matches = g_bulk_matches * pl.cast(g_bulk_first >= 0, pl.INT32)
            if g_first_len == WIN:
                for g_sub in pl.range(WIN // GATHER_RUN_TILE):
                    for g_dr in pl.unroll(GATHER_RUN_TILE):
                        g_row = g_sub * GATHER_RUN_TILE + g_dr
                        g_slot_i32 = pl.read(window_swa_indices, [g_t0, g_row])
                        g_matches = pl.cast(g_slot_i32 == g_bulk_first + g_row, pl.INT32)
                        g_bulk_matches = g_bulk_matches * g_matches
                for g_token in pl.unroll(S - 1):
                    g_t = g_t0 + g_token + 1
                    g_len = pl.read(window_swa_lens, [g_t])
                    g_slot_i32 = pl.read(window_swa_indices, [g_t, g_len - 1])
                    g_matches = pl.cast(g_slot_i32 == g_bulk_first + WIN + g_token, pl.INT32)
                    g_bulk_matches = g_bulk_matches * g_matches
                    g_bulk_matches = g_bulk_matches * pl.cast(g_len == WIN, pl.INT32)
            if g_bulk_matches == 1:
                g_bulk_src = pl.cast(g_bulk_first, pl.INDEX)
                g_bulk_rows = ori_kv_flat[g_bulk_src : g_bulk_src + REQUEST_KV_ROWS, 0:HEAD_DIM]
                raw_kv[g_base : g_base + REQUEST_KV_ROWS, 0:HEAD_DIM] = g_bulk_rows
            else:
                for g_sub in pl.range(WIN // GATHER_RUN_TILE):
                    g_sr0 = g_sub * GATHER_RUN_TILE
                    g_sdst = g_base + g_sr0
                    if g_sr0 + GATHER_RUN_TILE <= g_first_len:
                        g_first = pl.read(window_swa_indices, [g_t0, g_sr0])
                        g_run_matches = pl.cast(g_first >= 0, pl.INT32)
                        for g_dr in pl.unroll(GATHER_RUN_TILE):
                            g_slot_i32 = pl.read(window_swa_indices, [g_t0, g_sr0 + g_dr])
                            g_run_matches = g_run_matches * pl.cast(g_slot_i32 == g_first + g_dr, pl.INT32)
                        if g_run_matches == 1:
                            g_run_src = pl.cast(g_first, pl.INDEX)
                            g_run_rows = ori_kv_flat[g_run_src : g_run_src + GATHER_RUN_TILE, 0:HEAD_DIM]
                            raw_kv[g_sdst : g_sdst + GATHER_RUN_TILE, 0:HEAD_DIM] = g_run_rows
                        else:
                            for g_dr in pl.range(GATHER_RUN_TILE):
                                g_slot_i32 = pl.read(window_swa_indices, [g_t0, g_sr0 + g_dr])
                                if g_slot_i32 >= 0:
                                    g_slot = pl.cast(g_slot_i32, pl.INDEX)
                                    g_dst = g_sdst + g_dr
                                    g_slot_row = ori_kv_flat[g_slot : g_slot + 1, 0:HEAD_DIM]
                                    raw_kv[g_dst : g_dst + 1, 0:HEAD_DIM] = g_slot_row
                    else:
                        for g_dr in pl.range(GATHER_RUN_TILE):
                            g_row = g_sr0 + g_dr
                            if g_row < g_first_len:
                                g_slot_i32 = pl.read(window_swa_indices, [g_t0, g_row])
                                if g_slot_i32 >= 0:
                                    g_slot = pl.cast(g_slot_i32, pl.INDEX)
                                    g_dst = g_base + g_row
                                    g_slot_row = ori_kv_flat[g_slot : g_slot + 1, 0:HEAD_DIM]
                                    raw_kv[g_dst : g_dst + 1, 0:HEAD_DIM] = g_slot_row

                for g_token in pl.unroll(S - 1):
                    g_t = g_t0 + g_token + 1
                    g_len = pl.read(window_swa_lens, [g_t])
                    g_slot_i32 = pl.read(window_swa_indices, [g_t, g_len - 1])
                    if g_slot_i32 >= 0:
                        g_slot = pl.cast(g_slot_i32, pl.INDEX)
                        g_dst = g_base + g_first_len + g_token
                        g_slot_row = ori_kv_flat[g_slot : g_slot + 1, 0:HEAD_DIM]
                        raw_kv[g_dst : g_dst + 1, 0:HEAD_DIM] = g_slot_row

        with pl.spmd(t_dim // VALID_TOKEN_TILE, name_hint="hca_raw_valid") as raw_valid_tid:
            valid_block = pl.tile.get_block_idx()
            valid_t0 = valid_block * VALID_TOKEN_TILE
            valid_col = pl.cast(pl.tile.arange(0, [1, WIN], dtype=pl.INT32), target_type=pl.FP32)
            valid_zero = pl.tile.full([VALID_TOKEN_TILE, WIN], dtype=pl.FP32, value=0.0)
            valid_cols = pl.col_expand_add(valid_zero, valid_col)
            valid_lens_i32 = pl.load(window_swa_lens, [valid_t0], [VALID_TOKEN_TILE], target_memory=pl.MemorySpace.Vec)
            valid_lens = pl.cast(pl.reshape(valid_lens_i32, [VALID_TOKEN_TILE, 1]), target_type=pl.FP32)
            valid_mask = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(valid_cols, valid_lens)), 0.0), 1.0)
            pl.store(valid_mask, [valid_t0, 0], raw_valid)

        with pl.spmd(HALF_ROPE // ROPE_TILE, name_hint="rope_cs") as rope_cs_tid:
            cp = pl.tile.get_block_idx()
            cp_r0 = cp * ROPE_TILE
            cp_c0 = 2 * cp_r0
            cs_one = pl.full([ROPE_CS_T_TILE, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
            cs_index_i32 = pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32)
            cs_index = pl.cast(cs_index_i32, target_type=pl.FP32)
            cs_col = pl.col_expand_mul(cs_one, cs_index)
            cs_dup = pl.mul(cs_col, 0.5)
            cs_dup_idx = pl.cast(cs_dup, target_type=pl.INT32, mode="trunc")
            cs_dup_f = pl.cast(cs_dup_idx, target_type=pl.FP32)
            cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
            cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
            for cs_rb in pl.range(rope_cs_blocks):
                cs_t0 = cs_rb * ROPE_CS_T_TILE
                cs_cos_bf16 = freqs_cos[cs_t0 : cs_t0 + ROPE_CS_T_TILE, cp_r0 : cp_r0 + ROPE_TILE]
                cs_sin_bf16 = freqs_sin[cs_t0 : cs_t0 + ROPE_CS_T_TILE, cp_r0 : cp_r0 + ROPE_TILE]
                cs_cos = pl.cast(cs_cos_bf16, target_type=pl.FP32)
                cs_sin = pl.cast(cs_sin_bf16, target_type=pl.FP32)
                cs_cos_dup = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
                cs_sin_dup = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
                cs_sin_signed = pl.mul(cs_sin_dup, cs_sign)
                rope_cos_il[cs_t0 : cs_t0 + ROPE_CS_T_TILE, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_dup
                rope_sin_signed[cs_t0 : cs_t0 + ROPE_CS_T_TILE, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = cs_sin_signed

        # One raw-window block per query, pipelined across queries.
        raw_transfer_rows = RAW_WORKERS * QK_TRANSFER_SLOTS * H
        raw_score_transfer = pl.create_tensor([raw_transfer_rows, ATTN_K_TILE], dtype=pl.FP32)
        raw_probability_transfer = pl.create_tensor([raw_transfer_rows, ATTN_K_TILE], dtype=pl.BF16)
        raw_ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)

        with pl.spmd(RAW_WORKERS, name_hint="hca_raw_attn", deps=[raw_gather_tid, raw_valid_tid], allow_early_resolve=True) as raw_heads_tid:
            raw_qk_task = pl.tile.get_block_idx()
            pl.system.set_ffts(raw_ffts_workspace)
            raw_qk_count = pl.max((t_dim - raw_qk_task + RAW_WORKERS - 1) // RAW_WORKERS, 0)
            for raw_qk_tick in pl.range(raw_qk_count + QK_PRE_LAUNCH):
                if raw_qk_tick < raw_qk_count:
                    raw_qk_t = raw_qk_task + raw_qk_tick * RAW_WORKERS
                    raw_qk_slot = raw_qk_task * QK_TRANSFER_SLOTS + raw_qk_tick % QK_TRANSFER_SLOTS
                    raw_qk_row = raw_qk_slot * H
                    raw_qk_request = raw_qk_t // S
                    raw_qk_token = raw_qk_t % S
                    raw_qk_first_len = pl.read(window_swa_lens, [raw_qk_request * S])
                    raw_qk_drop = pl.max(raw_qk_first_len + raw_qk_token - WIN, 0)
                    raw_qk_base = raw_qk_request * REQUEST_KV_ROWS + raw_qk_drop
                    raw_qk_q = pl.load(q_flat, [raw_qk_t * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                    raw_qk_kv = pl.load(raw_kv, [raw_qk_base, 0], [ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                    raw_qk_scores = pl.matmul(raw_qk_q, pl.tile.transpose_view(raw_qk_kv), out_dtype=pl.FP32)
                    pl.store(raw_qk_scores, [raw_qk_row, 0], raw_score_transfer)
                    pl.system.sync_set(QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if raw_qk_tick >= QK_PRE_LAUNCH:
                    raw_pv_item = raw_qk_tick - QK_PRE_LAUNCH
                    raw_pv_t = raw_qk_task + raw_pv_item * RAW_WORKERS
                    raw_pv_slot = raw_qk_task * QK_TRANSFER_SLOTS + raw_pv_item % QK_TRANSFER_SLOTS
                    raw_pv_request = raw_pv_t // S
                    raw_pv_first_len = pl.read(window_swa_lens, [raw_pv_request * S])
                    raw_pv_drop = pl.max(raw_pv_first_len + raw_pv_t % S - WIN, 0)
                    raw_pv_base = raw_pv_request * REQUEST_KV_ROWS + raw_pv_drop
                    pl.system.sync_wait(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                    raw_pv_probability = pl.load(
                        raw_probability_transfer, [raw_pv_slot * H, 0], [H, ATTN_K_TILE], target_memory=pl.MemorySpace.Mat,
                    )
                    raw_pv_kv = pl.load(raw_kv, [raw_pv_base, 0], [ATTN_K_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
                    raw_pv_output = pl.matmul(raw_pv_probability, raw_pv_kv, out_dtype=pl.FP32)
                    pl.store(raw_pv_output, [raw_pv_t * H, 0], stream_heads)

            for raw_qk_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(raw_ffts_workspace)
                raw_qk_head = raw_qk_aiv * (H // 2)
                raw_qk_reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                for raw_qk_item in pl.range(raw_qk_count):
                    raw_qk_t = raw_qk_task + raw_qk_item * RAW_WORKERS
                    raw_qk_slot = raw_qk_task * QK_TRANSFER_SLOTS + raw_qk_item % QK_TRANSFER_SLOTS
                    raw_qk_row = raw_qk_slot * H + raw_qk_head
                    pl.system.sync_wait(QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    raw_qk_scores_half = pl.load(raw_score_transfer, [raw_qk_row, 0], [H // 2, ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                    raw_qk_valid_row = pl.load(raw_valid, [raw_qk_t, 0], [1, ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                    raw_qk_bias = pl.mul(pl.sub(raw_qk_valid_row, 1.0), -NEG_INF)
                    raw_qk_scores_half = pl.col_expand_add(pl.mul(raw_qk_scores_half, SOFTMAX_SCALE), raw_qk_bias)
                    raw_qk_mi = pl.row_max(raw_qk_scores_half, raw_qk_reduce_tmp)
                    raw_qk_exp = pl.exp(pl.row_expand_sub(raw_qk_scores_half, raw_qk_mi))
                    raw_qk_exp = pl.col_expand_mul(raw_qk_exp, raw_qk_valid_row)
                    raw_qk_li = pl.row_sum(raw_qk_exp, raw_qk_reduce_tmp)
                    raw_qk_probability = pl.cast(raw_qk_exp, target_type=pl.BF16, mode="rint")
                    pl.store(raw_qk_probability, [raw_qk_row, 0], raw_probability_transfer)
                    pl.store(raw_qk_mi, [raw_qk_t * H + raw_qk_head, 0], stream_state_m)
                    pl.store(raw_qk_li, [raw_qk_t * H + raw_qk_head, 0], stream_state_l)
                    pl.system.sync_set(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)

        raw_branch_tids[0] = raw_heads_tid
        rope_cs_tids[0] = rope_cs_tid

    with pl.scope():
        cmp_work_kv = pl.create_tensor([cmp_gather_count * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16)
        with pl.spmd(cmp_gather_count, name_hint="hca_cmp_work_gather", deps=[cmp_cache_ready_dep]) as cmp_gather_tid:
            gather_item = pl.tile.get_block_idx()
            gather_request = gather_item // cmp_work_count
            gather_work = gather_item - gather_request * cmp_work_count
            gather_first_col = gather_work * CMP_PAGES_PER_WORK
            gather_dst0 = gather_item * ATTN_K_TILE
            for gather_page in pl.unroll(CMP_PAGES_PER_WORK):
                gather_page_col = gather_first_col + gather_page
                gather_dst = gather_dst0 + gather_page * BLOCK_SIZE
                gather_zero_rows = pl.full([BLOCK_SIZE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                cmp_work_kv[gather_dst : gather_dst + BLOCK_SIZE, 0:HEAD_DIM] = gather_zero_rows
                if gather_page_col < cmp_table_blocks:
                    gather_page_i32 = pl.read(cmp_block_table, [gather_request, gather_page_col])
                    if gather_page_i32 >= 0:
                        if gather_page_i32 < cmp_block_num:
                            gather_page_id = pl.cast(gather_page_i32, pl.INDEX)
                            gather_src = gather_page_id * BLOCK_SIZE
                            gather_page_rows = cmp_kv_flat[gather_src : gather_src + BLOCK_SIZE, 0:HEAD_DIM]
                            cmp_work_kv[gather_dst : gather_dst + BLOCK_SIZE, 0:HEAD_DIM] = gather_page_rows

        # Seed the maximum from the sink and publish one compressed state per query.
        # The sink contributes to the denominator only in the final raw/compressed merge.
        attn_sink_col = pl.reshape(attn_sink, [H, 1])
        transfer_slots = NUM_QK_CORES * QK_TRANSFER_SLOTS
        transfer_heads = transfer_slots * H
        score_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.FP32)
        probability_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.BF16)
        pv_transfer = pl.create_tensor([transfer_heads, HEAD_DIM], dtype=pl.FP32)
        mi_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
        li_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
        ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
        with pl.spmd(NUM_QK_CORES, name_hint="hca_cmp_qk_pv", deps=[cmp_gather_tid], allow_early_resolve=True) as cmp_qk_tid:
            qk_core = pl.tile.get_block_idx()
            pl.system.set_ffts(ffts_workspace)
            for qk_t in pl.range(qk_core, t_dim, NUM_QK_CORES):
                qk_request = qk_t // S
                qk_position = pl.max(pl.read(position_ids, [qk_t]), -1)
                qk_kv_len = pl.max(pl.read(kv_seq_lens, [qk_request]), 0)
                qk_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min((qk_position + 1) // COMPRESS_RATIO, qk_kv_len // COMPRESS_RATIO))
                qk_blocks = pl.min(cmp_work_count, (qk_rows + ATTN_K_TILE - 1) // ATTN_K_TILE)
                qk_q = pl.load(
                    q_flat, [qk_t * H, 0], [H, HEAD_DIM], target_memory=pl.MemorySpace.Mat,
                )
                for qk_tick in pl.range(qk_blocks + QK_PRE_LAUNCH):
                    if qk_tick < qk_blocks:
                        qk_sb = qk_tick
                        qk_first_page = pl.read(cmp_block_table, [qk_request, qk_sb * CMP_PAGES_PER_WORK])
                        if pl.min(qk_first_page + 1, cmp_block_num - qk_first_page) > 0:
                            qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                            qk_kv_row = (qk_request * cmp_work_count + qk_sb) * ATTN_K_TILE
                            qk_transfer_row = qk_slot * H
                            qk_kv = pl.load(
                                cmp_work_kv, [qk_kv_row, 0], [ATTN_K_TILE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            qk_scores = pl.matmul(qk_q, pl.tile.transpose_view(qk_kv), out_dtype=pl.FP32)
                            pl.store(qk_scores, [qk_transfer_row, 0], score_transfer)
                            pl.system.sync_set(
                                QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX,
                                ffts_mode=2, core_type=pl.KernelType.AIC,
                            )
                    if qk_tick >= QK_PRE_LAUNCH:
                        pv_sb = qk_tick - QK_PRE_LAUNCH
                        pv_first_page = pl.read(cmp_block_table, [qk_request, pv_sb * CMP_PAGES_PER_WORK])
                        if pl.min(pv_first_page + 1, cmp_block_num - pv_first_page) > 0:
                            pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                            pv_kv_row = (qk_request * cmp_work_count + pv_sb) * ATTN_K_TILE
                            pv_transfer_row = pv_slot * H
                            pl.system.sync_wait(QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                            pv_probability = pl.load(
                                probability_transfer, [pv_transfer_row, 0], [H, ATTN_K_TILE],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pv_kv = pl.load(
                                cmp_work_kv, [pv_kv_row, 0], [ATTN_K_TILE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            pv_output = pl.matmul(pv_probability, pv_kv, out_dtype=pl.FP32)
                            pl.store(pv_output, [pv_transfer_row, 0], pv_transfer)
                            pl.system.sync_set(
                                QK_PV_READY_EVENT, pipe=pl.PipeType.FIX,
                                ffts_mode=2, core_type=pl.KernelType.AIC,
                            )

                for qk_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    pl.system.set_ffts(ffts_workspace)
                    qk_lane_head = qk_aiv * (H // 2)
                    qk_reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    running_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                    running_l = pl.tile.muls(running_m, 0.0)
                    running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                    running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                    for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                        qk_blocks + QK_PRE_LAUNCH,
                        init_values=(running_m, running_l, running_left, running_right),
                    ):
                        if qk_tick < qk_blocks:
                            qk_sb = qk_tick
                            qk_first_page = pl.read(cmp_block_table, [qk_request, qk_sb * CMP_PAGES_PER_WORK])
                            if pl.min(qk_first_page + 1, cmp_block_num - qk_first_page) > 0:
                                qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                                qk_transfer_row = qk_slot * H
                                pl.system.sync_wait(QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                                qk_scores_half = pl.load(
                                    score_transfer, [qk_transfer_row + qk_lane_head, 0], [H // 2, ATTN_K_TILE],
                                    target_memory=pl.MemorySpace.Vec,
                                )
                                qk_scaled = pl.mul(qk_scores_half, SOFTMAX_SCALE)
                                qk_valid_rows = pl.min(ATTN_K_TILE, qk_rows - qk_sb * ATTN_K_TILE)
                                qk_valid_scores = pl.set_validshape(qk_scaled, H // 2, qk_valid_rows)
                                qk_masked = pl.fillpad(qk_valid_scores, pad_value=pl.PadValue.min)
                                qk_mi = pl.row_max(qk_masked, qk_reduce_tmp)
                                qk_exp = pl.exp(pl.row_expand_sub(qk_masked, qk_mi))
                                qk_li = pl.row_sum(qk_exp, qk_reduce_tmp)
                                qk_probability = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                                pl.store(qk_probability, [qk_transfer_row + qk_lane_head, 0], probability_transfer)
                                pl.store(qk_mi, [qk_transfer_row + qk_lane_head, 0], mi_transfer)
                                pl.store(qk_li, [qk_transfer_row + qk_lane_head, 0], li_transfer)
                                pl.system.sync_set(
                                    QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3,
                                    ffts_mode=2, core_type=pl.KernelType.AIV,
                                )
                        if qk_tick >= QK_PRE_LAUNCH:
                            pv_sb = qk_tick - QK_PRE_LAUNCH
                            pv_first_page = pl.read(cmp_block_table, [qk_request, pv_sb * CMP_PAGES_PER_WORK])
                            if pl.min(pv_first_page + 1, cmp_block_num - pv_first_page) > 0:
                                pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                                pv_transfer_row = pv_slot * H
                                pl.system.sync_wait(QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                                pv_m = pl.load(mi_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                                pv_l = pl.load(li_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec)
                                next_m = pl.maximum(m_iter, pv_m)
                                alpha = pl.exp(pl.sub(m_iter, next_m))
                                beta = pl.exp(pl.sub(pv_m, next_m))
                                next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, pv_l))
                                pv_left = pl.load(
                                    pv_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, HEAD_DIM // 2],
                                    target_memory=pl.MemorySpace.Vec,
                                )
                                next_left = pl.add(pl.row_expand_mul(left_iter, alpha), pl.row_expand_mul(pv_left, beta))
                                pv_right = pl.load(
                                    pv_transfer, [pv_transfer_row + qk_lane_head, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2],
                                    target_memory=pl.MemorySpace.Vec,
                                )
                                next_right = pl.add(pl.row_expand_mul(right_iter, alpha), pl.row_expand_mul(pv_right, beta))
                                m_valid, l_valid, left_valid, right_valid = pl.yield_(next_m, next_l, next_left, next_right)
                            else:
                                m_valid, l_valid, left_valid, right_valid = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                            m_after, l_after, left_after, right_after = pl.yield_(m_valid, l_valid, left_valid, right_valid)
                        else:
                            m_after, l_after, left_after, right_after = pl.yield_(m_iter, l_iter, left_iter, right_iter)
                        running_m, running_l, running_left, running_right = pl.yield_(m_after, l_after, left_after, right_after)
                    qk_output_row = qk_t * H + qk_lane_head
                    pl.store(running_m, [qk_output_row, 0], cmp_partial_m)
                    pl.store(running_l, [qk_output_row, 0], cmp_partial_l)
                    pl.store(running_left, [qk_output_row, 0], cmp_partial_o)
                    pl.store(running_right, [qk_output_row, HEAD_DIM // 2], cmp_partial_o)

        cmp_branch_tids[0] = cmp_qk_tid

    return (
        stream_state_m, stream_state_l, stream_heads,
        cmp_partial_m, cmp_partial_l, cmp_partial_o,
        rope_cos_il, rope_sin_signed,
        raw_branch_tids[0], cmp_branch_tids[0], rope_cs_tids[0],
    )


@pl.jit.inline(auto_scope=False)
def sparse_attn_hca_tp1(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    cache_ready_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write HCA heads as grouped ``[T_PAD, O_GROUP_IN]`` slabs."""
    (
        stream_state_m, stream_state_l, stream_heads,
        cmp_partial_m, cmp_partial_l, cmp_partial_o,
        rope_cos_il, rope_sin_signed,
        raw_tid, cmp_tid, rope_tid,
    ) = sparse_attn_hca(
        q, ori_kv, window_swa_indices, window_swa_lens, cmp_kv, cmp_block_table, position_ids, kv_seq_lens, attn_sink, freqs_cos,
        freqs_sin, cache_ready_dep, cache_ready_dep,
    )
    t_dim = pl.tensor.dim(stream_state_m, 0) // H
    stream_block_count = t_dim * (H // H_TILE)
    attn_sink_col = pl.reshape(attn_sink, [H, 1])

    with pl.spmd(
        MERGE_WORKERS, name_hint="hca_stream_merge_pack", deps=[raw_tid, cmp_tid, rope_tid],
    ) as heads_tid:
        worker = pl.tile.get_block_idx()
        for stream_idx in pl.range(worker, stream_block_count, MERGE_WORKERS):
            merge_t = stream_idx // (H // H_TILE)
            merge_h_tile = stream_idx - merge_t * (H // H_TILE)
            merge_h0 = merge_h_tile * H_TILE
            merge_state_row = merge_t * H + merge_h0
            stream_m = pl.load(stream_state_m, [merge_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_l = pl.load(stream_state_l, [merge_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_o = pl.load(stream_heads, [merge_state_row, 0], [H_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec)
            # Compressed QK/PV publishes one online-softmax state per query.
            stream_cmp_m = pl.load(cmp_partial_m, [merge_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_cmp_l = pl.load(cmp_partial_l, [merge_state_row, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_cmp_o = pl.load(
                cmp_partial_o, [merge_state_row, 0], [H_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Vec,
            )
            stream_m_new = pl.maximum(stream_m, stream_cmp_m)
            stream_alpha = pl.exp(pl.sub(stream_m, stream_m_new))
            stream_beta = pl.exp(pl.sub(stream_cmp_m, stream_m_new))
            stream_l = pl.add(pl.mul(stream_alpha, stream_l), pl.mul(stream_beta, stream_cmp_l))
            stream_o_scaled = pl.row_expand_mul(stream_o, stream_alpha)
            stream_cmp_o_scaled = pl.row_expand_mul(stream_cmp_o, stream_beta)
            stream_o = pl.add(stream_o_scaled, stream_cmp_o_scaled)
            stream_m = stream_m_new
            stream_sink = pl.load(attn_sink_col, [merge_h0, 0], [H_TILE, 1], target_memory=pl.MemorySpace.Vec)
            stream_sink_tile = pl.add(pl.sub(stream_m, stream_m), stream_sink)
            stream_denom = pl.add(stream_l, pl.exp(pl.sub(stream_sink_tile, stream_m)))
            stream_output = pl.row_expand_div(stream_o, stream_denom)
            pl.store(stream_output, [merge_state_row, 0], stream_heads)
            packed_stream_output = stream_heads[merge_state_row : merge_state_row + H_TILE, 0:HEAD_DIM]
            stream_bf16 = pl.cast(packed_stream_output, target_type=pl.BF16, mode="rint")
            stream_rope = packed_stream_output[0:H_TILE, NOPE_DIM:HEAD_DIM]
            stream_cos_il = rope_cos_il[merge_t : merge_t + 1, 0:ROPE_DIM]
            stream_sin_signed = rope_sin_signed[merge_t : merge_t + 1, 0:ROPE_DIM]
            stream_swap_one = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=1.0)
            stream_swap_index = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
            stream_swap_col = pl.col_expand_mul(stream_swap_one, stream_swap_index)
            stream_swap_dup = pl.cast(pl.mul(stream_swap_col, 0.5), target_type=pl.INT32, mode="trunc")
            stream_swap_dup_f = pl.cast(stream_swap_dup, target_type=pl.FP32)
            stream_swap_lane = pl.sub(stream_swap_col, pl.mul(stream_swap_dup_f, 2.0))
            stream_swap = pl.sub(pl.add(stream_swap_col, 1.0), pl.mul(stream_swap_lane, 2.0))
            stream_swap_row = pl.cast(stream_swap, target_type=pl.INT32)
            stream_swap_zero = pl.full([H_TILE, ROPE_DIM], dtype=pl.INT32, value=0)
            stream_swap_idx = pl.col_expand_add(stream_swap_zero, stream_swap_row)
            stream_swapped = pl.gather(stream_rope, dim=-1, index=stream_swap_idx)
            stream_rope_cos = pl.col_expand_mul(stream_rope, stream_cos_il)
            stream_swap_sin = pl.col_expand_mul(stream_swapped, stream_sin_signed)
            stream_rot = pl.add(stream_rope_cos, stream_swap_sin)
            stream_rope_bf16 = pl.cast(stream_rot, target_type=pl.BF16, mode="rint")
            stream_full_bf16 = pl.concat(stream_bf16[0:H_TILE, 0:NOPE_DIM], stream_rope_bf16)
            for stream_hi in pl.unroll(H_TILE):
                stream_head = merge_h0 + stream_hi
                stream_pack_row = (stream_head // HEADS_PER_GROUP) * T_PAD + merge_t
                stream_pack_col = (stream_head % HEADS_PER_GROUP) * HEAD_DIM
                stream_head_row = stream_full_bf16[stream_hi : stream_hi + 1, 0:HEAD_DIM]
                o_packed_heads[
                    stream_pack_row : stream_pack_row + 1, stream_pack_col : stream_pack_col + HEAD_DIM,
                ] = stream_head_row

    return o_packed_heads, heads_tid


@pl.jit
def sparse_attn_hca_test(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    o_packed_heads: pl.Out[pl.Tensor[[O_GROUPS, T_PAD, O_GROUP_IN], pl.BF16]],
):
    q.bind_dynamic(0, T_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(1, CMP_TABLE_BLOCKS_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)

    cache_ready_dep = pl.system.task_dummy(deps=[])
    o_packed_flat = pl.reshape(o_packed_heads, [O_GROUPS * T_PAD, O_GROUP_IN])
    o_packed_flat, _heads_tid = sparse_attn_hca_tp1(
        q, ori_kv, window_swa_indices, window_swa_lens, cmp_kv, cmp_block_table, position_ids, kv_seq_lens,
        attn_sink, freqs_cos, freqs_sin, o_packed_flat, cache_ready_dep,
    )
    return o_packed_heads


def golden_sparse_attn(tensors):
    """Torch reference for the HCA sparse-attention heads."""
    import torch

    q = tensors["q"].float()
    tokens = q.shape[0]
    batch = tokens // S
    ori_kv = tensors["ori_kv"].float()
    window_swa_indices = tensors["window_swa_indices"]
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    position_ids = tensors["position_ids"].to(torch.int64)
    kv_seq_lens = tensors["kv_seq_lens"].to(torch.int64)
    attn_sink = tensors["attn_sink"].float()
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()

    o = torch.zeros(tokens, H, HEAD_DIM)

    for t in range(tokens):
        b = t // S
        item_rows = []
        item_valid = []
        raw_rows = []
        raw_valid = []

        for raw in window_swa_indices[t].tolist():
            slot = int(raw)
            if slot >= 0:
                blk_id = slot // BLOCK_SIZE
                intra = slot % BLOCK_SIZE
                raw_rows.append(ori_kv[blk_id, intra, 0])
                raw_valid.append(True)
            else:
                raw_rows.append(torch.zeros(HEAD_DIM, dtype=ori_kv.dtype))
                raw_valid.append(False)
        raw_rows = torch.stack(raw_rows, dim=0)
        raw_valid = torch.tensor(raw_valid, dtype=torch.bool)
        for row_begin in range(0, WIN, ATTN_K_TILE):
            item_rows.append(raw_rows[row_begin : row_begin + ATTN_K_TILE])
            item_valid.append(raw_valid[row_begin : row_begin + ATTN_K_TILE])

        position_rows = (int(position_ids[t].item()) + 1) // COMPRESS_RATIO
        cache_rows = int(kv_seq_lens[b].item()) // COMPRESS_RATIO
        compressed_rows = min(position_rows, cache_rows, HCA_MAX_COMPRESSED_ROWS)
        for row_begin in range(0, max(compressed_rows, 0), ATTN_K_TILE):
            valid_rows = min(ATTN_K_TILE, compressed_rows - row_begin)
            rows = []
            valid = []
            for lane in range(ATTN_K_TILE):
                logical_row = row_begin + lane
                if lane < valid_rows:
                    logical_page = logical_row // BLOCK_SIZE
                    physical_page = -1
                    if logical_page < cmp_block_table.shape[1]:
                        physical_page = int(cmp_block_table[b, logical_page].item())
                    if 0 <= physical_page < cmp_kv.shape[0]:
                        rows.append(cmp_kv[physical_page, logical_row % BLOCK_SIZE, 0])
                        valid.append(True)
                        continue
                rows.append(torch.zeros(HEAD_DIM, dtype=cmp_kv.dtype))
                valid.append(False)
            item_rows.append(torch.stack(rows, dim=0))
            item_valid.append(torch.tensor(valid, dtype=torch.bool))

        q_t = q[t]

        block_mi = []
        block_li = []
        block_oi = []
        for kv_tile, valid_tile in zip(item_rows, item_valid):
            scores = (q_t @ kv_tile.T) * SOFTMAX_SCALE
            scores = scores.masked_fill(~valid_tile.unsqueeze(0), NEG_INF)
            mi = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - mi).masked_fill(~valid_tile.unsqueeze(0), 0.0)
            li = exp_scores.sum(dim=-1, keepdim=True)
            oi = exp_scores.to(torch.bfloat16).float() @ kv_tile.to(torch.bfloat16).float()
            block_mi.append(mi)
            block_li.append(li)
            block_oi.append(oi)

        score_max = block_mi[0]
        li = block_li[0]
        oi_num = block_oi[0]
        for mi_cur, li_cur, oi_cur in zip(block_mi[1:], block_li[1:], block_oi[1:]):
            score_max_new = torch.maximum(score_max, mi_cur)
            alpha = torch.exp(score_max - score_max_new)
            beta = torch.exp(mi_cur - score_max_new)
            li = alpha * li + beta * li_cur
            oi_num = alpha * oi_num + beta * oi_cur
            score_max = score_max_new

        denom = li + torch.exp(attn_sink.unsqueeze(-1) - score_max)
        o[t] = oi_num / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    # Pack as [group, T_PAD, group-input]; rows past the runtime token count are
    # capacity padding the kernel never writes.
    packed = tensors["o_packed_heads"]
    packed[:, :tokens] = o.float().view(tokens, O_GROUPS, O_GROUP_IN).permute(1, 0, 2).to(torch.bfloat16)

def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    mixed_topk_fixture: bool = False,
    cache_window_replacement_fixture: bool = False,
    batch: int = B,
    compressed_rows: int = 128,
):
    """Build deterministic demo tensors for the HCA standalone harness."""
    import torch
    from golden import TensorSpec
    from utils import block_table

    tokens = batch * S

    if batch < 1 or batch > B:
        raise ValueError(f"HCA sparse-attention batch must be in [1, {B}], got {batch}")
    if tokens % ROPE_CS_T_TILE != 0:
        raise ValueError(
            f"HCA sparse-attention token count {tokens} must be divisible by "
            f"ROPE_CS_T_TILE={ROPE_CS_T_TILE}",
        )

    if mixed_topk_fixture:
        if batch != 4:
            raise ValueError("mixed HCA length fixture requires batch=4")
        compressed_rows_by_request = torch.tensor([128, 128, 1024, 4096], dtype=torch.int32)
    else:
        compressed_rows_by_request = torch.full((batch,), compressed_rows, dtype=torch.int32)

    rows_below = bool((compressed_rows_by_request < 0).any())
    rows_above = bool((compressed_rows_by_request > HCA_MAX_COMPRESSED_ROWS).any())
    if rows_below or rows_above:
        raise ValueError(
            f"compressed_rows must be in [0, {HCA_MAX_COMPRESSED_ROWS}], "
            f"got {compressed_rows_by_request.tolist()}",
        )
    pages_per_request = ((compressed_rows_by_request.to(torch.int64) + BLOCK_SIZE - 1) // BLOCK_SIZE)
    table_blocks = max(int(pages_per_request.max().item()), 1)
    required_pages = int(pages_per_request.sum().item())
    if required_pages > CMP_BLOCK_NUM:
        raise ValueError(
            f"HCA compressed pool needs {required_pages} pages, "
            f"capacity is {CMP_BLOCK_NUM}",
        )

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        q = torch.rand(tokens, H, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        kv = torch.rand(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            table = init_window_block_table()
            sentinel_page = int(table[0, WIN // BLOCK_SIZE].item())
            sentinel_row = WIN % BLOCK_SIZE
            kv[sentinel_page, sentinel_row, 0].fill_(8.0)
        if cache_window_replacement_fixture:
            kv[0, 16, 0].fill_(0.0)
            kv[0, 16, 0, 0] = 4.0
        return kv

    def init_window_swa_indices():
        """Build physical cache-row indices for standalone window raw slots."""
        tbl = init_window_block_table()
        indices = torch.full((tokens, WIN), -1, dtype=torch.int32)
        lens = init_window_swa_lens()
        for t in range(tokens):
            b = t // S
            token = t % S
            valid_len = int(lens[t].item())
            for raw in range(valid_len):
                logical_row = raw if short_window_fixture else token + raw
                blk = int(tbl[b, logical_row // BLOCK_SIZE].item())
                if blk >= 0:
                    indices[t, raw] = blk * BLOCK_SIZE + logical_row % BLOCK_SIZE
        return indices

    def init_window_swa_lens():
        """Build the valid raw-window length for each speculative query."""
        lens = torch.full((tokens,), WIN, dtype=torch.int32)
        if short_window_fixture:
            for t in range(tokens):
                lens[t] = min(17 + t % S, WIN)
        return lens

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_window_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        return block_table(batch=batch, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        table = torch.full((batch, table_blocks), -1, dtype=torch.int32)
        cursor = 0
        for request in range(batch):
            for logical_page in range(int(pages_per_request[request].item())):
                table[request, logical_page] = (cursor * 7 + 3) % CMP_BLOCK_NUM
                cursor += 1
        return table

    def query_compressed_rows():
        rows = compressed_rows_by_request.repeat_interleave(S)
        if short_window_fixture or cache_window_replacement_fixture:
            rows.zero_()
        if causal_regression_fixture:
            rows[0] = 0
        return rows

    def init_position_ids():
        rows = query_compressed_rows().to(torch.int64)
        return torch.clamp(rows * COMPRESS_RATIO - 1, min=0).to(torch.int32)

    def init_kv_seq_lens():
        rows = query_compressed_rows().reshape(batch, S)
        return (rows.max(dim=1).values.to(torch.int64) * COMPRESS_RATIO).to(torch.int32)

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        angles = torch.arange(tokens * HALF_ROPE).reshape(tokens, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        angles = torch.arange(tokens * HALF_ROPE).reshape(tokens, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    return [
        TensorSpec("q", [tokens, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("window_swa_indices", [tokens, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("window_swa_lens", [tokens], torch.int32, init_value=init_window_swa_lens),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [batch, table_blocks], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=init_position_ids),
        TensorSpec("kv_seq_lens", [batch], torch.int32, init_value=init_kv_seq_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("o_packed_heads", [O_GROUPS, T_PAD, O_GROUP_IN], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "-b", "--batch", type=int, default=B,
        help=f"runtime request count in [1, {B}] (the compile-time upper bound). The token axis "
             "is pl.dynamic, so one compiled program serves every value.",
    )
    parser.add_argument(
        "--compressed-rows", type=int, default=128,
        help="visible ratio-128 rows per query; use 8192 for the 1M ceiling",
    )
    parser.add_argument(
        "--causal-regression-fixture", action="store_true", default=False,
        help="Amplify the S=2 future-window-slot regression.",
    )
    parser.add_argument(
        "--short-window-fixture", action="store_true", default=False,
        help="Use a short-window topk row with valid prefix + -1 padding.",
    )
    parser.add_argument(
        "--mixed-topk-fixture", action="store_true", default=False,
        help="Use B=4 compressed histories for 16K, 16K, 128K, and 512K.",
    )
    parser.add_argument(
        "--cache-window-replacement-fixture", action="store_true", default=False,
        help="Place a sentinel row inside the cache window prefix.",
    )
    parser.add_argument("--save-data", action="store_true", help="Save inputs and golden outputs for replay.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument(
        "--enable-dep-gen", action="store_true", default=False,
        help="Capture PTO2 dependency edges (deps.json); the swimlane converter draws "
             "fanout/fanin arrows from the sibling file.",
    )
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")

    if args.mixed_topk_fixture:
        workload = "compressed_rows/request=[128,128,1024,4096]"
    else:
        work_per_query = (args.compressed_rows + ATTN_K_TILE - 1) // ATTN_K_TILE
        workload = f"compressed_rows={args.compressed_rows} work/query={work_per_query}"
    print(f"compress_ratio={COMPRESS_RATIO} {workload}", flush=True)

    result = run(
        fn=sparse_attn_hca_test,
        specs=build_tensor_specs(
            args.causal_regression_fixture, args.short_window_fixture, args.mixed_topk_fixture,
            args.cache_window_replacement_fixture, batch=args.batch, compressed_rows=args.compressed_rows,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        save_data=args.save_data,
        config=dict(
            dump_passes=args.dump_passes, platform=args.platform, device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane, enable_dep_gen=args.enable_dep_gen,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "o_packed_heads": ratio_allclose(atol=1e-4, rtol=1.0 / 128, valid_rows=args.batch * S, valid_axis=1),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
