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
ATTN_D_TILE = 256
RAW_H_TILE = 32
H_TILE = 16
QK_M_TILE = 32
CMP_PAGES_PER_WORK = ATTN_K_TILE // BLOCK_SIZE
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
ROPE_CS_T_TILE = 8
T_PAD = ((T + 16 - 1) // 16) * 16
MERGE_WORKERS = 48
ATTENTION_PUBLISH_WORKERS = 48
ATTENTION_PUBLISH_T_TILE = 8
LOCAL_O_GROUPS = O_GROUPS // TP
GROUP_T_PAD = TP * T_PAD
ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD
PUBLISH_GROUPS = H_TILE // HEADS_PER_GROUP

if WIN != ATTN_K_TILE:
    raise ValueError("HCA raw window must form one baseline-sized attention tile")
if HCA_MAX_COMPRESSED_ROWS > HCA_COMPRESSED_POOL_ROWS:
    raise ValueError("HCA compressed rows exceed the configured pool")
if ATTN_K_TILE % BLOCK_SIZE != 0:
    raise ValueError("HCA work must contain complete cache pages")
if H % RAW_H_TILE != 0:
    raise ValueError("HCA raw head tiles must divide the attention head count")
if H % QK_M_TILE != 0 or QK_M_TILE % H_TILE != 0:
    raise ValueError("HCA head tiles must divide the attention head count")
if BLOCK_SIZE % GATHER_RUN_TILE != 0:
    raise ValueError("a contiguous gather run must stay inside one cache block")
if HEAD_DIM != 2 * ATTN_D_TILE:
    raise ValueError("HCA stream matmuls require two bounded head-width tiles")
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
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    cache_ready_dep: pl.Scalar[pl.TASK_ID],
):
    """Merge raw and compressed attention into grouped inverse-RoPE heads."""
    t_dim = pl.tensor.dim(q, 0)
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    cmp_table_blocks = pl.tensor.dim(cmp_block_table, 1)
    cmp_work_count = (cmp_table_blocks + CMP_PAGES_PER_WORK - 1) // CMP_PAGES_PER_WORK
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [t_dim * H, HEAD_DIM])
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    request_count = pl.tensor.dim(cmp_block_table, 0)
    raw_gather_count = request_count
    raw_block_count = t_dim * (H // RAW_H_TILE)
    stream_block_count = t_dim * (H // H_TILE)
    cmp_gather_count = request_count * cmp_work_count
    cmp_qk_block_count = t_dim * cmp_work_count
    cmp_partial_rows = t_dim * (H // H_TILE) * cmp_work_count * H_TILE

    stream_state_m = pl.create_tensor([t_dim * H, 1], dtype=pl.FP32)
    stream_state_l = pl.create_tensor([t_dim * H, 1], dtype=pl.FP32)
    stream_heads = pl.create_tensor([t_dim * H, HEAD_DIM], dtype=pl.FP32)
    cmp_partial_m = pl.create_tensor([cmp_partial_rows, 8], dtype=pl.FP32)
    cmp_partial_l = pl.create_tensor([cmp_partial_rows, 8], dtype=pl.FP32)
    cmp_partial_o = pl.create_tensor([cmp_partial_rows, HEAD_DIM], dtype=pl.FP32)
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    raw_branch_tids = pl.array.create(1, pl.TASK_ID)
    cmp_branch_tids = pl.array.create(1, pl.TASK_ID)
    rope_cs_tids = pl.array.create(1, pl.TASK_ID)

    with pl.scope():
        raw_kv = pl.create_tensor([request_count * REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16)
        raw_valid = pl.create_tensor([t_dim, WIN], dtype=pl.FP32)
        with pl.spmd(raw_gather_count, name_hint="hca_gather_kv", deps=[cache_ready_dep]) as raw_gather_tid:
            g_req = pl.tile.get_block_idx()
            g_t0 = g_req * S
            g_base = g_req * REQUEST_KV_ROWS
            g_first_len = pl.read(window_swa_lens, [g_t0])
            raw_kv[
                g_base : g_base + REQUEST_KV_ROWS,
                0:HEAD_DIM,
            ] = pl.full([REQUEST_KV_ROWS, HEAD_DIM], dtype=pl.BF16, value=0.0)

            for g_sub in pl.range(WIN // GATHER_RUN_TILE):
                g_sr0 = g_sub * GATHER_RUN_TILE
                g_sdst = g_base + g_sr0
                if g_sr0 + GATHER_RUN_TILE <= g_first_len:
                    g_first = pl.read(window_swa_indices, [g_t0, g_sr0])
                    g_run_matches = pl.cast(g_first >= 0, pl.INT32)
                    for g_dr in pl.unroll(GATHER_RUN_TILE):
                        g_slot_i32 = pl.read(window_swa_indices, [g_t0, g_sr0 + g_dr])
                        g_run_matches = g_run_matches * pl.cast(
                            g_slot_i32 == g_first + g_dr,
                            pl.INT32,
                        )
                    if g_run_matches == 1:
                        g_run_src = pl.cast(g_first, pl.INDEX)
                        raw_kv[
                            g_sdst : g_sdst + GATHER_RUN_TILE,
                            0:HEAD_DIM,
                        ] = ori_kv_flat[
                            g_run_src : g_run_src + GATHER_RUN_TILE,
                            0:HEAD_DIM,
                        ]
                    else:
                        for g_dr in pl.range(GATHER_RUN_TILE):
                            g_slot_i32 = pl.read(window_swa_indices, [g_t0, g_sr0 + g_dr])
                            if g_slot_i32 >= 0:
                                g_slot = pl.cast(g_slot_i32, pl.INDEX)
                                g_dst = g_sdst + g_dr
                                raw_kv[
                                    g_dst : g_dst + 1,
                                    0:HEAD_DIM,
                                ] = ori_kv_flat[
                                    g_slot : g_slot + 1,
                                    0:HEAD_DIM,
                                ]
                else:
                    for g_dr in pl.range(GATHER_RUN_TILE):
                        g_row = g_sr0 + g_dr
                        if g_row < g_first_len:
                            g_slot_i32 = pl.read(window_swa_indices, [g_t0, g_row])
                            if g_slot_i32 >= 0:
                                g_slot = pl.cast(g_slot_i32, pl.INDEX)
                                g_dst = g_base + g_row
                                raw_kv[
                                    g_dst : g_dst + 1,
                                    0:HEAD_DIM,
                                ] = ori_kv_flat[
                                    g_slot : g_slot + 1,
                                    0:HEAD_DIM,
                                ]

            for g_token in pl.unroll(S - 1):
                g_t = g_t0 + g_token + 1
                g_len = pl.read(window_swa_lens, [g_t])
                g_slot_i32 = pl.read(window_swa_indices, [g_t, g_len - 1])
                if g_slot_i32 >= 0:
                    g_slot = pl.cast(g_slot_i32, pl.INDEX)
                    g_dst = g_base + g_first_len + g_token
                    raw_kv[
                        g_dst : g_dst + 1,
                        0:HEAD_DIM,
                    ] = ori_kv_flat[
                        g_slot : g_slot + 1,
                        0:HEAD_DIM,
                    ]

        with pl.spmd(t_dim // VALID_TOKEN_TILE, name_hint="hca_raw_valid") as raw_valid_tid:
            valid_block = pl.tile.get_block_idx()
            valid_t0 = valid_block * VALID_TOKEN_TILE
            valid_col = pl.cast(
                pl.tile.arange(0, [1, WIN], dtype=pl.INT32),
                target_type=pl.FP32,
            )
            valid_zero = pl.tile.full([VALID_TOKEN_TILE, WIN], dtype=pl.FP32, value=0.0)
            valid_cols = pl.col_expand_add(valid_zero, valid_col)
            valid_lens_i32 = pl.load(
                window_swa_lens,
                [valid_t0],
                [VALID_TOKEN_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            valid_lens = pl.cast(
                pl.reshape(valid_lens_i32, [VALID_TOKEN_TILE, 1]),
                target_type=pl.FP32,
            )
            valid_mask = pl.minimum(
                pl.maximum(pl.neg(pl.row_expand_sub(valid_cols, valid_lens)), 0.0),
                1.0,
            )
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

        with pl.spmd(
            raw_block_count,
            name_hint="hca_raw_attn",
            deps=[raw_gather_tid, raw_valid_tid],
        ) as raw_heads_tid:
            stream_idx = pl.tile.get_block_idx()
            stream_t = stream_idx // (H // RAW_H_TILE)
            stream_h_tile = stream_idx - stream_t * (H // RAW_H_TILE)
            stream_h0 = stream_h_tile * RAW_H_TILE
            stream_state_row = stream_t * H + stream_h0
            stream_request = stream_t // S
            stream_token = stream_t - stream_request * S
            stream_first_t = stream_request * S
            stream_first_len = pl.read(window_swa_lens, [stream_first_t])
            stream_drop = pl.max(stream_first_len + stream_token - WIN, 0)
            stream_raw_base = stream_request * REQUEST_KV_ROWS + stream_drop
            stream_q = pl.load(q_flat, [stream_state_row, 0], [RAW_H_TILE, HEAD_DIM], target_memory=pl.MemorySpace.Mat)
            stream_row_max_tmp = pl.create_tile([RAW_H_TILE, RAW_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            stream_row_sum_tmp = pl.create_tile([RAW_H_TILE, RAW_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            stream_raw_kv = pl.load(
                raw_kv,
                [stream_raw_base, 0],
                [RAW_K_TILE, HEAD_DIM],
                target_memory=pl.MemorySpace.Mat,
            )
            stream_raw_kv_t = pl.tile.transpose_view(stream_raw_kv)
            stream_raw_kv_t_left = stream_raw_kv_t[0:ATTN_D_TILE, 0:RAW_K_TILE]
            stream_raw_kv_t_right = stream_raw_kv_t[ATTN_D_TILE:HEAD_DIM, 0:RAW_K_TILE]
            stream_raw_valid_row = pl.load(
                raw_valid,
                [stream_t, 0],
                [1, RAW_K_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            stream_raw_valid_zero = pl.tile.full([RAW_H_TILE, RAW_K_TILE], dtype=pl.FP32, value=0.0)
            stream_raw_valid = pl.col_expand_add(stream_raw_valid_zero, stream_raw_valid_row)
            stream_raw_bias = pl.mul(pl.sub(stream_raw_valid, 1.0), -NEG_INF)
            stream_raw_scores = pl.matmul(stream_q[:, 0:ATTN_D_TILE], stream_raw_kv_t_left, out_dtype=pl.FP32)
            stream_raw_scores = pl.matmul_acc(
                stream_raw_scores,
                stream_q[:, ATTN_D_TILE:HEAD_DIM],
                stream_raw_kv_t_right,
            )
            stream_raw_scores = pl.add(pl.mul(stream_raw_scores, SOFTMAX_SCALE), stream_raw_bias)
            stream_raw_mi_col = pl.row_max(stream_raw_scores, stream_row_max_tmp)
            stream_raw_exp = pl.exp(pl.row_expand_sub(stream_raw_scores, stream_raw_mi_col))
            stream_raw_exp = pl.mul(stream_raw_exp, stream_raw_valid)
            stream_raw_li_col = pl.row_sum(stream_raw_exp, stream_row_sum_tmp)
            stream_raw_exp_bf16 = pl.cast(stream_raw_exp, target_type=pl.BF16, mode="rint")
            stream_raw_oi_left = pl.matmul(stream_raw_exp_bf16, stream_raw_kv[:, 0:ATTN_D_TILE], out_dtype=pl.FP32)
            stream_raw_oi_right = pl.matmul(
                stream_raw_exp_bf16,
                stream_raw_kv[:, ATTN_D_TILE : 2 * ATTN_D_TILE],
                out_dtype=pl.FP32,
            )
            pl.store(stream_raw_mi_col, [stream_state_row, 0], stream_state_m)
            pl.store(stream_raw_li_col, [stream_state_row, 0], stream_state_l)
            pl.store(stream_raw_oi_left, [stream_state_row, 0], stream_heads)
            pl.store(stream_raw_oi_right, [stream_state_row, ATTN_D_TILE], stream_heads)

        raw_branch_tids[0] = raw_heads_tid
        rope_cs_tids[0] = rope_cs_tid

    with pl.scope():
        cmp_work_kv = pl.create_tensor([cmp_gather_count * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16)
        with pl.spmd(cmp_gather_count, name_hint="hca_cmp_work_gather", deps=[cache_ready_dep]) as cmp_gather_tid:
            gather_item = pl.tile.get_block_idx()
            gather_request = gather_item // cmp_work_count
            gather_work = gather_item - gather_request * cmp_work_count
            gather_first_col = gather_work * CMP_PAGES_PER_WORK
            gather_dst0 = gather_item * ATTN_K_TILE
            for gather_page in pl.unroll(CMP_PAGES_PER_WORK):
                gather_page_col = gather_first_col + gather_page
                gather_dst = gather_dst0 + gather_page * BLOCK_SIZE
                cmp_work_kv[
                    gather_dst : gather_dst + BLOCK_SIZE,
                    0:HEAD_DIM,
                ] = pl.full(
                    [BLOCK_SIZE, HEAD_DIM],
                    dtype=pl.BF16,
                    value=0.0,
                )
                if gather_page_col < cmp_table_blocks:
                    gather_page_i32 = pl.read(cmp_block_table, [gather_request, gather_page_col])
                    if gather_page_i32 >= 0:
                        if gather_page_i32 < cmp_block_num:
                            gather_page_id = pl.cast(gather_page_i32, pl.INDEX)
                            gather_src = gather_page_id * BLOCK_SIZE
                            cmp_work_kv[
                                gather_dst : gather_dst + BLOCK_SIZE,
                                0:HEAD_DIM,
                            ] = cmp_kv_flat[
                                gather_src : gather_src + BLOCK_SIZE,
                                0:HEAD_DIM,
                            ]

        with pl.spmd(cmp_qk_block_count, name_hint="hca_cmp_qk_pv", deps=[cmp_gather_tid]) as cmp_qk_tid:
            qk_item = pl.tile.get_block_idx()
            qk_t = qk_item // cmp_work_count
            qk_work = qk_item - qk_t * cmp_work_count
            qk_request = qk_t // S
            qk_work_row = qk_work * ATTN_K_TILE
            qk_token_base = qk_t * (H // H_TILE) * cmp_work_count * H_TILE
            qk_neutral_m = pl.tile.full([H_TILE, 8], dtype=pl.FP32, value=NEG_INF)
            qk_neutral_l = pl.tile.full([H_TILE, 8], dtype=pl.FP32, value=0.0)
            qk_neutral_o = pl.tile.full([H_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            for qk_h_idx in pl.unroll(H // H_TILE):
                qk_row = (qk_token_base + qk_h_idx * cmp_work_count * H_TILE + qk_work * H_TILE)
                pl.store(qk_neutral_m, [qk_row, 0], cmp_partial_m)
                pl.store(qk_neutral_l, [qk_row, 0], cmp_partial_l)
                pl.store(qk_neutral_o, [qk_row, 0], cmp_partial_o)

            qk_rows = pl.cast(0, pl.INDEX)
            qk_position_i32 = pl.read(position_ids, [qk_t])
            if qk_request < request_count:
                qk_kv_len_i32 = pl.read(kv_seq_lens, [qk_request])
                if qk_position_i32 >= 0:
                    if qk_kv_len_i32 >= 0:
                        qk_position = pl.cast(qk_position_i32, pl.INDEX)
                        qk_kv_len = pl.cast(qk_kv_len_i32, pl.INDEX)
                        qk_position_rows = (qk_position + 1) // COMPRESS_RATIO
                        qk_kv_rows = qk_kv_len // COMPRESS_RATIO
                        qk_rows = pl.min(HCA_MAX_COMPRESSED_ROWS, pl.min(qk_position_rows, qk_kv_rows))

            if qk_work_row < qk_rows:
                qk_first_col = qk_work * CMP_PAGES_PER_WORK
                qk_first_page_i32 = pl.read(cmp_block_table, [qk_request, qk_first_col])
                if qk_first_page_i32 >= 0:
                    if qk_first_page_i32 < cmp_block_num:
                        qk_work_src = (qk_request * cmp_work_count + qk_work) * ATTN_K_TILE
                        qk_kv = pl.load(
                            cmp_work_kv,
                            [qk_work_src, 0],
                            [ATTN_K_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        qk_valid_rows = pl.min(ATTN_K_TILE, qk_rows - qk_work_row)
                        qk_kv_t = pl.tile.transpose_view(qk_kv)
                        qk_row_max_tmp = pl.create_tile(
                            [QK_M_TILE, ATTN_K_TILE],
                            dtype=pl.FP32,
                            target_memory=pl.MemorySpace.Vec,
                        )
                        qk_row_sum_tmp = pl.create_tile(
                            [QK_M_TILE, ATTN_K_TILE],
                            dtype=pl.FP32,
                            target_memory=pl.MemorySpace.Vec,
                        )
                        for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
                            qk_h0 = qk_hb * QK_M_TILE
                            qk_head_row = qk_t * H + qk_h0
                            qk_q = pl.load(
                                q_flat,
                                [qk_head_row, 0],
                                [QK_M_TILE, HEAD_DIM],
                                target_memory=pl.MemorySpace.Mat,
                            )
                            qk_scores = pl.matmul(qk_q, qk_kv_t, out_dtype=pl.FP32)
                            qk_scores = pl.mul(qk_scores, SOFTMAX_SCALE)
                            qk_scores = pl.set_validshape(qk_scores, QK_M_TILE, qk_valid_rows)
                            qk_scores = pl.fillpad(qk_scores, pad_value=pl.PadValue.min)
                            qk_mi = pl.row_max(qk_scores, qk_row_max_tmp)
                            qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                            qk_li = pl.row_sum(qk_exp, qk_row_sum_tmp)
                            qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                            qk_oi_left = pl.matmul(qk_exp_bf16, qk_kv[:, 0:ATTN_D_TILE], out_dtype=pl.FP32)
                            qk_oi_right = pl.matmul(qk_exp_bf16, qk_kv[:, ATTN_D_TILE:HEAD_DIM], out_dtype=pl.FP32)
                            qk_oi = pl.concat(qk_oi_left, qk_oi_right)
                            qk_h_idx0 = qk_hb * (QK_M_TILE // H_TILE)
                            qk_row0 = (qk_token_base + qk_h_idx0 * cmp_work_count * H_TILE + qk_work * H_TILE)
                            qk_row1 = qk_row0 + cmp_work_count * H_TILE
                            pl.store(qk_mi[0:H_TILE, 0:1], [qk_row0, 0], cmp_partial_m)
                            pl.store(qk_li[0:H_TILE, 0:1], [qk_row0, 0], cmp_partial_l)
                            pl.store(qk_oi[0:H_TILE, 0:HEAD_DIM], [qk_row0, 0], cmp_partial_o)
                            pl.store(qk_mi[H_TILE:QK_M_TILE, 0:1], [qk_row1, 0], cmp_partial_m)
                            pl.store(qk_li[H_TILE:QK_M_TILE, 0:1], [qk_row1, 0], cmp_partial_l)
                            pl.store(qk_oi[H_TILE:QK_M_TILE, 0:HEAD_DIM], [qk_row1, 0], cmp_partial_o)

        cmp_branch_tids[0] = cmp_qk_tid

    with pl.spmd(
        MERGE_WORKERS,
        name_hint="hca_stream_merge_pack",
        deps=[raw_branch_tids[0], cmp_branch_tids[0], rope_cs_tids[0]],
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
            stream_token_base = merge_t * (H // H_TILE) * cmp_work_count * H_TILE
            for stream_work in pl.range(cmp_work_count):
                stream_partial_row = (
                    stream_token_base
                    + merge_h_tile * cmp_work_count * H_TILE
                    + stream_work * H_TILE
                )
                stream_cmp_m_aligned = pl.load(
                    cmp_partial_m,
                    [stream_partial_row, 0],
                    [H_TILE, 8],
                    target_memory=pl.MemorySpace.Vec,
                )
                stream_cmp_l_aligned = pl.load(
                    cmp_partial_l,
                    [stream_partial_row, 0],
                    [H_TILE, 8],
                    target_memory=pl.MemorySpace.Vec,
                )
                stream_cmp_m = stream_cmp_m_aligned[0:H_TILE, 0:1]
                stream_cmp_l = stream_cmp_l_aligned[0:H_TILE, 0:1]
                stream_cmp_o = pl.load(
                    cmp_partial_o,
                    [stream_partial_row, 0],
                    [H_TILE, HEAD_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                stream_m_new = pl.maximum(stream_m, stream_cmp_m)
                stream_alpha = pl.exp(pl.sub(stream_m, stream_m_new))
                stream_beta = pl.exp(pl.sub(stream_cmp_m, stream_m_new))
                stream_l = pl.add(
                    pl.mul(stream_alpha, stream_l),
                    pl.mul(stream_beta, stream_cmp_l),
                )
                stream_o = pl.add(
                    pl.row_expand_mul(stream_o, stream_alpha),
                    pl.row_expand_mul(stream_cmp_o, stream_beta),
                )
                stream_m = stream_m_new
            stream_sink = pl.load(
                attn_sink_col,
                [merge_h0, 0],
                [H_TILE, 1],
                target_memory=pl.MemorySpace.Vec,
            )
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
            stream_swap_dup = pl.cast(
                pl.mul(stream_swap_col, 0.5),
                target_type=pl.INT32,
                mode="trunc",
            )
            stream_swap_dup_f = pl.cast(stream_swap_dup, target_type=pl.FP32)
            stream_swap_lane = pl.sub(stream_swap_col, pl.mul(stream_swap_dup_f, 2.0))
            stream_swap = pl.sub(
                pl.add(stream_swap_col, 1.0),
                pl.mul(stream_swap_lane, 2.0),
            )
            stream_swap_row = pl.cast(stream_swap, target_type=pl.INT32)
            stream_swap_zero = pl.full([H_TILE, ROPE_DIM], dtype=pl.INT32, value=0)
            stream_swap_idx = pl.col_expand_add(stream_swap_zero, stream_swap_row)
            stream_swapped = pl.gather(stream_rope, dim=-1, index=stream_swap_idx)
            stream_rot = pl.add(
                pl.col_expand_mul(stream_rope, stream_cos_il),
                pl.col_expand_mul(stream_swapped, stream_sin_signed),
            )
            stream_rope_bf16 = pl.cast(stream_rot, target_type=pl.BF16, mode="rint")
            stream_full_bf16 = pl.concat(
                stream_bf16[0:H_TILE, 0:NOPE_DIM],
                stream_rope_bf16,
            )
            for stream_hi in pl.unroll(H_TILE):
                stream_head = merge_h0 + stream_hi
                stream_pack_row = (stream_head // HEADS_PER_GROUP) * T_PAD + merge_t
                stream_pack_col = (stream_head % HEADS_PER_GROUP) * HEAD_DIM
                o_packed_heads[
                    stream_pack_row : stream_pack_row + 1,
                    stream_pack_col : stream_pack_col + HEAD_DIM,
                ] = stream_full_bf16[stream_hi : stream_hi + 1, 0:HEAD_DIM]

    return o_packed_heads, heads_tid


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
    o_packed_heads, heads_tid = sparse_attn_hca(
        q,
        ori_kv,
        window_swa_indices,
        window_swa_lens,
        cmp_kv,
        cmp_block_table,
        position_ids,
        kv_seq_lens,
        attn_sink,
        freqs_cos,
        freqs_sin,
        o_packed_heads,
        cache_ready_dep,
    )

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
        q, ori_kv, window_swa_indices, window_swa_lens,
        cmp_kv, cmp_block_table,
        position_ids, kv_seq_lens,
        attn_sink, freqs_cos, freqs_sin,
        o_packed_flat, cache_ready_dep,
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

        compressed_rows = min(
            (int(position_ids[t].item()) + 1) // COMPRESS_RATIO,
            int(kv_seq_lens[b].item()) // COMPRESS_RATIO,
            HCA_MAX_COMPRESSED_ROWS,
        )
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
        compressed_rows_by_request = torch.full(
            (batch,), compressed_rows, dtype=torch.int32,
        )

    if bool((compressed_rows_by_request < 0).any()) or bool(
        (compressed_rows_by_request > HCA_MAX_COMPRESSED_ROWS).any(),
    ):
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
    parser.add_argument("-b", "--batch", type=int, default=B,
                        help=f"runtime request count in [1, {B}] (the compile-time "
                             "upper bound). The token axis is pl.dynamic, so one compiled program "
                             "serves every value.")
    parser.add_argument("--compressed-rows", type=int, default=128,
                        help="visible ratio-128 rows per query; use 8192 for the 1M ceiling")
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression.")
    parser.add_argument("--short-window-fixture", action="store_true", default=False,
                        help="Use a short-window topk row with valid prefix + -1 padding.")
    parser.add_argument("--mixed-topk-fixture", action="store_true", default=False,
                        help="Use B=4 compressed histories for 16K, 16K, 128K, and 512K.")
    parser.add_argument("--cache-window-replacement-fixture", action="store_true", default=False,
                        help="Place a sentinel row inside the cache window prefix.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")

    if args.mixed_topk_fixture:
        workload = "compressed_rows/request=[128,128,1024,4096]"
    else:
        workload = (
            f"compressed_rows={args.compressed_rows} "
            f"work/query={(args.compressed_rows + ATTN_K_TILE - 1) // ATTN_K_TILE}"
        )
    print(f"compress_ratio={COMPRESS_RATIO} {workload}", flush=True)

    result = run(
        fn=sparse_attn_hca_test,
        specs=build_tensor_specs(
            args.causal_regression_fixture,
            args.short_window_fixture,
            args.mixed_topk_fixture,
            args.cache_window_replacement_fixture,
            batch=args.batch,
            compressed_rows=args.compressed_rows,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "o_packed_heads": ratio_allclose(
                atol=1e-4, rtol=1.0 / 128,
                valid_rows=args.batch * S, valid_axis=1,
            ),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
