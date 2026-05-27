# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill sparse attention standalone bring-up.

This file is intentionally self-contained. The first milestone is a correct
prefill golden/reference and harness; the PyPTO kernel will be filled after
the contract is stable.

Current standalone contract:
- `ori_kv` is the causal prompt KV cache.
- `cmp_kv` is an optional compressed-cache tail selected by `cmp_sparse_indices`.
- `cmp_sparse_indices` uses the prefill/model.py concatenated-KV convention:
  raw indices `< S` read prompt KV, raw indices `>= S` read compressed KV.
- `seqused_kv[b]` is the valid prompt length for batch item `b`.
"""

import pypto.language as pl

from config import FLASH as M, BLOCK_SIZE, INT8_AMAX_EPS, INT8_SCALE_MAX


# Standalone prefill target shape for correctness bring-up.
B = 1
S = 128
T = B * S

# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
IDX_TOPK = M.index_topk
WIN = M.sliding_window
TOPK = WIN + IDX_TOPK
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# cache shapes
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 0
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, S // 4))
PREFILL_SPARSE_TOPK = min(TOPK, min(M.sliding_window, S) + PREFILL_MAX_COMPRESSED)
ORI_MAX_BLOCKS = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = max(1, (PREFILL_MAX_COMPRESSED + BLOCK_SIZE - 1) // BLOCK_SIZE)
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

# RoPE selector shapes kept to match the decode sparse-attn interface.
ROPE_CHUNK = 16
ROPE_INTERLEAVE_CHUNK = 2 * ROPE_CHUNK

# Correctness-first kernel tiling. These mirror the proven decode sparse-attn
# shapes where possible, while padding prompt-K tiles so S=16 can still use
# cube-friendly 64-column attention blocks without out-of-bounds slices.
GATHER_TOKEN_TILE = 8
ATTN_TOKEN_TILE = 32
ROPE_TOKEN_TILE = 8
ROPE_PACK_TOKEN_TILE = 16
MATMUL_ROW_PAD = 16
PV_HEAD_TILE = 16
MERGE_NORM_TOKEN_TILE = 16
PREFILL_ATTN_TILE = 64
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE
ROPE_PACK_SPMD_BLOCKS = ((T + ROPE_PACK_TOKEN_TILE - 1) // ROPE_PACK_TOKEN_TILE) * O_GROUPS
A_K_CHUNK = 128
A_N_CHUNK = 128
B_K_CHUNK = 128
B_N_CHUNK = 128 if T >= 128 else 256
QUANT_CHUNK = 128 if T >= 128 else (128 if T >= 64 else 256)
QUANT_TOKEN_TILE = 32
PROJ_TOKEN_TILE = 128 if T >= 128 else T


@pl.jit.inline
def prefill_sparse_attn(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    even_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    A_K_BLOCKS = O_GROUP_IN // A_K_CHUNK
    A_N_BLOCKS = O_LORA // A_N_CHUNK
    A_AMAX_BLOCKS = O_GROUPS * A_N_BLOCKS
    B_K_BLOCKS = (O_GROUPS * O_LORA) // B_K_CHUNK
    B_N_BLOCKS = D // B_N_CHUNK

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [B * ORI_MAX_BLOCKS])
    cmp_block_table_flat = pl.reshape(cmp_block_table, [B * CMP_MAX_BLOCKS])

    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    prefill_raw_scores = pl.create_tensor([T * H * PREFILL_ATTN_BLOCKS, PREFILL_ATTN_TILE], dtype=pl.FP32)
    prefill_exp = pl.create_tensor([T * H * PREFILL_ATTN_BLOCKS, PREFILL_ATTN_TILE], dtype=pl.BF16)
    prefill_blk_mi = pl.create_tensor([T * H * PREFILL_ATTN_BLOCKS, 1], dtype=pl.FP32)
    prefill_blk_li = pl.create_tensor([T * H * PREFILL_ATTN_BLOCKS, 1], dtype=pl.FP32)
    prefill_blk_oi0 = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.FP32)
    prefill_blk_oi1 = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.FP32)
    prefill_blk_oi2 = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.FP32)
    attn_rope_stage = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.BF16)
    even_select_stage = pl.create_tensor([ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], dtype=pl.BF16)
    odd_select_stage = pl.create_tensor([ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], dtype=pl.BF16)
    rope_even_interleave_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    rope_odd_interleave_buf = pl.create_tensor([T * H, ROPE_DIM], dtype=pl.FP32)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)

    # Stage 1: gather the per-token sparse prompt/compressed KV rows.
    for gather_t0 in pl.parallel(0, T, GATHER_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_gather_prompt_kv_tile"):
            zero_kv_row = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
            for gather_dt in pl.range(GATHER_TOKEN_TILE):
                gather_t = gather_t0 + gather_dt
                gather_b = gather_t // S
                gather_s = gather_t - gather_b * S
                gather_seq_len = pl.read(seqused_kv, [gather_b])
                if gather_s < gather_seq_len:
                    for gather_k in pl.range(PREFILL_SPARSE_PAD):
                        gather_raw = pl.read(cmp_sparse_indices, [gather_t, gather_k])
                        gather_dst_row = gather_t * PREFILL_SPARSE_PAD + gather_k
                        if gather_raw >= 0:
                            if gather_raw < S:
                                gather_ori_slot = gather_raw
                                gather_block_slot = gather_ori_slot // BLOCK_SIZE
                                gather_block_pos = gather_b * ORI_MAX_BLOCKS + gather_block_slot
                                gather_blk = pl.cast(pl.read(ori_block_table_flat, [gather_block_pos]), pl.INDEX)
                                gather_intra = gather_ori_slot - gather_block_slot * BLOCK_SIZE
                                gather_src_row = gather_blk * BLOCK_SIZE + gather_intra
                                sparse_kv = pl.assemble(
                                    sparse_kv,
                                    ori_kv_flat[gather_src_row : gather_src_row + 1, 0 : HEAD_DIM],
                                    [gather_dst_row, 0],
                                )
                            else:
                                gather_cmp_slot = gather_raw - S
                                gather_cmp_block_slot = gather_cmp_slot // BLOCK_SIZE
                                gather_cmp_block_pos = gather_b * CMP_MAX_BLOCKS + gather_cmp_block_slot
                                gather_cmp_blk = pl.cast(pl.read(cmp_block_table_flat, [gather_cmp_block_pos]), pl.INDEX)
                                gather_cmp_intra = gather_cmp_slot - gather_cmp_block_slot * BLOCK_SIZE
                                gather_cmp_src_row = gather_cmp_blk * BLOCK_SIZE + gather_cmp_intra
                                sparse_kv = pl.assemble(
                                    sparse_kv,
                                    cmp_kv_flat[gather_cmp_src_row : gather_cmp_src_row + 1, 0 : HEAD_DIM],
                                    [gather_dst_row, 0],
                                )
                        else:
                            sparse_kv = pl.assemble(sparse_kv, zero_kv_row, [gather_dst_row, 0])
                else:
                    for gather_k in pl.range(PREFILL_SPARSE_PAD):
                        gather_dst_row = gather_t * PREFILL_SPARSE_PAD + gather_k
                        sparse_kv = pl.assemble(sparse_kv, zero_kv_row, [gather_dst_row, 0])

    # Stage 2: causal prefill attention, tiled across context rows.
    for attn_t0 in pl.parallel(0, T, ATTN_TOKEN_TILE):
        for h0 in pl.parallel(0, H, MATMUL_ROW_PAD):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attn_qk_matmul_tile"):
                for qk_dt in pl.range(ATTN_TOKEN_TILE):
                    qk_t = attn_t0 + qk_dt
                    qk_b = qk_t // S
                    qk_s = qk_t - qk_b * S
                    qk_seq_len = pl.read(seqused_kv, [qk_b])
                    if qk_s < qk_seq_len:
                        qk_head_row = qk_t * H + h0
                        qk_q_batch = q_flat[qk_head_row : qk_head_row + MATMUL_ROW_PAD, 0 : HEAD_DIM]
                        qk_kv_base = qk_t * PREFILL_SPARSE_PAD

                        for qk_sb in pl.range(PREFILL_ATTN_BLOCKS):
                            qk_tile_start = qk_sb * PREFILL_ATTN_TILE
                            qk_tile_valid = 0
                            for qk_valid_i in pl.range(PREFILL_ATTN_TILE):
                                qk_valid_raw = pl.read(cmp_sparse_indices, [qk_t, qk_tile_start + qk_valid_i])
                                if qk_valid_raw >= 0:
                                    qk_tile_valid = qk_valid_i + 1
                            if qk_tile_valid > 0:
                                qk_kv_tile = sparse_kv[
                                    qk_kv_base + qk_tile_start : qk_kv_base + qk_tile_start + PREFILL_ATTN_TILE,
                                    0 : HEAD_DIM,
                                ]
                                qk_raw_scores = pl.matmul(qk_q_batch, qk_kv_tile, b_trans=True, out_dtype=pl.FP32)
                                qk_block_row = qk_t * H * PREFILL_ATTN_BLOCKS + qk_sb * H + h0
                                prefill_raw_scores = pl.assemble(prefill_raw_scores, qk_raw_scores, [qk_block_row, 0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attn_softmax_tile"):
                for softmax_dt in pl.range(ATTN_TOKEN_TILE):
                    softmax_t = attn_t0 + softmax_dt
                    softmax_b = softmax_t // S
                    softmax_s = softmax_t - softmax_b * S
                    softmax_seq_len = pl.read(seqused_kv, [softmax_b])
                    if softmax_s < softmax_seq_len:

                        for softmax_sb in pl.range(PREFILL_ATTN_BLOCKS):
                            softmax_tile_start = softmax_sb * PREFILL_ATTN_TILE
                            softmax_tile_valid = 0
                            for softmax_valid_i in pl.range(PREFILL_ATTN_TILE):
                                softmax_valid_raw = pl.read(
                                    cmp_sparse_indices,
                                    [softmax_t, softmax_tile_start + softmax_valid_i],
                                )
                                if softmax_valid_raw >= 0:
                                    softmax_tile_valid = softmax_valid_i + 1
                            if softmax_tile_valid > 0:
                                softmax_block_row = softmax_t * H * PREFILL_ATTN_BLOCKS + softmax_sb * H + h0
                                softmax_raw_scores = prefill_raw_scores[
                                    softmax_block_row : softmax_block_row + MATMUL_ROW_PAD,
                                    0 : PREFILL_ATTN_TILE,
                                ]
                                softmax_scores_valid = pl.slice(
                                    pl.mul(softmax_raw_scores, SOFTMAX_SCALE),
                                    [MATMUL_ROW_PAD, PREFILL_ATTN_TILE],
                                    [0, 0],
                                    valid_shape=[MATMUL_ROW_PAD, softmax_tile_valid],
                                )
                                softmax_scores = pl.fillpad(softmax_scores_valid, pad_value=pl.PadValue.min)
                                softmax_mi = pl.row_max(softmax_scores)
                                softmax_exp_scores = pl.exp(pl.row_expand_sub(softmax_scores, softmax_mi))
                                softmax_exp_scores_bf16 = pl.cast(softmax_exp_scores, target_type=pl.BF16)
                                softmax_li = pl.row_sum(pl.cast(softmax_exp_scores_bf16, target_type=pl.FP32))
                                prefill_blk_mi = pl.assemble(prefill_blk_mi, softmax_mi, [softmax_block_row, 0])
                                prefill_blk_li = pl.assemble(prefill_blk_li, softmax_li, [softmax_block_row, 0])
                                prefill_exp = pl.assemble(prefill_exp, softmax_exp_scores_bf16, [softmax_block_row, 0])

            for pv_h_delta in pl.parallel(0, MATMUL_ROW_PAD, PV_HEAD_TILE):
                pv_h0 = h0 + pv_h_delta

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attn_pv_head_tile"):
                    for pv_dt in pl.range(ATTN_TOKEN_TILE):
                        pv_t = attn_t0 + pv_dt
                        pv_b = pv_t // S
                        pv_s = pv_t - pv_b * S
                        pv_seq_len = pl.read(seqused_kv, [pv_b])
                        if pv_s < pv_seq_len:
                            pv_kv_base = pv_t * PREFILL_SPARSE_PAD

                            for pv_sb in pl.range(PREFILL_ATTN_BLOCKS):
                                pv_tile_start = pv_sb * PREFILL_ATTN_TILE
                                pv_tile_valid_raw = pl.read(cmp_sparse_indices, [pv_t, pv_tile_start])
                                if pv_tile_valid_raw >= 0:
                                    pv_kv_tile = sparse_kv[
                                        pv_kv_base + pv_tile_start : pv_kv_base + pv_tile_start + PREFILL_ATTN_TILE,
                                        0 : HEAD_DIM,
                                    ]
                                    pv_block_row = pv_t * H * PREFILL_ATTN_BLOCKS + pv_sb * H + pv_h0
                                    pv_exp_scores = prefill_exp[
                                        pv_block_row : pv_block_row + PV_HEAD_TILE,
                                        0 : PREFILL_ATTN_TILE,
                                    ]
                                    pv_oi = pl.matmul(pv_exp_scores, pv_kv_tile, out_dtype=pl.FP32)
                                    pv_head_row = pv_t * H + pv_h0
                                    if pv_sb == 0:
                                        prefill_blk_oi0 = pl.assemble(prefill_blk_oi0, pv_oi, [pv_head_row, 0])
                                    if pv_sb == 1:
                                        prefill_blk_oi1 = pl.assemble(prefill_blk_oi1, pv_oi, [pv_head_row, 0])
                                    if pv_sb == 2:
                                        prefill_blk_oi2 = pl.assemble(prefill_blk_oi2, pv_oi, [pv_head_row, 0])

                for merge_norm_t_delta in pl.parallel(0, ATTN_TOKEN_TILE, MERGE_NORM_TOKEN_TILE):
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attn_merge_norm_head_tile"):
                        zero_head_tile = pl.full([PV_HEAD_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                        for merge_norm_dt in pl.range(MERGE_NORM_TOKEN_TILE):
                            merge_norm_t = attn_t0 + merge_norm_t_delta + merge_norm_dt
                            merge_norm_b = merge_norm_t // S
                            merge_norm_s = merge_norm_t - merge_norm_b * S
                            merge_norm_seq_len = pl.read(seqused_kv, [merge_norm_b])
                            merge_norm_head_row = merge_norm_t * H + pv_h0
                            if merge_norm_s < merge_norm_seq_len:
                                merge_norm_block_row0 = merge_norm_t * H * PREFILL_ATTN_BLOCKS + pv_h0
                                merge_norm_mi = prefill_blk_mi[
                                    merge_norm_block_row0 : merge_norm_block_row0 + PV_HEAD_TILE,
                                    0 : 1,
                                ]
                                merge_norm_li = prefill_blk_li[
                                    merge_norm_block_row0 : merge_norm_block_row0 + PV_HEAD_TILE,
                                    0 : 1,
                                ]
                                merge_norm_oi = prefill_blk_oi0[
                                    merge_norm_head_row : merge_norm_head_row + PV_HEAD_TILE,
                                    0 : HEAD_DIM,
                                ]

                                if PREFILL_ATTN_BLOCKS > 1:
                                    merge_norm_tile_start1 = PREFILL_ATTN_TILE
                                    merge_norm_block1_raw = pl.read(cmp_sparse_indices, [merge_norm_t, merge_norm_tile_start1])
                                    if merge_norm_block1_raw >= 0:
                                        merge_norm_block_row1 = merge_norm_t * H * PREFILL_ATTN_BLOCKS + H + pv_h0
                                        merge_norm_cur_mi = prefill_blk_mi[
                                            merge_norm_block_row1 : merge_norm_block_row1 + PV_HEAD_TILE,
                                            0 : 1,
                                        ]
                                        merge_norm_cur_li = prefill_blk_li[
                                            merge_norm_block_row1 : merge_norm_block_row1 + PV_HEAD_TILE,
                                            0 : 1,
                                        ]
                                        merge_norm_cur_oi = prefill_blk_oi1[
                                            merge_norm_head_row : merge_norm_head_row + PV_HEAD_TILE,
                                            0 : HEAD_DIM,
                                        ]
                                        merge_norm_mi_new = pl.maximum(merge_norm_mi, merge_norm_cur_mi)
                                        merge_norm_alpha = pl.exp(pl.sub(merge_norm_mi, merge_norm_mi_new))
                                        merge_norm_beta = pl.exp(pl.sub(merge_norm_cur_mi, merge_norm_mi_new))
                                        merge_norm_li = pl.add(
                                            pl.mul(merge_norm_alpha, merge_norm_li),
                                            pl.mul(merge_norm_beta, merge_norm_cur_li),
                                        )
                                        merge_norm_oi = pl.add(
                                            pl.row_expand_mul(merge_norm_oi, merge_norm_alpha),
                                            pl.row_expand_mul(merge_norm_cur_oi, merge_norm_beta),
                                        )
                                        merge_norm_mi = merge_norm_mi_new

                                if PREFILL_ATTN_BLOCKS > 2:
                                    merge_norm_tile_start2 = 2 * PREFILL_ATTN_TILE
                                    merge_norm_block2_raw = pl.read(cmp_sparse_indices, [merge_norm_t, merge_norm_tile_start2])
                                    if merge_norm_block2_raw >= 0:
                                        merge_norm_block_row2 = merge_norm_t * H * PREFILL_ATTN_BLOCKS + 2 * H + pv_h0
                                        merge_norm_cur_mi2 = prefill_blk_mi[
                                            merge_norm_block_row2 : merge_norm_block_row2 + PV_HEAD_TILE,
                                            0 : 1,
                                        ]
                                        merge_norm_cur_li2 = prefill_blk_li[
                                            merge_norm_block_row2 : merge_norm_block_row2 + PV_HEAD_TILE,
                                            0 : 1,
                                        ]
                                        merge_norm_cur_oi2 = prefill_blk_oi2[
                                            merge_norm_head_row : merge_norm_head_row + PV_HEAD_TILE,
                                            0 : HEAD_DIM,
                                        ]
                                        merge_norm_mi_new2 = pl.maximum(merge_norm_mi, merge_norm_cur_mi2)
                                        merge_norm_alpha2 = pl.exp(pl.sub(merge_norm_mi, merge_norm_mi_new2))
                                        merge_norm_beta2 = pl.exp(pl.sub(merge_norm_cur_mi2, merge_norm_mi_new2))
                                        merge_norm_li = pl.add(
                                            pl.mul(merge_norm_alpha2, merge_norm_li),
                                            pl.mul(merge_norm_beta2, merge_norm_cur_li2),
                                        )
                                        merge_norm_oi = pl.add(
                                            pl.row_expand_mul(merge_norm_oi, merge_norm_alpha2),
                                            pl.row_expand_mul(merge_norm_cur_oi2, merge_norm_beta2),
                                        )
                                        merge_norm_mi = merge_norm_mi_new2

                                merge_norm_sink_bias = pl.reshape(attn_sink[pv_h0 : pv_h0 + PV_HEAD_TILE], [PV_HEAD_TILE, 1])
                                merge_norm_sink_tile = pl.add(pl.sub(merge_norm_mi, merge_norm_mi), merge_norm_sink_bias)
                                merge_norm_denom = pl.add(
                                    merge_norm_li,
                                    pl.exp(pl.sub(merge_norm_sink_tile, merge_norm_mi)),
                                )
                                merge_norm_out = pl.row_expand_div(merge_norm_oi, merge_norm_denom)
                                attn_stage_row = pl.cast(
                                    merge_norm_out[0 : PV_HEAD_TILE, 0 : HEAD_DIM],
                                    target_type=pl.BF16,
                                )
                            else:
                                attn_stage_row = zero_head_tile

                            attn_rope_stage = pl.assemble(
                                attn_rope_stage,
                                attn_stage_row[0 : PV_HEAD_TILE, NOPE_DIM:HEAD_DIM],
                                [merge_norm_head_row, 0],
                            )

                            for merge_norm_head_i in pl.range(PV_HEAD_TILE):
                                merge_norm_global_head = pv_h0 + merge_norm_head_i
                                merge_norm_g = merge_norm_global_head // HEADS_PER_GROUP
                                merge_norm_hh = merge_norm_global_head - merge_norm_g * HEADS_PER_GROUP
                                merge_norm_pack_row = merge_norm_g * T + merge_norm_t
                                merge_norm_head_col = merge_norm_hh * HEAD_DIM
                                o_packed = pl.assemble(
                                    o_packed,
                                    attn_stage_row[merge_norm_head_i : merge_norm_head_i + 1, 0:NOPE_DIM],
                                    [merge_norm_pack_row, merge_norm_head_col],
                                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rope_selector_copy"):
        even_select_stage[0:ROPE_INTERLEAVE_CHUNK, 0:ROPE_CHUNK] = even_select_local[
            0:ROPE_INTERLEAVE_CHUNK,
            0:ROPE_CHUNK,
        ]
        odd_select_stage[0:ROPE_INTERLEAVE_CHUNK, 0:ROPE_CHUNK] = odd_select_local[
            0:ROPE_INTERLEAVE_CHUNK,
            0:ROPE_CHUNK,
        ]

    # Stage 3: inverse RoPE on the rope slice of the attention output.
    for rope_apply_t0 in pl.parallel(0, T, ROPE_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rope_apply_assemble_tile"):
            for rope_apply_dt in pl.range(ROPE_TOKEN_TILE):
                rope_apply_t = rope_apply_t0 + rope_apply_dt
                rope_apply_head_row = rope_apply_t * H

                for rope_asm_r0 in pl.range(0, HALF_ROPE, ROPE_CHUNK):
                    cos_chunk = pl.cast(
                        freqs_cos[rope_apply_t : rope_apply_t + 1, rope_asm_r0 : rope_asm_r0 + ROPE_CHUNK],
                        target_type=pl.FP32,
                    )
                    sin_chunk = pl.cast(
                        freqs_sin[rope_apply_t : rope_apply_t + 1, rope_asm_r0 : rope_asm_r0 + ROPE_CHUNK],
                        target_type=pl.FP32,
                    )
                    rope_tile = attn_rope_stage[
                        rope_apply_head_row : rope_apply_head_row + H,
                        2 * rope_asm_r0 : 2 * rope_asm_r0 + ROPE_INTERLEAVE_CHUNK,
                    ]
                    rope_apply_even_chunk = pl.matmul(rope_tile, even_select_stage, out_dtype=pl.FP32)
                    rope_apply_odd_chunk = pl.matmul(rope_tile, odd_select_stage, out_dtype=pl.FP32)
                    rope_even_acc = pl.add(
                        pl.col_expand_mul(rope_apply_even_chunk, cos_chunk),
                        pl.col_expand_mul(rope_apply_odd_chunk, sin_chunk),
                    )
                    rope_odd_acc = pl.sub(
                        pl.col_expand_mul(rope_apply_odd_chunk, cos_chunk),
                        pl.col_expand_mul(rope_apply_even_chunk, sin_chunk),
                    )
                    rope_rot_even_chunk = pl.cast(rope_even_acc, target_type=pl.BF16, mode="rint")
                    rope_rot_odd_chunk = pl.cast(rope_odd_acc, target_type=pl.BF16, mode="rint")
                    rope_even_interleave = pl.matmul(
                        rope_rot_even_chunk,
                        even_select_local,
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    rope_odd_interleave = pl.matmul(
                        rope_rot_odd_chunk,
                        odd_select_local,
                        b_trans=True,
                        out_dtype=pl.FP32,
                    )
                    rope_even_interleave_buf = pl.assemble(
                        rope_even_interleave_buf,
                        rope_even_interleave,
                        [rope_apply_head_row, 2 * rope_asm_r0],
                    )
                    rope_odd_interleave_buf = pl.assemble(
                        rope_odd_interleave_buf,
                        rope_odd_interleave,
                        [rope_apply_head_row, 2 * rope_asm_r0],
                    )

    for rope_pack_block in pl.spmd(ROPE_PACK_SPMD_BLOCKS, name_hint="prefill_rope_pack_group_spmd"):
        rope_pack_token_block = rope_pack_block // O_GROUPS
        rope_pack_g = rope_pack_block - rope_pack_token_block * O_GROUPS
        rope_combine_t0 = rope_pack_token_block * ROPE_PACK_TOKEN_TILE

        for rope_combine_dt in pl.range(ROPE_PACK_TOKEN_TILE):
            rope_combine_t = rope_combine_t0 + rope_combine_dt
            if rope_combine_t < T:
                rope_pack_head_row = rope_combine_t * H + rope_pack_g * HEADS_PER_GROUP
                rope_even_tile = rope_even_interleave_buf[
                    rope_pack_head_row : rope_pack_head_row + HEADS_PER_GROUP,
                    0 : ROPE_DIM,
                ]
                rope_odd_tile = rope_odd_interleave_buf[
                    rope_pack_head_row : rope_pack_head_row + HEADS_PER_GROUP,
                    0 : ROPE_DIM,
                ]
                rope_full = pl.cast(
                    pl.add(rope_even_tile, rope_odd_tile),
                    target_type=pl.BF16,
                )
                rope_pack_row = rope_pack_g * T + rope_combine_t
                for rope_pack_hh in pl.range(HEADS_PER_GROUP):
                    rope_pack_head_col = rope_pack_hh * HEAD_DIM + NOPE_DIM
                    o_packed = pl.assemble(
                        o_packed,
                        rope_full[rope_pack_hh : rope_pack_hh + 1, 0:ROPE_DIM],
                        [rope_pack_row, rope_pack_head_col],
                    )

    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.BF16)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)
    o_r_amax_parts = pl.create_tensor([A_AMAX_BLOCKS, T], dtype=pl.FP32)
    o_r_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)

    # Stage 5: grouped BF16 projection `o_packed @ wo_a^T`.
    for g in pl.parallel(0, O_GROUPS, 1):
        row_base_o = g * T
        out_col_g = g * O_LORA

        for nb in pl.parallel(0, A_N_BLOCKS, 1):
            n0 = nb * A_N_CHUNK

            for proj_t0 in pl.parallel(0, T, PROJ_TOKEN_TILE):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_stage_a_accum_tile"):
                    xa0_chunk = o_packed[row_base_o + proj_t0:row_base_o + proj_t0 + PROJ_TOKEN_TILE, 0:A_K_CHUNK]
                    wa0_chunk = wo_a[g:g + 1, n0:n0 + A_N_CHUNK, 0:A_K_CHUNK]
                    acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                    for kb in pl.pipeline(1, A_K_BLOCKS, stage=2):
                        k0 = kb * A_K_CHUNK
                        xa_k_chunk = o_packed[
                            row_base_o + proj_t0:row_base_o + proj_t0 + PROJ_TOKEN_TILE,
                            k0:k0 + A_K_CHUNK,
                        ]
                        wa_k_chunk = wo_a[g:g + 1, n0:n0 + A_N_CHUNK, k0:k0 + A_K_CHUNK]
                        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_stage_a_store_amax_tile"):
                    acc_a_2d = pl.reshape(acc_a, [PROJ_TOKEN_TILE, A_N_CHUNK])
                    acc_a_bf16 = pl.cast(acc_a_2d, target_type=pl.BF16)
                    o_r[proj_t0:proj_t0 + PROJ_TOKEN_TILE, out_col_g + n0:out_col_g + n0 + A_N_CHUNK] = acc_a_bf16
                    acc_a_f32 = pl.cast(acc_a_bf16, target_type=pl.FP32)
                    acc_a_abs = pl.maximum(acc_a_f32, pl.neg(acc_a_f32))
                    acc_a_amax = pl.reshape(pl.row_max(acc_a_abs), [1, PROJ_TOKEN_TILE])
                    amax_part_row = g * A_N_BLOCKS + nb
                    o_r_amax_parts[
                        amax_part_row:amax_part_row + 1,
                        proj_t0:proj_t0 + PROJ_TOKEN_TILE,
                    ] = acc_a_amax

    # Stage 6: per-row symmetric INT8 activation quantization.
    for quant_t0 in pl.parallel(0, T, QUANT_TOKEN_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_stage_b_quant_tile"):
            or_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for ab in pl.range(0, A_AMAX_BLOCKS, 1):
                or_a_part = o_r_amax_parts[ab:ab + 1, quant_t0:quant_t0 + QUANT_TOKEN_TILE]
                or_amax = pl.maximum(or_amax, or_a_part)
            or_sq_row = pl.div(pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
            or_scale_dq = pl.reshape(pl.recip(or_sq_row), [QUANT_TOKEN_TILE, 1])
            o_r_scale_dq[quant_t0:quant_t0 + QUANT_TOKEN_TILE, 0:1] = or_scale_dq
            or_sq_col = pl.reshape(or_sq_row, [QUANT_TOKEN_TILE, 1])
            for k1 in pl.range(0, O_GROUPS * O_LORA, QUANT_CHUNK):
                or_q_f32 = pl.cast(o_r[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_CHUNK], target_type=pl.FP32)
                or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
                or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="rint")
                or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
                o_r_i8[quant_t0:quant_t0 + QUANT_TOKEN_TILE, k1:k1 + QUANT_CHUNK] = pl.cast(
                    or_q_half,
                    target_type=pl.INT8,
                    mode="trunc",
                )

    # Stage 7: INT8 output projection and dequantization.
    for nb in pl.parallel(0, B_N_BLOCKS, 1):
        n0 = nb * B_N_CHUNK

        for proj_t0 in pl.parallel(0, T, PROJ_TOKEN_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_stage_b_accum_tile"):
                xb0_chunk = o_r_i8[proj_t0:proj_t0 + PROJ_TOKEN_TILE, 0:B_K_CHUNK]
                wb0_chunk = wo_b[n0:n0 + B_N_CHUNK, 0:B_K_CHUNK]
                acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.INT32)
                for kb in pl.pipeline(1, B_K_BLOCKS, stage=2):
                    k0 = kb * B_K_CHUNK
                    xb_k_chunk = o_r_i8[proj_t0:proj_t0 + PROJ_TOKEN_TILE, k0:k0 + B_K_CHUNK]
                    wb_k_chunk = wo_b[n0:n0 + B_N_CHUNK, k0:k0 + B_K_CHUNK]
                    acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_stage_b_store_tile"):
                wb_scale_chunk = pl.reshape(wo_b_scale[n0:n0 + B_N_CHUNK], [1, B_N_CHUNK])
                attn_chunk = pl.cast(acc_b, target_type=pl.FP32, mode="none")
                attn_scale_tile = o_r_scale_dq[proj_t0:proj_t0 + PROJ_TOKEN_TILE, 0:1]
                attn_chunk = pl.col_expand_mul(pl.row_expand_mul(attn_chunk, attn_scale_tile), wb_scale_chunk)
                attn_out[proj_t0:proj_t0 + PROJ_TOKEN_TILE, n0:n0 + B_N_CHUNK] = pl.cast(
                    attn_chunk,
                    target_type=pl.BF16,
                    mode="rint",
                )

    return attn_out


@pl.jit
def prefill_sparse_attn_test(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    even_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    return prefill_sparse_attn(
        q,
        ori_kv,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        attn_sink,
        seqused_kv,
        freqs_cos,
        freqs_sin,
        even_select_local,
        odd_select_local,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the W8A8C16 activation path."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis."""
    import torch

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def get_prefill_cmp_valid(compress_ratio: int) -> int:
    """Map standalone prefill ratio modes to visible compressed-cache length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return min(IDX_TOPK, S // compress_ratio)
    if compress_ratio == 128:
        return min(IDX_TOPK, S // compress_ratio)
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")


def golden_prefill_sparse_attn(tensors):
    """Torch reference for the first standalone prefill sparse-attn contract."""
    import torch

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    ori_block_table = tensors["ori_block_table"]
    cmp_kv = tensors.get("cmp_kv")
    cmp_block_table = tensors.get("cmp_block_table")
    cmp_sparse_indices = tensors.get("cmp_sparse_indices")
    attn_sink = tensors["attn_sink"].float()
    seqused_kv = tensors.get("seqused_kv")
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)
    for t in range(T):
        b = t // S
        s = t - b * S
        seq_len = int(seqused_kv[b].item()) if seqused_kv is not None else S
        if s >= seq_len:
            continue

        gathered = []
        if cmp_sparse_indices is None:
            for raw in range(min(s + 1, seq_len)):
                blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv[blk_id, intra, 0])
        else:
            for raw_i in cmp_sparse_indices[t, :PREFILL_SPARSE_TOPK].tolist():
                raw = int(raw_i)
                if raw < 0:
                    continue
                if raw < S:
                    if raw >= seq_len:
                        continue
                    blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                    intra = raw % BLOCK_SIZE
                    gathered.append(ori_kv[blk_id, intra, 0])
                else:
                    cmp_slot = raw - S
                    if cmp_kv is None or cmp_block_table is None:
                        continue
                    if cmp_slot >= CMP_BLOCK_NUM * BLOCK_SIZE:
                        continue
                    blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
                    intra = cmp_slot % BLOCK_SIZE
                    gathered.append(cmp_kv.float()[blk_id, intra, 0])

        if not gathered:
            continue
        kv_b = torch.stack(gathered, dim=0)

        mi = None
        li = None
        oi = None
        for tile_start in range(0, kv_b.shape[0], PREFILL_ATTN_TILE):
            kv_tile = kv_b[tile_start : tile_start + PREFILL_ATTN_TILE]
            scores = (q[t] @ kv_tile.T) * SOFTMAX_SCALE
            cur_mi = scores.max(dim=-1, keepdim=True).values
            exp_scores_bf16 = torch.exp(scores - cur_mi).to(torch.bfloat16)
            cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)
            cur_oi = exp_scores_bf16.float() @ kv_tile.to(torch.bfloat16).float()
            if mi is None:
                mi = cur_mi
                li = cur_li
                oi = cur_oi
            else:
                mi_new = torch.maximum(mi, cur_mi)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(cur_mi - mi_new)
                li = alpha * li + beta * cur_li
                oi = oi * alpha + cur_oi * beta
                mi = mi_new

        if mi is not None:
            denom = li + torch.exp(attn_sink.unsqueeze(-1) - mi)
            o[t] = oi / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = cos[:, :HALF_ROPE].unsqueeze(1)
    sin_half = sin[:, :HALF_ROPE].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(B, S, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)

    tensors["attn_out"][:] = out.to(torch.bfloat16)


def build_tensor_specs(compress_ratio: int = DEFAULT_COMPRESS_RATIO):
    """Build deterministic tensors for the prefill standalone harness."""
    import torch
    from golden import TensorSpec

    cmp_valid = get_prefill_cmp_valid(compress_ratio)

    def seeded_uniform(shape, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.rand(*shape, generator=generator) - 0.5

    def init_q():
        return seeded_uniform((T, H, HEAD_DIM), 1) / 8.0

    def init_ori_kv():
        return seeded_uniform((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 2) / 8.0

    def init_ori_block_table():
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_cmp_kv():
        return seeded_uniform((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 3) / 8.0

    def init_cmp_block_table():
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl

    def init_cmp_sparse_indices():
        idx = torch.full((T, TOPK), -1, dtype=torch.int32)
        for t in range(T):
            b = t // S
            s = t - b * S
            del b
            win_start = max(0, s - M.sliding_window + 1)
            window = torch.arange(win_start, s + 1, dtype=torch.int32)
            idx[t, : window.numel()] = window
            if compress_ratio:
                comp_count = min(cmp_valid, (s + 1) // compress_ratio)
                if comp_count > 0:
                    comp = torch.arange(comp_count, dtype=torch.int32) + S
                    idx[t, window.numel() : window.numel() + comp_count] = comp
        return idx

    def init_attn_sink():
        return torch.zeros(H)

    def init_seqused_kv():
        return torch.full((B,), S, dtype=torch.int32)

    def init_cos():
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)

    def init_sin():
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    def init_even_select_local():
        matrix = torch.zeros((ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK))
        for i in range(ROPE_CHUNK):
            matrix[2 * i, i] = 1
        return matrix

    def init_odd_select_local():
        matrix = torch.zeros((ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK))
        for i in range(ROPE_CHUNK):
            matrix[2 * i + 1, i] = 1
        return matrix

    def init_wo_a():
        return seeded_uniform((O_GROUPS, O_LORA, O_GROUP_IN), 4) / (O_GROUP_IN ** 0.5)

    wo_b_bf16 = (seeded_uniform((D, O_GROUPS * O_LORA), 5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)

    def init_wo_b():
        return wo_b_i8

    def init_wo_b_scale():
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], torch.int32, init_value=init_seqused_kv),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("even_select_local", [ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [ROPE_INTERLEAVE_CHUNK, ROPE_CHUNK], torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument(
        "--enable-l2-swimlane",
        nargs="?",
        const=4,
        default=0,
        type=int,
        metavar="PERF_LEVEL",
    )
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_sparse_attn_test,
        specs=build_tensor_specs(args.compress_ratio),
        golden_fn=golden_prefill_sparse_attn,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_pmu=args.enable_pmu,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "attn_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
