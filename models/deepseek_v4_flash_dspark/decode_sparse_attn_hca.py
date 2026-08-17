# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 HCA sparse attention with grouped output projection (decode).

Ratio-128 ragged compressed pages plus three-state raw sources; no indexer.
The SWA and CSA variants live in sibling modules."""


import pypto.language as pl

from config import (
    BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_LOCAL_REQUESTS,
    DECODE_SEQ,
    FLASH as M,
    HCA_KV_POOL_BLOCKS,
    HCA_ROWS_PER_SHARD,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    SWA_SOURCE_INVALID,
    SWA_SOURCE_OVERLAY_BASE,
    TP,
)


# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis (block tables)
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
HCA_WORK_DYN = pl.dynamic("HCA_WORK_DYN")
HCA_PAGES_DYN = pl.dynamic("HCA_PAGES_DYN")
HCA_REQUEST_OFFSETS_DYN = pl.dynamic("HCA_REQUEST_OFFSETS_DYN")
HCA_QUERY_OFFSETS_DYN = pl.dynamic("HCA_QUERY_OFFSETS_DYN")

# model config
B = DECODE_BATCH // TP
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
SOFTMAX_SCALE = M.softmax_scale
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

COMPRESS_RATIO = 128
NEG_INF = -1.0e20

# tiling
H_TILE = 16
QK_M_TILE = 32           # qk_pv M rows per QK/PV matmul; QK_M_TILE/H_TILE-way KV L1->L0 reuse
ATTN_K_TILE = HCA_ROWS_PER_SHARD
ROPE_TILE = 16
ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE
A_K_TILE = 256           # proj_a cube K frag
PROJ_A_MM_N_TILE = 128   # proj_a cube N frag
T_PAD = ((T + 16 - 1) // 16) * 16  # T padded up to the 16-row cube M floor
MM_T_TILE = T_PAD  # one cube M tile spans every token row of the T_PAD-strided scratch
ROPE_CS_T_TILE = 8    # rope cos/sin row block; T is a multiple of 8 by the batch contract
PROJ_A_ROW_TILE = 16  # proj_a cube M; row-blocked so unwritten pad rows never enter the matmul
PA_N_FRAGS = O_LORA // PROJ_A_MM_N_TILE
B_K_TILE = 256           # proj_b_mm cube K frag
PROJ_B_MM_N_TILE = 256   # proj_b_mm cube N frag; writes grouped INT32 partials
PROJ_B_ACT_N_TILE = 512  # proj_b_act vector N frag; keeps the O_GROUPS-way accumulate inside UB
PROJ_B_ACT_N_REGS = D // PROJ_B_ACT_N_TILE
QUANT_TOKEN_TILE = 8     # fused per-group amax+quant row tile
PROJ_B_D_TILE = 512      # proj_b_mm D chunk per task; its N frags loop inside the task
PROJ_B_ACT_T_TILE = 8    # proj_b_act inner token tile for the O_GROUPS-way INT32->FP32 accumulate
PROJ_B_ACT_TASK_T_TILE = 8   # proj_b_act token block per task

assert WIN == ATTN_K_TILE == 128


@pl.jit.inline(auto_scope=False)
def sparse_attn_hca_heads(
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
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    cmp_dep: pl.Scalar[pl.TASK_ID],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Write HCA heads as ``[group, T_PAD, O_GROUP_IN]`` slabs.

    Three-state raw sources (persistent/invalid/overlay) plus exact packed
    compressed-shard work are gathered in a single full-active pass and
    merged per query.  The returned task ID covers every write to the
    packed output tensor.
    """
    total_t = pl.tensor.dim(q, 0)
    request_count = pl.tensor.dim(request_epochs, 0)
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    page_count = pl.tensor.dim(hca_pages, 0)
    ori_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_flat = pl.reshape(cmp_kv, [cmp_block_num * BLOCK_SIZE, HEAD_DIM])
    q_flat = pl.reshape(q, [total_t * H, HEAD_DIM])

    t_dim = total_t
    work_begin_i32 = pl.read(hca_query_work_offsets, [0])
    work_end_i32 = pl.read(hca_query_work_offsets, [t_dim])
    work_base = pl.cast(work_begin_i32, pl.INDEX)
    work_count = pl.cast(work_end_i32 - work_begin_i32, pl.INDEX)
    item_count = t_dim + work_count
    t_hblocks = t_dim * (H // H_TILE)
    rope_cs_blocks = t_dim // ROPE_CS_T_TILE

    partial_mi = pl.create_tensor([item_count * H, 1], dtype=pl.FP32)
    partial_li = pl.create_tensor([item_count * H, 1], dtype=pl.FP32)
    partial_oi = pl.create_tensor([item_count * H, HEAD_DIM], dtype=pl.FP32)
    packed_kv = pl.create_tensor(
        [item_count * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16
    )
    packed_valid = pl.create_tensor([item_count, ATTN_K_TILE], dtype=pl.FP32)
    # Three-state gather: persistent raw KV, invalid (zero), or current-step
    # overlay.  Compressed-shard lanes resolve ragged pages through
    # range/epoch checks before physical addressing.
    with pl.spmd(
        item_count, name_hint="hca_gather_kv", deps=[cmp_dep]
    ) as gather_tid:
        item = pl.tile.get_block_idx()
        query = pl.cast(item, pl.INDEX)
        work = work_base + item - t_dim
        is_raw = item < t_dim
        if not is_raw:
            query = pl.cast(pl.read(hca_work_query_ids, [work]), pl.INDEX)
        request = pl.cast(-1, pl.INDEX)
        if query >= 0:
            if query < total_t:
                request = pl.cast(pl.read(query_request_ids, [query]), pl.INDEX)
        row_begin = pl.cast(0, pl.INT32)
        valid_rows = pl.cast(ATTN_K_TILE, pl.INT32)
        if not is_raw:
            row_begin = pl.read(hca_work_row_begin, [work])
            valid_rows = pl.read(hca_work_valid_rows, [work])
        page_begin = pl.cast(0, pl.INT32)
        page_total = pl.cast(0, pl.INT32)
        valid_begin = pl.cast(0, pl.INT32)
        valid_end = pl.cast(0, pl.INT32)
        head = pl.cast(0, pl.INT32)
        request_epoch = pl.cast(-1, pl.INT32)
        if request >= 0:
            if request < request_count:
                candidate_page_begin = pl.read(hca_page_offsets, [request])
                candidate_page_end = pl.read(hca_page_offsets, [request + 1])
                if candidate_page_begin >= 0:
                    if candidate_page_end >= candidate_page_begin:
                        if candidate_page_end <= page_count:
                            page_begin = candidate_page_begin
                            page_total = candidate_page_end - candidate_page_begin
                            valid_begin = pl.read(hca_windows, [request, 0])
                            valid_end = pl.read(hca_windows, [request, 1])
                            head = pl.read(hca_windows, [request, 2])
                            request_epoch = pl.read(request_epochs, [request])
        for lane in pl.range(ATTN_K_TILE):
            dst = item * ATTN_K_TILE + lane
            source_valid = pl.cast(0, pl.INT32)
            source_row = pl.cast(0, pl.INDEX)
            source_overlay = pl.cast(-1, pl.INDEX)
            if is_raw:
                source = pl.read(swa_sources, [query, lane])
                if source >= 0:
                    if source < ori_block_num * BLOCK_SIZE:
                        source_valid = pl.cast(1, pl.INT32)
                        source_row = pl.cast(source, pl.INDEX)
                else:
                    if source <= SWA_SOURCE_OVERLAY_BASE:
                        source_overlay = pl.cast(
                            SWA_SOURCE_OVERLAY_BASE - source, pl.INDEX
                        )
                        if source_overlay >= 0:
                            if source_overlay < total_t:
                                overlay_request = pl.read(
                                    query_request_ids, [source_overlay]
                                )
                                if overlay_request == request:
                                    if source_overlay <= query:
                                        source_valid = pl.cast(2, pl.INT32)
            else:
                if lane < valid_rows:
                    logical_row = row_begin + lane
                    if logical_row >= valid_begin:
                        if logical_row < valid_end:
                            logical_page_base = valid_begin // BLOCK_SIZE
                            relative_page = (
                                logical_row // BLOCK_SIZE - logical_page_base
                            )
                            if page_total > 0:
                                if relative_page >= 0:
                                    if relative_page < page_total:
                                        if head >= 0:
                                            if head < page_total:
                                                page_index = (
                                                    head + relative_page
                                                ) % page_total
                                                page_entry = page_begin + page_index
                                                if page_entry >= 0:
                                                    if page_entry < page_count:
                                                        physical_page = pl.read(
                                                            hca_pages, [page_entry, 0]
                                                        )
                                                        page_epoch = pl.read(
                                                            hca_pages, [page_entry, 1]
                                                        )
                                                        if physical_page >= 0:
                                                            if physical_page < cmp_block_num:
                                                                if page_epoch == request_epoch:
                                                                    source_valid = pl.cast(
                                                                        1, pl.INT32
                                                                    )
                                                                    source_row = pl.cast(
                                                                        physical_page * BLOCK_SIZE
                                                                        + logical_row % BLOCK_SIZE,
                                                                        pl.INDEX,
                                                                    )
            if source_valid == 1:
                if is_raw:
                    packed_kv[dst : dst + 1, 0:HEAD_DIM] = ori_flat[
                        source_row : source_row + 1, 0:HEAD_DIM
                    ]
                else:
                    packed_kv[dst : dst + 1, 0:HEAD_DIM] = cmp_flat[
                        source_row : source_row + 1, 0:HEAD_DIM
                    ]
                pl.write(packed_valid, [item, lane], 1.0)
            elif source_valid == 2:
                packed_kv[dst : dst + 1, 0:HEAD_DIM] = current_kv[
                    source_overlay : source_overlay + 1, 0:HEAD_DIM
                ]
                pl.write(packed_valid, [item, lane], 1.0)
            else:
                packed_kv[dst : dst + 1, 0:HEAD_DIM] = pl.full(
                    [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                )
                pl.write(packed_valid, [item, lane], 0.0)

    # qk_pv writes per-tile (mi, li, oi) to GM; merge_norm reads them back.
    with pl.spmd(
        item_count,
        name_hint="qk_pv",
        deps=[gather_tid],
        allow_early_resolve=True,
    ) as _qk_tid:
        item = pl.tile.get_block_idx()
        query = pl.cast(item, pl.INDEX)
        if item >= t_dim:
            work = work_base + item - t_dim
            query = pl.cast(pl.read(hca_work_query_ids, [work]), pl.INDEX)
        kv_tile = packed_kv[
            item * ATTN_K_TILE : (item + 1) * ATTN_K_TILE, 0:HEAD_DIM
        ]
        valid_row = packed_valid[item : item + 1, 0:ATTN_K_TILE]
        bias = pl.mul(pl.sub(valid_row, 1.0), -NEG_INF)
        for head_block in pl.pipeline(H // QK_M_TILE, stage=2):
            head0 = head_block * QK_M_TILE
            q_tile = q_flat[
                query * H + head0 : query * H + head0 + QK_M_TILE,
                0:HEAD_DIM,
            ]
            scores = pl.mul(
                pl.matmul(q_tile, kv_tile, b_trans=True, out_dtype=pl.FP32),
                SOFTMAX_SCALE,
            )
            scores = pl.add(
                scores,
                pl.col_expand(
                    pl.full(
                        [QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0
                    ),
                    bias,
                ),
            )
            mi = pl.row_max(scores)
            exp_scores = pl.mul(
                pl.exp(pl.row_expand_sub(scores, mi)),
                pl.col_expand(
                    pl.full(
                        [QK_M_TILE, ATTN_K_TILE], dtype=pl.FP32, value=0.0
                    ),
                    valid_row,
                ),
            )
            li = pl.row_sum(exp_scores)
            oi = pl.matmul(
                pl.cast(exp_scores, target_type=pl.BF16, mode="rint"),
                kv_tile,
                out_dtype=pl.FP32,
            )
            partial_mi[
                item * H + head0 : item * H + head0 + QK_M_TILE, 0:1
            ] = mi
            partial_li[
                item * H + head0 : item * H + head0 + QK_M_TILE, 0:1
            ] = li
            partial_oi[
                item * H + head0 : item * H + head0 + QK_M_TILE, 0:HEAD_DIM
            ] = oi

    # Precompute the head-invariant interleaved cos and sign*sin once: they
    # depend only on (token, column), not head.  Hoisted above merge_norm.
    rope_cos_il = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T_PAD, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([H_TILE, ROPE_DIM], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_swap"):
        sw_col = pl.col_expand_mul(
            pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(
                pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        sw_dup_f = pl.cast(
            pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(
            pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)),
            target_type=pl.INT32,
        )

    for cp in pl.spmd(HALF_ROPE // ROPE_TILE, name_hint="rope_cs"):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(
            pl.full(
                [ROPE_CS_T_TILE, ROPE_INTERLEAVE_TILE],
                dtype=pl.FP32,
                value=1.0,
            ),
            pl.cast(
                pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32),
                target_type=pl.FP32,
            ),
        )
        cs_dup_f = pl.cast(
            pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        for cs_rb in pl.range(rope_cs_blocks):
            cs_t0 = cs_rb * ROPE_CS_T_TILE
            cs_cos = pl.cast(
                freqs_cos[
                    cs_t0 : cs_t0 + ROPE_CS_T_TILE,
                    cp_r0 : cp_r0 + ROPE_TILE,
                ],
                target_type=pl.FP32,
            )
            cs_sin = pl.cast(
                freqs_sin[
                    cs_t0 : cs_t0 + ROPE_CS_T_TILE,
                    cp_r0 : cp_r0 + ROPE_TILE,
                ],
                target_type=pl.FP32,
            )
            rope_cos_il[
                cs_t0 : cs_t0 + ROPE_CS_T_TILE,
                cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE,
            ] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
            rope_sin_signed[
                cs_t0 : cs_t0 + ROPE_CS_T_TILE,
                cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE,
            ] = pl.mul(pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)

    # Online-softmax merge across raw + compressed-shard partials, sink-norm,
    # then fused inverse RoPE.  One spmd block per (token, head-tile) so the
    # merge fans out over that many AIVs.
    with pl.spmd(t_hblocks, name_hint="merge_norm") as merge_tid:
        merge_item = pl.tile.get_block_idx()
        local_query = merge_item // (H // H_TILE)
        query = pl.cast(local_query, pl.INDEX)
        head_index = merge_item - local_query * (H // H_TILE)
        head0 = head_index * H_TILE
        raw_row = local_query * H + head0
        merge_mi = partial_mi[raw_row : raw_row + H_TILE, 0:1]
        merge_li = partial_li[raw_row : raw_row + H_TILE, 0:1]
        merge_oi = partial_oi[raw_row : raw_row + H_TILE, 0:HEAD_DIM]
        work_begin = pl.cast(
            pl.read(hca_query_work_offsets, [query]), pl.INDEX
        )
        merge_work_end = pl.cast(
            pl.read(hca_query_work_offsets, [query + 1]), pl.INDEX
        )
        for work in pl.pipeline(work_begin, merge_work_end, stage=2):
            row = (t_dim + work - work_base) * H + head0
            cur_mi = partial_mi[row : row + H_TILE, 0:1]
            cur_li = partial_li[row : row + H_TILE, 0:1]
            cur_oi = partial_oi[row : row + H_TILE, 0:HEAD_DIM]
            merge_mi_new = pl.maximum(merge_mi, cur_mi)
            alpha = pl.exp(pl.sub(merge_mi, merge_mi_new))
            beta = pl.exp(pl.sub(cur_mi, merge_mi_new))
            merge_li = pl.add(
                pl.mul(alpha, merge_li), pl.mul(beta, cur_li)
            )
            merge_oi = pl.add(
                pl.row_expand_mul(merge_oi, alpha),
                pl.row_expand_mul(cur_oi, beta),
            )
            merge_mi = merge_mi_new
        sink = pl.reshape(attn_sink[head0 : head0 + H_TILE], [H_TILE, 1])
        denom = pl.add(merge_li, pl.exp(pl.sub(sink, merge_mi)))
        full = pl.row_expand_div(merge_oi, denom)
        full_bf16 = pl.cast(full, target_type=pl.BF16, mode="rint")
        rope = full[0:H_TILE, NOPE_DIM:HEAD_DIM]
        swapped = pl.gather(
            rope, dim=-1, index=rope_swap_idx[0:H_TILE, 0:ROPE_DIM]
        )
        rotated = pl.add(
            pl.col_expand_mul(
                rope,
                rope_cos_il[local_query : local_query + 1, 0:ROPE_DIM],
            ),
            pl.col_expand_mul(
                swapped,
                rope_sin_signed[
                    local_query : local_query + 1, 0:ROPE_DIM
                ],
            ),
        )
        rope_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")
        merged_bf16 = pl.concat(full_bf16[:, :NOPE_DIM], rope_bf16)
        for lane in pl.unroll(H_TILE):
            packed_row = (
                ((head0 + lane) // HEADS_PER_GROUP) * T_PAD + query
            )
            packed_col = ((head0 + lane) % HEADS_PER_GROUP) * HEAD_DIM
            o_packed_heads[
                packed_row : packed_row + 1,
                packed_col : packed_col + HEAD_DIM,
            ] = merged_bf16[lane : lane + 1, :]

    return o_packed_heads, merge_tid


@pl.jit.inline
def sparse_attn_hca_local_o_proj(
    o_packed_heads: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Project local-token, full-group HCA heads into BF16 hidden rows."""
    t_dim = pl.tensor.dim(attn_out, 0)
    act_t_blks = t_dim // PROJ_B_ACT_TASK_T_TILE
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE

    o_packed = pl.reshape(o_packed_heads, [O_GROUPS * T_PAD, O_GROUP_IN])
    # Back-to-back grouped output projection: proj_a[g] -> quant[g] -> proj_b[g]
    # pipelines per group, because the PER-GROUP amax keeps the quant reduction
    # inside one O_LORA group instead of barriering the whole row. manual_scope
    # suppresses auto-dep, so every edge is explicit: proj_a waits on merge_norm,
    # quant[g] on proj_a[g], proj_b[g] on quant[g]. proj_b_act combines the group
    # partials and is the consolidated attn_out writer.
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    # [G, T] so each group's per-row scale is a contiguous row.
    act_scale_dq = pl.create_tensor([O_GROUPS, T_PAD], dtype=pl.FP32)
    # Per-group INT32 partials: proj_b_mm writes group g's contribution to output
    # channel n at partials[:, g*D + n]. No atomic-add -> no zero-seed.
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        # One proj_a SPMD grid per output group.
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T_PAD
            out_col_g = g * O_LORA
            with pl.spmd(proj_a_rows * PA_N_FRAGS, name_hint="proj_a_mm", deps=[heads_dep],
                         allow_early_resolve=True) as pa_tid:
                pa_unit = pl.tile.get_block_idx()
                pa_rb = pa_unit // PA_N_FRAGS  # row block outermost
                nf = pa_unit - pa_rb * PA_N_FRAGS
                pa_r0 = pa_rb * PROJ_A_ROW_TILE
                pa_rows = pl.min(PROJ_A_ROW_TILE, t_dim - pa_r0)
                pa_src0 = row_base_o + pa_r0
                n0 = nf * PROJ_A_MM_N_TILE
                xa0_chunk = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE])
                wa0_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                    k0 = kb * A_K_TILE
                    xa_k_chunk = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE])
                    wa_k_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                # acc_a is 3D (wo_a keeps its group axis), which subscript-write cannot express.
                o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])

            # Per-group proj_a -> quant -> proj_b dependency chain.
            col_g = g * O_LORA
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="quant", deps=[pa_tid], allow_early_resolve=True) as q_tid:
                for qt in pl.pipeline(0, t_dim, QUANT_TOKEN_TILE, stage=2):
                    oc_amax = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    g_abs = pl.abs(oc_amax)
                    g_row_max = pl.row_max(g_abs)
                    g_row_max = pl.reshape(g_row_max, [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    act_scale_dq[g : g + 1, qt : qt + QUANT_TOKEN_TILE] = pl.recip(g_sq_row)
                    g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = oq_i8
                # Zero the rows past the runtime token count; proj_b_mm reads the full T_PAD extent.
                for zt in pl.range(t_dim, T_PAD, QUANT_TOKEN_TILE):
                    zero_half = pl.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                    o_r_i8_pad[zt : zt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = pl.cast(
                        zero_half, target_type=pl.INT8, mode="trunc")

            # One proj_b SPMD grid per output group.
            with pl.spmd(D // PROJ_B_D_TILE, name_hint="proj_b_mm", deps=[q_tid], allow_early_resolve=True) as pb_tid:
                dc = pl.tile.get_block_idx()
                d0 = dc * PROJ_B_D_TILE
                for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    acc_b = pl.matmul(
                        o_r_i8_pad[:, col_g : col_g + B_K_TILE],
                        wo_b[n0 : n0 + PROJ_B_MM_N_TILE, col_g : col_g + B_K_TILE],
                        b_trans=True,
                        out_dtype=pl.INT32,
                    )
                    for kb in pl.pipeline(1, O_LORA // B_K_TILE, stage=2):
                        k0 = col_g + kb * B_K_TILE
                        acc_b = pl.matmul_acc(
                            acc_b,
                            o_r_i8_pad[:, k0 : k0 + B_K_TILE],
                            wo_b[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE],
                            b_trans=True,
                        )
                    partials[0:MM_T_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
            proj_b_tids[g] = pb_tid

    # proj_b_act sums the O_GROUPS INT32 partials -- each dequantized by its group's
    # per-row act scale -- then applies the per-channel weight scale -> BF16. Explicit
    # deps on the eight proj_b grids bridge manual_scope -> the return's auto-dep.
    with pl.spmd(act_t_blks * PROJ_B_ACT_N_REGS, name_hint="proj_b_act",
                 deps=[proj_b_tids[i] for i in range(O_GROUPS)], allow_early_resolve=True) as act_tid:
        act_idx = pl.tile.get_block_idx()
        tblk = act_idx // PROJ_B_ACT_N_REGS  # token block outermost
        nreg = act_idx - tblk * PROJ_B_ACT_N_REGS
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, t0 + PROJ_B_ACT_TASK_T_TILE, PROJ_B_ACT_T_TILE):
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb : b_tb + PROJ_B_ACT_T_TILE, p_col0 : p_col0 + PROJ_B_ACT_N_TILE]
                g_scale_row = act_scale_dq[act_g : act_g + 1, b_tb : b_tb + PROJ_B_ACT_T_TILE]
                g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])
                p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
                p_g_scaled = pl.row_expand_mul(p_g_f32, g_scale)
                acc = pl.add(acc, p_g_scaled)
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
            attn_out[b_tb : b_tb + PROJ_B_ACT_T_TILE, ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE] = out_bf16

    return act_tid


@pl.jit.inline
def sparse_attn_hca(
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
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    cmp_dep: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Compute HCA sparse attention and the grouped output projection.

    Returns the output-projection completion, which transitively depends on
    all raw and compressed reads and serves as the delayed raw-commit fence.
    """
    o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)
    o_packed_heads, heads_dep = sparse_attn_hca_heads(
        q, ori_kv, current_kv, swa_sources,
        cmp_kv, query_request_ids,
        hca_pages, hca_page_offsets, hca_windows, request_epochs,
        hca_query_work_offsets, hca_work_query_ids, hca_work_row_begin, hca_work_valid_rows,
        attn_sink, freqs_cos, freqs_sin,
        o_packed_heads, cmp_dep,
    )
    projection_done = sparse_attn_hca_local_o_proj(
        o_packed_heads,
        wo_a, wo_b, wo_b_scale,
        attn_out, heads_dep,
    )
    return projection_done


@pl.jit
def sparse_attn_test(
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
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
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
    attn_out.bind_dynamic(0, T_DYN)

    dep = pl.system.task_dummy(deps=[])
    sparse_attn_hca(
        q,
        ori_kv,
        current_kv,
        swa_sources,
        cmp_kv,
        query_request_ids,
        hca_pages,
        hca_page_offsets,
        hca_windows,
        request_epochs,
        hca_query_work_offsets,
        hca_work_query_ids,
        hca_work_row_begin,
        hca_work_valid_rows,
        attn_sink,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        dep,
    )
    return attn_out


def golden_sparse_attn(tensors):
    """Torch reference: three-state raw plus packed compressed merge, grouped o_proj."""
    import torch

    q = tensors["q"].float()
    ori = tensors["ori_kv"].float().reshape(-1, HEAD_DIM)
    current = tensors["current_kv"].float()
    cmp_kv = tensors["cmp_kv"].float().reshape(-1, HEAD_DIM)
    sources = tensors["swa_sources"]
    request_ids = tensors["query_request_ids"]
    pages = tensors["hca_pages"]
    page_offsets = tensors["hca_page_offsets"]
    windows = tensors["hca_windows"]
    epochs = tensors["request_epochs"]
    work_offsets = tensors["hca_query_work_offsets"]
    work_rows = tensors["hca_work_row_begin"]
    work_valid = tensors["hca_work_valid_rows"]
    sink = tensors["attn_sink"].float()
    tokens = q.shape[0]
    o = torch.zeros(tokens, H, HEAD_DIM)

    def partial(query, kv_rows, valid):
        kv_tile = torch.stack(kv_rows).float()
        valid_t = torch.tensor(valid, dtype=torch.bool)
        scores = (q[query] @ kv_tile.T) * SOFTMAX_SCALE
        scores = scores.masked_fill(~valid_t.unsqueeze(0), NEG_INF)
        mi = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - mi).masked_fill(
            ~valid_t.unsqueeze(0), 0.0
        )
        li = exp_scores.sum(dim=-1, keepdim=True)
        oi = exp_scores.to(torch.bfloat16).float() @ kv_tile.to(torch.bfloat16).float()
        return mi, li, oi

    for query in range(tokens):
        raw_rows = []
        raw_valid = []
        for source in sources[query].tolist():
            if source >= 0:
                if 0 <= int(source) < int(ori.shape[0]):
                    raw_rows.append(ori[source])
                    raw_valid.append(True)
                else:
                    raw_rows.append(torch.zeros(HEAD_DIM))
                    raw_valid.append(False)
            elif source <= SWA_SOURCE_OVERLAY_BASE:
                overlay = SWA_SOURCE_OVERLAY_BASE - int(source)
                same_request = (
                    0 <= overlay < tokens
                    and int(request_ids[overlay]) == int(request_ids[query])
                )
                if same_request and overlay <= query:
                    raw_rows.append(current[overlay])
                    raw_valid.append(True)
                else:
                    # Reject cross-request and future overlays.
                    raw_rows.append(torch.zeros(HEAD_DIM))
                    raw_valid.append(False)
            else:
                raw_rows.append(torch.zeros(HEAD_DIM))
                raw_valid.append(False)
        mi, li, oi = partial(query, raw_rows, raw_valid)
        request = int(request_ids[query])
        for work in range(int(work_offsets[query]), int(work_offsets[query + 1])):
            row_begin = int(work_rows[work])
            valid_rows = int(work_valid[work])
            kv_rows = []
            valid = []
            for lane in range(ATTN_K_TILE):
                if lane >= valid_rows:
                    kv_rows.append(torch.zeros(HEAD_DIM))
                    valid.append(False)
                    continue
                row = row_begin + lane
                begin = 0
                end = 0
                if 0 <= request and request + 1 < int(page_offsets.shape[0]):
                    begin = int(page_offsets[request])
                    end = int(page_offsets[request + 1])
                total = end - begin
                row_valid = False
                row_value = torch.zeros(HEAD_DIM)
                if (
                    0 <= request < int(windows.shape[0])
                    and total > 0
                    and 0 <= begin <= end <= int(pages.shape[0])
                ):
                    valid_begin = int(windows[request, 0])
                    valid_end = int(windows[request, 1])
                    head = int(windows[request, 2])
                    rel = row // BLOCK_SIZE - valid_begin // BLOCK_SIZE
                    if (
                        valid_begin <= row < valid_end
                        and 0 <= head < total
                        and 0 <= rel < total
                    ):
                        entry = begin + (head + rel) % total
                        if 0 <= entry < int(pages.shape[0]):
                            page_epoch = int(pages[entry, 1])
                            physical = int(pages[entry, 0])
                            source_row = physical * BLOCK_SIZE + row % BLOCK_SIZE
                            if (
                                0 <= physical < int(tensors["cmp_kv"].shape[0])
                                and page_epoch == int(epochs[request])
                                and 0 <= source_row < int(cmp_kv.shape[0])
                            ):
                                row_value = cmp_kv[source_row]
                                row_valid = True
                kv_rows.append(row_value)
                valid.append(row_valid)
            cur_mi, cur_li, cur_oi = partial(query, kv_rows, valid)
            mi_new = torch.maximum(mi, cur_mi)
            alpha = torch.exp(mi - mi_new)
            beta = torch.exp(cur_mi - mi_new)
            li = alpha * li + beta * cur_li
            oi = alpha * oi + beta * cur_oi
            mi = mi_new
        o[query] = oi / (li + torch.exp(sink.unsqueeze(-1) - mi))

    cos = tensors["freqs_cos"].float()[:, :HALF_ROPE].unsqueeze(1)
    sin = tensors["freqs_sin"].float()[:, :HALF_ROPE].unsqueeze(1)
    pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    even, odd = pair[..., 0], pair[..., 1]
    inv_even = (even * cos + odd * sin).to(torch.bfloat16).float()
    inv_odd = (odd * cos - even * sin).to(torch.bfloat16).float()
    o = torch.cat(
        [
            o[..., :NOPE_DIM],
            torch.stack([inv_even, inv_odd], dim=-1).flatten(-2),
        ],
        dim=-1,
    ).to(torch.bfloat16)
    wo_a = tensors["wo_a"].float()
    projected = torch.einsum(
        "tgd,grd->tgr",
        o.float().reshape(tokens, O_GROUPS, O_GROUP_IN),
        wo_a,
    )
    projected = projected.reshape(tokens, O_GROUPS, O_LORA)
    amax = projected.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = INT8_SCALE_MAX / amax
    quant = (
        torch.round(projected * scale)
        .to(torch.int32)
        .to(torch.float16)
        .to(torch.int8)
    )
    weight = tensors["wo_b"].reshape(D, O_GROUPS, O_LORA)
    out = torch.zeros(tokens, D)
    for group in range(O_GROUPS):
        out += (
            quant[:, group].to(torch.int32)
            @ weight[:, group].to(torch.int32).T
        ).float() / scale[:, group]
    out *= tensors["wo_b_scale"].float().unsqueeze(0)
    tensors["attn_out"][:] = out.to(torch.bfloat16)

def build_tensor_specs(
    causal_regression_fixture: bool = False,
    short_window_fixture: bool = False,
    mixed_topk_fixture: bool = False,
    cache_window_replacement_fixture: bool = False,
    batch: int = B,
):
    """Build deterministic demo tensors for the HCA standalone harness."""
    import torch
    from golden import TensorSpec
    from utils import block_table, quant_w_per_channel, swa_indices_and_lens

    if batch < 1 or batch > B:
        raise ValueError(f"batch must be in [1, {B}], got {batch}")

    tokens = batch * S

    if short_window_fixture or causal_regression_fixture or cache_window_replacement_fixture:
        rows_by_request = [0] * batch
    else:
        rows_by_request = [ATTN_K_TILE] * batch

    raw_pages_per_request = WIN // BLOCK_SIZE
    raw_block_table = block_table(
        batch=batch,
        table_blocks=raw_pages_per_request,
        physical_blocks=batch * raw_pages_per_request,
    )
    raw_positions = (
        torch.arange(WIN - S, WIN, dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    canonical_sources, canonical_lens = swa_indices_and_lens(
        raw_positions,
        raw_block_table,
        block_size=BLOCK_SIZE,
        window=WIN,
    )

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        q = torch.rand(tokens, H, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            q[0].fill_(1.0)
        return q

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        kv = torch.rand(
            batch * raw_pages_per_request, BLOCK_SIZE, 1, HEAD_DIM
        ) - 0.5
        if cache_window_replacement_fixture:
            page = int(raw_block_table[0, 16 // BLOCK_SIZE])
            row = 16 % BLOCK_SIZE
            kv[page, row, 0].fill_(0.0)
            kv[page, row, 0, 0] = 4.0
        return kv

    def init_current_kv():
        """Initialize the current-step KV overlay."""
        kv = torch.rand(tokens, HEAD_DIM) - 0.5
        if causal_regression_fixture:
            kv.zero_()
            for request in range(batch):
                kv[request * S] = 4.0
                for local in range(1, S):
                    kv[request * S + local] = -4.0
        return kv

    def init_swa_sources():
        """Build three-state raw sources from physical rows + causal overlay."""
        sources = canonical_sources.clone()
        for request in range(batch):
            for local in range(S):
                query = request * S + local
                overlay_begin = int(canonical_lens[query]) - local - 1
                for overlay_local in range(local + 1):
                    overlay_query = request * S + overlay_local
                    sources[query, overlay_begin + overlay_local] = (
                        SWA_SOURCE_OVERLAY_BASE - overlay_query
                    )
        if mixed_topk_fixture:
            sources[:, : WIN // 2] = SWA_SOURCE_INVALID
        if causal_regression_fixture:
            sources.fill_(SWA_SOURCE_INVALID)
            for request in range(batch):
                first = request * S
                for local in range(S):
                    for overlay_local in range(local + 1):
                        sources[first + local, WIN - local - 1 + overlay_local] = (
                            SWA_SOURCE_OVERLAY_BASE - (first + overlay_local)
                        )
        return sources

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return torch.rand(HCA_KV_POOL_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM) - 0.5

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_query_request_ids():
        return torch.arange(batch, dtype=torch.int32).repeat_interleave(S)

    def init_hca_pages_and_work():
        page_ids = []
        page_offsets = [0]
        windows = []
        work_query_ids = []
        work_rows = []
        work_valid_rows = []
        work_offsets = [0]
        physical_page = 0
        for request, rows in enumerate(rows_by_request):
            pages = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
            if physical_page + pages > HCA_KV_POOL_BLOCKS:
                raise ValueError("sparse HCA fixture exceeds the global page pool")
            ids = list(range(physical_page, physical_page + pages))
            physical_page += pages
            head = 0
            page_ids.extend((page, 7) for page in ids)
            page_offsets.append(len(page_ids))
            windows.append((0, rows, head))
            for local in range(S):
                query = request * S + local
                row = 0
                while row < rows:
                    work_query_ids.append(query)
                    work_rows.append(row)
                    work_valid_rows.append(min(ATTN_K_TILE, rows - row))
                    row += ATTN_K_TILE
                work_offsets.append(len(work_query_ids))
        return (
            torch.tensor(page_ids, dtype=torch.int32).reshape(-1, 2),
            torch.tensor(page_offsets, dtype=torch.int32),
            torch.tensor(windows, dtype=torch.int32),
            torch.tensor(work_offsets, dtype=torch.int32),
            torch.tensor(work_query_ids, dtype=torch.int32),
            torch.tensor(work_rows, dtype=torch.int32),
            torch.tensor(work_valid_rows, dtype=torch.int32),
        )

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

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights."""
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) / (O_GROUP_IN ** 0.5)

    wo_b_bf16 = ((torch.rand(D, O_GROUPS * O_LORA) - 0.5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

    def init_wo_b():
        """Initialize the second-stage output-projection weights in per-channel INT8 form."""
        return wo_b_i8

    def init_wo_b_scale():
        """Initialize the dequant scales paired with the INT8 second-stage weights."""
        return wo_b_scale

    hca_pages, hca_page_offsets, hca_windows, work_offsets, work_qids, work_rb, work_vr = init_hca_pages_and_work()
    cmp_kv_value = init_cmp_kv()

    return [
        TensorSpec("q", [tokens, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [batch * (WIN // BLOCK_SIZE), BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("current_kv", [tokens, HEAD_DIM], torch.bfloat16, init_value=init_current_kv),
        TensorSpec("swa_sources", [tokens, WIN], torch.int32, init_value=init_swa_sources),
        TensorSpec("cmp_kv", [cmp_kv_value.shape[0], BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=lambda: cmp_kv_value),
        TensorSpec("query_request_ids", [tokens], torch.int32, init_value=init_query_request_ids),
        TensorSpec("hca_pages", [hca_pages.shape[0], 2], torch.int32, init_value=lambda: hca_pages),
        TensorSpec("hca_page_offsets", [hca_page_offsets.shape[0]], torch.int32, init_value=lambda: hca_page_offsets),
        TensorSpec("hca_windows", [batch, 3], torch.int32, init_value=lambda: hca_windows),
        TensorSpec("request_epochs", [batch], torch.int32, init_value=lambda: torch.full((batch,), 7, dtype=torch.int32)),
        TensorSpec("hca_query_work_offsets", [work_offsets.shape[0]], torch.int32, init_value=lambda: work_offsets),
        TensorSpec("hca_work_query_ids", [work_qids.shape[0]], torch.int32, init_value=lambda: work_qids),
        TensorSpec("hca_work_row_begin", [work_rb.shape[0]], torch.int32, init_value=lambda: work_rb),
        TensorSpec("hca_work_valid_rows", [work_vr.shape[0]], torch.int32, init_value=lambda: work_vr),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [tokens, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [tokens, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=B,
                        help=f"runtime request count up to {B} (the compile-time "
                             "upper bound). The token axis is pl.dynamic, so one compiled program "
                             "serves every value.")
    parser.add_argument("--causal-regression-fixture", action="store_true", default=False,
                        help="Amplify the S=2 future-window-slot regression.")
    parser.add_argument("--short-window-fixture", action="store_true", default=False,
                        help="Use a short-window topk row with valid prefix + -1 padding.")
    parser.add_argument("--mixed-topk-fixture", action="store_true", default=False,
                        help="Use -1-padded window slots with valid compressed raw indices.")
    parser.add_argument("--cache-window-replacement-fixture", action="store_true", default=False,
                        help="Place a sentinel row inside the cache window prefix.")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="Capture PTO2 dependency edges (deps.json); the swimlane "
                             "converter draws fanout/fanin arrows from the sibling file.")
    parser.add_argument("--enable-pmu", nargs="?", const=2, default=0, type=int, choices=[0, 1, 2, 4])
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch < 1 or args.batch > B:
        parser.error(f"--batch must be in [1, {B}], got {args.batch}")

    print(f"compress_ratio={COMPRESS_RATIO} -> ATTN_K_TILE={ATTN_K_TILE}", flush=True)

    result = run_jit(
        fn=sparse_attn_test,
        specs=build_tensor_specs(
            args.causal_regression_fixture,
            args.short_window_fixture,
            args.mixed_topk_fixture,
            args.cache_window_replacement_fixture,
            batch=args.batch,
        ),
        golden_fn=golden_sparse_attn,
        golden_data=args.golden_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=args.enable_dep_gen,
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
