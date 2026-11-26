# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Owner-zero B=1 Drafter attention with one actual head group per rank."""

import pypto.language as pl
from .qkv_proj_rope import (
    D,Q_LORA,HEAD_DIM,ROPE_DIM,ORI_BLOCK_NUM_DYN,BLOCK_SIZE,
    BIAS_B_TILE,NEG_INF,QPROJ_T_PAD,Q_ROPE_T_TILE,Q_ROPE_WORKERS,ROPE_DIM_SCALE,q_proj_qr,
)
from .kv_tp8 import draft_kv_projection_tp8,draft_kv_norm_tp8
from .qproj_tp8 import q_proj_q_matmul_tp8,q_proj_q_dequant_tp8
from .swa_local_tp8 import (
    T,B,INDEX_WIDTH,SPARSE_BLOCKS,ATTN_K_TILE,O_GROUPS,LOCAL_T_PAD,HEADS_PER_GROUP,local_attention,
)


@pl.jit.inline
def prepare_live_rope(
    rope_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[T, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T, ROPE_DIM], pl.INT32],
):
    """Build the head-invariant interleaved cos / sign-folded sin / swap-index rope rows."""
    t_dim = pl.tensor.dim(rope_cos, 0)
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    # The tail guard below writes the outputs from inside a conditional region.
    # Binding them to the scalar extent keeps the region result off the dynamic
    # symbol, which has no definition in an inlining caller that passes a
    # statically shaped token axis.
    rope_cos_il_view = pl.reshape(rope_cos_il, [t_dim, ROPE_DIM])
    rope_sin_signed_view = pl.reshape(rope_sin_signed, [t_dim, ROPE_DIM])
    rope_swap_idx_view = pl.reshape(rope_swap_idx, [t_dim, ROPE_DIM])

    token_tiles = (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE
    for qrp_worker in pl.spmd(
        pl.min(Q_ROPE_WORKERS, token_tiles), name_hint="q_rope_prepare", allow_early_resolve=True
    ):
        # The interleave lane, swap index and sign fold only depend on the column
        # index, so they are built once per worker instead of once per tile.
        qrp_ones = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_dup_idx = pl.cast(qrp_dup_f, target_type=pl.INT32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_swap_idx = pl.cast(qrp_swap_f, target_type=pl.INT32)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        for qrp_idx in pl.range(qrp_worker, token_tiles, pl.min(Q_ROPE_WORKERS, token_tiles)):
            qrp_t0 = qrp_idx * Q_ROPE_T_TILE
            qrp_valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - qrp_t0)
            if qrp_valid_rows == Q_ROPE_T_TILE:
                qrp_cos_rows_full = rope_cos_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :]
                qrp_sin_rows_full = rope_sin_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :]
                qrp_cos_full = pl.cast(qrp_cos_rows_full, target_type=pl.FP32)
                qrp_sin_full = pl.cast(qrp_sin_rows_full, target_type=pl.FP32)
                qrp_cos_il_full = pl.gather(qrp_cos_full, dim=-1, index=qrp_dup_idx)
                qrp_sin_il_full = pl.gather(qrp_sin_full, dim=-1, index=qrp_dup_idx)
                qrp_sin_signed_full = pl.mul(qrp_sin_il_full, qrp_sign)
                rope_cos_il_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_cos_il_full
                rope_sin_signed_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_sin_signed_full
                rope_swap_idx_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_swap_idx
            else:
                qrp_cos_rows_tail = pl.load(
                    rope_cos_view,
                    [qrp_t0, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[qrp_valid_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                qrp_sin_rows_tail = pl.load(
                    rope_sin_view,
                    [qrp_t0, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[qrp_valid_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                qrp_tail_col = pl.col_expand_mul(
                    pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                    pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                )
                qrp_tail_dup_f = pl.cast(
                    pl.cast(pl.mul(qrp_tail_col, 0.5), target_type=pl.INT32, mode="trunc"),
                    target_type=pl.FP32,
                )
                qrp_tail_lane = pl.sub(qrp_tail_col, pl.mul(qrp_tail_dup_f, 2.0))
                qrp_tail_swap_f = pl.sub(pl.add(qrp_tail_col, 1.0), pl.mul(qrp_tail_lane, 2.0))
                # Row-major flattening offsets stay in fp32: col-expand is defined for
                # half / float only.
                qrp_row_seed = pl.mul(
                    pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                    ROPE_DIM_SCALE,
                )
                qrp_row_grid = pl.col_expand_mul(
                    pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.FP32, value=1.0),
                    qrp_row_seed,
                )
                qrp_row_offset = pl.transpose(qrp_row_grid, axis1=0, axis2=1)
                qrp_dup_idx_tail = pl.cast(pl.add(qrp_tail_dup_f, qrp_row_offset), target_type=pl.INT32)
                qrp_gather_tmp = pl.create_tile(
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    dtype=pl.INT32,
                    target_memory=pl.MemorySpace.Vec,
                )
                qrp_cos_il_tail = pl.tile.gather(
                    pl.cast(qrp_cos_rows_tail, target_type=pl.FP32),
                    qrp_dup_idx_tail,
                    qrp_gather_tmp,
                )
                qrp_sin_il_tail = pl.tile.gather(
                    pl.cast(qrp_sin_rows_tail, target_type=pl.FP32),
                    qrp_dup_idx_tail,
                    qrp_gather_tmp,
                )
                qrp_tail_sign = pl.sub(pl.mul(qrp_tail_lane, 2.0), 1.0)
                qrp_sin_signed_tail = pl.mul(qrp_sin_il_tail, qrp_tail_sign)
                pl.store(
                    pl.set_validshape(qrp_cos_il_tail, qrp_valid_rows, ROPE_DIM),
                    [qrp_t0, 0],
                    rope_cos_il_view,
                )
                pl.store(
                    pl.set_validshape(qrp_sin_signed_tail, qrp_valid_rows, ROPE_DIM),
                    [qrp_t0, 0],
                    rope_sin_signed_view,
                )
                pl.store(
                    pl.set_validshape(
                        pl.cast(qrp_tail_swap_f, target_type=pl.INT32),
                        qrp_valid_rows,
                        ROPE_DIM,
                    ),
                    [qrp_t0, 0],
                    rope_swap_idx_view,
                )


@pl.jit.inline
def attention_local_tp8(
    x: pl.Tensor[[T, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[Q_LORA, 8 * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[8 * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos_local: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin_local: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    kv_slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[B, INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[8], pl.FP32],
    o_packed_heads: pl.Tensor[[O_GROUPS * LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16],
):
    kv_tokens = T
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([T, ROPE_DIM], dtype=pl.INT32)
    prepare_live_rope(
        freqs_cos_local,
        freqs_sin_local,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
    )

    q_work = pl.create_tensor([T, 8, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qr_i8 = pl.create_tensor([QPROJ_T_PAD,Q_LORA],dtype=pl.INT8)
    qr_dq = pl.create_tensor([QPROJ_T_PAD,1],dtype=pl.FP32,manual_dep=True)
    q_proj_qr(x,wq_a,gamma_cq,qr,qr_scale,qr_i8,qr_dq)
    q_dep = pl.system.task_dummy(deps=[])
    partial = pl.create_tensor([QPROJ_T_PAD,8*HEAD_DIM],dtype=pl.INT32)
    local_rows=pl.const(5,pl.INDEX)
    start=pl.const(0,pl.INDEX)
    q_proj_q_matmul_tp8(wq_b,qr_i8,partial,local_rows,q_dep)
    # The five-row Q dequant already masks the RoPE tail. Q and KV use the
    # same positions on this owner-zero path, so both consume this preparation.
    q_proj_q_dequant_tp8(wq_b_scale,rope_cos_il,rope_sin_signed,q_work,
                        qr_dq,partial,start,local_rows)
    q=pl.slice(q_work,[T,8,HEAD_DIM],[0,0,0])

    # DSA-CP keeps Q on its token owner while every rank updates the complete
    # group KV stream before running attention for its local queries.
    late_dep = pl.system.task_dummy(deps=[])
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    query_kv_raw=pl.create_tensor([16,512],dtype=pl.FP32)
    query_base=pl.const(0,pl.INDEX)
    query_rows=pl.const(5,pl.INDEX)
    kv_projected=draft_kv_projection_tp8(x,wkv,query_kv_raw,query_rows,query_base,query_rows,late_dep)
    draft_kv_norm_tp8(query_kv_raw,gamma_ckv,rope_cos_il,rope_sin_signed,
                     rope_swap_idx,kv,query_rows,query_base,query_rows,kv_projected)


    # Commit the block's own KV and build the visible-length mask in one task; the
    # gather below reads those rows back through swa_indices.
    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([B, INDEX_WIDTH], dtype=pl.FP32)
    sink_matrix=pl.create_tensor([1,16],dtype=pl.FP32)
    actual_sink=pl.reshape(attn_sink,[1,8])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_kv_commit_valid_bias"):
        pl.store(pl.load(actual_sink,[0,0],[1,8]),[0,0],sink_matrix)
        pl.store(pl.tile.full([1,8],dtype=pl.FP32,value=0.0),[0,8],sink_matrix)
        for write_t in pl.range(kv_tokens):
            write_row_i64 = pl.read(kv_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]
        v_col = pl.cast(pl.arange(0, [1, INDEX_WIDTH], dtype=pl.INT32), target_type=pl.FP32)
        for v_blk in pl.range((B + BIAS_B_TILE - 1) // BIAS_B_TILE):
            v_b0 = v_blk * BIAS_B_TILE
            v_rows = pl.min(BIAS_B_TILE, B - v_b0)
            v_col_m = pl.col_expand(pl.full([BIAS_B_TILE, INDEX_WIDTH], dtype=pl.FP32, value=0.0), v_col)
            v_lens_i32 = pl.slice(swa_lens, [BIAS_B_TILE], [v_b0], valid_shape=[v_rows])
            v_lens = pl.cast(pl.reshape(v_lens_i32, [BIAS_B_TILE, 1]), target_type=pl.FP32)
            v_valid = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0), 1.0)
            v_idx_i32 = pl.slice(
                swa_indices, [BIAS_B_TILE, INDEX_WIDTH], [v_b0, 0], valid_shape=[v_rows, INDEX_WIDTH]
            )
            v_idx = pl.cast(v_idx_i32, target_type=pl.FP32)
            v_mapped = pl.minimum(pl.maximum(pl.add(v_idx, 1.0), 0.0), 1.0)
            v_valid = pl.mul(v_valid, v_mapped)
            v_bias = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)
            v_bias = pl.set_validshape(v_bias, v_rows, INDEX_WIDTH)
            sparse_bias[v_b0 : v_b0 + BIAS_B_TILE, 0:INDEX_WIDTH] = v_bias

    # One index row per request: build_dspark_swa_indices repeats it across the
    # block's query rows, so the gather runs B times, not T.
    visible_kv = pl.create_tensor([B * INDEX_WIDTH, HEAD_DIM], dtype=pl.BF16)
    for g_task in pl.spmd(B * SPARSE_BLOCKS, name_hint="dspark_gather_kv"):
        g_b = g_task // SPARSE_BLOCKS
        g_c0 = (g_task % SPARSE_BLOCKS) * ATTN_K_TILE
        g_base = g_b * INDEX_WIDTH
        if g_c0 < pl.read(swa_lens, [g_b]):
            for g_c in pl.range(g_c0, g_c0 + ATTN_K_TILE):
                g_slot_i32 = pl.read(swa_indices, [g_b, g_c])
                g_dst = g_base + g_c
                if g_slot_i32 >= 0:
                    g_slot = pl.cast(g_slot_i32, pl.INDEX)
                    visible_kv[g_dst : g_dst + 1, 0:HEAD_DIM] = kv_cache_flat[g_slot : g_slot + 1, 0:HEAD_DIM]
                else:
                    visible_kv[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full(
                        [1, HEAD_DIM], dtype=pl.BF16, value=0.0
                    )

    # Sixteen physical Cube rows carry eight actual heads and eight dummy heads.
    q_flat=pl.reshape(q,[T*8,HEAD_DIM])
    q_pad=pl.create_tensor([T*16,HEAD_DIM],dtype=pl.BF16)
    sink=pl.reshape(sink_matrix,[16])
    with pl.spmd(T,name_hint="draft_tp8_pad_local_q"):
        token=pl.tile.get_block_idx()
        start=token*16
        source=token*8
        q_pad[start:start+8,:]=q_flat[source:source+8,:]
        q_pad[start+8:start+16,:]=pl.full([8,HEAD_DIM],dtype=pl.BF16,value=0.0)
    padded=pl.reshape(q_pad,[T,16,HEAD_DIM])
    local_attention(padded,visible_kv,sparse_bias,swa_lens,sink,rope_cos_il,rope_sin_signed,rope_swap_idx,o_packed_heads)
    return kv_cache,o_packed_heads
