# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""One real Drafter attention Q head shard, with the original arithmetic."""

import pypto.language as pl

from .qkv_proj_rope import (
    EPS,
    HEAD_DIM,
    NOPE_DIM,
    QKV_PROJ_ROPE_T_DYN,
    QPROJ_MM_N_TILE,
    QPROJ_M_TILE,
    QPROJ_TAIL_M_TILE,
    QPROJ_T_PAD,
    Q_DEQUANT_WORKERS,
    Q_LORA,
    Q_ROPE_H_TILE,
    Q_ROPE_T_TILE,
    ROPE_DIM,
    ROPE_DIM_SCALE,
)

HEADS_LOCAL = 8
QPROJ_TP8_WORKERS = 16

@pl.jit.inline(auto_scope=False)
def q_proj_q_matmul_tp8(
    wq_b: pl.Tensor[[Q_LORA, HEADS_LOCAL * HEAD_DIM], pl.INT8, pl.NZ],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    q_proj_i32: pl.Tensor[[QPROJ_T_PAD, HEADS_LOCAL * HEAD_DIM], pl.INT32],
    tile_rows: pl.Scalar[pl.INDEX],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    """Project one bounded Q tile and expose its cube task ID."""
    qproj_t_matmul = ((tile_rows + QPROJ_TAIL_M_TILE - 1) // QPROJ_TAIL_M_TILE) * QPROJ_TAIL_M_TILE
    qproj_full_rows = (tile_rows // QPROJ_M_TILE) * QPROJ_M_TILE
    with pl.spmd(
        QPROJ_TP8_WORKERS,
        name_hint="qproj_matmul",
        deps=[qproj_dep],
    ) as qproj_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
        qproj_worker = pl.tile.get_block_idx()
        for qproj_n_idx in pl.range(
            qproj_worker,
            (HEADS_LOCAL * HEAD_DIM) // QPROJ_MM_N_TILE,
            QPROJ_TP8_WORKERS,
        ):
            w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
            wq_full = wq_b[0:Q_LORA, w_col0 : w_col0 + QPROJ_MM_N_TILE]
            for t0 in pl.range(0, qproj_full_rows, QPROJ_M_TILE):
                qr_full = qr_i8_matmul[t0 : t0 + QPROJ_M_TILE, 0:Q_LORA]
                col_acc = pl.matmul(qr_full, wq_full, out_dtype=pl.INT32)
                q_proj_i32[t0 : t0 + QPROJ_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc

            tail_w_col0 = w_col0
            for tail_t0 in pl.range(qproj_full_rows, qproj_t_matmul, QPROJ_TAIL_M_TILE):
                qproj_tail_rows = pl.min(QPROJ_TAIL_M_TILE, tile_rows - tail_t0)
                qr_i8_tail = pl.slice(
                    qr_i8_matmul,
                    [QPROJ_TAIL_M_TILE, Q_LORA],
                    [tail_t0, 0],
                    valid_shape=[qproj_tail_rows, Q_LORA],
                )
                tail_acc = pl.matmul(qr_i8_tail, wq_full, out_dtype=pl.INT32)
                q_proj_i32[
                    tail_t0 : tail_t0 + QPROJ_TAIL_M_TILE,
                    tail_w_col0 : tail_w_col0 + QPROJ_MM_N_TILE,
                ] = tail_acc
    return qproj_tid


@pl.jit.inline(auto_scope=False)
def q_proj_q_dequant_tp8(
    wq_b_scale: pl.Tensor[[HEADS_LOCAL * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.FP32],
    q: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, HEADS_LOCAL, HEAD_DIM], pl.BF16],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    q_proj_i32: pl.Tensor[[QPROJ_T_PAD, HEADS_LOCAL * HEAD_DIM], pl.INT32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Dequantize, normalize, and rotate the five live owner-zero rows."""
    t_dim = pl.tensor.dim(q, 0)
    q_flat = pl.reshape(q, [t_dim, HEADS_LOCAL * HEAD_DIM])
    for dq_worker in pl.spmd(
        Q_DEQUANT_WORKERS,
        name_hint="qproj_dequant_rms_nope_rope",
        allow_early_resolve=True,
    ):
        dq_head_tile = Q_ROPE_H_TILE
        if (
            tile_rows >= Q_DEQUANT_WORKERS * Q_ROPE_T_TILE
            and tile_rows % (Q_DEQUANT_WORKERS * Q_ROPE_T_TILE) == 0
        ):
            dq_head_tile = HEADS_LOCAL
        for dq_work in pl.range(
            dq_worker,
            ((tile_rows + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE) * (HEADS_LOCAL // dq_head_tile),
            Q_DEQUANT_WORKERS,
        ):
            hg = (dq_work % (HEADS_LOCAL // dq_head_tile)) * dq_head_tile
            tg = (dq_work // (HEADS_LOCAL // dq_head_tile)) * Q_ROPE_T_TILE
            out_tg = tile_base + tg
            valid_tail_rows = tile_rows - tg
            # q_proj and its dequant scale are padded to the cube boundary.
            # Keep the math on a static eight-row tile and crop public stores.
            qr_scale_dq_tail = pl.load(
                qr_scale_pad_store,
                [tg, 0],
                [Q_ROPE_T_TILE, 1],
                target_memory=pl.MemorySpace.Vec,
            )
            q_cos_il_tail = pl.load(
                rope_cos_il,
                [out_tg, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shape=[valid_tail_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            q_sin_signed_tail = pl.load(
                rope_sin_signed,
                [out_tg, 0],
                [Q_ROPE_T_TILE, ROPE_DIM],
                valid_shape=[valid_tail_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            q_col = pl.col_expand_mul(
                pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
            )
            q_dup_f = pl.cast(pl.cast(pl.mul(q_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
            q_lane = pl.sub(q_col, pl.mul(q_dup_f, 2.0))
            q_swap_f = pl.sub(pl.add(q_col, 1.0), pl.mul(q_lane, 2.0))
            # Row-major flattening offsets stay in fp32: col-expand is defined
            # for half / float only.
            q_row_seed = pl.mul(
                pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                ROPE_DIM_SCALE,
            )
            q_row_grid = pl.col_expand_mul(
                pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.FP32, value=1.0),
                q_row_seed,
            )
            q_row_offset = pl.transpose(q_row_grid, axis1=0, axis2=1)
            q_swap_idx_tail = pl.cast(pl.add(q_swap_f, q_row_offset), target_type=pl.INT32)
            q_head_reduce_tmp = pl.create_tile(
                [Q_ROPE_T_TILE, HEAD_DIM],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            q_gather_tmp = pl.create_tile(
                [Q_ROPE_T_TILE, ROPE_DIM],
                dtype=pl.INT32,
                target_memory=pl.MemorySpace.Vec,
            )
            for h_inner_tail in pl.range(dq_head_tile):
                h_tail = hg + h_inner_tail
                h0_tail = h_tail * HEAD_DIM
                q_head_acc_tail = pl.load(
                    q_proj_i32,
                    [tg, h0_tail],
                    [Q_ROPE_T_TILE, HEAD_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_head_scale_input_tail = pl.load(
                    wq_b_scale,
                    [h0_tail],
                    [HEAD_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_head_scale_tail = pl.reshape(q_head_scale_input_tail, [1, HEAD_DIM])
                q_head_acc_fp32_tail = pl.cast(q_head_acc_tail, target_type=pl.FP32, mode="none")
                q_head_row_scaled_tail = pl.row_expand_mul(q_head_acc_fp32_tail, qr_scale_dq_tail)
                q_head_dq_tail = pl.col_expand_mul(q_head_row_scaled_tail, q_head_scale_tail)

                q_head_sq_tail = pl.mul(q_head_dq_tail, q_head_dq_tail)
                q_head_sq_sum_tail = pl.row_sum(q_head_sq_tail, q_head_reduce_tmp)
                q_head_inv_rms_tail = pl.recip(
                    pl.sqrt(pl.add(pl.mul(q_head_sq_sum_tail, 1.0 / HEAD_DIM), EPS)),
                )

                q_nope_normed_tail = pl.row_expand_mul(q_head_dq_tail[:, 0:NOPE_DIM], q_head_inv_rms_tail)
                q_nope_bf16_tail = pl.cast(q_nope_normed_tail, target_type=pl.BF16, mode="rint")
                q_nope_valid = pl.set_validshape(q_nope_bf16_tail, valid_tail_rows, NOPE_DIM)
                pl.store(q_nope_valid, [out_tg, h0_tail], q_flat)

                q_rope_chunk_raw_tail = q_head_dq_tail[:, NOPE_DIM:HEAD_DIM]
                q_rope_chunk_tail = pl.row_expand_mul(q_rope_chunk_raw_tail, q_head_inv_rms_tail)
                q_rope_swapped_tail = pl.tile.gather(q_rope_chunk_tail, q_swap_idx_tail, q_gather_tmp)
                q_rope_base_tail = pl.mul(q_rope_chunk_tail, q_cos_il_tail)
                q_rope_delta_tail = pl.mul(q_rope_swapped_tail, q_sin_signed_tail)
                q_rope_rot_tail = pl.add(q_rope_base_tail, q_rope_delta_tail)
                q_rope_bf16_tail = pl.cast(q_rope_rot_tail, target_type=pl.BF16, mode="rint")
                q_rope_valid = pl.set_validshape(q_rope_bf16_tail, valid_tail_rows, ROPE_DIM)
                pl.store(q_rope_valid, [out_tg, h0_tail + NOPE_DIM], q_flat)
    return q
