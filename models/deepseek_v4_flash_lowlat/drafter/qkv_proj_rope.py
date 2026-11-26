# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Drafter projection, attention and context-parallel exchange."""

import pypto.language as pl
from .config import FLASH as M, INT8_SCALE_MAX, INT8_AMAX_EPS
from math import gcd
import sys
from . import config
import pypto.language.distributed as pld
from .config import DECODE_TOKENS
from .config import BLOCK_SIZE, DECODE_SEQ, MOE_TOKENS
from ..kv_proj_rope import kv_proj_rope


QKV_PROJ_ROPE_T_DYN = pl.dynamic("NATIVE_D_QKV_Q_T_DYN")  # T = B * S


ROPE_T_DYN = pl.dynamic("NATIVE_D_QKV_ROPE_T_DYN")

PREFILL_DENSE_TILE = 512

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

ROPE_DIM_SCALE = float(ROPE_DIM)

NOPE_DIM = M.nope_head_dim

Q_LORA = M.q_lora_rank

EPS = M.rms_norm_eps

QPROJ_MM_N_TILE = 256  # qproj output-column tile

Q_LORA_TILE = 256  # qr rms-norm / quant N granularity


QUANT_TILE = 256

T_TILE = 8

MATMUL_T_TILE = 16

QR_M_TILE = MATMUL_T_TILE  # qr_proj token (M) tile; cube rows must be a 16-row boxed tile

QR_DENSE_M_TILE = 64

QR_N_TILE = 128  # qr_proj Q_LORA (N) per matmul

QR_K_TILE = 256  # qr_proj D (K) reduction tile   | divides QR_SPLIT_K_TILE

QR_OK = 2  # qr_proj split-K factor         | D//QR_OK cores share each N-group

QR_SPLIT_K_TILE = D // QR_OK  # qr_proj K per split (=2048)


QPROJ_M_TILE = 64  # dense qproj token tile; 64 KiB L0C accumulator

QPROJ_WORKERS = 24

QPROJ_TAIL_M_TILE = 64

QPROJ_T_PAD = ((PREFILL_DENSE_TILE + QPROJ_TAIL_M_TILE - 1) // QPROJ_TAIL_M_TILE) * QPROJ_TAIL_M_TILE


Q_ROPE_T_TILE = 8

Q_ROPE_WORKERS = 16

QR_NORM_WORKERS = 16

Q_ROPE_H_TILE = 4  # heads per fused qproj dequant/rms/rope task

Q_DEQUANT_WORKERS = 48

assert QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C Acc cap

assert QPROJ_M_TILE % QPROJ_TAIL_M_TILE == 0


@pl.jit.inline
def rope_prepare(
    rope_cos: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.INT32],
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


@pl.jit.inline(auto_scope=False)
def q_proj_qr(
    x: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, 1], pl.FP32],
    qr_i8_matmul: pl.Out[pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8]],
    qr_scale_pad_store: pl.Out[pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32]],
):
    """Q LoRA, RMSNorm and quantization -- the half the indexer query chain needs.

    The two staging tensors are held by the caller so the qproj half can be issued
    separately; they are rewritten per dense tile, so a caller that runs more than one
    tile iteration must let the qproj half finish before the next one starts.
    Callers must bound each paired QR/Q invocation to PREFILL_DENSE_TILE rows.
    """
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            x_view = pl.reshape(x, [t_dim, D])
            qr_t_matmul = ((tile_rows + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
            qr_full_rows = (tile_rows // QR_DENSE_M_TILE) * QR_DENSE_M_TILE
            qproj_t_matmul = ((tile_rows + QPROJ_TAIL_M_TILE - 1) // QPROJ_TAIL_M_TILE) * QPROJ_TAIL_M_TILE

            # Split-K qr_proj (M=t_dim, K=D=4096, N=Q_LORA=1024): QR_N_TILE N-groups expanded
            # QR_OK-fold into cube blocks that atomic-add their K partials into a zero-seeded
            # output. Seeded on-core, not through create_tensor init_value=0.
            qr_fp32 = pl.create_tensor([qr_t_matmul, Q_LORA], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
                for ts0 in pl.range(0, qr_t_matmul, QR_M_TILE):
                    for nseed0 in pl.range(0, Q_LORA, QR_N_TILE):
                        qr_seed = pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0)
                        qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = qr_seed

            for qbg_idx in pl.spmd(
                (Q_LORA // QR_N_TILE) * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True
            ):
                # Weight reads bypass L2.
                pl.set_cache_policy(wq_a, pl.CachePolicy.BYPASS)
                q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
                qr_k_base = (qbg_idx % QR_OK) * QR_SPLIT_K_TILE
                for dense_t0 in pl.range(0, qr_full_rows, QR_DENSE_M_TILE):
                    dense_x0 = tile_base + dense_t0
                    dense_acc = pl.create_tensor([QR_DENSE_M_TILE, QR_N_TILE], dtype=pl.FP32)
                    for dense_k in pl.pipeline(0, QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                        dense_d0 = qr_k_base + dense_k * QR_K_TILE
                        dense_x = x_view[
                            dense_x0 : dense_x0 + QR_DENSE_M_TILE, dense_d0 : dense_d0 + QR_K_TILE
                        ]
                        dense_w = wq_a[dense_d0 : dense_d0 + QR_K_TILE, q_a_col0 : q_a_col0 + QR_N_TILE]
                        dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_w, init_cond=(dense_k == 0))
                    qr_fp32 = pl.assemble(qr_fp32, dense_acc, [dense_t0, q_a_col0], atomic=pl.AtomicType.Add)
                for t0 in pl.range(qr_full_rows, qr_t_matmul, QR_M_TILE):
                    q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
                    for db in pl.pipeline(QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                        qr_d0 = qr_k_base + db * QR_K_TILE
                        qr_rows = pl.min(QR_M_TILE, tile_rows - t0)
                        x_t0 = tile_base + t0
                        q_x_chunk_bf16 = pl.slice(
                            x_view,
                            [QR_M_TILE, QR_K_TILE],
                            [x_t0, qr_d0],
                            valid_shape=[qr_rows, QR_K_TILE],
                        )
                        w_chunk = wq_a[qr_d0 : qr_d0 + QR_K_TILE, q_a_col0 : q_a_col0 + QR_N_TILE]
                        q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk, init_cond=(db == 0))
                    qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=pl.AtomicType.Add)

            qr_view = pl.reshape(qr, [t_dim, Q_LORA])
            qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])

            # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
            qr_token_tiles = (tile_rows + T_TILE - 1) // T_TILE
            for tg_idx_worker in pl.spmd(
                pl.min(qr_token_tiles, QR_NORM_WORKERS),
                name_hint="qr_rms_norm_quant",
                allow_early_resolve=True,
            ):
                for tg_idx in pl.range(
                    tg_idx_worker, qr_token_tiles, pl.min(qr_token_tiles, QR_NORM_WORKERS)
                ):
                    tg = tg_idx * T_TILE
                    valid_rows = pl.min(T_TILE, tile_rows - tg)
                    out_tg = tile_base + tg
                    qr_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                    qr_amax_g = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                    for qr_rms_col0 in pl.pipeline(0, Q_LORA, Q_LORA_TILE, stage=2):
                        qr_rms_chunk = qr_fp32[tg : tg + T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE]
                        qr_rms_sq = pl.mul(qr_rms_chunk, qr_rms_chunk)
                        qr_rms_row_sum = pl.reshape(pl.row_sum(qr_rms_sq), [1, T_TILE])
                        qr_sq_sum = pl.add(qr_sq_sum, qr_rms_row_sum)
                        gamma_rms_cast = pl.cast(
                            gamma_cq[qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE], target_type=pl.FP32
                        )
                        gamma_rms_chunk = pl.reshape(gamma_rms_cast, [1, Q_LORA_TILE])
                        qr_g = pl.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
                        qr_g_abs = pl.abs(qr_g)
                        qr_g_row_max = pl.reshape(pl.row_max(qr_g_abs), [1, T_TILE])
                        qr_amax_g = pl.maximum(qr_amax_g, qr_g_row_max)
                    qr_inv_rms = pl.rsqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS), high_precision=True)
                    qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])
                    qr_amax_floor = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    qr_amax_normed = pl.mul(qr_inv_rms, qr_amax_g)
                    qr_tile_amax = pl.maximum(qr_amax_floor, qr_amax_normed)

                    qr_scale_quant_row = pl.div(
                        pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax
                    )
                    qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
                    qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])
                    qr_scale_pad_store = pl.assemble(qr_scale_pad_store, qr_tile_scale_dq, [tg, 0])
                    if valid_rows == T_TILE:
                        qr_scale_view[out_tg : out_tg + T_TILE, :] = qr_tile_scale_dq
                    else:
                        qr_scale_tail = pl.load(
                            qr_scale_pad_store,
                            [tg, 0],
                            [T_TILE, 1],
                            valid_shape=[valid_rows, 1],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        pl.store(qr_scale_tail, [out_tg, 0], qr_scale_view)

                    for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
                        qr_chunk = qr_fp32[tg : tg + T_TILE, qa : qa + QUANT_TILE]
                        gamma_q_cast = pl.cast(gamma_cq[qa : qa + QUANT_TILE], target_type=pl.FP32)
                        gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
                        qr_q_normed = pl.col_expand_mul(
                            pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk
                        )
                        qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
                        qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
                        qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
                        qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
                        qr_i8_matmul[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8
                        if valid_rows == T_TILE:
                            qr_view[out_tg : out_tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8
                        else:
                            qr_q_tail = pl.load(
                                qr_i8_matmul,
                                [tg, qa],
                                [T_TILE, QUANT_TILE],
                                valid_shape=[valid_rows, QUANT_TILE],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            pl.store(qr_q_tail, [out_tg, qa], qr_view)


@pl.jit.inline(auto_scope=False)
def q_proj_q_matmul(
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    q_proj_i32: pl.Tensor[[QPROJ_T_PAD, H * HEAD_DIM], pl.INT32],
    tile_rows: pl.Scalar[pl.INDEX],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    """Project one bounded Q tile and expose its cube task ID."""
    qproj_t_matmul = ((tile_rows + QPROJ_TAIL_M_TILE - 1) // QPROJ_TAIL_M_TILE) * QPROJ_TAIL_M_TILE
    qproj_full_rows = (tile_rows // QPROJ_M_TILE) * QPROJ_M_TILE
    with pl.spmd(
        QPROJ_WORKERS,
        name_hint="qproj_matmul",
        deps=[qproj_dep],
    ) as qproj_tid:
        # Weight reads bypass L2.
        pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
        qproj_worker = pl.tile.get_block_idx()
        for qproj_n_idx in pl.range(
            qproj_worker,
            (H * HEAD_DIM) // QPROJ_MM_N_TILE,
            QPROJ_WORKERS,
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
def q_proj_q_dequant(
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, H, HEAD_DIM], pl.BF16],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    q_proj_i32: pl.Tensor[[QPROJ_T_PAD, H * HEAD_DIM], pl.INT32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Dequantize, normalize, and rotate one projected Q tile."""
    t_dim = pl.tensor.dim(q, 0)
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
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
            dq_head_tile = H
        elif (
            tile_rows >= Q_DEQUANT_WORKERS * Q_ROPE_T_TILE * 16 // H
            and tile_rows % (Q_DEQUANT_WORKERS * Q_ROPE_T_TILE * 16 // H) == 0
        ):
            dq_head_tile = 16
        for dq_work in pl.range(
            dq_worker,
            ((tile_rows + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE) * (H // dq_head_tile),
            Q_DEQUANT_WORKERS,
        ):
            hg = (dq_work % (H // dq_head_tile)) * dq_head_tile
            tg = (dq_work // (H // dq_head_tile)) * Q_ROPE_T_TILE
            out_tg = tile_base + tg
            if tg + Q_ROPE_T_TILE <= tile_rows:
                qr_scale_dq_t = qr_scale_pad_store[tg : tg + Q_ROPE_T_TILE, :]
                q_cos_il = rope_cos_il[out_tg : out_tg + Q_ROPE_T_TILE, :]
                q_sin_signed = rope_sin_signed[out_tg : out_tg + Q_ROPE_T_TILE, :]
                q_swap_idx = rope_swap_idx[out_tg : out_tg + Q_ROPE_T_TILE, :]
                for h_inner in pl.pipeline(dq_head_tile, stage=2):
                    h = hg + h_inner
                    h0 = h * HEAD_DIM
                    q_head_acc = q_proj_i32[tg : tg + Q_ROPE_T_TILE, h0 : h0 + HEAD_DIM]
                    q_head_scale = pl.reshape(wq_b_scale[h0 : h0 + HEAD_DIM], [1, HEAD_DIM])
                    q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                    q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                    q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                    q_head_sq = pl.mul(q_head_dq, q_head_dq)
                    q_head_sq_row = pl.row_sum(q_head_sq)
                    q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                    q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                    q_head_var = pl.add(q_head_sq_mean, EPS)
                    q_head_inv_rms = pl.rsqrt(q_head_var, high_precision=True)
                    q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])

                    q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                    q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = q_nope_bf16

                    q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                    q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                    q_rope_swapped = pl.gather(q_rope_chunk, dim=-1, index=q_swap_idx)
                    q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                    q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                    q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                    q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + HEAD_DIM] = q_rope_bf16
            else:
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


@pl.jit.inline(auto_scope=False)
def q_proj_q(
    x: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, D], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, H, HEAD_DIM], pl.BF16],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    """Q projection and its dequant + RMSNorm + RoPE over bounded tiles."""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            q_proj_i32 = pl.create_tensor([QPROJ_T_PAD, H * HEAD_DIM], dtype=pl.INT32)
            _qproj_tid = q_proj_q_matmul(
                wq_b,
                qr_i8_matmul,
                q_proj_i32,
                tile_rows,
                qproj_dep,
            )
            q_proj_q_dequant(
                wq_b_scale,
                rope_cos_il,
                rope_sin_signed,
                rope_swap_idx,
                q,
                qr_scale_pad_store,
                q_proj_i32,
                tile_base,
                tile_rows,
            )
    return q


@pl.jit.inline(auto_scope=False)
def q_proj_rope(
    x: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    rope_cos_il: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, H, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[QKV_PROJ_ROPE_T_DYN, 1], pl.FP32],
):
    """Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles."""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            # Consume each tile's QR and scales before staging the next tile.
            x_tile = pl.slice(x, [tile_rows, D], [tile_base, 0])
            qr_tile = pl.slice(qr, [tile_rows, Q_LORA], [tile_base, 0])
            qr_scale_tile = pl.slice(qr_scale, [tile_rows, 1], [tile_base, 0])
            q_tile = pl.slice(q, [tile_rows, H, HEAD_DIM], [tile_base, 0, 0])
            cos_tile = pl.slice(rope_cos_il, [tile_rows, ROPE_DIM], [tile_base, 0])
            sin_tile = pl.slice(rope_sin_signed, [tile_rows, ROPE_DIM], [tile_base, 0])
            swap_tile = pl.slice(rope_swap_idx, [tile_rows, ROPE_DIM], [tile_base, 0])
            qr_i8_matmul = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
            # The quant scale rides the qr_i8 -> qproj_matmul -> dequant chain.
            qr_scale_pad_store = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32, manual_dep=True)
            q_proj_qr(x_tile, wq_a, gamma_cq, qr_tile, qr_scale_tile, qr_i8_matmul, qr_scale_pad_store)
            q_seq_dep = pl.system.task_dummy(deps=[])
            q_proj_q(
                x_tile,
                wq_b,
                wq_b_scale,
                cos_tile,
                sin_tile,
                swap_tile,
                q_tile,
                qr_i8_matmul,
                qr_scale_pad_store,
                q_seq_dep,
            )


_TP_CHOICES = (1, 2, 4)

_TP_DEFAULT = 2


def _parse_tp_argv():
    for index, arg in enumerate(sys.argv):
        if arg == "--tp" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--tp="):
            return int(arg.split("=", 1)[1])
    return _TP_DEFAULT


DECODE_O_PROJ_TP_SIZE = _parse_tp_argv()

if DECODE_O_PROJ_TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {DECODE_O_PROJ_TP_SIZE})")

config.TP = DECODE_O_PROJ_TP_SIZE







O_LORA = M.o_lora_rank

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

LOCAL_T = DECODE_TOKENS // DECODE_O_PROJ_TP_SIZE

LOCAL_O_GROUPS = O_GROUPS // DECODE_O_PROJ_TP_SIZE

LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA

DECODE_O_PROJ_T_DYN = pl.dynamic("NATIVE_D_T_DYN")

TOKEN_TILE = 16

COMM_ROW_TILE = 8

ATTENTION_PUBLISH_WORKERS = 48

O_RS_REDUCE_WORKERS = 48  # 128 row blocks; 8 left 40 of 48 AIV idle

O_RS_PUBLISH_WORKERS = 32

O_RS_DEQUANT_WORKERS = 12  # per owner; 4 owners -> one AIV wave

O_RS_PUT_T_TILE = 8

O_RS_D_TILE = gcd(4096, D)

LOCAL_T_PAD = (LOCAL_T + TOKEN_TILE - 1) // TOKEN_TILE * TOKEN_TILE

T_PAD = LOCAL_T_PAD

GROUP_T_PAD = DECODE_O_PROJ_TP_SIZE * LOCAL_T_PAD

ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD

O_WINDOW_ROWS = DECODE_O_PROJ_TP_SIZE * LOCAL_T_PAD


def _token_tile(capacity, ceiling):
    """Largest TOKEN_TILE multiple up to `ceiling` that divides `capacity`."""
    for tile in range(min(ceiling, capacity) // TOKEN_TILE * TOKEN_TILE, 0, -TOKEN_TILE):
        if capacity % tile == 0:
            return tile
    return TOKEN_TILE


PROJ_B_MM_T_TILE = _token_tile(T_PAD, 128)

O_A_T_TILE = 128

O_A_K_TILE = 256

O_A_N_TILE = 128

QUANT_T_TILE = 8

O_A_QUANT_WORKERS = 6  # per owner-group; 4 x 2 x 6 -> one AIV wave

O_B_T_TILE = _token_tile(LOCAL_T_PAD, 128)

O_B_K_TILE = 256

O_B_N_TILE = 256

O_B_D_TILE = 512

ACT_T_TILE = 16

ACT_N_TILE = 512

if DECODE_TOKENS % DECODE_O_PROJ_TP_SIZE != 0:
    raise ValueError(f"decode tokens {DECODE_TOKENS} must be divisible by TP size {DECODE_O_PROJ_TP_SIZE}")

if O_GROUPS % DECODE_O_PROJ_TP_SIZE != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {DECODE_O_PROJ_TP_SIZE}")

if O_GROUP_IN % O_A_K_TILE != 0:
    raise ValueError(f"O-A input {O_GROUP_IN} must be divisible by K tile {O_A_K_TILE}")

if O_LORA % O_A_N_TILE != 0:
    raise ValueError(f"O-A output {O_LORA} must be divisible by N tile {O_A_N_TILE}")

if O_LORA % O_B_K_TILE != 0:
    raise ValueError(f"O-B group width {O_LORA} must be divisible by K tile {O_B_K_TILE}")

if D % O_B_D_TILE != 0 or O_B_D_TILE % O_B_N_TILE != 0:
    raise ValueError("O-B output tiles must divide the hidden dimension")

if D % ACT_N_TILE != 0:
    raise ValueError(f"O-B activation tile {ACT_N_TILE} must divide hidden size {D}")

if D % O_RS_D_TILE != 0:
    raise ValueError(f"O-B ReduceScatter tile {O_RS_D_TILE} must divide hidden size {D}")

if O_RS_REDUCE_WORKERS % DECODE_O_PROJ_TP_SIZE != 0:
    raise ValueError(f"TP size {DECODE_O_PROJ_TP_SIZE} must divide O-B ReduceScatter workers {O_RS_REDUCE_WORKERS}")

if O_RS_PUBLISH_WORKERS % DECODE_O_PROJ_TP_SIZE != 0:
    raise ValueError(f"TP size {DECODE_O_PROJ_TP_SIZE} must divide O-B publish workers {O_RS_PUBLISH_WORKERS}")

if LOCAL_T_PAD % O_B_T_TILE != 0:
    raise ValueError(f"O-B token tile {O_B_T_TILE} must divide local token capacity {LOCAL_T_PAD}")

if GROUP_T_PAD % O_B_T_TILE != 0:
    raise ValueError(f"O-B token tile {O_B_T_TILE} must divide token capacity {GROUP_T_PAD}")

if T_PAD % PROJ_B_MM_T_TILE != 0:
    raise ValueError(f"proj_b_mm token tile {PROJ_B_MM_T_TILE} must divide token capacity {T_PAD}")


@pl.jit.incore
def o_group_a2a_gather(
    local_groups_out: pl.Out[pl.Tensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16]],
    exchange_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    group_t: pl.Scalar[pl.INT32],
):
    """Copy exchanged attention groups into the local projection buffer."""
    worker = pl.tile.get_block_idx()
    for local_group in pl.range(LOCAL_O_GROUPS):
        group_base_row = local_group * GROUP_T_PAD
        for group_row in pl.range(worker, group_t, ATTENTION_PUBLISH_WORKERS):
            copy_row = group_base_row + group_row
            local_groups_out[
                copy_row : copy_row + 1,
                0:O_GROUP_IN,
            ] = exchange_window[
                copy_row : copy_row + 1,
                0:O_GROUP_IN,
            ]
    return local_groups_out


@pl.jit.inline
def o_group_a2a(
    local_groups_out: pl.Tensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    exchange_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    exchange_signal: pld.DistributedTensor[[DECODE_O_PROJ_TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    publish_dep: pl.Scalar[pl.TASK_ID],
    publish_count: pl.Scalar[pl.INT32],
):
    """Finish a non-overlapping producer-fused exchange and release its window."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_group_a2a_wait", deps=[publish_dep]) as wait_tid:
        expected = pl.cast(publish_count, pl.INT32)
        for source_tp in pl.range(DECODE_O_PROJ_TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=exchange_signal, offsets=[source_tp, 0], expected=expected, cmp=pld.WaitCmp.Ge
                )

    group_t = DECODE_O_PROJ_TP_SIZE * local_t
    # Specialize the explicit SPMD callee on a scratch output.
    if False:
        gather_specialize = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        o_group_a2a_gather(gather_specialize, exchange_window, group_t)
    local_groups_out, gather_tid = pl.spmd_submit(
        self.o_group_a2a_gather,  # noqa: F821 - materialized by @pl.jit
        local_groups_out,
        pl.no_dep(exchange_window),
        group_t,
        core_num=ATTENTION_PUBLISH_WORKERS,
        deps=[wait_tid],
    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="o_group_a2a_complete",
        deps=[gather_tid],
        no_dep_args=[exchange_signal],
    ):
        completion_anchor = pl.read(local_groups_out, [0, 0])
        for peer_tp in pl.range(DECODE_O_PROJ_TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=exchange_signal,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(publish_count + 1, pl.INT32)
        for source_tp in pl.range(DECODE_O_PROJ_TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=exchange_signal,
                    offsets=[source_tp, 0],
                    expected=completion_expected,
                    cmp=pld.WaitCmp.Ge,
                )

        reset_value = pl.cast(-completion_expected, pl.INT32)
        self_rank = group_base + tp_rank
        for source_tp in pl.range(DECODE_O_PROJ_TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=exchange_signal,
                    peer=self_rank,
                    offsets=[source_tp, 0],
                    value=reset_value,
                    op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(local_groups_out, [0, 0], completion_anchor)
    return local_groups_out, exchange_signal


@pl.jit.incore
def tp_o_rs_reduce(
    local_out: pl.Out[pl.Tensor[[DECODE_O_PROJ_T_DYN, D], pl.BF16]],
    reduce_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    local_t: pl.Scalar[pl.INT32],
):
    """Sum the received O-projection partials for the local tokens."""
    worker = pl.tile.get_block_idx()
    for block in pl.range(worker, local_t * (D // O_RS_D_TILE), O_RS_REDUCE_WORKERS):
        local_row = block // (D // O_RS_D_TILE)
        d_block = block - local_row * (D // O_RS_D_TILE)
        d0 = d_block * O_RS_D_TILE
        own_partial = pl.load(reduce_window, [local_row, d0], [1, O_RS_D_TILE])
        reduce_acc = pl.cast(own_partial, target_type=pl.FP32, mode="none")
        for source_tp in pl.range(1, DECODE_O_PROJ_TP_SIZE):
            source_row = source_tp * LOCAL_T_PAD + local_row
            source_partial = pl.load(reduce_window, [source_row, d0], [1, O_RS_D_TILE])
            source_fp32 = pl.cast(source_partial, target_type=pl.FP32, mode="none")
            reduce_acc = pl.add(reduce_acc, source_fp32)
        reduced = pl.cast(reduce_acc, target_type=pl.BF16, mode="rint")
        pl.store(reduced, [local_row, d0], local_out)
    return local_out


@pl.jit.inline
def o_proj_reduce_scatter(
    attention_local_groups: pl.Tensor[[LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_GROUPS, D, O_LORA], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    local_t: pl.Scalar[pl.INT32],
    local_out: pl.Tensor[[DECODE_O_PROJ_T_DYN, D], pl.BF16],
    reduce_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    reduce_signal: pld.DistributedTensor[[DECODE_O_PROJ_TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Project O-B tiles directly into their ReduceScatter owner windows."""
    group_t = DECODE_O_PROJ_TP_SIZE * local_t
    o_a_rows = (group_t + O_A_T_TILE - 1) // O_A_T_TILE
    o_b_rows = (group_t + O_B_T_TILE - 1) // O_B_T_TILE
    o_b_group_t = o_b_rows * O_B_T_TILE
    owner_rows = (local_t + ACT_T_TILE - 1) // ACT_T_TILE

    attn_2d = pl.reshape(attention_local_groups, [LOCAL_O_GROUPS * GROUP_T_PAD, O_GROUP_IN])
    # Owner-private buffers and group-local A -> quant -> B dependencies.
    publish_all = pl.create_tensor([O_WINDOW_ROWS, D], dtype=pl.BF16)
    put_rows = (local_t + O_RS_PUT_T_TILE - 1) // O_RS_PUT_T_TILE
    own_a_rows = (local_t + O_A_T_TILE - 1) // O_A_T_TILE
    own_b_rows = (local_t + O_B_T_TILE - 1) // O_B_T_TILE
    own_b_t = own_b_rows * O_B_T_TILE
    own_quant_blocks = (local_t + QUANT_T_TILE - 1) // QUANT_T_TILE
    own_pad_blocks = (own_b_t + QUANT_T_TILE - 1) // QUANT_T_TILE
    own_act_rows = (local_t + ACT_T_TILE - 1) // ACT_T_TILE

    for owner in pl.parallel(DECODE_O_PROJ_TP_SIZE):
        own_base = owner * local_t
        own_a_fp32 = pl.create_tensor([LOCAL_T_PAD, LOCAL_O_WIDTH], dtype=pl.FP32, manual_dep=True)
        own_a_i8 = pl.create_tensor([LOCAL_T_PAD, LOCAL_O_WIDTH], dtype=pl.INT8, manual_dep=True)
        # The quant scale rides the own_a_i8 -> tp_o_b -> dequant chain.
        own_scale = pl.create_tensor([LOCAL_O_GROUPS, LOCAL_T_PAD], dtype=pl.FP32, manual_dep=True)
        own_b_i32 = pl.create_tensor([LOCAL_T_PAD, LOCAL_O_GROUPS * D], dtype=pl.INT32)

        for local_group in pl.parallel(LOCAL_O_GROUPS):
            attention_row = local_group * GROUP_T_PAD + own_base
            o_a_col = local_group * O_LORA

            with pl.spmd(own_a_rows * (O_LORA // O_A_N_TILE), name_hint="tp_o_a") as pa_tid:
                # Weight reads bypass L2.
                pl.set_cache_policy(wo_a, pl.CachePolicy.BYPASS)
                pa_unit = pl.tile.get_block_idx()
                pa_rb = pa_unit // (O_LORA // O_A_N_TILE)
                pa_nb = pa_unit % (O_LORA // O_A_N_TILE)
                pa_t0 = pa_rb * O_A_T_TILE
                pa_n0 = pa_nb * O_A_N_TILE
                pa_rows = pl.min(O_A_T_TILE, local_t - pa_t0)
                pa_src = attention_row + pa_t0
                pa_wrow = o_a_col + pa_n0
                pa_x0 = pl.slice(
                    attn_2d, [O_A_T_TILE, O_A_K_TILE], [pa_src, 0], valid_shape=[pa_rows, O_A_K_TILE]
                )
                pa_w0 = wo_a[local_group : local_group + 1, pa_n0 : pa_n0 + O_A_N_TILE, 0:O_A_K_TILE]
                pa_acc = pl.matmul(pa_x0, pa_w0, b_trans=True, out_dtype=pl.FP32)
                for pa_k0 in pl.pipeline(O_A_K_TILE, O_GROUP_IN, O_A_K_TILE, stage=2):
                    pa_xk = pl.slice(
                        attn_2d, [O_A_T_TILE, O_A_K_TILE], [pa_src, pa_k0], valid_shape=[pa_rows, O_A_K_TILE]
                    )
                    pa_wk = wo_a[
                        local_group : local_group + 1, pa_n0 : pa_n0 + O_A_N_TILE, pa_k0 : pa_k0 + O_A_K_TILE
                    ]
                    pa_acc = pl.matmul_acc(pa_acc, pa_xk, pa_wk, b_trans=True)
                pa_acc_2d = pl.reshape(pa_acc, [O_A_T_TILE, O_A_N_TILE])
                pa_valid = pl.set_validshape(pa_acc_2d, pa_rows, O_A_N_TILE)
                own_a_fp32[pa_t0 : pa_t0 + O_A_T_TILE, pa_wrow : pa_wrow + O_A_N_TILE] = pa_valid

            with pl.spmd(O_A_QUANT_WORKERS, name_hint="tp_o_a_quant", deps=[pa_tid]) as q_tid:
                qz_worker = pl.tile.get_block_idx()
                for qz_blk in pl.range(qz_worker, own_quant_blocks, O_A_QUANT_WORKERS):
                    qz_t = qz_blk * QUANT_T_TILE
                    qz_rows = pl.min(QUANT_T_TILE, local_t - qz_t)
                    qz_tile = pl.slice(own_a_fp32, [QUANT_T_TILE, O_LORA], [qz_t, o_a_col])
                    qz_amax = pl.reshape(pl.row_max(pl.abs(qz_tile)), [1, QUANT_T_TILE])
                    qz_floor = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    qz_amax = pl.maximum(qz_floor, qz_amax)
                    qz_max = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    qz_sq = pl.div(qz_max, qz_amax)
                    qz_sdq = pl.recip(qz_sq)
                    own_scale[local_group : local_group + 1, qz_t : qz_t + QUANT_T_TILE] = pl.set_validshape(
                        qz_sdq, 1, qz_rows
                    )
                    qz_sq_col = pl.reshape(qz_sq, [QUANT_T_TILE, 1])
                    qz_scaled = pl.row_expand_mul(qz_tile, qz_sq_col)
                    qz_i32 = pl.cast(qz_scaled, target_type=pl.INT32, mode="rint")
                    qz_f16 = pl.cast(qz_i32, target_type=pl.FP16, mode="round")
                    qz_i8 = pl.cast(qz_f16, target_type=pl.INT8, mode="trunc")
                    own_a_i8[qz_t : qz_t + QUANT_T_TILE, o_a_col : o_a_col + O_LORA] = pl.set_validshape(
                        qz_i8, qz_rows, O_LORA
                    )
                for qz_pad in pl.range(own_quant_blocks + qz_worker, own_pad_blocks, O_A_QUANT_WORKERS):
                    qz_pt = qz_pad * QUANT_T_TILE
                    qz_prows = pl.min(QUANT_T_TILE, own_b_t - qz_pt)
                    qz_zero = pl.full([QUANT_T_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                    qz_zero_i8 = pl.cast(qz_zero, target_type=pl.INT8, mode="trunc")
                    own_a_i8[qz_pt : qz_pt + QUANT_T_TILE, o_a_col : o_a_col + O_LORA] = pl.set_validshape(
                        qz_zero_i8, qz_prows, O_LORA
                    )

            with pl.spmd(own_b_rows * (D // O_B_D_TILE), name_hint="tp_o_b", deps=[q_tid]):
                # Weight reads bypass L2.
                pl.set_cache_policy(wo_b, pl.CachePolicy.BYPASS)
                pb_unit = pl.tile.get_block_idx()
                pb_tb = pb_unit // (D // O_B_D_TILE)
                pb_db = pb_unit % (D // O_B_D_TILE)
                pb_t0 = pb_tb * O_B_T_TILE
                pb_d0 = pb_db * O_B_D_TILE
                for pb_n0 in pl.range(pb_d0, pb_d0 + O_B_D_TILE, O_B_N_TILE):
                    pb_x0 = own_a_i8[pb_t0 : pb_t0 + O_B_T_TILE, o_a_col : o_a_col + O_B_K_TILE]
                    pb_w0 = wo_b[local_group : local_group + 1, pb_n0 : pb_n0 + O_B_N_TILE, 0:O_B_K_TILE]
                    pb_acc = pl.matmul(pb_x0, pb_w0, b_trans=True, out_dtype=pl.INT32)
                    for pb_k0 in pl.pipeline(O_B_K_TILE, O_LORA, O_B_K_TILE, stage=2):
                        pb_bk = o_a_col + pb_k0
                        pb_xk = own_a_i8[pb_t0 : pb_t0 + O_B_T_TILE, pb_bk : pb_bk + O_B_K_TILE]
                        pb_wk = wo_b[
                            local_group : local_group + 1,
                            pb_n0 : pb_n0 + O_B_N_TILE,
                            pb_k0 : pb_k0 + O_B_K_TILE,
                        ]
                        pb_acc = pl.matmul_acc(pb_acc, pb_xk, pb_wk, b_trans=True)
                    pb_acc_2d = pl.reshape(pb_acc, [O_B_T_TILE, O_B_N_TILE])
                    pb_col = local_group * D + pb_n0
                    own_b_i32[pb_t0 : pb_t0 + O_B_T_TILE, pb_col : pb_col + O_B_N_TILE] = pb_acc_2d

        with pl.spmd(
            O_RS_DEQUANT_WORKERS,
            name_hint="tp_o_b_dequant",
            optimizations=[pl.cross_core_slot(slot_num=2)],
        ):
            dq_worker = pl.tile.get_block_idx()
            for dq_blk in pl.range(dq_worker, own_act_rows * (D // ACT_N_TILE), O_RS_DEQUANT_WORKERS):
                dq_rb = dq_blk // (D // ACT_N_TILE)
                dq_nb = dq_blk - dq_rb * (D // ACT_N_TILE)
                dq_row = dq_rb * ACT_T_TILE
                dq_n0 = dq_nb * ACT_N_TILE
                dq_rows = pl.min(ACT_T_TILE, local_t - dq_row)
                dq_acc = pl.full([ACT_T_TILE, ACT_N_TILE], dtype=pl.FP32, value=0.0)
                for dq_group in pl.pipeline(LOCAL_O_GROUPS, stage=2):
                    dq_col = dq_group * D + dq_n0
                    dq_i32 = pl.slice(
                        own_b_i32,
                        [ACT_T_TILE, ACT_N_TILE],
                        [dq_row, dq_col],
                        valid_shape=[dq_rows, ACT_N_TILE],
                    )
                    dq_fp32 = pl.cast(dq_i32, target_type=pl.FP32, mode="none")
                    dq_srow = pl.slice(
                        own_scale, [1, ACT_T_TILE], [dq_group, dq_row], valid_shape=[1, dq_rows]
                    )
                    dq_scol = pl.reshape(dq_srow, [ACT_T_TILE, 1])
                    dq_acc = pl.add(dq_acc, pl.row_expand_mul(dq_fp32, dq_scol))
                dq_wscale = pl.reshape(wo_b_scale[dq_n0 : dq_n0 + ACT_N_TILE], [1, ACT_N_TILE])
                dq_bf16 = pl.cast(pl.col_expand_mul(dq_acc, dq_wscale), target_type=pl.BF16, mode="rint")
                dq_stage = owner * LOCAL_T_PAD + dq_row
                publish_all[dq_stage : dq_stage + ACT_T_TILE, dq_n0 : dq_n0 + ACT_N_TILE] = pl.set_validshape(
                    dq_bf16, dq_rows, ACT_N_TILE
                )

    with pl.spmd(
        O_RS_PUBLISH_WORKERS,
        name_hint="tp_o_b_publish",
    ) as publish_tid:
        pub_worker = pl.tile.get_block_idx()
        # Flatten (owner, row block) into one work list: put_rows alone is under
        # the worker count, so an owner-outer loop leaves a third of the workers
        # idle while the rest each issue TP_SIZE puts.
        for pub_blk in pl.range(pub_worker, DECODE_O_PROJ_TP_SIZE * put_rows, O_RS_PUBLISH_WORKERS):
            pub_owner = pub_blk // put_rows
            pub_row_block = pub_blk - pub_owner * put_rows
            pub_owner_row = pub_row_block * O_RS_PUT_T_TILE
            pub_rows = pl.min(O_RS_PUT_T_TILE, local_t - pub_owner_row)
            pub_src_row = pub_owner * LOCAL_T_PAD + pub_owner_row
            pub_dst_row = tp_rank * LOCAL_T_PAD + pub_owner_row
            pld.tensor.put(
                dst=reduce_window,
                peer=group_base + pub_owner,
                src=publish_all,
                dst_offsets=[pub_dst_row, 0],
                src_offsets=[pub_src_row, 0],
                shape=[pub_rows, D],
                chunk_rows=O_RS_PUT_T_TILE,
                chunk_cols=D,
            )

        for notify_owner in pl.range(DECODE_O_PROJ_TP_SIZE):
            if notify_owner != tp_rank:
                pld.system.notify(
                    target=reduce_signal,
                    peer=group_base + notify_owner,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="tp_o_rs_wait", deps=[publish_tid]) as wait_tid:
        expected = pl.cast(O_RS_PUBLISH_WORKERS, pl.INT32)
        for source_tp in pl.range(DECODE_O_PROJ_TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=reduce_signal, offsets=[source_tp, 0], expected=expected, cmp=pld.WaitCmp.Ge
                )

    # Specialize the explicit SPMD callee on a scratch output.
    if False:
        reduce_output_rows = pl.tensor.dim(local_out, 0)
        reduce_specialize = pl.create_tensor([reduce_output_rows, D], dtype=pl.BF16)
        tp_o_rs_reduce(reduce_specialize, reduce_window, local_t)
    local_out, reduce_tid = pl.spmd_submit(
        self.tp_o_rs_reduce,  # noqa: F821 - materialized by @pl.jit
        local_out,
        pl.no_dep(reduce_window),
        local_t,
        core_num=O_RS_REDUCE_WORKERS,
        deps=[wait_tid],
    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="tp_o_rs_complete",
        deps=[reduce_tid],
        no_dep_args=[reduce_signal],
    ):
        completion_anchor = pl.read(local_out, [0, 0])
        for peer_tp in pl.range(DECODE_O_PROJ_TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=reduce_signal,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

        completion_expected = pl.cast(O_RS_PUBLISH_WORKERS + 1, pl.INT32)
        for source_tp in pl.range(DECODE_O_PROJ_TP_SIZE):
            if source_tp != tp_rank:
                pld.system.wait(
                    signal=reduce_signal,
                    offsets=[source_tp, 0],
                    expected=completion_expected,
                    cmp=pld.WaitCmp.Ge,
                )

        reset_value = pl.cast(-(O_RS_PUBLISH_WORKERS + 1), pl.INT32)
        self_rank = group_base + tp_rank
        for source_tp in pl.range(DECODE_O_PROJ_TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=reduce_signal,
                    peer=self_rank,
                    offsets=[source_tp, 0],
                    value=reset_value,
                    op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(local_out, [0, 0], completion_anchor)
    return local_out, reduce_signal


def _parse_int_argv(flag: str, default: int) -> int:
    for index, arg in enumerate(sys.argv):
        if arg == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return default


PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE = _parse_int_argv("--tp", _TP_DEFAULT)

if PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE})")




CP_Q_T_DYN = pl.dynamic("NATIVE_D_PREFILL_CP_Q_T_DYN")

CP_KV_T_DYN = pl.dynamic("NATIVE_D_PREFILL_CP_KV_T_DYN")


PREFILL_GROUP_CAP = 8192


READBACK_ROW_TILE = 16

READBACK_WORKERS = 16


def _prefill_cp_token_allgather_step(
    hidden_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    group_out: pl.Out[pl.Tensor[[CP_KV_T_DYN, D], pl.BF16]],
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    prior_cycle_gate: pl.Scalar[pl.TASK_ID],
):
    """Gather rank-major rows and retire the complete two-phase signal epoch."""
    hidden_local.bind_dynamic(0, CP_Q_T_DYN)
    group_out.bind_dynamic(0, CP_KV_T_DYN)
    local_rows = pl.tensor.dim(hidden_local, 0)
    local_t = pl.cast(local_rows, pl.INT32)
    target_row = tp_rank * local_t

    # Each peer has an independent destination and arrival signal.
    with pl.spmd(
        PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE,
        name_hint="prefill_cp_token_allgather_push",
        allow_early_resolve=True,
        deps=[prior_cycle_gate],
    ) as _push_tid:
        peer_tp = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=gather_window,
            peer=group_base + peer_tp,
            src=hidden_local,
            dst_offsets=[target_row, 0],
            src_offsets=[0, 0],
            shape=[local_t, D],
            chunk_rows=COMM_ROW_TILE,
            chunk_cols=D,
            pipeline=True,
        )
        if peer_tp != tp_rank:
            pld.system.notify(
                target=gather_signal,
                peer=group_base + peer_tp,
                offsets=[tp_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

    # Register the peer payload conditions as deferred completion.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_payload_wait",
        deps=[prior_cycle_gate],
    ) as _payload_wait_tid:
        for source_tp in pl.range(PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal,
                    offsets=[source_tp, 0],
                    expected=pl.cast(1, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Fixed readers own strided row tiles; release the window after every reader.
    group_rows = PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE * local_rows
    full_rows = (group_rows // READBACK_ROW_TILE) * READBACK_ROW_TILE
    with pl.spmd(
        READBACK_WORKERS,
        name_hint="prefill_cp_token_allgather_readback",
        deps=[_push_tid, _payload_wait_tid],
    ) as _readback_tid:
        worker = pl.tile.get_block_idx()
        for tile_row in pl.range(
            worker * READBACK_ROW_TILE,
            full_rows,
            READBACK_WORKERS * READBACK_ROW_TILE,
        ):
            window_tile = gather_window[tile_row : tile_row + READBACK_ROW_TILE, 0:D]
            group_out[tile_row : tile_row + READBACK_ROW_TILE, 0:D] = window_tile
        for tail_row in pl.range(full_rows + worker, group_rows, READBACK_WORKERS):
            window_row = gather_window[tail_row : tail_row + 1, 0:D]
            group_out[tail_row : tail_row + 1, 0:D] = window_row
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_readback_notify",
        deps=[_readback_tid],
    ) as _readback_notify_tid:
        for peer_tp in pl.range(PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_readback_wait",
        deps=[prior_cycle_gate],
    ) as _readback_wait_tid:
        for source_tp in pl.range(PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=gather_signal,
                    offsets=[source_tp, 0],
                    expected=pl.cast(2, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Retire peer credits and anchor output consumption to signal retirement.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_token_allgather_retire",
        deps=[_readback_notify_tid, _readback_wait_tid],
    ):
        completion_anchor = pl.read(group_out, [0, 0])
        reset_value = pl.cast(-2, pl.INT32)
        for source_tp in pl.range(PREFILL_CP_TOKEN_ALLGATHER_TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=gather_signal,
                    peer=group_base + tp_rank,
                    offsets=[source_tp, 0],
                    value=reset_value,
                    op=pld.NotifyOp.AtomicAdd,
                )
        pl.write(group_out, [0, 0], completion_anchor)

    return group_out, gather_signal


prefill_cp_token_allgather_step = pl.jit.inline(_prefill_cp_token_allgather_step)


ORI_BLOCK_NUM_DYN = pl.dynamic("NATIVE_D_DSPARK_ATTENTION_ORI_BLOCK_NUM_DYN")

KV_T_DYN = pl.dynamic("NATIVE_D_DSPARK_ATTENTION_KV_T_DYN")

B = MOE_TOKENS // DECODE_SEQ

S = 5  # anchor-first draft query rows per request

T = B * S







WIN = M.sliding_window



SOFTMAX_SCALE = M.softmax_scale

VISIBLE_ROWS = WIN + S  # trailing window + whole draft block

ATTN_K_TILE = 64

NUM_QK_CORES = 24

QK_PRE_LAUNCH = 2

QK_TRANSFER_SLOTS = QK_PRE_LAUNCH + 1

QK_SCORE_READY_EVENT = 0

QK_PROB_READY_EVENT = 1

QK_PV_READY_EVENT = 2

SPARSE_BLOCKS = (VISIBLE_ROWS + ATTN_K_TILE - 1) // ATTN_K_TILE

INDEX_WIDTH = SPARSE_BLOCKS * ATTN_K_TILE

BIAS_B_TILE = 8  # 2 request blocks

H_TILE = 16  # 4 head blocks

NEG_INF = -1.0e20


@pl.jit.inline
def dspark_attention(
    x: pl.Tensor[[T, D], pl.BF16],
    kv_x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16, pl.NZ],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos_local: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin_local: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_position_ids: pl.Tensor[[KV_T_DYN], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    kv_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[B, INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    o_packed_heads: pl.Tensor[[O_GROUPS * LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16],
):
    kv_tokens = pl.tensor.dim(kv_position_ids, 0)
    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([T, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(
        freqs_cos_local,
        freqs_sin_local,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
    )

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    q_proj_rope(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        gamma_cq,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        q,
        qr,
        qr_scale,
    )

    kv_rope_cos_il = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.FP32)
    kv_rope_sin_signed = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.FP32)
    kv_rope_swap_idx = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(
        freqs_cos,
        freqs_sin,
        kv_rope_cos_il,
        kv_rope_sin_signed,
        kv_rope_swap_idx,
    )

    # DSA-CP keeps Q on its token owner while every rank updates the complete
    # group KV stream before running attention for its local queries.
    late_dep = pl.system.task_dummy(deps=[])
    kv = pl.create_tensor([kv_tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(
        kv_x,
        wkv,
        gamma_ckv,
        kv_rope_cos_il,
        kv_rope_sin_signed,
        kv_rope_swap_idx,
        kv,
        late_dep,
    )

    # Commit the block's own KV and build the visible-length mask in one task; the
    # gather below reads those rows back through swa_indices.
    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([B, INDEX_WIDTH], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_kv_commit_valid_bias"):
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

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    # Save online-softmax state for the fused publish phase after QK/PV.
    sparse_mi = pl.create_tensor([T * H, 1], dtype=pl.FP32)
    sparse_li = pl.create_tensor([T * H, 1], dtype=pl.FP32)
    sparse_oi = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.FP32)
    attn_sink_col = pl.reshape(attn_sink, [H, 1])
    o_packed_flat = pl.reshape(
        o_packed_heads,
        [O_GROUPS * LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM],
    )
    transfer_slots = NUM_QK_CORES * QK_TRANSFER_SLOTS
    transfer_heads = transfer_slots * H
    score_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([transfer_heads, ATTN_K_TILE], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([transfer_heads, HEAD_DIM], dtype=pl.FP32)
    mi_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    li_transfer = pl.create_tensor([transfer_heads, 1], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(NUM_QK_CORES, name_hint="dspark_qk_pv") as qk_tid:
        qk_core = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts_workspace)
        for qk_t in pl.range(qk_core, T, NUM_QK_CORES):
            qk_b = qk_t // S
            qk_q = pl.load(
                q_flat,
                [qk_t * H, 0],
                [H, HEAD_DIM],
                target_memory=pl.MemorySpace.Mat,
            )
            for qk_tick in pl.range(SPARSE_BLOCKS + QK_PRE_LAUNCH):
                if qk_tick < SPARSE_BLOCKS:
                    qk_sb = qk_tick
                    if qk_sb * ATTN_K_TILE < pl.read(swa_lens, [qk_b]):
                        qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                        qk_kv_row = qk_b * INDEX_WIDTH + qk_sb * ATTN_K_TILE
                        qk_transfer_row = qk_slot * H
                        qk_kv = pl.load(
                            visible_kv,
                            [qk_kv_row, 0],
                            [ATTN_K_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        qk_scores = pl.matmul(qk_q, pl.tile.transpose_view(qk_kv), out_dtype=pl.FP32)
                        pl.store(qk_scores, [qk_transfer_row, 0], score_transfer)
                        pl.system.sync_set(
                            QK_SCORE_READY_EVENT,
                            pipe=pl.PipeType.FIX,
                            ffts_mode=2,
                            core_type=pl.KernelType.AIC,
                        )
                if qk_tick >= QK_PRE_LAUNCH:
                    pv_sb = qk_tick - QK_PRE_LAUNCH
                    if pv_sb * ATTN_K_TILE < pl.read(swa_lens, [qk_b]):
                        pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                        pv_kv_row = qk_b * INDEX_WIDTH + pv_sb * ATTN_K_TILE
                        pv_transfer_row = pv_slot * H
                        pl.system.sync_wait(
                            QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC
                        )
                        pv_probability = pl.load(
                            probability_transfer,
                            [pv_transfer_row, 0],
                            [H, ATTN_K_TILE],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_kv = pl.load(
                            visible_kv,
                            [pv_kv_row, 0],
                            [ATTN_K_TILE, HEAD_DIM],
                            target_memory=pl.MemorySpace.Mat,
                        )
                        pv_output = pl.matmul(pv_probability, pv_kv, out_dtype=pl.FP32)
                        pl.store(pv_output, [pv_transfer_row, 0], pv_transfer)
                        pl.system.sync_set(
                            QK_PV_READY_EVENT,
                            pipe=pl.PipeType.FIX,
                            ffts_mode=2,
                            core_type=pl.KernelType.AIC,
                        )

            for qk_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                pl.system.set_ffts(ffts_workspace)
                qk_lane_head = qk_aiv * (H // 2)
                qk_reduce_tmp = pl.create_tile(
                    [H // 2, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                running_m = pl.load(
                    attn_sink_col, [qk_lane_head, 0], [H // 2, 1], target_memory=pl.MemorySpace.Vec
                )
                running_l = pl.tile.muls(running_m, 0.0)
                running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
                for qk_tick, (m_iter, l_iter, left_iter, right_iter) in pl.range(
                    SPARSE_BLOCKS + QK_PRE_LAUNCH,
                    init_values=(running_m, running_l, running_left, running_right),
                ):
                    if qk_tick < SPARSE_BLOCKS:
                        qk_sb = qk_tick
                        if qk_sb * ATTN_K_TILE < pl.read(swa_lens, [qk_b]):
                            qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_sb % QK_TRANSFER_SLOTS
                            qk_transfer_row = qk_slot * H
                            qk_s0 = qk_sb * ATTN_K_TILE
                            pl.system.sync_wait(
                                QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                            )
                            qk_scores_half = pl.load(
                                score_transfer,
                                [qk_transfer_row + qk_lane_head, 0],
                                [H // 2, ATTN_K_TILE],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            qk_bias = pl.load(
                                sparse_bias,
                                [qk_b, qk_s0],
                                [1, ATTN_K_TILE],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            qk_scaled = pl.mul(qk_scores_half, SOFTMAX_SCALE)
                            qk_masked = pl.col_expand_add(qk_scaled, qk_bias)
                            qk_mi = pl.row_max(qk_masked, qk_reduce_tmp)
                            qk_exp = pl.exp(pl.row_expand_sub(qk_masked, qk_mi))
                            qk_li = pl.row_sum(qk_exp, qk_reduce_tmp)
                            qk_probability = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                            pl.store(
                                qk_probability, [qk_transfer_row + qk_lane_head, 0], probability_transfer
                            )
                            pl.store(qk_mi, [qk_transfer_row + qk_lane_head, 0], mi_transfer)
                            pl.store(qk_li, [qk_transfer_row + qk_lane_head, 0], li_transfer)
                            pl.system.sync_set(
                                QK_PROB_READY_EVENT,
                                pipe=pl.PipeType.MTE3,
                                ffts_mode=2,
                                core_type=pl.KernelType.AIV,
                            )
                    if qk_tick >= QK_PRE_LAUNCH:
                        pv_sb = qk_tick - QK_PRE_LAUNCH
                        if pv_sb * ATTN_K_TILE < pl.read(swa_lens, [qk_b]):
                            pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_sb % QK_TRANSFER_SLOTS
                            pv_transfer_row = pv_slot * H
                            pl.system.sync_wait(
                                QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                            )
                            pv_m = pl.load(
                                mi_transfer,
                                [pv_transfer_row + qk_lane_head, 0],
                                [H // 2, 1],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            pv_l = pl.load(
                                li_transfer,
                                [pv_transfer_row + qk_lane_head, 0],
                                [H // 2, 1],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            next_m = pl.maximum(m_iter, pv_m)
                            alpha = pl.exp(pl.sub(m_iter, next_m))
                            beta = pl.exp(pl.sub(pv_m, next_m))
                            next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, pv_l))
                            pv_left = pl.load(
                                pv_transfer,
                                [pv_transfer_row + qk_lane_head, 0],
                                [H // 2, HEAD_DIM // 2],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            next_left = pl.add(
                                pl.row_expand_mul(left_iter, alpha), pl.row_expand_mul(pv_left, beta)
                            )
                            pv_right = pl.load(
                                pv_transfer,
                                [pv_transfer_row + qk_lane_head, HEAD_DIM // 2],
                                [H // 2, HEAD_DIM // 2],
                                target_memory=pl.MemorySpace.Vec,
                            )
                            next_right = pl.add(
                                pl.row_expand_mul(right_iter, alpha), pl.row_expand_mul(pv_right, beta)
                            )
                            m_valid, l_valid, left_valid, right_valid = pl.yield_(
                                next_m, next_l, next_left, next_right
                            )
                        else:
                            m_valid, l_valid, left_valid, right_valid = pl.yield_(
                                m_iter, l_iter, left_iter, right_iter
                            )
                        m_after, l_after, left_after, right_after = pl.yield_(
                            m_valid, l_valid, left_valid, right_valid
                        )
                    else:
                        m_after, l_after, left_after, right_after = pl.yield_(
                            m_iter, l_iter, left_iter, right_iter
                        )
                    running_m, running_l, running_left, running_right = pl.yield_(
                        m_after, l_after, left_after, right_after
                    )
                qk_output_row = qk_t * H + qk_lane_head
                pl.store(running_m, [qk_output_row, 0], sparse_mi)
                pl.store(running_l, [qk_output_row, 0], sparse_li)
                pl.store(running_left, [qk_output_row, 0], sparse_oi)
                pl.store(running_right, [qk_output_row, HEAD_DIM // 2], sparse_oi)

        # Publish this core's completed query rows without launching another task.
        for merge_aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            merge_lane_head0 = merge_aiv * (H // 2)
            merge_swap_col = pl.load(rope_swap_idx, [0, 0], [1, ROPE_DIM])
            merge_swap_offset = pl.add(pl.cast(merge_swap_col, target_type=pl.FP32), NOPE_DIM)
            merge_swap_grid = pl.col_expand_add(
                pl.tile.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0),
                merge_swap_offset,
            )
            merge_row_ids = pl.tile.arange(0, [1, H_TILE], dtype=pl.INT32)
            merge_row_offsets = pl.reshape(
                pl.mul(pl.cast(merge_row_ids, target_type=pl.FP32), HEAD_DIM),
                [H_TILE, 1],
            )
            merge_swap_flat = pl.row_expand_add(merge_swap_grid, merge_row_offsets)
            merge_swap_indices = pl.cast(merge_swap_flat, target_type=pl.INT32)
            merge_gather_tmp = pl.create_tile([H_TILE, ROPE_DIM], dtype=pl.INT32)
            for merge_t in pl.range(qk_core, T, NUM_QK_CORES):
                merge_cos = pl.load(rope_cos_il, [merge_t, 0], [1, ROPE_DIM])
                merge_sin = pl.load(rope_sin_signed, [merge_t, 0], [1, ROPE_DIM])
                for merge_tile in pl.unroll((H // 2) // H_TILE):
                    merge_lane_head = merge_tile * H_TILE
                    merge_head0 = merge_lane_head0 + merge_lane_head
                    merge_partial_row0 = merge_t * H + merge_head0
                    merge_m = pl.load(sparse_mi, [merge_partial_row0, 0], [H_TILE, 1])
                    merge_l = pl.load(sparse_li, [merge_partial_row0, 0], [H_TILE, 1])
                    merge_out = pl.load(sparse_oi, [merge_partial_row0, 0], [H_TILE, HEAD_DIM])
                    merge_sink = pl.load(attn_sink_col, [merge_head0, 0], [H_TILE, 1])
                    merge_sink_exp = pl.exp(pl.sub(merge_sink, merge_m))
                    merge_denom = pl.add(merge_l, merge_sink_exp)
                    merge_normed = pl.row_expand_div(merge_out, merge_denom)
                    merge_normed_bf16 = pl.cast(merge_normed, target_type=pl.BF16, mode="rint")
                    merge_nope = merge_normed_bf16[:, 0:NOPE_DIM]
                    merge_rope = merge_normed[:, NOPE_DIM:HEAD_DIM]
                    merge_swapped = pl.tile.gather(
                        merge_normed,
                        merge_swap_indices,
                        merge_gather_tmp,
                    )
                    inverse_cos = pl.col_expand_mul(merge_rope, merge_cos)
                    inverse_sin = pl.col_expand_mul(merge_swapped, merge_sin)
                    inverse_rope = pl.sub(inverse_cos, inverse_sin)
                    inverse_rope_bf16 = pl.cast(inverse_rope, target_type=pl.BF16, mode="rint")

                    for merge_group in pl.unroll(H_TILE // HEADS_PER_GROUP):
                        group = merge_head0 // HEADS_PER_GROUP + merge_group
                        group_head = merge_group * HEADS_PER_GROUP
                        packed_row = (group * LOCAL_T_PAD + merge_t) * HEADS_PER_GROUP
                        packed_nope = pl.slice(
                            merge_nope,
                            [HEADS_PER_GROUP, NOPE_DIM],
                            [group_head, 0],
                        )
                        packed_rope = pl.slice(
                            inverse_rope_bf16,
                            [HEADS_PER_GROUP, ROPE_DIM],
                            [group_head, 0],
                        )
                        pl.store(packed_nope, [packed_row, 0], o_packed_flat)
                        pl.store(packed_rope, [packed_row, NOPE_DIM], o_packed_flat)

    return kv_cache, o_packed_heads
