# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Q/KV LoRA + RoPE (dynamic shape): projects token-major
attention-normalized inputs for both decode and prefill attention paths."""

import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    TP,
    PREFILL_BATCH,
    PREFILL_SEQ,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)


# Dynamic shape variables. The q branch, the kv branch and the rope tables each
# carry their own token axis.
T_DYN = pl.dynamic("QKV_Q_T_DYN")  # T = B * S
KV_T_DYN = pl.dynamic("QKV_KV_T_DYN")
ROPE_T_DYN = pl.dynamic("QKV_ROPE_T_DYN")

# Bounded physical-row tile for Q/KV projection scratch.
PREFILL_DENSE_TILE = 512


# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_DIM_SCALE = float(ROPE_DIM)
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings

# tiling
Q_PROJ_TILE = 128  # qproj K-tile (Q_LORA reduction)
QPROJ_MM_N_TILE = 512  # qproj output-column tile
Q_LORA_TILE = 256  # qr rms-norm / quant N granularity
KV_TILE = 64  # kv rms-norm / rope / NOPE N granularity
QUANT_TILE = 256
T_TILE = 8
MATMUL_T_TILE = 16
QR_M_TILE = MATMUL_T_TILE  # qr_proj token (M) tile; cube rows must be a 16-row boxed tile
QR_N_TILE = 128  # qr_proj Q_LORA (N) per matmul
QR_K_TILE = 256  # qr_proj D (K) reduction tile   | divides QR_SPLIT_K_TILE
QR_OK = 2  # qr_proj split-K factor         | D//QR_OK cores share each N-group
QR_SPLIT_K_TILE = D // QR_OK  # qr_proj K per split (=2048)
KV_M_TILE = MATMUL_T_TILE  # kv_proj token (M) tile; decode pads from 8 real rows to 16
KV_N_TILE = 128  # kv_proj HEAD_DIM (N) per matmul
KV_K_TILE = 256  # kv_proj D (K) reduction tile   | divides KV_SPLIT_K_TILE
KV_OK = 2  # kv_proj split-K factor         | D//KV_OK cores share each N-group
KV_OM = 4  # kv_proj split-M factor        | M-tiles fan out KV_OM-fold
KV_SPLIT_K_TILE = D // KV_OK  # kv_proj K per split (=2048)
QPROJ_M_TILE = 64  # dense qproj token tile; fills the 128 KiB L0C accumulator
QPROJ_TAIL_M_TILE = MATMUL_T_TILE  # partial-M path validated by decode/small physical T
KV_RMS_T_TILE = 16  # kv rms-norm + rope fused token (T) tile
Q_ROPE_T_TILE = 8
# q_rope_prepare runs on persistent workers, capped at the tile count so the
# short call sites do not dispatch more blocks than they have work.
# 24 workers, not one per AIV lane: the task holds only ~273 core-us, so 48
# serial block dispatches cost more than the single wave they buy.
Q_ROPE_WORKERS = 24
Q_ROPE_H_TILE = 4  # heads per fused qproj dequant/rms/rope task
assert QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C Acc cap
assert QPROJ_M_TILE % QPROJ_TAIL_M_TILE == 0


@pl.jit.inline
def materialize_rope_rows_dynamic(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    rope_cos_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
):
    """Materialize every physical position row."""
    t_dim = pl.tensor.dim(position_ids, 0)
    token_tiles = (t_dim + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
    for rope_t0 in pl.spmd(token_tiles, name_hint="qkv_rope_rows_dynamic"):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < t_dim:
                rope_pos = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
                rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]


@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
):
    """Materialize the active prefix of a capacity-shaped position tensor."""
    t_dim = pl.tensor.dim(position_ids, 0)
    token_tiles = (t_dim + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
    for rope_t0 in pl.spmd(token_tiles, name_hint="qkv_rope_rows"):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < t_dim and rope_t < num_tokens:
                rope_pos = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
                rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]


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
def q_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
):
    """Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles."""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            x_view = pl.reshape(x, [t_dim, D])
            qr_t_matmul = ((tile_rows + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
            qproj_t_matmul = ((tile_rows + QPROJ_TAIL_M_TILE - 1) // QPROJ_TAIL_M_TILE) * QPROJ_TAIL_M_TILE
            qproj_full_rows = (tile_rows // QPROJ_M_TILE) * QPROJ_M_TILE

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
                q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
                qr_k_base = (qbg_idx % QR_OK) * QR_SPLIT_K_TILE
                for t0 in pl.range(0, qr_t_matmul, QR_M_TILE):
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
                        if db == 0:
                            q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                        else:
                            q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
                    qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=pl.AtomicType.Add)

            qr_view = pl.reshape(qr, [t_dim, Q_LORA])
            qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
            qr_i8_matmul = pl.create_tensor([qproj_t_matmul, Q_LORA], dtype=pl.INT8)
            # The quant scale rides the qr_i8 -> qproj_matmul -> dequant chain.
            qr_scale_pad_store = pl.create_tensor([qproj_t_matmul, 1], dtype=pl.FP32, manual_dep=True)

            # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
            qr_token_tiles = (tile_rows + T_TILE - 1) // T_TILE
            for tg_idx in pl.spmd(qr_token_tiles, name_hint="qr_rms_norm_quant", allow_early_resolve=True):
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
                    gamma_rms_cast = pl.cast(gamma_cq[qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE], target_type=pl.FP32)
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

                qr_scale_quant_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax)
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
                    qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
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

            # Pure-matmul qproj scope (cube, INT32 -> GM), unmixed with downstream vector work.
            q_proj_i32 = pl.create_tensor([qproj_t_matmul, H * HEAD_DIM], dtype=pl.INT32)
            # Full 64-row cube tiles use the dense path.  The A2/A3 partial-M path is
            # not numerically reliable at this tile size, so the final incomplete
            # 64-row block is lowered through the established 16-row cube shape.
            for qproj_n_idx in pl.spmd(
                (H * HEAD_DIM) // QPROJ_MM_N_TILE,
                name_hint="qproj_matmul",
            ):
                w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
                for t0 in pl.range(0, qproj_full_rows, QPROJ_M_TILE):
                    col_acc = pl.create_tensor([QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
                    for qr_proj_col0 in pl.pipeline(0, Q_LORA, Q_PROJ_TILE, stage=2):
                        qr_i8_chunk = qr_i8_matmul[
                            t0 : t0 + QPROJ_M_TILE,
                            qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE,
                        ]
                        wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                        if qr_proj_col0 == 0:
                            col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                        else:
                            col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
                    q_proj_i32[t0 : t0 + QPROJ_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc

                tail_w_col0 = w_col0
                for tail_t0 in pl.range(qproj_full_rows, qproj_t_matmul, QPROJ_TAIL_M_TILE):
                    qproj_tail_rows = pl.min(QPROJ_TAIL_M_TILE, tile_rows - tail_t0)
                    tail_acc = pl.create_tensor([QPROJ_TAIL_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
                    for tail_qr_col0 in pl.pipeline(0, Q_LORA, Q_PROJ_TILE, stage=2):
                        qr_i8_tail = pl.slice(
                            qr_i8_matmul,
                            [QPROJ_TAIL_M_TILE, Q_PROJ_TILE],
                            [tail_t0, tail_qr_col0],
                            valid_shape=[qproj_tail_rows, Q_PROJ_TILE],
                        )
                        wq_tail = wq_b[
                            tail_qr_col0 : tail_qr_col0 + Q_PROJ_TILE,
                            tail_w_col0 : tail_w_col0 + QPROJ_MM_N_TILE,
                        ]
                        if tail_qr_col0 == 0:
                            tail_acc = pl.matmul(qr_i8_tail, wq_tail, out_dtype=pl.INT32)
                        else:
                            tail_acc = pl.matmul_acc(tail_acc, qr_i8_tail, wq_tail)
                    q_proj_i32[
                        tail_t0 : tail_t0 + QPROJ_TAIL_M_TILE,
                        tail_w_col0 : tail_w_col0 + QPROJ_MM_N_TILE,
                    ] = tail_acc

            # Fused qproj dequant, per-head RMSNorm, NOPE writeback, and interleaved RoPE.
            # RoPE: out[j] = inv_rms * (x[j] * cos[j] + x[j^1] * sign[j] * sin[j]).
            q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
            for hg_idx in pl.spmd(
                H // Q_ROPE_H_TILE, name_hint="qproj_dequant_rms_nope_rope", allow_early_resolve=True
            ):
                hg = hg_idx * Q_ROPE_H_TILE
                for tg in pl.range(0, tile_rows, Q_ROPE_T_TILE):
                    out_tg = tile_base + tg
                    if tg + Q_ROPE_T_TILE <= tile_rows:
                        qr_scale_dq_t = qr_scale_pad_store[tg : tg + Q_ROPE_T_TILE, :]
                        q_cos_il = rope_cos_il[out_tg : out_tg + Q_ROPE_T_TILE, :]
                        q_sin_signed = rope_sin_signed[out_tg : out_tg + Q_ROPE_T_TILE, :]
                        q_swap_idx = rope_swap_idx[out_tg : out_tg + Q_ROPE_T_TILE, :]
                        for h_inner in pl.pipeline(Q_ROPE_H_TILE, stage=2):
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
                        for h_inner_tail in pl.range(Q_ROPE_H_TILE):
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


@pl.jit.inline(auto_scope=False)
def kv_proj_rope(
    x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.INT32],
    kv: pl.Tensor[[KV_T_DYN, HEAD_DIM], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """KV LoRA, RMSNorm, and RoPE over bounded dense tiles."""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            x_view = pl.reshape(x, [t_dim, D])
            t_matmul = ((tile_rows + MATMUL_T_TILE - 1) // MATMUL_T_TILE) * MATMUL_T_TILE

            # Split-K kv_proj: KV_N_TILE N-groups expanded KV_OK-fold into cube blocks that
            # atomic-add their K partials into a zero-seeded output.
            kv_fp32 = pl.create_tensor([t_matmul, HEAD_DIM], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
                for kts0 in pl.range(0, t_matmul, KV_M_TILE):
                    for kvseed0 in pl.range(0, HEAD_DIM, KV_N_TILE):
                        kv_seed = pl.full([KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0)
                        kv_fp32[kts0 : kts0 + KV_M_TILE, kvseed0 : kvseed0 + KV_N_TILE] = kv_seed

            # `late_dep` fences kv_proj one hop behind rms_norm so qr_proj_matmul takes the cores first.
            with pl.spmd(
                (HEAD_DIM // KV_N_TILE) * KV_OK * KV_OM, name_hint="kv_proj_matmul", deps=[late_dep],
            ) as _kv_tid:
                kbg = pl.tile.get_block_idx()
                kv_col0 = (kbg // (KV_OK * KV_OM)) * KV_N_TILE
                kv_k_base = ((kbg // KV_OM) % KV_OK) * KV_SPLIT_K_TILE
                kv_m_group = kbg % KV_OM
                for t0 in pl.range(kv_m_group * KV_M_TILE, t_matmul, KV_OM * KV_M_TILE):
                    kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
                    for db in pl.pipeline(KV_SPLIT_K_TILE // KV_K_TILE, stage=2):
                        d0 = kv_k_base + db * KV_K_TILE
                        kv_rows = pl.min(KV_M_TILE, tile_rows - t0)
                        x_t0 = tile_base + t0
                        kv_x_chunk_bf16 = pl.slice(
                            x_view,
                            [KV_M_TILE, KV_K_TILE],
                            [x_t0, d0],
                            valid_shape=[kv_rows, KV_K_TILE],
                        )
                        wkv_chunk = wkv[d0 : d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                        if db == 0:
                            kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                        else:
                            kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
                    kv_fp32 = pl.assemble(kv_fp32, kv_acc, [t0, kv_col0], atomic=pl.AtomicType.Add)

            kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])

            # Fused KV RMSNorm + interleaved (CANN A3) RoPE, one spmd task per
            # [KV_RMS_T_TILE, HEAD_DIM] row block. NOPE columns [0:NOPE_DIM) and rope columns
            # [NOPE_DIM:HEAD_DIM) are disjoint, so each task writes a conflict-free row block.
            kv_token_tiles = (tile_rows + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
            for tg_idx in pl.spmd(kv_token_tiles, name_hint="kv_rms_norm_rope"):
                tg = tg_idx * KV_RMS_T_TILE
                valid_rows = pl.min(KV_RMS_T_TILE, tile_rows - tg)
                out_tg = tile_base + tg
                if valid_rows == KV_RMS_T_TILE:
                    kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
                    for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                        kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
                        kv_sq = pl.mul(kv_chunk, kv_chunk)
                        kv_row_sum = pl.reshape(pl.row_sum(kv_sq), [1, KV_RMS_T_TILE])
                        kv_sq_sum = pl.add(kv_sq_sum, kv_row_sum)
                    kv_inv_rms = pl.rsqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS), high_precision=True)
                    kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

                    for n0 in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                        kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
                        gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
                        gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
                        kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                        kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
                        kv_view[out_tg : out_tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16

                    gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
                    gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
                    kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
                    kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
                    kv_cos_il_full = rope_cos_il[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_sin_signed_full = rope_sin_signed[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swap_idx_full = rope_swap_idx[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swapped_full = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx_full)
                    kv_rope_rot_full = pl.add(
                        pl.mul(kv_rope_norm_chunk, kv_cos_il_full),
                        pl.mul(kv_swapped_full, kv_sin_signed_full),
                    )
                    kv_rope_i16_full = pl.cast(kv_rope_rot_full, target_type=pl.BF16, mode="rint")
                    kv_view[out_tg : out_tg + KV_RMS_T_TILE, NOPE_DIM:HEAD_DIM] = kv_rope_i16_full
                else:
                    kv_reduce_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    kv_sq_sum_tail = pl.tile.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
                    for kv_sq_col0_tail in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                        kv_chunk_tail = pl.load(
                            kv_fp32,
                            [tg, kv_sq_col0_tail],
                            [KV_RMS_T_TILE, KV_TILE],
                            valid_shape=[valid_rows, KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        kv_sq_tail = pl.mul(kv_chunk_tail, kv_chunk_tail)
                        kv_row_sum_tail = pl.reshape(pl.row_sum(kv_sq_tail, kv_reduce_tmp), [1, KV_RMS_T_TILE])
                        kv_sq_sum_tail = pl.add(kv_sq_sum_tail, kv_row_sum_tail)
                    kv_inv_rms_tail = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum_tail, 1.0 / HEAD_DIM), EPS)))
                    kv_inv_rms_t_tail = pl.reshape(kv_inv_rms_tail, [KV_RMS_T_TILE, 1])

                    for n0_tail in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                        kv_chunk_tail = pl.load(
                            kv_fp32,
                            [tg, n0_tail],
                            [KV_RMS_T_TILE, KV_TILE],
                            valid_shape=[valid_rows, KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        gamma_kv_input_tail = pl.load(
                            gamma_ckv,
                            [n0_tail],
                            [KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        gamma_kv_cast_tail = pl.cast(gamma_kv_input_tail, target_type=pl.FP32)
                        gamma_kv_chunk_tail = pl.reshape(gamma_kv_cast_tail, [1, KV_TILE])
                        kv_normed_tail = pl.col_expand_mul(
                            pl.row_expand_mul(kv_chunk_tail, kv_inv_rms_t_tail),
                            gamma_kv_chunk_tail,
                        )
                        kv_normed_bf16_tail = pl.cast(kv_normed_tail, target_type=pl.BF16, mode="rint")
                        kv_normed_valid = pl.set_validshape(kv_normed_bf16_tail, valid_rows, KV_TILE)
                        pl.store(kv_normed_valid, [out_tg, n0_tail], kv_view)

                    gamma_rope_input_tail = pl.load(
                        gamma_ckv,
                        [NOPE_DIM],
                        [ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    gamma_rope_cast_tail = pl.cast(gamma_rope_input_tail, target_type=pl.FP32)
                    gamma_rope_tail = pl.reshape(gamma_rope_cast_tail, [1, ROPE_DIM])
                    kv_rope_chunk_tail = pl.load(
                        kv_fp32,
                        [tg, NOPE_DIM],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_rope_norm_tail = pl.col_expand_mul(
                        pl.row_expand_mul(kv_rope_chunk_tail, kv_inv_rms_t_tail),
                        gamma_rope_tail,
                    )
                    kv_cos_il_tail = pl.load(
                        rope_cos_il,
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_sin_signed_tail = pl.load(
                        rope_sin_signed,
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_col = pl.col_expand_mul(
                        pl.tile.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                        pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                    )
                    kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
                    kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))
                    kv_swap_f = pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0))
                    # Row-major flattening offsets stay in fp32: col-expand is defined
                    # for half / float only.
                    kv_row_seed = pl.mul(
                        pl.cast(pl.tile.arange(0, [1, KV_RMS_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                        ROPE_DIM_SCALE,
                    )
                    kv_row_grid = pl.col_expand_mul(
                        pl.tile.full([ROPE_DIM, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0),
                        kv_row_seed,
                    )
                    kv_row_offset = pl.transpose(kv_row_grid, axis1=0, axis2=1)
                    kv_swap_idx_tail = pl.cast(pl.add(kv_swap_f, kv_row_offset), target_type=pl.INT32)
                    kv_gather_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, ROPE_DIM],
                        dtype=pl.INT32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_swapped_tail = pl.tile.gather(kv_rope_norm_tail, kv_swap_idx_tail, kv_gather_tmp)
                    kv_rope_rot_tail = pl.add(
                        pl.mul(kv_rope_norm_tail, kv_cos_il_tail),
                        pl.mul(kv_swapped_tail, kv_sin_signed_tail),
                    )
                    kv_rope_i16_tail = pl.cast(kv_rope_rot_tail, target_type=pl.BF16, mode="rint")
                    kv_rope_valid = pl.set_validshape(kv_rope_i16_tail, valid_rows, ROPE_DIM)
                    pl.store(kv_rope_valid, [out_tg, NOPE_DIM], kv_view)


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """Fused q + kv projection: both branches share one token axis and one rope table."""
    t_dim = pl.tensor.dim(x, 0)
    q_rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos, rope_sin, q_rope_cos_il, q_rope_sin_signed, q_rope_swap_idx)
    q_proj_rope(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        gamma_cq,
        q_rope_cos_il,
        q_rope_sin_signed,
        q_rope_swap_idx,
        q,
        qr,
        qr_scale,
    )
    kv_proj_rope(
        x,
        wkv,
        gamma_ckv,
        q_rope_cos_il,
        q_rope_sin_signed,
        q_rope_swap_idx,
        kv,
        late_dep,
    )
    return q


@pl.jit
def qkv_proj_rope_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Out[pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    q.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)

    # Standalone: no rms_norm producer to fence.
    late_dep = pl.system.task_dummy(deps=[])
    qkv_proj_rope(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos,
        rope_sin,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        late_dep,
    )
    return q


# Split-branch coverage geometry: q on a CP-local token slice, kv on the full run.
SPLIT_T_FULL = PREFILL_BATCH * PREFILL_SEQ
SPLIT_T_LOCAL = SPLIT_T_FULL // 4


@pl.jit
def q_kv_split_test(
    x_local: pl.Tensor[[SPLIT_T_LOCAL, D], pl.BF16],
    x_full: pl.Tensor[[SPLIT_T_FULL, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos_local: pl.Tensor[[SPLIT_T_LOCAL, ROPE_DIM], pl.BF16],
    rope_sin_local: pl.Tensor[[SPLIT_T_LOCAL, ROPE_DIM], pl.BF16],
    rope_cos_full: pl.Tensor[[SPLIT_T_FULL, ROPE_DIM], pl.BF16],
    rope_sin_full: pl.Tensor[[SPLIT_T_FULL, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Out[pl.Tensor[[SPLIT_T_LOCAL, H, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[SPLIT_T_LOCAL, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[SPLIT_T_LOCAL, 1], pl.FP32]],
    kv: pl.Out[pl.Tensor[[SPLIT_T_FULL, HEAD_DIM], pl.BF16]],
):
    """Drive the q and kv branches at different token counts, as CP does."""
    q_cos_il = pl.create_tensor([SPLIT_T_LOCAL, ROPE_DIM], dtype=pl.FP32)
    q_sin_signed = pl.create_tensor([SPLIT_T_LOCAL, ROPE_DIM], dtype=pl.FP32)
    q_swap_idx = pl.create_tensor([SPLIT_T_LOCAL, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_local, rope_sin_local, q_cos_il, q_sin_signed, q_swap_idx)
    q_proj_rope(
        x_local,
        wq_a,
        wq_b,
        wq_b_scale,
        gamma_cq,
        q_cos_il,
        q_sin_signed,
        q_swap_idx,
        q,
        qr,
        qr_scale,
    )

    kv_cos_il = pl.create_tensor([SPLIT_T_FULL, ROPE_DIM], dtype=pl.FP32)
    kv_sin_signed = pl.create_tensor([SPLIT_T_FULL, ROPE_DIM], dtype=pl.FP32)
    kv_swap_idx = pl.create_tensor([SPLIT_T_FULL, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_full, rope_sin_full, kv_cos_il, kv_sin_signed, kv_swap_idx)
    # Standalone: no rms_norm producer to fence.
    late_dep = pl.system.task_dummy(deps=[])
    kv_proj_rope(
        x_full,
        wkv,
        gamma_ckv,
        kv_cos_il,
        kv_sin_signed,
        kv_swap_idx,
        kv,
        late_dep,
    )
    return q


def golden_q_kv_split(tensors):
    """Reference: the fused golden on the full run, sliced to each branch's rows."""
    import torch

    full = dict(tensors)
    full["x"] = tensors["x_full"]
    full["rope_cos"] = tensors["rope_cos_full"]
    full["rope_sin"] = tensors["rope_sin_full"]
    full["q"] = torch.zeros(SPLIT_T_FULL, H, HEAD_DIM, dtype=torch.bfloat16)
    full["kv"] = torch.zeros(SPLIT_T_FULL, HEAD_DIM, dtype=torch.bfloat16)
    full["qr"] = torch.zeros(SPLIT_T_FULL, Q_LORA, dtype=torch.int8)
    full["qr_scale"] = torch.zeros(SPLIT_T_FULL, 1, dtype=torch.float32)
    golden_qkv_proj_rope(full)

    tensors["q"][:] = full["q"][:SPLIT_T_LOCAL]
    tensors["qr"][:] = full["qr"][:SPLIT_T_LOCAL]
    tensors["qr_scale"][:] = full["qr_scale"][:SPLIT_T_LOCAL]
    tensors["kv"][:] = full["kv"]


def build_split_tensor_specs():
    import torch
    from golden import TensorSpec

    base = {spec.name: spec for spec in build_tensor_specs(1, SPLIT_T_FULL)}
    x_full = base["x"].init_value()
    rope_cos_full = base["rope_cos"].init_value()
    rope_sin_full = base["rope_sin"].init_value()
    x_loc = x_full[:SPLIT_T_LOCAL].clone()
    rope_cos_loc = rope_cos_full[:SPLIT_T_LOCAL].clone()
    rope_sin_loc = rope_sin_full[:SPLIT_T_LOCAL].clone()

    return [
        TensorSpec("x_local", [SPLIT_T_LOCAL, D], torch.bfloat16, init_value=lambda: x_loc),
        TensorSpec("x_full", [SPLIT_T_FULL, D], torch.bfloat16, init_value=lambda: x_full),
        base["wq_a"],
        base["wq_b"],
        base["wq_b_scale"],
        base["wkv"],
        TensorSpec(
            "rope_cos_local", [SPLIT_T_LOCAL, ROPE_DIM], torch.bfloat16, init_value=lambda: rope_cos_loc
        ),
        TensorSpec(
            "rope_sin_local", [SPLIT_T_LOCAL, ROPE_DIM], torch.bfloat16, init_value=lambda: rope_sin_loc
        ),
        TensorSpec(
            "rope_cos_full", [SPLIT_T_FULL, ROPE_DIM], torch.bfloat16, init_value=lambda: rope_cos_full
        ),
        TensorSpec(
            "rope_sin_full", [SPLIT_T_FULL, ROPE_DIM], torch.bfloat16, init_value=lambda: rope_sin_full
        ),
        base["gamma_cq"],
        base["gamma_ckv"],
        TensorSpec("q", [SPLIT_T_LOCAL, H, HEAD_DIM], torch.bfloat16),
        TensorSpec("qr", [SPLIT_T_LOCAL, Q_LORA], torch.int8),
        TensorSpec("qr_scale", [SPLIT_T_LOCAL, 1], torch.float32),
        TensorSpec("kv", [SPLIT_T_FULL, HEAD_DIM], torch.bfloat16),
    ]


def golden_qkv_proj_rope(tensors):
    """Torch reference: Q/KV LoRA + RoPE for an already attention-normalized input."""
    import torch
    from utils import int8_quant_per_row

    x = tensors["x"].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv = tensors["wkv"].float()
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    def rms_norm(x, gamma, eps=EPS):
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x * inv * gamma

    def matmul_bf16_input_fp32(a, b):
        a_fp32 = a.to(torch.bfloat16).float()
        b_fp32 = b.to(torch.bfloat16).float()
        return torch.matmul(a_fp32, b_fp32).float()

    def apply_rope(x_rope, cos, sin):
        # x_rope: [T, ..., ROPE_DIM] with interleaved even/odd rotary pairs.
        x_pair = x_rope.unflatten(-1, (-1, 2))
        x_even, x_odd = x_pair[..., 0], x_pair[..., 1]
        cos_v = cos[..., :ROPE_HALF]
        sin_v = sin[..., :ROPE_HALF]
        while cos_v.ndim < x_even.ndim:
            cos_v = cos_v.unsqueeze(-2)
            sin_v = sin_v.unsqueeze(-2)
        y_even = (x_even * cos_v - x_odd * sin_v).to(torch.bfloat16)
        y_odd = (x_even * sin_v + x_odd * cos_v).to(torch.bfloat16)
        return torch.stack([y_even, y_odd], dim=-1).flatten(-2)

    t_dim = x.shape[0]
    token_x = x.view(t_dim, D)

    # Q path
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)  # [T, Q_LORA]
    # W8A8C16: wq_b W8 per-output-channel int8; qr_out A8 per-token int8.
    # flash: also quantizes wq_a/wkv to fp8 (default Linear dtype).
    qr_i8, qr_scale = int8_quant_per_row(qr_out.float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(t_dim, H, HEAD_DIM)
    inv = torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_full = q_full * inv  # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    # KV path
    kv_full = rms_norm(matmul_bf16_input_fp32(token_x, wkv), gamma_ckv)  # [T, HEAD_DIM]
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)  # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:] = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_i8
    tensors["qr_scale"][:] = qr_scale


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x():
        return torch.empty([T, D], dtype=torch.bfloat16).uniform_(-1, 1)

    def init_wq_a():
        return torch.empty([D, Q_LORA], dtype=torch.bfloat16).uniform_(-0.1, 0.1)

    def init_wq_b():
        return torch.empty([Q_LORA, H * HEAD_DIM], dtype=torch.bfloat16).uniform_(-0.1, 0.1)

    def init_wkv():
        return torch.empty([D, HEAD_DIM], dtype=torch.bfloat16).uniform_(-0.1, 0.1)

    def init_cos():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)

    def init_sin():
        return torch.empty([T, ROPE_DIM], dtype=torch.bfloat16).uniform_(-1, 1)

    def init_gamma_cq():
        return torch.empty([Q_LORA], dtype=torch.bfloat16).uniform_(-1, 1)

    def init_gamma_ckv():
        return torch.empty([HEAD_DIM], dtype=torch.bfloat16).uniform_(-1, 1)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wq_b_scale = wq_b_scale.view(H * HEAD_DIM)

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("rope_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("rope_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16),
        TensorSpec("kv", [T, HEAD_DIM], torch.bfloat16),
        TensorSpec("qr", [T, Q_LORA], torch.int8),
        TensorSpec("qr_scale", [T, 1], torch.float32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    MODES = {
        "decode": (DECODE_BATCH // TP, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["decode", "prefill", "split", "all"],
        default="all",
        help="decode / prefill batch sizes, 'split' for mismatched q and kv rows, or 'all'.",
    )
    parser.add_argument(
        "--enable-chip-swimlane",
        type=int,
        choices=[0, 1, 2, 4],
        default=0,
        help="chip swimlane level: 0=off, 1=per-kernel AICore timing "
        "(prints the per-function Task Statistics table), 2=+AICPU timing.",
    )
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) + ["split"] if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        if mode_name == "split":
            fn, specs, golden = q_kv_split_test, build_split_tensor_specs(), golden_q_kv_split
            print(f"--- qkv_proj_rope split: q rows={SPLIT_T_LOCAL}, kv rows={SPLIT_T_FULL} ---")
        else:
            B, S = MODES[mode_name]
            fn, specs, golden = qkv_proj_rope_test, build_tensor_specs(B, S), golden_qkv_proj_rope
            print(f"--- qkv_proj_rope {mode_name}: B={B}, S={S} ---")
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden,
            # W8A8C16 q_proj adds INT8 quant/dequant round-off before per-head RMSNorm.
            rtol=5e-3,
            atol=5e-3,
            # Precision reference: pypto mla_prolog —
            # cann-recipes-infer/ops/pypto_python/example/test_mla_prolog_pypto.py
            compare_fn={
                "q": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "qr": ratio_allclose(atol=1, rtol=0, max_error_ratio=0),
                "qr_scale": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
