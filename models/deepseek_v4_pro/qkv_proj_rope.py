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

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)

# tiling
Q_PROJ_TILE = 128
QPROJ_MM_N_TILE = 1024
QPROJ_BLOCK_TILE = 4
Q_LORA_TILE = 512
KV_TILE = 64
QUANT_TILE = 512
T_TILE = 8
MATMUL_T_TILE = 16
QR_M_TILE = MATMUL_T_TILE
QR_N_TILE = 128
QR_K_TILE = 256
QR_SPLIT_TILE = 2
QR_K_SPLIT_TILE = D // QR_SPLIT_TILE
KV_M_TILE = MATMUL_T_TILE
KV_N_TILE = 128
KV_K_TILE = 128
KV_SPLIT_TILE = 4
KV_K_SPLIT_TILE = D // KV_SPLIT_TILE
QPROJ_M_TILE = MATMUL_T_TILE
KV_RMS_T_TILE = 8
Q_ROPE_T_TILE = 8
Q_ROPE_H_TILE = 4


def _even_pipeline_trip(name: str, trip: int) -> None:
    """Require even split-K pipeline trips for A5 accumulator-buffer selection."""
    assert trip % 2 == 0, (
        f"{name} pipeline trip count must be even, got {trip}; an odd count makes "
        "codegen emit an unsupported acc->acc pto.tmov. Retune the K tile or split-K factor."
    )


_even_pipeline_trip("qr_proj", QR_K_SPLIT_TILE // QR_K_TILE)
_even_pipeline_trip("kv_proj", KV_K_SPLIT_TILE // KV_K_TILE)
assert QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C accumulator capacity


@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(position_ids, 0)
    for rope_t0 in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="qkv_rope_rows"):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < num_tokens:
                rope_pos_i32 = pl.read(position_ids, [rope_t])
                rope_pos = pl.cast(rope_pos_i32, pl.INDEX)
                rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]

@pl.jit.inline
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
    # Task resolution is ordered across the kernel.
    t_dim = pl.tensor.dim(x, 0)
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    q_rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    for qrp_idx in pl.spmd(t_dim // Q_ROPE_T_TILE, name_hint="q_rope_prepare"):
        qrp_t0 = qrp_idx * Q_ROPE_T_TILE
        qrp_cos = pl.cast(rope_cos_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, 0 : ROPE_DIM // 2], target_type=pl.FP32)
        qrp_sin = pl.cast(rope_sin_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, 0 : ROPE_DIM // 2], target_type=pl.FP32)
        qrp_cos_il = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
        qrp_cos_il = pl.tensor.scatter(qrp_cos, mask_pattern=pl.tile.MaskPattern.P0101, dst=qrp_cos_il)
        qrp_cos_il = pl.tensor.scatter(qrp_cos, mask_pattern=pl.tile.MaskPattern.P1010, dst=qrp_cos_il)
        q_rope_cos_il[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_cos_il
        qrp_sin_neg = pl.neg(qrp_sin)
        qrp_sin_signed = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
        qrp_sin_signed = pl.tensor.scatter(qrp_sin_neg, mask_pattern=pl.tile.MaskPattern.P0101, dst=qrp_sin_signed)
        qrp_sin_signed = pl.tensor.scatter(qrp_sin, mask_pattern=pl.tile.MaskPattern.P1010, dst=qrp_sin_signed)
        q_rope_sin_signed[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_sin_signed

    x_view = pl.reshape(x, [t_dim, D])
    t_matmul = pl.max(t_dim, MATMUL_T_TILE)
    qr_partials = pl.create_tensor([QR_SPLIT_TILE * T_MAX, Q_LORA], dtype=pl.FP32)
    for qbg_idx in pl.spmd((Q_LORA // QR_N_TILE) * QR_SPLIT_TILE, name_hint="qr_proj_matmul"):
        q_a_col0 = (qbg_idx // QR_SPLIT_TILE) * QR_N_TILE
        qr_split = qbg_idx % QR_SPLIT_TILE
        qr_k_base = qr_split * QR_K_SPLIT_TILE
        for tc in pl.range(t_matmul // QR_M_TILE):
            t0 = tc * QR_M_TILE
            q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(QR_K_SPLIT_TILE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                qr_rows = pl.min(QR_M_TILE, t_dim - t0)
                q_x_chunk_bf16 = pl.slice(x_view, [QR_M_TILE, QR_K_TILE], [t0, qr_d0], valid_shape=[qr_rows, QR_K_TILE])
                w_chunk = wq_a[qr_d0 : qr_d0 + QR_K_TILE, q_a_col0 : q_a_col0 + QR_N_TILE]
                if db == 0:
                    q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_partial_t0 = qr_split * T_MAX + t0
            qr_partials[qr_partial_t0 : qr_partial_t0 + QR_M_TILE, q_a_col0 : q_a_col0 + QR_N_TILE] = q_acc

    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
    qr_i8_matmul = pl.create_tensor([T_MAX, Q_LORA], dtype=pl.INT8)
    for tg_idx in pl.spmd(t_dim // T_TILE, name_hint="qr_rms_norm_quant"):
        tg = tg_idx * T_TILE
        qr_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        qr_amax_g = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for qr_rms_qb in pl.pipeline(Q_LORA // Q_LORA_TILE, stage=2):
            qr_rms_col0 = qr_rms_qb * Q_LORA_TILE
            qr_rms_chunk = qr_partials[tg : tg + T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE]
            for qr_rms_split in pl.range(1, QR_SPLIT_TILE):
                qr_rms_p0 = qr_rms_split * T_MAX + tg
                qr_rms_partial = qr_partials[qr_rms_p0 : qr_rms_p0 + T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE]
                qr_rms_chunk = pl.add(qr_rms_chunk, qr_rms_partial)
            qr_sq = pl.mul(qr_rms_chunk, qr_rms_chunk)
            qr_sq_row = pl.row_sum(qr_sq)
            qr_sq_partial = pl.reshape(qr_sq_row, [1, T_TILE])
            qr_sq_sum = pl.add(qr_sq_sum, qr_sq_partial)
            gamma_rms_cast = pl.cast(gamma_cq[qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE], target_type=pl.FP32)
            gamma_rms_chunk = pl.reshape(gamma_rms_cast, [1, Q_LORA_TILE])
            qr_g = pl.col_expand_mul(qr_rms_chunk, gamma_rms_chunk)
            qr_g_abs = pl.abs(qr_g)
            qr_amax_row = pl.row_max(qr_g_abs)
            qr_amax_partial = pl.reshape(qr_amax_row, [1, T_TILE])
            qr_amax_g = pl.maximum(qr_amax_g, qr_amax_partial)
        qr_sq_mean = pl.mul(qr_sq_sum, 1.0 / Q_LORA)
        qr_rms_arg = pl.add(qr_sq_mean, EPS)
        qr_inv_rms = pl.rsqrt(qr_rms_arg, high_precision=True)
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])
        qr_amax_floor = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        qr_amax_normed = pl.mul(qr_inv_rms, qr_amax_g)
        qr_tile_amax = pl.maximum(qr_amax_floor, qr_amax_normed)

        qr_scale_max = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        qr_scale_quant_row = pl.div(qr_scale_max, qr_tile_amax)
        qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
        qr_scale_recip = pl.recip(qr_scale_quant_row)
        qr_tile_scale_dq = pl.reshape(qr_scale_recip, [T_TILE, 1])
        qr_scale_view[tg : tg + T_TILE, :] = qr_tile_scale_dq

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = qr_partials[tg : tg + T_TILE, qa : qa + QUANT_TILE]
            for qr_q_split in pl.range(1, QR_SPLIT_TILE):
                qr_q_p0 = qr_q_split * T_MAX + tg
                qr_q_partial = qr_partials[qr_q_p0 : qr_q_p0 + T_TILE, qa : qa + QUANT_TILE]
                qr_chunk = pl.add(qr_chunk, qr_q_partial)
            gamma_q_cast = pl.cast(gamma_cq[qa : qa + QUANT_TILE], target_type=pl.FP32)
            gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
            qr_q_rms = pl.row_expand_mul(qr_chunk, qr_inv_rms_t)
            qr_q_normed = pl.col_expand_mul(qr_q_rms, gamma_q_chunk)
            qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
            qr_view[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8
            qr_i8_matmul[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8

    q_proj_i32 = pl.create_tensor([T_MAX, H * HEAD_DIM], dtype=pl.INT32)
    # RoPE: out[j] = inv_rms * (x[j] * cos[j] + x[j^1] * sign[j] * sin[j]).
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])

    for hg_idx in pl.spmd(((H * HEAD_DIM) // QPROJ_MM_N_TILE) // QPROJ_BLOCK_TILE, name_hint="qproj_matmul"):
        hg = hg_idx * QPROJ_BLOCK_TILE
        for h_inner in pl.range(QPROJ_BLOCK_TILE):
            w_col0 = (hg + h_inner) * QPROJ_MM_N_TILE
            for tc in pl.range(t_matmul // QPROJ_M_TILE):
                t0 = tc * QPROJ_M_TILE
                col_acc = pl.create_tensor([QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
                for qr_proj_col0 in pl.pipeline(0, Q_LORA, Q_PROJ_TILE, stage=2):
                    qr_i8_chunk = qr_i8_matmul[t0 : t0 + QPROJ_M_TILE, qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE]
                    wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                    if qr_proj_col0 == 0:
                        col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                    else:
                        col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
                # --- fused epilogue: dequant + RMS + NoPE/RoPE, on col_acc ---
                for sub_h in pl.range(QPROJ_MM_N_TILE // HEAD_DIM):
                    sub_c0 = sub_h * HEAD_DIM
                    h0 = w_col0 + sub_c0
                    q_head_scale = pl.reshape(wq_b_scale[h0 : h0 + HEAD_DIM], [1, HEAD_DIM])
                    # Cast the WHOLE accumulator tile first. An Acc tile must be a
                    # whole number of 16x16 fractal boxes, so its rows cannot be
                    # sliced; once it is FP32 it is an ordinary vector tile and the
                    # row slice below is free.
                    q_head_acc = col_acc[0:QPROJ_M_TILE, sub_c0 : sub_c0 + HEAD_DIM]
                    q_head_fp32_full = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                    # The epilogue can only touch the rows this accumulator holds, so
                    # it walks the QPROJ_M_TILE rows of the current cube tile rather
                    # than the whole token axis. t_matmul rounds t_dim up to
                    # QPROJ_M_TILE, so the last tile can run past the real row count.
                    for fq_sub in pl.range(QPROJ_M_TILE // Q_ROPE_T_TILE):
                        q_row0 = fq_sub * Q_ROPE_T_TILE
                        fq_tg = t0 + q_row0
                        if fq_tg < t_dim:
                            qr_scale_dq_t = qr_scale_view[fq_tg : fq_tg + Q_ROPE_T_TILE, :]
                            q_cos_il = q_rope_cos_il[fq_tg : fq_tg + Q_ROPE_T_TILE, :]
                            q_sin_signed = q_rope_sin_signed[fq_tg : fq_tg + Q_ROPE_T_TILE, :]
                            q_head_acc_fp32 = q_head_fp32_full[q_row0 : q_row0 + Q_ROPE_T_TILE, 0:HEAD_DIM]
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
                            q_flat[fq_tg : fq_tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = q_nope_bf16

                            q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                            q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                            q_rope_col0 = h0 + NOPE_DIM
                            q_rope_even = pl.gather(q_rope_chunk, mask_pattern=pl.tile.MaskPattern.P0101)
                            q_rope_odd = pl.gather(q_rope_chunk, mask_pattern=pl.tile.MaskPattern.P1010)
                            q_rope_swapped = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
                            q_rope_swapped = pl.tensor.scatter(
                                q_rope_odd, mask_pattern=pl.tile.MaskPattern.P0101, dst=q_rope_swapped,
                            )
                            q_rope_swapped = pl.tensor.scatter(
                                q_rope_even, mask_pattern=pl.tile.MaskPattern.P1010, dst=q_rope_swapped,
                            )
                            q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                            q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                            q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                            q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                            q_flat[fq_tg : fq_tg + Q_ROPE_T_TILE, q_rope_col0 : q_rope_col0 + ROPE_DIM] = q_rope_bf16

    kv_partials = pl.create_tensor([KV_SPLIT_TILE * T_MAX, HEAD_DIM], dtype=pl.FP32)
    with pl.spmd((HEAD_DIM // KV_N_TILE) * KV_SPLIT_TILE, name_hint="kv_proj_matmul", deps=[late_dep]) as _kv_tid:
        kbg = pl.tile.get_block_idx()
        kv_col0 = (kbg // KV_SPLIT_TILE) * KV_N_TILE
        kv_split = kbg % KV_SPLIT_TILE
        kv_k_base = kv_split * KV_K_SPLIT_TILE
        for tc in pl.range(t_matmul // KV_M_TILE):
            t0 = tc * KV_M_TILE
            kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(KV_K_SPLIT_TILE // KV_K_TILE, stage=2):
                d0 = kv_k_base + db * KV_K_TILE
                kv_rows = pl.min(KV_M_TILE, t_dim - t0)
                kv_x_chunk_bf16 = pl.slice(x_view, [KV_M_TILE, KV_K_TILE], [t0, d0], valid_shape=[kv_rows, KV_K_TILE])
                wkv_chunk = wkv[d0 : d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                if db == 0:
                    kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_partial_t0 = kv_split * T_MAX + t0
            kv_partials[kv_partial_t0 : kv_partial_t0 + KV_M_TILE, kv_col0 : kv_col0 + KV_N_TILE] = kv_acc

    kv_fp32 = pl.create_tensor([T_MAX, HEAD_DIM], dtype=pl.FP32)
    for kv_reduce_idx in pl.spmd((t_matmul // KV_M_TILE) * (HEAD_DIM // KV_N_TILE), name_hint="kv_proj_reduce"):
        kv_t0 = (kv_reduce_idx // (HEAD_DIM // KV_N_TILE)) * KV_M_TILE
        kv_col0 = (kv_reduce_idx % (HEAD_DIM // KV_N_TILE)) * KV_N_TILE
        kv_total = kv_partials[kv_t0 : kv_t0 + KV_M_TILE, kv_col0 : kv_col0 + KV_N_TILE]
        for kv_split in pl.range(1, KV_SPLIT_TILE):
            kv_partial_t0 = kv_split * T_MAX + kv_t0
            kv_partial = kv_partials[kv_partial_t0 : kv_partial_t0 + KV_M_TILE, kv_col0 : kv_col0 + KV_N_TILE]
            kv_total = pl.add(kv_total, kv_partial)
        kv_fp32[kv_t0 : kv_t0 + KV_M_TILE, kv_col0 : kv_col0 + KV_N_TILE] = kv_total

    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
    for tg_idx in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="kv_rms_norm_rope"):
        tg = tg_idx * KV_RMS_T_TILE
        kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
            kv_sq = pl.mul(kv_chunk, kv_chunk)
            kv_sq_row = pl.row_sum(kv_sq)
            kv_sq_partial = pl.reshape(kv_sq_row, [1, KV_RMS_T_TILE])
            kv_sq_sum = pl.add(kv_sq_sum, kv_sq_partial)
        kv_sq_mean = pl.mul(kv_sq_sum, 1.0 / HEAD_DIM)
        kv_rms_arg = pl.add(kv_sq_mean, EPS)
        kv_inv_rms = pl.rsqrt(kv_rms_arg, high_precision=True)
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

        for n0 in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
            gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
            gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
            kv_rms = pl.row_expand_mul(kv_chunk, kv_inv_rms_t)
            kv_normed = pl.col_expand_mul(kv_rms, gamma_kv_chunk)
            kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
            kv_view[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16

        # RoPE: out[j] = n[j] * cos_il[j] + n[j^1] * sign[j] * sin_il[j].
        gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
        gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
        kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
        kv_rope_rms = pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t)
        kv_rope_norm_chunk = pl.col_expand_mul(kv_rope_rms, gamma_rope)
        for kv_rope_row in pl.range(KV_RMS_T_TILE):
            kv_rope_t = tg + kv_rope_row
            kv_cos = pl.cast(rope_cos_view[kv_rope_t : kv_rope_t + 1, 0:ROPE_HALF], target_type=pl.FP32)
            kv_sin = pl.cast(rope_sin_view[kv_rope_t : kv_rope_t + 1, 0:ROPE_HALF], target_type=pl.FP32)
            kv_rope_norm_row = kv_rope_norm_chunk[kv_rope_row : kv_rope_row + 1, :]
            kv_rope_even = pl.gather(kv_rope_norm_row, mask_pattern=pl.tile.MaskPattern.P0101)
            kv_rope_odd = pl.gather(kv_rope_norm_row, mask_pattern=pl.tile.MaskPattern.P1010)
            kv_even_base = pl.mul(kv_rope_even, kv_cos)
            kv_odd_neg = pl.neg(kv_rope_odd)
            kv_even_delta = pl.mul(kv_odd_neg, kv_sin)
            kv_even_rot = pl.add(kv_even_base, kv_even_delta)
            kv_odd_base = pl.mul(kv_rope_odd, kv_cos)
            kv_odd_delta = pl.mul(kv_rope_even, kv_sin)
            kv_odd_rot = pl.add(kv_odd_base, kv_odd_delta)
            kv_rope_rot = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            kv_rope_rot = pl.tensor.scatter(kv_even_rot, mask_pattern=pl.tile.MaskPattern.P0101, dst=kv_rope_rot)
            kv_rope_rot = pl.tensor.scatter(kv_odd_rot, mask_pattern=pl.tile.MaskPattern.P1010, dst=kv_rope_rot)
            kv_rope_i16 = pl.cast(kv_rope_rot, target_type=pl.BF16, mode="rint")
            kv_view[kv_rope_t : kv_rope_t + 1, NOPE_DIM : NOPE_DIM + ROPE_DIM] = kv_rope_i16

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

    late_dep = pl.system.task_dummy(deps=[])
    qkv_proj_rope(
        x,
        wq_a, wq_b, wq_b_scale, wkv,
        rope_cos, rope_sin,
        gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale,
        late_dep,
    )
    return q


_A5_FP32_VECTOR_LANES = 64
_A5_CUBE_ACC_K = 16
_A5_CUBE_N_TILE = 128
_A5_CUBE_K_GROUP_TILE = 64


def _golden_a5_trowsum_fp32(values):
    """Mirror A5 TROWSUM's FP32 reduction order along the last dimension."""
    import torch

    if values.dtype != torch.float32:
        raise ValueError(f"A5 FP32 TROWSUM golden requires float32, got {values.dtype}")
    if values.shape[-1] % _A5_FP32_VECTOR_LANES != 0:
        raise ValueError(
            f"A5 FP32 TROWSUM width must be divisible by {_A5_FP32_VECTOR_LANES}, "
            f"got {values.shape[-1]}"
        )

    groups = values.reshape(*values.shape[:-1], -1, _A5_FP32_VECTOR_LANES)
    while groups.shape[-1] > 1:
        pairs = groups.reshape(*groups.shape[:-1], -1, 2)
        groups = pairs[..., 0] + pairs[..., 1]

    group_sums = groups[..., 0]
    total = torch.zeros_like(group_sums[..., :1])
    for group in range(group_sums.shape[-1]):
        total += group_sums[..., group:group + 1]
    return total


def _golden_a5_high_precision_rsqrt(value):
    """Match A5 high-precision FP32 rsqrt without host FP32 double rounding."""
    import torch

    return torch.rsqrt(value.to(torch.float64)).to(torch.float32)


def _golden_a5_cube_bf16_matmul(lhs, rhs):
    """Return ``lhs @ rhs`` in A5 BF16 Cube's K16 MAD accumulation order."""
    import torch

    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError("A5 Cube golden requires two rank-2 matrices")
    m_dim, k_dim = lhs.shape
    rhs_k_dim, n_dim = rhs.shape
    if k_dim != rhs_k_dim or k_dim % _A5_CUBE_ACC_K != 0:
        raise ValueError(
            f"A5 Cube golden requires equal K divisible by {_A5_CUBE_ACC_K}, "
            f"got {k_dim} and {rhs_k_dim}"
        )

    k_groups = k_dim // _A5_CUBE_ACC_K
    x_groups = lhs.to(torch.bfloat16).reshape(m_dim, k_groups, _A5_CUBE_ACC_K).double()
    w_groups = rhs.to(torch.bfloat16).T.contiguous().reshape(n_dim, k_groups, _A5_CUBE_ACC_K).double()
    out = torch.zeros(m_dim, n_dim, dtype=torch.float32, device=lhs.device)
    for n0 in range(0, n_dim, _A5_CUBE_N_TILE):
        n1 = min(n0 + _A5_CUBE_N_TILE, n_dim)
        acc = torch.zeros(m_dim, n1 - n0, dtype=torch.float32, device=lhs.device)
        for group0 in range(0, k_groups, _A5_CUBE_K_GROUP_TILE):
            group1 = min(group0 + _A5_CUBE_K_GROUP_TILE, k_groups)
            group_dots = torch.einsum(
                "mgk,ngk->mng",
                x_groups[:, group0:group1],
                w_groups[n0:n1, group0:group1],
            )
            for group in range(group1 - group0):
                acc = (acc.double() + group_dots[:, :, group]).float()
        out[:, n0:n1] = acc
    return out


def _golden_a5_split_k_bf16_matmul(lhs, rhs, *, splits, k_per_split):
    """Mirror per-split continuous K16 Cube MADs, then ascending split reduction."""
    if lhs.shape[-1] != splits * k_per_split:
        raise ValueError(
            f"split-K golden expected K={splits * k_per_split}, got {lhs.shape[-1]}"
        )
    total = None
    for split in range(splits):
        k0 = split * k_per_split
        partial = _golden_a5_cube_bf16_matmul(
            lhs[:, k0:k0 + k_per_split],
            rhs[k0:k0 + k_per_split],
        )
        total = partial if total is None else total + partial
    return total


def _golden_a5_chunked_rms_inv(values, *, chunk_size, eps):
    """A5 FP32 RMS: TROWSUM each chunk, add chunks ascending, then HP rsqrt."""
    import torch

    if values.dtype != torch.float32:
        raise ValueError(f"A5 RMS golden requires float32, got {values.dtype}")
    width = values.shape[-1]
    if width % chunk_size != 0:
        raise ValueError(f"A5 RMS width {width} must be divisible by chunk {chunk_size}")
    sq_sum = torch.zeros(
        *values.shape[:-1], 1, dtype=torch.float32, device=values.device
    )
    for k0 in range(0, width, chunk_size):
        chunk = values[..., k0:k0 + chunk_size]
        sq_sum += _golden_a5_trowsum_fp32(chunk * chunk)
    rms_arg = sq_sum * (1.0 / width) + eps
    return _golden_a5_high_precision_rsqrt(rms_arg)


def _golden_qr_rms_norm_quant(qr_fp32, gamma_cq):
    """Mirror QR's A5 RMS, raw-gamma amax association, and nested scaling."""
    import torch

    gamma_fp32 = gamma_cq.to(torch.bfloat16).float()
    qr_inv_rms = _golden_a5_chunked_rms_inv(qr_fp32, chunk_size=Q_LORA_TILE, eps=EPS)

    qr_amax_g = torch.zeros(*qr_fp32.shape[:-1], 1, dtype=torch.float32, device=qr_fp32.device)
    for k0 in range(0, qr_fp32.shape[-1], Q_LORA_TILE):
        qr_chunk = qr_fp32[..., k0:k0 + Q_LORA_TILE]
        gamma_chunk = gamma_fp32[k0:k0 + Q_LORA_TILE]
        qr_gamma = qr_chunk * gamma_chunk
        qr_gamma_amax = qr_gamma.abs().amax(dim=-1, keepdim=True)
        qr_amax_g = torch.maximum(qr_amax_g, qr_gamma_amax)

    qr_tile_amax = (qr_inv_rms * qr_amax_g).clamp_min(INT8_AMAX_EPS)
    qr_scale_quant = torch.full_like(qr_tile_amax, INT8_SCALE_MAX) / qr_tile_amax
    qr_scale_dequant = torch.ones_like(qr_scale_quant) / qr_scale_quant
    qr_normed = (qr_fp32 * qr_inv_rms) * gamma_fp32
    qr_i32 = torch.round(qr_normed * qr_scale_quant).to(torch.int32)
    qr_i8 = qr_i32.to(torch.float16).to(torch.int8)
    return qr_i8, qr_scale_dequant


def _golden_a5_q_head_rms_norm(q_full):
    """Mirror the fused Q dequant kernel's HEAD_DIM TROWSUM and HP rsqrt."""
    q_inv_rms = _golden_a5_chunked_rms_inv(q_full, chunk_size=HEAD_DIM, eps=EPS)
    return q_full * q_inv_rms


def golden_qkv_proj_rope(tensors):
    """Torch reference: Q/KV LoRA + RoPE for an already attention-normalized input."""
    import torch

    full_t_dim = tensors["x"].shape[0]
    active_t_dim = full_t_dim
    if "num_tokens" in tensors:
        active_t_dim = max(0, min(int(tensors["num_tokens"]), full_t_dim))
        tensors["q"].zero_()
        tensors["kv"].zero_()
        tensors["qr"].zero_()
        tensors["qr_scale"].zero_()
        if active_t_dim == 0:
            return

    x = tensors["x"][:active_t_dim].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv = tensors["wkv"].float()
    rope_cos = tensors["rope_cos"][:active_t_dim].float()
    rope_sin = tensors["rope_sin"][:active_t_dim].float()
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    def rms_norm(x, gamma, eps=EPS):
        inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        return x * inv * gamma

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

    t_dim = active_t_dim
    token_x = x.view(t_dim, D)

    qr_fp32 = _golden_a5_split_k_bf16_matmul(token_x, wq_a, splits=QR_SPLIT_TILE, k_per_split=QR_K_SPLIT_TILE)
    # W8A8C16: wq_b W8 per-output-channel int8; qr_out A8 per-token int8.
    qr_i8, qr_scale = _golden_qr_rms_norm_quant(qr_fp32, gamma_cq)
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(t_dim, H, HEAD_DIM)
    q_full = _golden_a5_q_head_rms_norm(q_full)  # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    kv_proj = _golden_a5_split_k_bf16_matmul(token_x, wkv, splits=KV_SPLIT_TILE, k_per_split=KV_K_SPLIT_TILE)
    kv_full = rms_norm(kv_proj, gamma_ckv)
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)               # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:active_t_dim] = q_out.to(torch.bfloat16)
    tensors["kv"][:active_t_dim] = kv_out.to(torch.bfloat16)
    tensors["qr"][:active_t_dim] = qr_i8
    tensors["qr_scale"][:active_t_dim] = qr_scale


def _reference_q_from_quantized_qr(
    qr,
    qr_scale,
    wq_b,
    wq_b_scale,
    rope_cos,
    rope_sin,
):
    """Recompute the Q path downstream of the emitted INT8 QR boundary."""
    import torch

    t_dim = qr.shape[0]
    q_i32 = torch.matmul(qr.to(torch.int32), wq_b.to(torch.int32))
    q_scaled = q_i32.float() * qr_scale.float()
    q_scaled = q_scaled * wq_b_scale.float().view(1, -1)
    q_full = q_scaled.view(t_dim, H, HEAD_DIM)
    q_full = _golden_a5_q_head_rms_norm(q_full)

    q_pair = q_full[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    q_even, q_odd = q_pair[..., 0], q_pair[..., 1]
    cos = rope_cos.float()[..., :ROPE_HALF].unsqueeze(-2)
    sin = rope_sin.float()[..., :ROPE_HALF].unsqueeze(-2)
    y_even = (q_even * cos - q_odd * sin).to(torch.bfloat16)
    y_odd = (q_even * sin + q_odd * cos).to(torch.bfloat16)
    q_rope = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    return torch.cat([q_full[..., :NOPE_DIM], q_rope], dim=-1).to(torch.bfloat16)


def quantized_qr_compare(
    *,
    max_code_step=1,
    max_changed_ratio=0.005,
    max_changed_per_row_ratio=0.005,
    max_show=10,
):
    """Bound both the magnitude and population of QR quantization-boundary changes."""
    import torch

    if max_code_step < 0:
        raise ValueError(f"max_code_step must be non-negative, got {max_code_step}")
    if not 0.0 <= max_changed_ratio <= 1.0:
        raise ValueError(
            f"max_changed_ratio must be in [0, 1], got {max_changed_ratio}"
        )
    if not 0.0 <= max_changed_per_row_ratio <= 1.0:
        raise ValueError(
            "max_changed_per_row_ratio must be in [0, 1], got "
            f"{max_changed_per_row_ratio}"
        )

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, inputs, rtol, atol
        actual = actual.cpu()
        expected = expected.cpu()
        if actual.shape != expected.shape:
            return False, (
                f"    QR shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )
        if actual.dtype != torch.int8 or expected.dtype != torch.int8:
            return False, (
                f"    QR comparator requires int8 tensors, got "
                f"{actual.dtype} and {expected.dtype}"
            )
        diff = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
        changed = diff != 0
        changed_count = int(changed.count_nonzero().item())
        changed_limit = round(max_changed_ratio * diff.numel())
        changed_per_row = changed.reshape(-1, changed.shape[-1]).sum(dim=-1)
        row_limit = int(max_changed_per_row_ratio * changed.shape[-1])
        overfull_rows = changed_per_row > row_limit
        overfull_row_count = int(overfull_rows.count_nonzero().item())
        too_large = diff > max_code_step
        too_large_count = int(too_large.count_nonzero().item())
        if (
            changed_count <= changed_limit
            and overfull_row_count == 0
            and too_large_count == 0
        ):
            return True, ""

        changed_indices = changed.flatten().nonzero(as_tuple=False).flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        flat_diff = diff.flatten()
        lines = []
        for index in changed_indices[:max_show].tolist():
            lines.append(
                f"      [{index}] actual={int(flat_actual[index])} "
                f"expected={int(flat_expected[index])} "
                f"code_step={int(flat_diff[index])}"
            )
        return False, (
            f"    QR quantization-boundary mismatch: changed={changed_count}/"
            f"{diff.numel()} (allowed<={max_changed_ratio:.4%}, "
            f"threshold={changed_limit}), code_step>{max_code_step}: "
            f"{too_large_count}, rows>{row_limit} changed codes: "
            f"{overfull_row_count}\n"
            + "\n".join(lines)
        )

    compare.__name__ = (
        f"quantized_qr_compare(max_code_step={max_code_step},"
        f"max_changed_ratio={max_changed_ratio},"
        f"max_changed_per_row_ratio={max_changed_per_row_ratio})"
    )
    return compare


def qr_scale_compare(
    *,
    atol=2.5e-5,
    rtol=5e-3,
    max_error_ratio=0.0,
    max_show=10,
):
    """Validate QR dequant scales with aggregate and per-row bounds."""
    import torch

    if atol < 0 or rtol < 0:
        raise ValueError("QR scale tolerances must be non-negative")
    if not 0.0 <= max_error_ratio <= 1.0:
        raise ValueError(
            f"max_error_ratio must be in [0, 1], got {max_error_ratio}"
        )
    scale_atol = atol
    scale_rtol = rtol

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, inputs, rtol, atol
        actual = actual.cpu().to(torch.float32)
        expected = expected.cpu().to(torch.float32)
        if actual.shape != expected.shape:
            return False, (
                f"    QR scale shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )
        if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
            return False, "    QR scales contain NaN or Inf"
        if (actual <= 0).any().item() or (expected <= 0).any().item():
            return False, "    QR scales must be positive"

        diff = (actual - expected).abs()
        tolerance = scale_atol + scale_rtol * expected.abs()
        bad = diff > tolerance
        bad_count = int(bad.count_nonzero().item())
        threshold = round(max_error_ratio * actual.numel())
        hard_bad = diff > (2.0 * tolerance)
        hard_bad_count = int(hard_bad.count_nonzero().item())
        if bad_count <= threshold and hard_bad_count == 0:
            return True, ""

        bad_indices = bad.flatten().nonzero(as_tuple=False).flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        flat_diff = diff.flatten()
        flat_tolerance = tolerance.flatten()
        lines = []
        for index in bad_indices[:max_show].tolist():
            lines.append(
                f"      [{index}] actual={float(flat_actual[index]):.8g} "
                f"expected={float(flat_expected[index]):.8g} "
                f"diff={float(flat_diff[index]):.4g} "
                f"tol={float(flat_tolerance[index]):.4g}"
            )
        return False, (
            f"    QR scale mismatch: bad={bad_count}/{actual.numel()} "
            f"(allowed<={max_error_ratio:.4%}, threshold={threshold}), "
            f"hard_bad={hard_bad_count}, atol={scale_atol}, rtol={scale_rtol}\n"
            + "\n".join(lines)
        )

    compare.__name__ = (
        f"qr_scale_compare(atol={scale_atol},rtol={scale_rtol},"
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


def q_from_runtime_qr_compare(
    *,
    atol=1e-4,
    rtol=1.0 / 128,
    max_error_ratio=0.005,
):
    """Validate Q against a reference conditioned on the emitted QR codes."""
    from golden import ratio_allclose

    base_compare = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=max_error_ratio)

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del expected
        required_outputs = ("qr", "qr_scale")
        required_inputs = ("wq_b", "wq_b_scale", "rope_cos", "rope_sin")
        missing_outputs = [name for name in required_outputs if name not in actual_outputs]
        missing_inputs = [name for name in required_inputs if name not in inputs]
        if missing_outputs or missing_inputs:
            return False, (
                "    conditioned Q comparator is missing "
                f"outputs={missing_outputs}, inputs={missing_inputs}"
            )

        conditioned = _reference_q_from_quantized_qr(
            actual_outputs["qr"].cpu(),
            actual_outputs["qr_scale"].cpu(),
            inputs["wq_b"].cpu(),
            inputs["wq_b_scale"].cpu(),
            inputs["rope_cos"].cpu(),
            inputs["rope_sin"].cpu(),
        )
        if actual.shape != conditioned.shape:
            return False, (
                f"    conditioned Q shape mismatch: actual={tuple(actual.shape)} "
                f"reference={tuple(conditioned.shape)}"
            )
        ok, detail = base_compare(
            actual,
            conditioned,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )
        if ok:
            return True, ""
        return False, "    Q downstream of emitted QR does not match:\n" + detail

    compare.__name__ = (
        f"q_from_runtime_qr_compare(atol={atol},rtol={rtol},"
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


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
        TensorSpec("x",         [T, D],                 torch.bfloat16, init_value=init_x),
        TensorSpec("wq_a",      [D, Q_LORA],            torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b",      [Q_LORA, H * HEAD_DIM], torch.int8,     init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv",       [D, HEAD_DIM],          torch.bfloat16, init_value=init_wkv),
        TensorSpec("rope_cos",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_cos),
        TensorSpec("rope_sin",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_sin),
        TensorSpec("gamma_cq",  [Q_LORA],               torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM],             torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("q",         [T, H, HEAD_DIM],       torch.bfloat16),
        TensorSpec("kv",        [T, HEAD_DIM],          torch.bfloat16),
        TensorSpec("qr",        [T, Q_LORA],            torch.int8),
        TensorSpec("qr_scale",  [T, 1],                 torch.float32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["decode", "prefill", "all"], default="all",
        help="Use decode or prefill batch sizes, or 'all' to test both.",
    )
    parser.add_argument(
        "--enable-chip-swimlane", type=int, choices=[0, 1, 2, 4], default=0,
        help="chip swimlane level: 0=off, 1=per-kernel AICore timing "
        "(prints the per-function Task Statistics table), 2=+AICPU timing.",
    )
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- qkv_proj_rope {mode_name}: B={B}, S={S} ---")
        result = run(
            fn=qkv_proj_rope_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_qkv_proj_rope,
            rtol=5e-3,
            atol=5e-3,
            compare_fn={
                "q":        q_from_runtime_qr_compare(atol=1e-4, rtol=1.0 / 128),
                "kv":       ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "qr":       quantized_qr_compare(max_code_step=1, max_changed_ratio=0.005),
                "qr_scale": qr_scale_compare(atol=2.5e-5, rtol=5e-3, max_error_ratio=0.0),
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
