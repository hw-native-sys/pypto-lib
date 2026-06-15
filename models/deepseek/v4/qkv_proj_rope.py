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

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS


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

# tiling
ROPE_TILE = 64
ROPE_PAIR_TILE = 32
HEAD_TILE = 64
Q_PROJ_OUT_TILE = 128
Q_PROJ_TILE = 512
Q_LORA_TILE = 32
D_TILE = 128
KV_TILE = 32
QUANT_TILE = 32
T_TILE = 8
QPROJ_T_TILE = 16
KV_RMS_T_TILE = 16
Q_ROPE_T_TILE = 32
KV_ROPE_T_TILE = 32
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % QPROJ_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % QPROJ_T_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % KV_RMS_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % KV_RMS_T_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % Q_ROPE_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % Q_ROPE_T_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % KV_ROPE_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % KV_ROPE_T_TILE == 0


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
                rope_pos = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
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
):
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])

    qr_fp32 = pl.create_tensor([t_dim, Q_LORA], dtype=pl.FP32)
    for qbg_idx in pl.spmd((Q_LORA // Q_LORA_TILE) // 2, name_hint="qr_proj_matmul"):
        qbg = qbg_idx * 2
        for q_inner in pl.pipeline(2, stage=2):
            q_a_col0 = (qbg + q_inner) * Q_LORA_TILE
            for tc in pl.range(t_dim // QPROJ_T_TILE):
                t0 = tc * QPROJ_T_TILE
                q_x_chunk_bf16 = x_view[t0 : t0 + QPROJ_T_TILE, 0:D_TILE]
                w_chunk = wq_a[0:D_TILE, q_a_col0 : q_a_col0 + Q_LORA_TILE]
                q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                for db in pl.pipeline(1, D // D_TILE, stage=2):
                    qr_d0 = db * D_TILE
                    q_x_chunk_bf16 = x_view[t0 : t0 + QPROJ_T_TILE, qr_d0 : qr_d0 + D_TILE]
                    w_chunk = wq_a[qr_d0 : qr_d0 + D_TILE, q_a_col0 : q_a_col0 + Q_LORA_TILE]
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
                qr_fp32[t0 : t0 + QPROJ_T_TILE, q_a_col0 : q_a_col0 + Q_LORA_TILE] = q_acc

    # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
    for tg_idx in pl.spmd(t_dim // T_TILE, name_hint="qr_rms_norm_quant"):
        tg = tg_idx * T_TILE
        qr_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        qr_tile_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for qr_rms_qb in pl.pipeline(Q_LORA // Q_LORA_TILE, stage=2):
            qr_rms_col0 = qr_rms_qb * Q_LORA_TILE
            qr_rms_chunk = qr_fp32[tg : tg + T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE]
            qr_sq_sum = pl.add(qr_sq_sum, pl.reshape(pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk)), [1, T_TILE]))
        qr_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)))
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])

        for qb in pl.pipeline(Q_LORA // Q_LORA_TILE, stage=2):
            qr_norm_col0 = qb * Q_LORA_TILE
            qr_norm_chunk = qr_fp32[tg : tg + T_TILE, qr_norm_col0 : qr_norm_col0 + Q_LORA_TILE]
            gamma_chunk = pl.reshape(
                pl.cast(gamma_cq[qr_norm_col0 : qr_norm_col0 + Q_LORA_TILE], target_type=pl.FP32),
                [1, Q_LORA_TILE],
            )
            qr_normed = pl.col_expand_mul(pl.row_expand_mul(qr_norm_chunk, qr_inv_rms_t), gamma_chunk)
            qr_normed_bf16 = pl.cast(qr_normed, target_type=pl.BF16, mode="rint")
            qr_norm_amax_f32 = pl.cast(qr_normed_bf16, target_type=pl.FP32)
            qr_norm_amax_abs = pl.maximum(qr_norm_amax_f32, pl.neg(qr_norm_amax_f32))
            qr_tile_amax = pl.maximum(qr_tile_amax, pl.reshape(pl.row_max(qr_norm_amax_abs), [1, T_TILE]))

        qr_scale_quant_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_tile_amax)
        qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
        qr_tile_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T_TILE, 1])
        qr_scale_view[tg : tg + T_TILE, :] = qr_tile_scale_dq

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = qr_fp32[tg : tg + T_TILE, qa : qa + QUANT_TILE]
            gamma_q_chunk = pl.reshape(
                pl.cast(gamma_cq[qa : qa + QUANT_TILE], target_type=pl.FP32),
                [1, QUANT_TILE],
            )
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
            qr_q_normed_bf16 = pl.cast(qr_q_normed, target_type=pl.BF16, mode="rint")
            qr_q_f32 = pl.cast(qr_q_normed_bf16, target_type=pl.FP32)
            qr_q_scaled = pl.row_expand_mul(qr_q_f32, qr_scale_quant_t)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr_view[tg : tg + T_TILE, qa : qa + QUANT_TILE] = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")

    # UN-MIXED qproj: pure-matmul scope (cube, INT32 -> GM) + separate dequant scope (vec).
    # The fused form pinned the dequant (vec) right after each matmul, so its AIV work ran in
    # qproj's window and stole AIV cores from the critical qr_proj_aiv. Split so the scheduler
    # can defer the off-critical-path dequant (q has large downstream slack) to when AIV is free.
    q_proj_fp32 = pl.create_tensor([t_dim, H * HEAD_DIM], dtype=pl.FP32)
    q_proj_i32 = pl.create_tensor([t_dim, H * HEAD_DIM], dtype=pl.INT32)
    for hg_idx in pl.spmd(((H * HEAD_DIM) // Q_PROJ_OUT_TILE) // 16, name_hint="qproj_matmul"):
        hg = hg_idx * 16
        for h_inner in pl.pipeline(16, stage=2):
            w_col0 = (hg + h_inner) * Q_PROJ_OUT_TILE
            for tc in pl.range(t_dim // QPROJ_T_TILE):
                t0 = tc * QPROJ_T_TILE
                qr_i8_chunk = qr_view[t0 : t0 + QPROJ_T_TILE, 0:Q_PROJ_TILE]
                wq_chunk = wq_b[0:Q_PROJ_TILE, w_col0 : w_col0 + Q_PROJ_OUT_TILE]
                col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                for qb in pl.pipeline(1, Q_LORA // Q_PROJ_TILE, stage=2):
                    qr_proj_col0 = qb * Q_PROJ_TILE
                    qr_i8_chunk = qr_view[t0 : t0 + QPROJ_T_TILE, qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE]
                    wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, w_col0 : w_col0 + Q_PROJ_OUT_TILE]
                    col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
                q_proj_i32[t0 : t0 + QPROJ_T_TILE, w_col0 : w_col0 + Q_PROJ_OUT_TILE] = col_acc

    for hg_idx in pl.spmd(((H * HEAD_DIM) // Q_PROJ_OUT_TILE) // 16, name_hint="qproj_dequant"):
        hg = hg_idx * 16
        for h_inner in pl.pipeline(16, stage=2):
            w_col0 = (hg + h_inner) * Q_PROJ_OUT_TILE
            w_scale = pl.reshape(wq_b_scale[w_col0 : w_col0 + Q_PROJ_OUT_TILE], [1, Q_PROJ_OUT_TILE])
            for tc in pl.pipeline(0, t_dim, QPROJ_T_TILE, stage=2):
                col_acc_t = q_proj_i32[tc : tc + QPROJ_T_TILE, w_col0 : w_col0 + Q_PROJ_OUT_TILE]
                col_fp32 = pl.cast(col_acc_t, target_type=pl.FP32, mode="none")
                qr_scale_dq_t = qr_scale_view[tc : tc + QPROJ_T_TILE, :]
                col_dequant = pl.col_expand_mul(pl.row_expand_mul(col_fp32, qr_scale_dq_t), w_scale)
                q_proj_fp32[tc : tc + QPROJ_T_TILE, w_col0 : w_col0 + Q_PROJ_OUT_TILE] = col_dequant

    # Per-head RMS, NOPE projection, and RoPE rotation, staged through
    # q_head_inv_rms_all and q_rope_pair_stage.
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    q_head_inv_rms_all = pl.create_tensor([H, t_dim], dtype=pl.FP32)
    for hg_idx in pl.spmd(H // 2, name_hint="q_head_rms_nope"):
        hg = hg_idx * 2
        for h_inner in pl.range(2):
            h = hg + h_inner
            h0 = h * HEAD_DIM
            for tg_idx in pl.range(t_dim // Q_ROPE_T_TILE):
                tg = tg_idx * Q_ROPE_T_TILE
                q_head_sq_sum = pl.full([1, Q_ROPE_T_TILE], dtype=pl.FP32, value=0.0)
                for db in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=2):
                    d0 = h0 + db * HEAD_TILE
                    q_head_chunk = q_proj_fp32[tg : tg + Q_ROPE_T_TILE, d0 : d0 + HEAD_TILE]
                    q_head_sq_sum = pl.add(
                        q_head_sq_sum,
                        pl.reshape(pl.row_sum(pl.mul(q_head_chunk, q_head_chunk)), [1, Q_ROPE_T_TILE]),
                    )
                q_head_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS)))
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])
                q_head_inv_rms_all[h : h + 1, tg : tg + Q_ROPE_T_TILE] = q_head_inv_rms

                for nb in pl.pipeline(NOPE_DIM // HEAD_TILE, stage=2):
                    n0 = nb * HEAD_TILE
                    q_nope_chunk = q_proj_fp32[tg : tg + Q_ROPE_T_TILE, h0 + n0 : h0 + n0 + HEAD_TILE]
                    q_normed = pl.row_expand_mul(q_nope_chunk, q_head_inv_rms_t)
                    q_flat[tg : tg + Q_ROPE_T_TILE, h0 + n0 : h0 + n0 + HEAD_TILE] = pl.cast(
                        q_normed, target_type=pl.BF16, mode="rint"
                    )

    # Per-head RoPE (CANN A3 rotate_interleaved): stay on the interleaved layout and
    # rotate via an i^1 swap gather + sign mask, dropping the de-interleave gather +
    # re-interleave scatter. The rotation indices/sign and the interleave-duplicated
    # cos/sin are built ENTIRELY IN-KERNEL (no host inputs): swap_idx (j^1), sign
    # ([-1,+1,...]) and dup_idx (j>>1) come from pl.arange per task, and cos_il/sin_il
    # are dup-gathered from rope_cos/rope_sin in-task. The prior in-task index-tile
    # tail clobber (-> tgather UB-OOB -> 507018 hang) that once forced these to be
    # kernel inputs is resolved. inv_rms is per-row so it factors out of the rotation
    # and is applied after; the writeback into q_flat is folded in (no FP32 GM stage).
    #   swapped = gather(x, j^1) = [x1,x0,x3,x2,...]; sign = [-1,+1,...]
    #   out = inv_rms * (x*cos_il + swapped*sign*sin_il)
    for hg_idx in pl.spmd(H // 2, name_hint="q_head_rope_fused"):
        hg = hg_idx * 2
        # In-kernel A3 index/sign build (per task, reused across the inner tg/h loop).
        q_ones = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        q_col = pl.col_expand_mul(q_ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        q_dup_f = pl.cast(pl.cast(pl.mul(q_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        q_dup_idx = pl.cast(q_dup_f, target_type=pl.INT32)                                       # j>>1
        q_lane = pl.sub(q_col, pl.mul(q_dup_f, 2.0))                                             # j%2
        q_swap_idx = pl.cast(pl.sub(pl.add(q_col, 1.0), pl.mul(q_lane, 2.0)), target_type=pl.INT32)  # j^1
        q_sign = pl.sub(pl.mul(q_lane, 2.0), 1.0)                                                # [-1,+1,...]
        # tg-outer / head-inner: cos_il/sin_il are head-independent, so dup-gather once
        # per tg and reuse for both heads.
        for tg_idx in pl.range(t_dim // Q_ROPE_T_TILE):
            tg = tg_idx * Q_ROPE_T_TILE
            q_cos_il = pl.gather(pl.cast(rope_cos_view[tg : tg + Q_ROPE_T_TILE, :], target_type=pl.FP32), dim=-1, index=q_dup_idx)
            q_sin_il = pl.gather(pl.cast(rope_sin_view[tg : tg + Q_ROPE_T_TILE, :], target_type=pl.FP32), dim=-1, index=q_dup_idx)
            for h_inner in pl.range(2):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_rope_inv_rms_chunk = pl.reshape(
                    q_head_inv_rms_all[h : h + 1, tg : tg + Q_ROPE_T_TILE], [Q_ROPE_T_TILE, 1]
                )
                q_rope_chunk = q_proj_fp32[tg : tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + NOPE_DIM + ROPE_DIM]
                q_rope_swapped = pl.gather(q_rope_chunk, dim=-1, index=q_swap_idx)
                q_rope_rot = pl.add(pl.mul(q_rope_chunk, q_cos_il), pl.mul(pl.mul(q_rope_swapped, q_sign), q_sin_il))
                q_flat[tg : tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + NOPE_DIM + ROPE_DIM] = pl.cast(
                    pl.row_expand_mul(q_rope_rot, q_rope_inv_rms_chunk), target_type=pl.BF16, mode="rint"
                )

    kv_fp32 = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    for kbg in pl.spmd((HEAD_DIM // KV_TILE) // 2, name_hint="kv_proj_matmul"):
        for k_inner in pl.pipeline(2, stage=2):
            kv_col0 = (kbg * 2 + k_inner) * KV_TILE
            for tc in pl.range(t_dim // QPROJ_T_TILE):
                t0 = tc * QPROJ_T_TILE
                kv_x_chunk_bf16 = x_view[t0 : t0 + QPROJ_T_TILE, 0:D_TILE]
                wkv_chunk = wkv[0:D_TILE, kv_col0 : kv_col0 + KV_TILE]
                kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                for db in pl.pipeline(1, D // D_TILE, stage=2):
                    d0 = db * D_TILE
                    kv_x_chunk_bf16 = x_view[t0 : t0 + QPROJ_T_TILE, d0 : d0 + D_TILE]
                    wkv_chunk = wkv[d0 : d0 + D_TILE, kv_col0 : kv_col0 + KV_TILE]
                    kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
                kv_fp32[t0 : t0 + QPROJ_T_TILE, kv_col0 : kv_col0 + KV_TILE] = kv_acc

    kv_inv_rms_tensor = pl.create_tensor([1, t_dim], dtype=pl.FP32)
    for tg_idx in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="kv_rms_norm"):
        tg = tg_idx * KV_RMS_T_TILE
        kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HEAD_DIM // KV_TILE, stage=2):
            kv_sq_col0 = kb * KV_TILE
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
            kv_sq_sum = pl.add(kv_sq_sum, pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, KV_RMS_T_TILE]))
        kv_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)))
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])
        kv_inv_rms_tensor[0:1, tg : tg + KV_RMS_T_TILE] = kv_inv_rms

        for nb in pl.pipeline(NOPE_DIM // KV_TILE, stage=2):
            n0 = nb * KV_TILE
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
            gamma_kv_chunk = pl.reshape(
                pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32),
                [1, KV_TILE],
            )
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv_view[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")

    # Fused KV RoPE: rmsnorm-scaled rope slice, gather-rotate-scatter, write back.
    # Per-T-tile spmd: each task owns one [KV_ROPE_T_TILE, ROPE_DIM] row block,
    # writing disjoint row ranges of kv -> no conflict. Was single-core
    # CORE_GROUP (the 507046 that previously blocked spmd here is resolved); each
    # task's chunk keeps Vec UB well under the 192 KB cap.
    for tg_idx in pl.spmd(t_dim // KV_ROPE_T_TILE, name_hint="kv_rope_fused"):
        tg = tg_idx * KV_ROPE_T_TILE
        gamma_rope = pl.reshape(
            pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32),
            [1, ROPE_DIM],
        )
        kv_rope_inv_rms_chunk = pl.reshape(
            kv_inv_rms_tensor[0:1, tg : tg + KV_ROPE_T_TILE], [KV_ROPE_T_TILE, 1]
        )
        kv_rope_chunk = kv_fp32[tg : tg + KV_ROPE_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
        # A3 interleaved swap-gather (same form as q_head_rope_fused), built in-kernel.
        # gamma is folded into kv_rope_norm_chunk BEFORE the swap so the swapped lane
        # n[j^1] correctly carries gamma[j^1] (gamma is per-column, does NOT commute);
        # inv_rms is per-row so it commutes. out[j] = n[j]*cos_il[j] + n[j^1]*sign[j]*sin_il[j].
        kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_rope_inv_rms_chunk), gamma_rope)
        kv_ones = pl.full([KV_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        kv_col = pl.col_expand_mul(kv_ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        kv_dup_idx = pl.cast(kv_dup_f, target_type=pl.INT32)                                       # j>>1
        kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))                                            # j%2
        kv_swap_idx = pl.cast(pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0)), target_type=pl.INT32)  # j^1
        kv_sign = pl.sub(pl.mul(kv_lane, 2.0), 1.0)                                                # [-1,+1,...]
        kv_cos_il = pl.gather(pl.cast(rope_cos_view[tg : tg + KV_ROPE_T_TILE, :], target_type=pl.FP32), dim=-1, index=kv_dup_idx)
        kv_sin_il = pl.gather(pl.cast(rope_sin_view[tg : tg + KV_ROPE_T_TILE, :], target_type=pl.FP32), dim=-1, index=kv_dup_idx)
        kv_swapped = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx)
        kv_rope_rot = pl.add(
            pl.mul(kv_rope_norm_chunk, kv_cos_il),
            pl.mul(pl.mul(kv_swapped, kv_sign), kv_sin_il),
        )
        kv_view[tg : tg + KV_ROPE_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM] = pl.cast(
            kv_rope_rot, target_type=pl.BF16, mode="rint"
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

    q = qkv_proj_rope(
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
    )
    return q


def golden_qkv_proj_rope(tensors):
    """Torch reference: Q/KV LoRA + RoPE for an already attention-normalized input."""
    import torch

    x = tensors["x"].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv = tensors["wkv"].float()
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    def int8_quant_per_row(x):
        rows = x.reshape(-1, x.shape[-1]).float()
        amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = rows * scale_quant
        out_i32 = torch.round(scaled).to(torch.int32)
        out_half = out_i32.to(torch.float16)
        out_i8 = out_half.to(torch.int8)
        return out_i8.reshape_as(x), (1.0 / scale_quant).reshape(*x.shape[:-1], 1)

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
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)   # [T, Q_LORA]
    # W8A8C16: wq_b W8 per-output-channel int8; qr_out A8 per-token int8.
    # flash: also quantizes wq_a/wkv to fp8 (default Linear dtype).
    qr_out_bf16 = qr_out.to(torch.bfloat16)
    qr_i8, qr_scale = int8_quant_per_row(qr_out_bf16.float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(t_dim, H, HEAD_DIM)
    inv = torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_full = q_full * inv                                            # per-head RMSNorm (no gamma)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos, rope_sin)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    # KV path
    kv_full = rms_norm(matmul_bf16_input_fp32(token_x, wkv), gamma_ckv)  # [T, HEAD_DIM]
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)               # add a pseudo head dim
    kv_rope = apply_rope(kv_rope_in, rope_cos, rope_sin).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:]  = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_i8
    tensors["qr_scale"][:] = qr_scale


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables, materialize_token_rope_tables

    T = B * S
    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, 4, dtype=torch.bfloat16)
    positions = torch.arange(T, dtype=torch.int32)
    shared_rope_cos, shared_rope_sin = materialize_token_rope_tables(shared_freqs_cos, shared_freqs_sin, positions)

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x():
        return (torch.randn(T, D) - 0.5)
    def init_wq_a():
        return (torch.randn(D, Q_LORA) - 0.5) / (D ** 0.5)
    def init_wq_b():
        return (torch.randn(Q_LORA, H * HEAD_DIM) - 0.5) / ((H * HEAD_DIM) ** 0.5)
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / (D ** 0.5)
    def init_cos():
        return shared_rope_cos.clone()
    def init_sin():
        return shared_rope_sin.clone()
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)

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
        TensorSpec("q",         [T, H, HEAD_DIM],       torch.bfloat16, is_output=True),
        TensorSpec("kv",        [T, HEAD_DIM],          torch.bfloat16, is_output=True),
        TensorSpec("qr",        [T, Q_LORA],            torch.int8,     is_output=True),
        TensorSpec("qr_scale",  [T, 1],                 torch.float32,  is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- qkv_proj_rope {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=qkv_proj_rope_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_qkv_proj_rope,
            # W8A8C16 q_proj adds INT8 quant/dequant round-off before per-head RMSNorm.
            rtol=5e-3,
            atol=5e-3,
            # Precision reference: pypto mla_prolog —
            # cann-recipes-infer/ops/pypto_python/example/test_mla_prolog_pypto.py
            compare_fn={
                "q":        ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "kv":       ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "qr":       ratio_allclose(atol=1, rtol=0, max_error_ratio=0),
                "qr_scale": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
