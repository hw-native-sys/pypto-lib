# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 single-token decode attn_norm fused + Q/KV LoRA + RoPE: produces (q, kv, qr) for the
attention body, with attn_norm fused at the front to save one GM round-trip."""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps

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


@pl.jit.inline
def qkv_proj_rope(
    x: pl.Tensor[[B, S, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
):
    x_flat = pl.reshape(x, [T, D])

    token_x_bf16 = pl.create_tensor([T, D], dtype=pl.BF16)
    for tg_idx in pl.spmd(T // T_TILE, name_hint="attn_norm"):
        tg = tg_idx * T_TILE
        x_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for rms_db in pl.pipeline(D // D_TILE, stage=2):
            rms_d0 = rms_db * D_TILE
            rms_x_chunk = pl.cast(x_flat[tg : tg + T_TILE, rms_d0 : rms_d0 + D_TILE], target_type=pl.FP32)
            x_sq_sum = pl.add(x_sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_chunk, rms_x_chunk)), [1, T_TILE]))
        x_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS)))
        x_inv_rms_t = pl.reshape(x_inv_rms, [T_TILE, 1])
        for apply_db in pl.pipeline(D // D_TILE, stage=2):
            apply_d0 = apply_db * D_TILE
            apply_x_chunk = pl.cast(x_flat[tg : tg + T_TILE, apply_d0 : apply_d0 + D_TILE], target_type=pl.FP32)
            norm_w_chunk = pl.reshape(norm_w[apply_d0 : apply_d0 + D_TILE], [1, D_TILE])
            x_normed = pl.col_expand_mul(pl.row_expand_mul(apply_x_chunk, x_inv_rms_t), norm_w_chunk)
            token_x_bf16[tg : tg + T_TILE, apply_d0 : apply_d0 + D_TILE] = pl.cast(x_normed, target_type=pl.BF16, mode="rint")

    qr_fp32 = pl.create_tensor([T, Q_LORA], dtype=pl.FP32)
    for qbg_idx in pl.spmd((Q_LORA // Q_LORA_TILE) // 2, name_hint="qr_proj_matmul"):
        qbg = qbg_idx * 2
        for q_inner in pl.pipeline(2, stage=2):
            q_a_col0 = (qbg + q_inner) * Q_LORA_TILE
            q_acc = pl.create_tensor([T, Q_LORA_TILE], dtype=pl.FP32)
            for db in pl.pipeline(0, D // D_TILE, stage=2):
                qr_d0 = db * D_TILE
                q_x_chunk_bf16 = token_x_bf16[:, qr_d0 : qr_d0 + D_TILE]
                w_chunk = wq_a[qr_d0 : qr_d0 + D_TILE, q_a_col0 : q_a_col0 + Q_LORA_TILE]
                if qr_d0 == 0:
                    q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                else:
                    q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_fp32[:, q_a_col0 : q_a_col0 + Q_LORA_TILE] = q_acc

    # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
    for tg_idx in pl.spmd(T // T_TILE, name_hint="qr_rms_norm_quant"):
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
        qr_scale[tg : tg + T_TILE, :] = qr_tile_scale_dq

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
            qr[tg : tg + T_TILE, qa : qa + QUANT_TILE] = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")

    q_proj_fp32 = pl.create_tensor([T, H * HEAD_DIM], dtype=pl.FP32)
    for hg_idx in pl.spmd(((H * HEAD_DIM) // Q_PROJ_OUT_TILE) // 16, name_hint="qproj"):
        hg = hg_idx * 16
        col_acc = pl.create_tensor([T, Q_PROJ_OUT_TILE], dtype=pl.INT32)
        for h_inner in pl.pipeline(16, stage=2):
            for qb in pl.pipeline(0, Q_LORA // Q_PROJ_TILE, stage=2):
                qr_proj_col0 = qb * Q_PROJ_TILE
                qr_i8_chunk = qr[:, qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE]
                wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, (hg + h_inner) * Q_PROJ_OUT_TILE : (hg + h_inner) * Q_PROJ_OUT_TILE + Q_PROJ_OUT_TILE]
                if qr_proj_col0 == 0:
                    col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                else:
                    col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
            w_col0 = (hg + h_inner) * Q_PROJ_OUT_TILE
            w_scale = pl.reshape(wq_b_scale[w_col0 : w_col0 + Q_PROJ_OUT_TILE], [1, Q_PROJ_OUT_TILE])
            for tc in pl.pipeline(0, T, QPROJ_T_TILE, stage=2):
                col_acc_t = col_acc[tc : tc + QPROJ_T_TILE, :]
                col_fp32 = pl.cast(col_acc_t, target_type=pl.FP32, mode="none")
                qr_scale_dq_t = qr_scale[tc : tc + QPROJ_T_TILE, :]
                col_dequant = pl.col_expand_mul(pl.row_expand_mul(col_fp32, qr_scale_dq_t), w_scale)
                q_proj_fp32[tc : tc + QPROJ_T_TILE, (hg + h_inner) * Q_PROJ_OUT_TILE : (hg + h_inner) * Q_PROJ_OUT_TILE + Q_PROJ_OUT_TILE] = col_dequant

    # Per-head RMS, NOPE projection, and RoPE rotation, staged through
    # q_head_inv_rms_all and q_rope_pair_stage.
    q_flat = pl.reshape(q, [T, H * HEAD_DIM])
    q_head_inv_rms_all = pl.create_tensor([H, T], dtype=pl.FP32)
    for hg_idx in pl.spmd(H // 2, name_hint="q_head_rms_nope"):
        hg = hg_idx * 2
        for h_inner in pl.range(2):
            h = hg + h_inner
            h0 = h * HEAD_DIM
            q_head_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
            for db in pl.pipeline(HEAD_DIM // HEAD_TILE, stage=2):
                d0 = h0 + db * HEAD_TILE
                q_head_chunk = q_proj_fp32[:, d0 : d0 + HEAD_TILE]
                q_head_sq_sum = pl.add(q_head_sq_sum, pl.reshape(pl.row_sum(pl.mul(q_head_chunk, q_head_chunk)), [1, T]))
            q_head_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS)))
            q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [T, 1])
            q_head_inv_rms_all[h : h + 1, :] = q_head_inv_rms

            for nb in pl.pipeline(NOPE_DIM // HEAD_TILE, stage=2):
                n0 = nb * HEAD_TILE
                q_nope_chunk = q_proj_fp32[:, h0 + n0 : h0 + n0 + HEAD_TILE]
                q_normed = pl.row_expand_mul(q_nope_chunk, q_head_inv_rms_t)
                q_flat[:, h0 + n0 : h0 + n0 + HEAD_TILE] = pl.cast(q_normed, target_type=pl.BF16, mode="rint")

    q_rope_pair_stage = pl.create_tensor([H * T, ROPE_DIM], dtype=pl.BF16)
    for hg_idx in pl.spmd(H // 8, name_hint="q_head_rope"):
        hg = hg_idx * 8
        q_head_inv_rms_t = pl.create_tensor([T, 1], dtype=pl.FP32)
        rope_cos_fp32 = pl.cast(rope_cos[:, :ROPE_HALF], target_type=pl.FP32)
        rope_sin_fp32 = pl.cast(rope_sin[:, :ROPE_HALF], target_type=pl.FP32)
        for h_inner in pl.range(8):
            q_head_inv_rms_t = pl.reshape(q_head_inv_rms_all[hg + h_inner : hg + h_inner + 1, :], [T, 1])
            q_rope = q_proj_fp32[:, (hg + h_inner) * HEAD_DIM + NOPE_DIM : (hg + h_inner) * HEAD_DIM + NOPE_DIM + ROPE_DIM]
            q_rope_norm = pl.row_expand_mul(q_rope, q_head_inv_rms_t)
            q_even = pl.tensor.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
            q_odd = pl.tensor.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
            q_rot_even = pl.sub(pl.mul(q_even, rope_cos_fp32), pl.mul(q_odd, rope_sin_fp32))
            q_rot_odd = pl.add(pl.mul(q_even, rope_sin_fp32), pl.mul(q_odd, rope_cos_fp32))
            q_rot_even_bf16 = pl.cast(q_rot_even, target_type=pl.BF16, mode="rint")
            q_rot_odd_bf16 = pl.cast(q_rot_odd, target_type=pl.BF16, mode="rint")
            q_rope_pair_stage[(hg + h_inner) * T : (hg + h_inner) * T + T, :ROPE_HALF] = q_rot_even_bf16
            q_rope_pair_stage[(hg + h_inner) * T : (hg + h_inner) * T + T, ROPE_HALF : ROPE_DIM] = q_rot_odd_bf16

    # Reassemble interleaved pairs (cube matmul) and write back to q_flat (vec cast).
    q_rope_grp_fp32 = pl.create_tensor([H * T, ROPE_DIM], dtype=pl.FP32)
    for hg_idx in pl.spmd(H // 8, name_hint="q_rope_reassemble"):
        hg = hg_idx * 8
        for h_inner in pl.pipeline(8, stage=2):
            even_chunk = q_rope_pair_stage[(hg + h_inner) * T : (hg + h_inner) * T + T, :ROPE_HALF]
            odd_chunk = q_rope_pair_stage[(hg + h_inner) * T : (hg + h_inner) * T + T, ROPE_HALF : ROPE_DIM]
            rot = pl.matmul(
                even_chunk,
                even_select_t[:, :],
                out_dtype=pl.FP32,
            )
            rot = pl.matmul_acc(
                rot,
                odd_chunk,
                odd_select_t[:, :],
            )
            q_rope_grp_fp32[(hg + h_inner) * T : (hg + h_inner) * T + T, :] = rot

    for hg_idx in pl.spmd(H // 8, name_hint="q_rope_write"):
        hg = hg_idx * 8
        for h_inner in pl.pipeline(8, stage=2):
            rot_fp32 = q_rope_grp_fp32[(hg + h_inner) * T : (hg + h_inner) * T + T, :]
            q_flat[:, (hg + h_inner) * HEAD_DIM + NOPE_DIM : (hg + h_inner) * HEAD_DIM + NOPE_DIM + ROPE_DIM] = pl.cast(rot_fp32, target_type=pl.BF16, mode="rint")

    q = pl.reshape(q_flat, [T, H, HEAD_DIM])

    kv_fp32 = pl.create_tensor([T, HEAD_DIM], dtype=pl.FP32)
    for kbg in pl.spmd((HEAD_DIM // KV_TILE) // 2, name_hint="kv_proj_matmul"):
        for k_inner in pl.pipeline(2, stage=2):
            kv_acc = pl.create_tensor([T, KV_TILE], dtype=pl.FP32)
            kv_col0 = (kbg * 2 + k_inner) * KV_TILE
            for db in pl.pipeline(0, D // D_TILE, stage=2):
                d0 = db * D_TILE
                kv_x_chunk_bf16 = token_x_bf16[:, d0 : d0 + D_TILE]
                wkv_chunk = wkv[d0 : d0 + D_TILE, kv_col0 : kv_col0 + KV_TILE]
                if d0 == 0:
                    kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                else:
                    kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_fp32[:, kv_col0 : kv_col0 + KV_TILE] = kv_acc

    kv_inv_rms_tensor = pl.create_tensor([1, T], dtype=pl.FP32)
    for tg_idx in pl.spmd(T // KV_RMS_T_TILE, name_hint="kv_rms_norm"):
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
            kv[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")

    kv_rot_even_tmp = pl.create_tensor([T, ROPE_HALF], dtype=pl.BF16)
    kv_rot_odd_tmp = pl.create_tensor([T, ROPE_HALF], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_apply"):
        kv_inv_rms_t = pl.reshape(kv_inv_rms_tensor[0:1, :], [T, 1])
        kv_rope = kv_fp32[:, NOPE_DIM : NOPE_DIM + ROPE_DIM]
        gamma_rope = pl.reshape(
            pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32),
            [1, ROPE_DIM],
        )
        kv_rope_norm = pl.col_expand_mul(pl.row_expand_mul(kv_rope, kv_inv_rms_t), gamma_rope)
        kv_even = pl.tensor.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
        kv_odd = pl.tensor.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
        cos = pl.cast(rope_cos[:, :ROPE_HALF], target_type=pl.FP32)
        sin = pl.cast(rope_sin[:, :ROPE_HALF], target_type=pl.FP32)
        kv_rot_even = pl.sub(pl.mul(kv_even, cos), pl.mul(kv_odd, sin))
        kv_rot_odd = pl.add(pl.mul(kv_even, sin), pl.mul(kv_odd, cos))
        kv_rot_even_tmp[:, :] = pl.cast(kv_rot_even, target_type=pl.BF16, mode="rint")
        kv_rot_odd_tmp[:, :] = pl.cast(kv_rot_odd, target_type=pl.BF16, mode="rint")

    kv_rope_full = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_reassemble"):
        for rope_col in pl.pipeline(0, ROPE_DIM, ROPE_TILE, stage=2):
            pair_col = rope_col // 2
            kv_rot_even_chunk = kv_rot_even_tmp[:, pair_col : pair_col + ROPE_PAIR_TILE]
            kv_rot_odd_chunk = kv_rot_odd_tmp[:, pair_col : pair_col + ROPE_PAIR_TILE]
            kv_rot_chunk = pl.matmul(
                kv_rot_even_chunk,
                even_select_t[pair_col : pair_col + ROPE_PAIR_TILE, rope_col : rope_col + ROPE_TILE],
                out_dtype=pl.FP32,
            )
            kv_rot_chunk = pl.matmul_acc(
                kv_rot_chunk,
                kv_rot_odd_chunk,
                odd_select_t[pair_col : pair_col + ROPE_PAIR_TILE, rope_col : rope_col + ROPE_TILE],
            )
            kv_rope_full[:, rope_col : rope_col + ROPE_TILE] = kv_rot_chunk

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_write"):
        kv[:, NOPE_DIM : NOPE_DIM + ROPE_DIM] = pl.cast(kv_rope_full, target_type=pl.BF16, mode="rint")

    return q


@pl.jit
def qkv_proj_rope_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Out[pl.Tensor[[T, H, HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
):
    q = qkv_proj_rope(
        x,
        norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos,
        rope_sin,
        even_select_t,
        odd_select_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )
    return q


def golden_qkv_proj_rope(tensors):
    """Torch reference: attn_norm fused, then Q/KV LoRA + RoPE (model.py 692, 495-504)."""
    import torch

    x = tensors["x"].float()
    norm_w = tensors["norm_w"].float()
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

    # attn_norm fused (model.py:692)
    token_x = rms_norm(x.view(T, D), norm_w)                        # [T, D]

    # Q path
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)   # [T, Q_LORA]
    # W8A8C16: wq_b W8 per-output-channel int8; qr_out A8 per-token int8.
    # flash: also quantizes wq_a/wkv to fp8 (default Linear dtype).
    qr_out_bf16 = qr_out.to(torch.bfloat16)
    qr_i8, qr_scale = int8_quant_per_row(qr_out_bf16.float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(T, H, HEAD_DIM)
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


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x():
        return (torch.randn(B, S, D) - 0.5)
    def init_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return (torch.randn(D, Q_LORA) - 0.5) / (D ** 0.5)
    def init_wq_b():
        return (torch.randn(Q_LORA, H * HEAD_DIM) - 0.5) / ((H * HEAD_DIM) ** 0.5)
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / (D ** 0.5)
    def init_cos():
        return torch.cos(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(T * ROPE_DIM).reshape(T, ROPE_DIM) * 1e-3)
    def init_even_select_t():
        m = torch.zeros((ROPE_HALF, ROPE_DIM))
        for i in range(ROPE_HALF):
            m[i, 2 * i] = 1
        return m
    def init_odd_select_t():
        m = torch.zeros((ROPE_HALF, ROPE_DIM))
        for i in range(ROPE_HALF):
            m[i, 2 * i + 1] = 1
        return m
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wq_b_scale = wq_b_scale.view(H * HEAD_DIM)

    return [
        TensorSpec("x",         [B, S, D],              torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w",    [D],                    torch.float32,  init_value=init_norm_w),
        TensorSpec("wq_a",      [D, Q_LORA],            torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b",      [Q_LORA, H * HEAD_DIM], torch.int8,     init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv",       [D, HEAD_DIM],          torch.bfloat16, init_value=init_wkv),
        TensorSpec("rope_cos",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_cos),
        TensorSpec("rope_sin",  [T, ROPE_DIM],          torch.bfloat16, init_value=init_sin),
        TensorSpec("even_select_t", [ROPE_HALF, ROPE_DIM], torch.bfloat16, init_value=init_even_select_t),
        TensorSpec("odd_select_t",  [ROPE_HALF, ROPE_DIM], torch.bfloat16, init_value=init_odd_select_t),
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

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=qkv_proj_rope_test,
        specs=build_tensor_specs(),
        golden_fn=golden_qkv_proj_rope,
        # W8A8C16 q_proj adds INT8 quant/dequant round-off before per-head RMSNorm.
        rtol=5e-3,
        atol=5e-3,
        compare_fn={
            "q":        ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "kv":       ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "qr":       ratio_allclose(atol=1, rtol=0, max_error_ratio=0),
            "qr_scale": ratio_allclose(atol=2.5e-5, rtol=5e-3),
        },
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
