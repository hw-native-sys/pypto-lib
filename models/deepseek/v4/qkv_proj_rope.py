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
B           = DECODE_BATCH
S           = DECODE_SEQ
T           = B * S
D           = M.hidden_size
H           = M.num_attention_heads
HEAD_DIM    = M.head_dim
ROPE_DIM    = M.qk_rope_head_dim
ROPE_HALF   = ROPE_DIM // 2
NOPE_DIM    = M.nope_head_dim
Q_LORA      = M.q_lora_rank
EPS         = M.rms_norm_eps

# tiling
ROPE_CHUNK  = 32
ROPE_PAIR_CHUNK = ROPE_CHUNK // 2
HEAD_CHUNK  = 64
HEAD_GROUP  = 8
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_CHUNK = 128
Q_LORA_TILE = 32
Q_LORA_CHUNK = Q_LORA_TILE
D_CHUNK     = 256 if T >= 64 else 512
KV_CHUNK    = 32
QUANT_CHUNK = 128 if T >= 64 else 256
assert (H * HEAD_DIM) % (HEAD_CHUNK * HEAD_GROUP) == 0, \
    "HEAD_BLOCKS must be divisible by HEAD_GROUP"
Q_BLOCKS      = Q_LORA // Q_LORA_TILE
Q_PROJ_BLOCKS = Q_LORA // Q_PROJ_CHUNK
HEAD_BLOCKS = (H * HEAD_DIM) // HEAD_CHUNK
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK
HEAD_GROUP_BLOCKS = (H * HEAD_DIM) // (HEAD_CHUNK * HEAD_GROUP)
D_BLOCKS = D // D_CHUNK
KV_BLOCKS = HEAD_DIM // KV_CHUNK


@pl.jit.inline
def qkv_proj_rope(
    x:         pl.Tensor[[B, S, D],              pl.BF16],
    norm_w:    pl.Tensor[[D],                    pl.FP32],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    rope_cos:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
    rope_sin:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t:  pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Tensor[[T, H, HEAD_DIM],        pl.BF16],
    kv:        pl.Tensor[[T, HEAD_DIM],           pl.BF16],
    qr:        pl.Tensor[[T, Q_LORA],             pl.INT8],
    qr_scale:  pl.Tensor[[T, 1],                  pl.FP32],
):
    x_flat = pl.reshape(x, [T, D])

    # Stage 0.1: fused attn_norm -> token_x_fp32
    token_x_fp32 = pl.create_tensor([T, D], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_norm_rms"):
        x_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for db in pl.range(D_BLOCKS):
            d0 = db * D_CHUNK
            x_chunk = pl.cast(pl.slice(x_flat, [T, D_CHUNK], [0, d0]), target_type=pl.FP32)
            x_sq_sum = pl.add(x_sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T]))
        x_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS)))

    x_inv_rms_t = pl.reshape(x_inv_rms, [T, 1])
    for db in pl.parallel(0, D_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="attn_norm_apply"):
            d0 = db * D_CHUNK
            x_chunk = pl.cast(pl.slice(x_flat, [T, D_CHUNK], [0, d0]), target_type=pl.FP32)
            norm_w_chunk = pl.reshape(pl.slice(norm_w, [D_CHUNK], [d0]), [1, D_CHUNK])
            x_normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, x_inv_rms_t), norm_w_chunk)
            token_x_fp32 = pl.assemble(token_x_fp32, x_normed, [0, d0])

    # Stage 0.2: pre-cast token_x for split AIV->AIC flow.
    token_x_bf16 = pl.create_tensor([T, D], dtype=pl.BF16)
    for db in pl.parallel(0, D_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="token_x_cast_bf16"):
            d0 = db * D_CHUNK
            x_chunk_fp32 = pl.slice(token_x_fp32, [T, D_CHUNK], [0, d0])
            token_x_bf16 = pl.assemble(token_x_bf16, pl.cast(x_chunk_fp32, target_type=pl.BF16, mode="rint"), [0, d0])

    # Stage 1/2.1: qr = rms_norm(token_x @ wq_a, gamma_cq)
    qr_fp32 = pl.create_tensor([T, Q_LORA], dtype=pl.FP32)
    for qb in pl.parallel(0, Q_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_matmul"):
            q_a_col0 = qb * Q_LORA_CHUNK
            d0_0 = 0
            x_chunk_bf16_0 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0_0])
            w_chunk_0 = pl.slice(wq_a, [D_CHUNK, Q_LORA_CHUNK], [d0_0, q_a_col0])
            q_acc = pl.matmul(x_chunk_bf16_0, w_chunk_0, out_dtype=pl.FP32)
            for db in pl.range(1, D_BLOCKS):
                d0 = db * D_CHUNK
                q_x_chunk_bf16 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0])
                w_chunk = pl.slice(wq_a, [D_CHUNK, Q_LORA_CHUNK], [d0, q_a_col0])
                q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [0, q_a_col0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_rms"):
        qr_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for qb in pl.range(Q_BLOCKS):
            qr_sq_col0 = qb * Q_LORA_CHUNK
            qr_chunk = pl.slice(qr_fp32, [T, Q_LORA_CHUNK], [0, qr_sq_col0])
            qr_sq_sum = pl.add(qr_sq_sum, pl.reshape(pl.row_sum(pl.mul(qr_chunk, qr_chunk)), [1, T]))
        qr_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)))

    qr_inv_rms_t = pl.reshape(qr_inv_rms, [T, 1])
    for qb in pl.parallel(0, Q_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_norm_apply"):
            qr_norm_col0 = qb * Q_LORA_CHUNK
            qr_chunk = pl.slice(qr_fp32, [T, Q_LORA_CHUNK], [0, qr_norm_col0])
            gamma_chunk = pl.reshape(
                pl.cast(pl.slice(gamma_cq, [Q_LORA_CHUNK], [qr_norm_col0]), target_type=pl.FP32),
                [1, Q_LORA_CHUNK],
            )
            qr_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_chunk)
            qr_fp32 = pl.assemble(qr_fp32, qr_normed, [0, qr_norm_col0])

    # Stage 2.2: pre-cast normalized qr for the W8A8 dynamic activation path.
    qr_bf16 = pl.create_tensor([T, Q_LORA], dtype=pl.BF16)
    for qb in pl.parallel(0, Q_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_cast_bf16"):
            qr_store_col0 = qb * Q_LORA_CHUNK
            qr_chunk_fp32 = pl.slice(qr_fp32, [T, Q_LORA_CHUNK], [0, qr_store_col0])
            qr_bf16 = pl.assemble(qr_bf16, pl.cast(qr_chunk_fp32, target_type=pl.BF16, mode="rint"), [0, qr_store_col0])

    # Stage 2.3: W8A8C16 activation path: quantize normalized qr per token.
    qr_scale_dq = pl.create_tensor([T, 1], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_quant"):
        qr_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for q0 in pl.range(0, Q_LORA, QUANT_CHUNK):
            qr_a_f32 = pl.cast(pl.slice(qr_bf16, [T, QUANT_CHUNK], [0, q0]), target_type=pl.FP32)
            qr_a_abs = pl.maximum(qr_a_f32, pl.neg(qr_a_f32))
            qr_a_max = pl.reshape(pl.row_max(qr_a_abs), [1, T])
            qr_amax = pl.maximum(qr_amax, qr_a_max)
        qr_scale_quant_row = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_amax)
        qr_scale_dq = pl.reshape(pl.recip(qr_scale_quant_row), [T, 1])
        qr_scale = pl.assemble(qr_scale, qr_scale_dq, [0, 0])
        qr_scale_quant = pl.reshape(qr_scale_quant_row, [T, 1])
        for q1 in pl.range(0, Q_LORA, QUANT_CHUNK):
            qr_q_f32 = pl.cast(pl.slice(qr_bf16, [T, QUANT_CHUNK], [0, q1]), target_type=pl.FP32)
            qr_q_scaled = pl.row_expand_mul(qr_q_f32, qr_scale_quant)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr = pl.assemble(qr, pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc"), [0, q1])

    # Stage 3: W8A8C16 q_proj = qr_i8 @ wq_b, then dequantize to FP32.
    q_proj_fp32 = pl.create_tensor([T, H * HEAD_DIM], dtype=pl.FP32)
    for hb in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, 1):
        h0 = hb * Q_PROJ_OUT_CHUNK
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qproj_matmul", optimization=pl.chunked_loop_optimizer):
            q0_0 = 0
            qr_i8_0 = pl.slice(qr, [T, Q_PROJ_CHUNK], [0, q0_0])
            wq_0 = pl.slice(wq_b, [Q_PROJ_CHUNK, Q_PROJ_OUT_CHUNK], [q0_0, h0])
            col_acc = pl.matmul(qr_i8_0, wq_0, out_dtype=pl.INT32)
            for qb in pl.range(1, Q_PROJ_BLOCKS):
                qr_proj_col0 = qb * Q_PROJ_CHUNK
                qr_i8_chunk = pl.slice(qr, [T, Q_PROJ_CHUNK], [0, qr_proj_col0])
                wq_chunk = pl.slice(wq_b, [Q_PROJ_CHUNK, Q_PROJ_OUT_CHUNK], [qr_proj_col0, h0])
                col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qproj_dequant", optimization=pl.chunked_loop_optimizer):
            col_fp32 = pl.cast(col_acc, target_type=pl.FP32, mode="none")
            w_scale = pl.slice(wq_b_scale, [1, Q_PROJ_OUT_CHUNK], [hb, 0])
            col_dequant = pl.col_expand_mul(pl.row_expand_mul(col_fp32, qr_scale_dq), w_scale)
            q_proj_fp32 = pl.assemble(q_proj_fp32, col_dequant, [0, h0])

    # Stage 4: per-head RMSNorm + RoPE on q
    q_flat = pl.reshape(q, [T, H * HEAD_DIM])
    for h in pl.parallel(0, H, 1):
        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="q_head_rms_rope"):
            h0 = h * HEAD_DIM
            q_head_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
            for db in pl.range(HEAD_DIM // HEAD_CHUNK):
                d0 = h0 + db * HEAD_CHUNK
                q_head_chunk = pl.slice(q_proj_fp32, [T, HEAD_CHUNK], [0, d0])
                q_head_sq_sum = pl.add(q_head_sq_sum, pl.reshape(pl.row_sum(pl.mul(q_head_chunk, q_head_chunk)), [1, T]))
            q_head_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS)))
            q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [T, 1])

            for nb in pl.range(NOPE_DIM // HEAD_CHUNK):
                n0 = nb * HEAD_CHUNK
                q_nope_chunk = pl.slice(q_proj_fp32, [T, HEAD_CHUNK], [0, h0 + n0])
                q_normed = pl.row_expand_mul(q_nope_chunk, q_head_inv_rms_t)
                q_flat = pl.assemble(q_flat, pl.cast(q_normed, target_type=pl.BF16, mode="rint"), [0, h0 + n0])

            q_rope = pl.slice(q_proj_fp32, [T, ROPE_DIM], [0, h0 + NOPE_DIM])
            q_rope_norm = pl.row_expand_mul(q_rope, q_head_inv_rms_t)
            q_even = pl.tensor.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
            q_odd = pl.tensor.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
            cos = pl.cast(pl.slice(rope_cos, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
            sin = pl.cast(pl.slice(rope_sin, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
            q_rot_even = pl.sub(pl.mul(q_even, cos), pl.mul(q_odd, sin))
            q_rot_odd = pl.add(pl.mul(q_even, sin), pl.mul(q_odd, cos))
            q_rot_even_bf16 = pl.cast(q_rot_even, target_type=pl.BF16, mode="rint")
            q_rot_odd_bf16 = pl.cast(q_rot_odd, target_type=pl.BF16, mode="rint")

        for rope_col in pl.range(0, ROPE_DIM, ROPE_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_rope_reassemble"):
                pair_col = rope_col // 2
                q_rot_even_chunk = pl.slice(q_rot_even_bf16, [T, ROPE_PAIR_CHUNK], [0, pair_col])
                q_rot_odd_chunk = pl.slice(q_rot_odd_bf16, [T, ROPE_PAIR_CHUNK], [0, pair_col])
                q_rot_chunk = pl.matmul(
                    q_rot_even_chunk,
                    pl.slice(even_select_t, [ROPE_PAIR_CHUNK, ROPE_CHUNK], [pair_col, rope_col]),
                    out_dtype=pl.FP32,
                )
                q_rot_chunk = pl.matmul_acc(
                    q_rot_chunk,
                    q_rot_odd_chunk,
                    pl.slice(odd_select_t, [ROPE_PAIR_CHUNK, ROPE_CHUNK], [pair_col, rope_col]),
                )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_rope_write"):
                h0 = h * HEAD_DIM
                q_flat = pl.assemble(q_flat, pl.cast(q_rot_chunk, target_type=pl.BF16, mode="rint"), [0, h0 + NOPE_DIM + rope_col])

    q = pl.reshape(q_flat, [T, H, HEAD_DIM])

    # Stage 5/6: kv = rms_norm(token_x @ wkv, gamma_ckv) + RoPE
    kv_fp32 = pl.create_tensor([T, HEAD_DIM], dtype=pl.FP32)
    for kb in pl.parallel(0, KV_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_matmul"):
            kv_col0 = kb * KV_CHUNK
            d0_0 = 0
            x_chunk_bf16_0 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0_0])
            wkv_chunk_0 = pl.slice(wkv, [D_CHUNK, KV_CHUNK], [d0_0, kv_col0])
            kv_acc = pl.matmul(x_chunk_bf16_0, wkv_chunk_0, out_dtype=pl.FP32)
            for db in pl.range(1, D_BLOCKS):
                d0 = db * D_CHUNK
                kv_x_chunk_bf16 = pl.slice(token_x_bf16, [T, D_CHUNK], [0, d0])
                wkv_chunk = pl.slice(wkv, [D_CHUNK, KV_CHUNK], [d0, kv_col0])
                kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [0, kv_col0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rms"):
        kv_sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for kb in pl.range(KV_BLOCKS):
            kv_sq_col0 = kb * KV_CHUNK
            kv_chunk = pl.slice(kv_fp32, [T, KV_CHUNK], [0, kv_sq_col0])
            kv_sq_sum = pl.add(kv_sq_sum, pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, T]))
        kv_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)))

    kv_inv_rms_t = pl.reshape(kv_inv_rms, [T, 1])
    for nb in pl.parallel(0, NOPE_DIM // KV_CHUNK, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_norm_nope"):
            n0 = nb * KV_CHUNK
            kv_chunk = pl.slice(kv_fp32, [T, KV_CHUNK], [0, n0])
            gamma_kv_chunk = pl.reshape(
                pl.cast(pl.slice(gamma_ckv, [KV_CHUNK], [n0]), target_type=pl.FP32),
                [1, KV_CHUNK],
            )
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv = pl.assemble(kv, pl.cast(kv_normed, target_type=pl.BF16, mode="rint"), [0, n0])
    kv_rot_even_tmp = pl.create_tensor([T, ROPE_HALF], dtype=pl.BF16)
    kv_rot_odd_tmp = pl.create_tensor([T, ROPE_HALF], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_apply"):
        kv_rope = pl.slice(kv_fp32, [T, ROPE_DIM], [0, NOPE_DIM])
        gamma_rope = pl.reshape(
            pl.cast(pl.slice(gamma_ckv, [ROPE_DIM], [NOPE_DIM]), target_type=pl.FP32),
            [1, ROPE_DIM],
        )
        kv_rope_norm = pl.col_expand_mul(pl.row_expand_mul(kv_rope, kv_inv_rms_t), gamma_rope)
        kv_even = pl.tensor.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
        kv_odd = pl.tensor.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
        cos = pl.cast(pl.slice(rope_cos, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
        sin = pl.cast(pl.slice(rope_sin, [T, ROPE_HALF], [0, 0]), target_type=pl.FP32)
        kv_rot_even = pl.sub(pl.mul(kv_even, cos), pl.mul(kv_odd, sin))
        kv_rot_odd = pl.add(pl.mul(kv_even, sin), pl.mul(kv_odd, cos))
        kv_rot_even_tmp = pl.assemble(kv_rot_even_tmp, pl.cast(kv_rot_even, target_type=pl.BF16, mode="rint"), [0, 0])
        kv_rot_odd_tmp = pl.assemble(kv_rot_odd_tmp, pl.cast(kv_rot_odd, target_type=pl.BF16, mode="rint"), [0, 0])

    for rope_col in pl.range(0, ROPE_DIM, ROPE_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_reassemble"):
            pair_col = rope_col // 2
            kv_rot_even_chunk = pl.slice(kv_rot_even_tmp, [T, ROPE_PAIR_CHUNK], [0, pair_col])
            kv_rot_odd_chunk = pl.slice(kv_rot_odd_tmp, [T, ROPE_PAIR_CHUNK], [0, pair_col])
            kv_rot_chunk = pl.matmul(
                kv_rot_even_chunk,
                pl.slice(even_select_t, [ROPE_PAIR_CHUNK, ROPE_CHUNK], [pair_col, rope_col]),
                out_dtype=pl.FP32,
            )
            kv_rot_chunk = pl.matmul_acc(
                kv_rot_chunk,
                kv_rot_odd_chunk,
                pl.slice(odd_select_t, [ROPE_PAIR_CHUNK, ROPE_CHUNK], [pair_col, rope_col]),
            )

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rope_write"):
            kv = pl.assemble(kv, pl.cast(kv_rot_chunk, target_type=pl.BF16, mode="rint"), [0, NOPE_DIM + rope_col])

    return q


@pl.jit
def qkv_proj_rope_test(
    x:         pl.Tensor[[B, S, D],              pl.BF16],
    norm_w:    pl.Tensor[[D],                    pl.FP32],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    rope_cos:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
    rope_sin:  pl.Tensor[[T, ROPE_DIM],          pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t:  pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Out[pl.Tensor[[T, H, HEAD_DIM], pl.BF16]],
    kv:        pl.Out[pl.Tensor[[T, HEAD_DIM],    pl.BF16]],
    qr:        pl.Out[pl.Tensor[[T, Q_LORA],      pl.INT8]],
    qr_scale:  pl.Out[pl.Tensor[[T, 1],           pl.FP32]],
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

    x         = tensors["x"].float()              # [B, S, D]
    norm_w    = tensors["norm_w"].float()          # [D]
    wq_a      = tensors["wq_a"].float()
    wq_b      = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv       = tensors["wkv"].float()
    rope_cos  = tensors["rope_cos"].float()
    rope_sin  = tensors["rope_sin"].float()
    gamma_cq  = tensors["gamma_cq"].float()
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
    wq_b_scale = wq_b_scale.view(Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK)

    return [
        TensorSpec("x",         [B, S, D],              torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w",    [D],                    torch.float32,  init_value=init_norm_w),
        TensorSpec("wq_a",      [D, Q_LORA],            torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b",      [Q_LORA, H * HEAD_DIM], torch.int8,     init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], torch.float32, init_value=lambda: wq_b_scale),
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
    from golden import RunConfig, ratio_allclose, run_jit

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
        config=RunConfig(
            # W8A8C16 q_proj adds INT8 quant/dequant round-off before per-head RMSNorm.
            rtol=5e-3,
            atol=5e-3,
            compare_fn={
                "q":        ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "kv":       ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "qr":       ratio_allclose(atol=1, rtol=0, max_error_ratio=0),
                "qr_scale": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
