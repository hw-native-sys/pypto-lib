# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill Q/KV projection + partial RoPE.

The kernel shape is [PREFILL_BATCH, PREFILL_SEQ] from config. The
implementation still splits B * S into smaller internal token chunks.
"""

import pypto.language as pl

from config import FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings

# Prefill QKV tiling. These constants intentionally live in this file instead
# of being imported from decode qkv, because some values depend on this
# kernel's own B/S/T shape.
HEAD_CHUNK = 64
HEAD_GROUP = 8
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_CHUNK = 512
Q_PROJ_GROUP = 8
Q_HEAD_RMS_GROUP = 2
Q_PROJ_DEQUANT_GROUP = 16
QR_PROJ_GROUP = 2
QR_NORM_GROUP = 8
ATTN_NORM_GROUP = 4
KV_PROJ_GROUP = 1
KV_PROJ_SPMD_GROUP = 2
ATTN_RMS_PARTIALS = 2
QR_RMS_PARTIALS = 2
Q_LORA_TILE = 32
Q_LORA_CHUNK = Q_LORA_TILE
D_CHUNK = 128 if T >= 128 else (256 if T >= 64 else 512)
KV_CHUNK = 32
QUANT_CHUNK = 32 if T >= 128 else (128 if T >= 64 else 256)
QUANT_APPLY_CHUNK = 256
assert (H * HEAD_DIM) % (HEAD_CHUNK * HEAD_GROUP) == 0, \
    "HEAD_BLOCKS must be divisible by HEAD_GROUP"
assert ((H * HEAD_DIM) // Q_PROJ_OUT_CHUNK) % Q_PROJ_GROUP == 0, \
    "Q_PROJ_HEAD_BLOCKS must be divisible by Q_PROJ_GROUP"
assert H % Q_HEAD_RMS_GROUP == 0, \
    "H must be divisible by Q_HEAD_RMS_GROUP"
assert ((H * HEAD_DIM) // Q_PROJ_OUT_CHUNK) % Q_PROJ_DEQUANT_GROUP == 0, \
    "Q_PROJ_HEAD_BLOCKS must be divisible by Q_PROJ_DEQUANT_GROUP"
assert (Q_LORA // Q_LORA_TILE) % QR_NORM_GROUP == 0, \
    "Q_BLOCKS must be divisible by QR_NORM_GROUP"
assert (Q_LORA // Q_LORA_TILE) % QR_PROJ_GROUP == 0, \
    "Q_BLOCKS must be divisible by QR_PROJ_GROUP"
assert (D // D_CHUNK) % ATTN_NORM_GROUP == 0, \
    "D_BLOCKS must be divisible by ATTN_NORM_GROUP"
assert (HEAD_DIM // KV_CHUNK) % KV_PROJ_GROUP == 0, \
    "KV_BLOCKS must be divisible by KV_PROJ_GROUP"
assert (HEAD_DIM // KV_CHUNK) % KV_PROJ_SPMD_GROUP == 0, \
    "KV_BLOCKS must be divisible by KV_PROJ_SPMD_GROUP"
assert (D // D_CHUNK) % ATTN_RMS_PARTIALS == 0, \
    "D_BLOCKS must be divisible by ATTN_RMS_PARTIALS"
assert (Q_LORA // Q_LORA_TILE) % QR_RMS_PARTIALS == 0, \
    "Q_BLOCKS must be divisible by QR_RMS_PARTIALS"
Q_BLOCKS = Q_LORA // Q_LORA_TILE
Q_PROJ_BLOCKS = Q_LORA // Q_PROJ_CHUNK
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK
D_BLOCKS = D // D_CHUNK
KV_BLOCKS = HEAD_DIM // KV_CHUNK

PREFILL_QKV_TOKEN_CHUNK = min(64, T)
PREFILL_QKV_CHUNKS = T // PREFILL_QKV_TOKEN_CHUNK
assert T % PREFILL_QKV_TOKEN_CHUNK == 0, "prefill qkv per-call token count must be divisible by the internal token chunk"


MAX_TOKENS = T
QKV_HEAD_CHUNK = HEAD_CHUNK
QKV_Q_PROJ_OUT_CHUNK = Q_PROJ_OUT_CHUNK
QKV_Q_PROJ_CHUNK = Q_PROJ_CHUNK
QKV_Q_PROJ_GROUP = Q_PROJ_GROUP
QKV_Q_HEAD_RMS_GROUP = Q_HEAD_RMS_GROUP
QKV_Q_PROJ_DEQUANT_GROUP = Q_PROJ_DEQUANT_GROUP
QKV_QR_PROJ_GROUP = QR_PROJ_GROUP
QKV_QR_NORM_GROUP = QR_NORM_GROUP
QKV_ATTN_NORM_GROUP = ATTN_NORM_GROUP
QKV_KV_PROJ_SPMD_GROUP = KV_PROJ_SPMD_GROUP
QKV_ATTN_RMS_PARTIALS = ATTN_RMS_PARTIALS
QKV_QR_RMS_PARTIALS = QR_RMS_PARTIALS
QKV_Q_LORA_CHUNK = Q_LORA_CHUNK
QKV_Q_LORA_TILE = Q_LORA_TILE
QKV_D_CHUNK = D_CHUNK
QKV_KV_CHUNK = KV_CHUNK
QKV_QUANT_CHUNK = QUANT_CHUNK
QKV_QUANT_APPLY_CHUNK = QUANT_APPLY_CHUNK
QKV_Q_BLOCKS = Q_BLOCKS
QKV_Q_PROJ_BLOCKS = Q_PROJ_BLOCKS
QKV_Q_PROJ_HEAD_BLOCKS = Q_PROJ_HEAD_BLOCKS
QKV_D_BLOCKS = D_BLOCKS
QKV_KV_BLOCKS = KV_BLOCKS
QKV_PREFILL_QKV_TOKEN_CHUNK = PREFILL_QKV_TOKEN_CHUNK
QKV_PREFILL_QKV_CHUNKS = PREFILL_QKV_CHUNKS
HCA_KV_STORE_TILE = 16

@pl.jit.inline
def prefill_qkv_proj_rope_core(
    x:         pl.Tensor[[T, D],                 pl.BF16],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    kv:        pl.Tensor[[T, HEAD_DIM],    pl.BF16],
    qr:        pl.Tensor[[T, Q_LORA],      pl.INT8],
    qr_scale:  pl.Tensor[[T, 1],           pl.FP32],
    rope_cos_t: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    # Stage -1: materialize absolute-position RoPE rows for packed tokens.
    # Token-major prefill uses per-token position_ids, so row order does not
    # have to match position order.
    for rope_t0 in pl.parallel(0, T, HCA_KV_STORE_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_qkv_rope_rows"):
            for rope_dt in pl.range(HCA_KV_STORE_TILE):
                rope_t = rope_t0 + rope_dt
                if rope_t < num_tokens:
                    rope_pos = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
                    rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                    rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]

    x_flat = x

    # Stage 0+: process this already attention-normalized token-major QKV
    # invocation in token chunks. The
    # internal chunk keeps the largest local tensors small enough for the QKV
    # projection, quantization, and RoPE scopes.
    for chunk_idx in pl.range(QKV_PREFILL_QKV_CHUNKS):
        t0 = chunk_idx * QKV_PREFILL_QKV_TOKEN_CHUNK
        x_tile = pl.slice(x_flat, [QKV_PREFILL_QKV_TOKEN_CHUNK, D], [t0, 0])

        kv_tile = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, HEAD_DIM], dtype=pl.BF16)
        qr_tile = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, Q_LORA], dtype=pl.INT8)
        qr_scale_tile = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, 1], dtype=pl.FP32)
        rope_cos_tile = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], dtype=pl.BF16)
        rope_sin_tile = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rope_chunk_materialize"):
            rope_cos_tile[:, :] = pl.slice(rope_cos_t, [QKV_PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], [t0, 0])
            rope_sin_tile[:, :] = pl.slice(rope_sin_t, [QKV_PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], [t0, 0])

        # Stage 1: LoRA A projection for the query path. Inputs are the
        # already attention-normalized BF16 activation; accumulation stays FP32.
        qr_fp32 = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, Q_LORA], dtype=pl.FP32)
        for qbg_idx in pl.spmd(QKV_Q_BLOCKS // QKV_QR_PROJ_GROUP, name_hint="prefill_qr_proj_matmul"):
            qbg = qbg_idx * QKV_QR_PROJ_GROUP
            for q_inner in pl.range(QKV_QR_PROJ_GROUP):
                q_a_col0 = (qbg + q_inner) * QKV_Q_LORA_CHUNK
                q_acc = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, QKV_Q_LORA_CHUNK], dtype=pl.FP32)
                for db in pl.pipeline(0, QKV_D_BLOCKS, stage=4):
                    qr_d0 = db * QKV_D_CHUNK
                    q_x_chunk_bf16 = x_tile[:, qr_d0 : qr_d0 + QKV_D_CHUNK]
                    w_chunk = wq_a[qr_d0 : qr_d0 + QKV_D_CHUNK, q_a_col0 : q_a_col0 + QKV_Q_LORA_CHUNK]
                    if qr_d0 == 0:
                        q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                    else:
                        q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
                qr_fp32[:, q_a_col0 : q_a_col0 + QKV_Q_LORA_CHUNK] = q_acc

        # Stage 2.1: RMSNorm reduction for qr. This is the DeepSeek q_a_layernorm
        # equivalent before quantizing qr for the W8A8C16 q_proj.
        QKV_Q_BLOCKS_PER_QR_PARTIAL = QKV_Q_BLOCKS // QKV_QR_RMS_PARTIALS
        qr_sq_partial = pl.create_tensor([QKV_QR_RMS_PARTIALS, QKV_PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32)
        for wgr in pl.parallel(0, QKV_QR_RMS_PARTIALS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_qr_rms_partial"):
                qr_rms_q_base = wgr * QKV_Q_BLOCKS_PER_QR_PARTIAL * QKV_Q_LORA_CHUNK
                qr_local_sum = pl.full([1, QKV_PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
                for qr_rms_qb in pl.range(QKV_Q_BLOCKS_PER_QR_PARTIAL):
                    qr_rms_col0 = qr_rms_q_base + qr_rms_qb * QKV_Q_LORA_CHUNK
                    qr_rms_chunk = qr_fp32[:, qr_rms_col0 : qr_rms_col0 + QKV_Q_LORA_CHUNK]
                    qr_local_sum = pl.add(
                        qr_local_sum,
                        pl.reshape(pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk)), [1, QKV_PREFILL_QKV_TOKEN_CHUNK]),
                    )
                qr_sq_partial[wgr : wgr + 1, :] = qr_local_sum

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_rms_final"):
            qr_sq_sum = pl.full([1, QKV_PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
            for w in pl.range(QKV_QR_RMS_PARTIALS):
                qr_sq_sum = pl.add(qr_sq_sum, qr_sq_partial[w : w + 1, :])
            qr_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)))

        # Stage 2.2: apply q_a_layernorm gamma, cast qr to BF16, and collect
        # per-row amax on the BF16 representation used by INT8 quantization.
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [QKV_PREFILL_QKV_TOKEN_CHUNK, 1])
        qr_bf16 = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, Q_LORA], dtype=pl.BF16)
        qr_amax_partial = pl.create_tensor([QKV_Q_BLOCKS, QKV_PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32)
        for qbg in pl.parallel(0, QKV_Q_BLOCKS, QKV_QR_NORM_GROUP):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_norm_apply"):
                local_amax = pl.full([1, QKV_PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for q_inner in pl.range(QKV_QR_NORM_GROUP):
                    qr_norm_col0 = (qbg + q_inner) * QKV_Q_LORA_CHUNK
                    qr_norm_chunk = qr_fp32[:, qr_norm_col0 : qr_norm_col0 + QKV_Q_LORA_CHUNK]
                    gamma_chunk = pl.reshape(
                        pl.cast(gamma_cq[qr_norm_col0 : qr_norm_col0 + QKV_Q_LORA_CHUNK], target_type=pl.FP32),
                        [1, QKV_Q_LORA_CHUNK],
                    )
                    qr_normed = pl.col_expand_mul(pl.row_expand_mul(qr_norm_chunk, qr_inv_rms_t), gamma_chunk)
                    qr_normed_bf16 = pl.cast(qr_normed, target_type=pl.BF16, mode="rint")
                    qr_bf16[:, qr_norm_col0 : qr_norm_col0 + QKV_Q_LORA_CHUNK] = qr_normed_bf16
                    qr_norm_amax_f32 = pl.cast(qr_normed_bf16, target_type=pl.FP32)
                    qr_norm_amax_abs = pl.maximum(qr_norm_amax_f32, pl.neg(qr_norm_amax_f32))
                    local_amax = pl.maximum(
                        local_amax,
                        pl.reshape(pl.row_max(qr_norm_amax_abs), [1, QKV_PREFILL_QKV_TOKEN_CHUNK]),
                    )
                qr_amax_partial[qbg : qbg + 1, :] = local_amax

        # Stage 2.3a: reduce per-row amax and produce the dequant scale stored
        # as qr_scale. qr_scale_quant_t is the reciprocal scale used only while
        # writing qr_tile below.
        qr_scale_quant_t = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, 1], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_quant_amax"):
            qr_amax = pl.full([1, QKV_PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for w in pl.range(0, QKV_Q_BLOCKS, QKV_QR_NORM_GROUP):
                qr_amax = pl.maximum(qr_amax, qr_amax_partial[w : w + 1, :])
            qr_scale_quant_row = pl.div(pl.full([1, QKV_PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_amax)
            qr_scale_tile[:, :] = pl.reshape(pl.recip(qr_scale_quant_row), [QKV_PREFILL_QKV_TOKEN_CHUNK, 1])
            qr_scale_quant_t[:, :] = pl.reshape(qr_scale_quant_row, [QKV_PREFILL_QKV_TOKEN_CHUNK, 1])

        # Stage 2.3b: quantize normalized qr to INT8. The round/cast sequence
        # mirrors the original qkv kernel so qr and qr_scale compare tightly.
        for qa in pl.parallel(0, Q_LORA, QKV_QUANT_APPLY_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_quant_apply"):
                for q1 in pl.range(0, QKV_QUANT_APPLY_CHUNK, QKV_QUANT_CHUNK):
                    qr_q_f32 = pl.cast(qr_bf16[:, qa + q1 : qa + q1 + QKV_QUANT_CHUNK], target_type=pl.FP32)
                    qr_q_scaled = pl.row_expand_mul(qr_q_f32, qr_scale_quant_t)
                    qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
                    qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
                    qr_tile[:, qa + q1 : qa + q1 + QKV_QUANT_CHUNK] = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")

        # Stage 3.1: KV projection from the same normalized activation.
        kv_fp32 = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, HEAD_DIM], dtype=pl.FP32)
        for kbg_idx in pl.spmd(QKV_KV_BLOCKS // QKV_KV_PROJ_SPMD_GROUP, name_hint="prefill_kv_proj_matmul"):
            kbg = kbg_idx * QKV_KV_PROJ_SPMD_GROUP
            for k_inner in pl.range(QKV_KV_PROJ_SPMD_GROUP):
                kv_acc = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, QKV_KV_CHUNK], dtype=pl.FP32)
                kv_col0 = (kbg + k_inner) * QKV_KV_CHUNK
                for db in pl.pipeline(0, QKV_D_BLOCKS, stage=4):
                    d0 = db * QKV_D_CHUNK
                    kv_x_chunk_bf16 = x_tile[:, d0 : d0 + QKV_D_CHUNK]
                    wkv_chunk = wkv[d0 : d0 + QKV_D_CHUNK, kv_col0 : kv_col0 + QKV_KV_CHUNK]
                    if d0 == 0:
                        kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                    else:
                        kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
                kv_fp32[:, kv_col0 : kv_col0 + QKV_KV_CHUNK] = kv_acc

        # Stage 3.2: apply kv_a_layernorm over the full 512-dim KV vector.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_rms"):
            kv_sq_sum = pl.full([1, QKV_PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
            for kb in pl.range(QKV_KV_BLOCKS):
                kv_sq_col0 = kb * QKV_KV_CHUNK
                kv_chunk = kv_fp32[:, kv_sq_col0 : kv_sq_col0 + QKV_KV_CHUNK]
                kv_sq_sum = pl.add(
                    kv_sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, QKV_PREFILL_QKV_TOKEN_CHUNK]),
                )
            kv_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)))

        # Stage 3.3: write the noPE part of KV after RMSNorm and gamma_ckv.
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [QKV_PREFILL_QKV_TOKEN_CHUNK, 1])
        for nb in pl.parallel(0, NOPE_DIM // QKV_KV_CHUNK, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_norm_nope"):
                n0 = nb * QKV_KV_CHUNK
                kv_chunk = kv_fp32[:, n0 : n0 + QKV_KV_CHUNK]
                gamma_kv_chunk = pl.reshape(
                    pl.cast(gamma_ckv[n0 : n0 + QKV_KV_CHUNK], target_type=pl.FP32),
                    [1, QKV_KV_CHUNK],
                )
                kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                kv_tile[:, n0 : n0 + QKV_KV_CHUNK] = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")

        # Stage 3.4: apply partial RoPE to the KV RoPE tail and scatter the
        # rotated even/odd pairs back into the interleaved DeepSeek head layout.
        kv_rot_even_tmp = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, ROPE_HALF], dtype=pl.BF16)
        kv_rot_odd_tmp = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, ROPE_HALF], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_rope_apply"):
            kv_rope = kv_fp32[:, NOPE_DIM : NOPE_DIM + ROPE_DIM]
            gamma_rope = pl.reshape(
                pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32),
                [1, ROPE_DIM],
            )
            kv_rope_norm = pl.col_expand_mul(pl.row_expand_mul(kv_rope, kv_inv_rms_t), gamma_rope)
            kv_even = pl.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
            kv_odd = pl.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
            cos = pl.cast(rope_cos_tile[:, :ROPE_HALF], target_type=pl.FP32)
            sin = pl.cast(rope_sin_tile[:, :ROPE_HALF], target_type=pl.FP32)
            kv_rot_even = pl.sub(pl.mul(kv_even, cos), pl.mul(kv_odd, sin))
            kv_rot_odd = pl.add(pl.mul(kv_even, sin), pl.mul(kv_odd, cos))
            kv_rot_even_tmp[:, :] = pl.cast(kv_rot_even, target_type=pl.BF16, mode="rint")
            kv_rot_odd_tmp[:, :] = pl.cast(kv_rot_odd, target_type=pl.BF16, mode="rint")

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_rope_scatter"):
            kv_rope_buf = pl.full([QKV_PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], dtype=pl.FP32, value=0.0)
            kv_rope_buf = pl.tensor.scatter(
                pl.cast(kv_rot_even_tmp, target_type=pl.FP32),
                mask_pattern=pl.tile.MaskPattern.P0101,
                dst=kv_rope_buf,
            )
            kv_rope_buf = pl.tensor.scatter(
                pl.cast(kv_rot_odd_tmp, target_type=pl.FP32),
                mask_pattern=pl.tile.MaskPattern.P1010,
                dst=kv_rope_buf,
            )
            kv_tile[:, NOPE_DIM : NOPE_DIM + ROPE_DIM] = pl.cast(kv_rope_buf, target_type=pl.BF16, mode="rint")

        # Stage 3.5: publish KV, qr, and qr_scale for this token chunk before
        # the q_proj path consumes qr_tile locally.
        kv = pl.assemble(kv, kv_tile, [t0, 0])
        qr = pl.assemble(qr, qr_tile, [t0, 0])
        qr_scale = pl.assemble(qr_scale, qr_scale_tile, [t0, 0])

        # Stage 4: W8A8C16 q_proj. INT8 qr_tile multiplies INT8 wq_b into INT32
        # accumulators; matmul and dequant are separate SPMD fan-outs to stay
        # within Vec buffer limits while avoiding a long AICPU dispatch trail.
        q_proj_fp32 = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, H * HEAD_DIM], dtype=pl.FP32)
        col_acc_all = pl.create_tensor([QKV_Q_PROJ_HEAD_BLOCKS * QKV_PREFILL_QKV_TOKEN_CHUNK, QKV_Q_PROJ_OUT_CHUNK], dtype=pl.INT32)
        for hg_idx in pl.spmd(QKV_Q_PROJ_HEAD_BLOCKS // QKV_Q_PROJ_GROUP, name_hint="prefill_qproj_matmul"):
            hg = hg_idx * QKV_Q_PROJ_GROUP
            col_acc = pl.create_tensor([QKV_PREFILL_QKV_TOKEN_CHUNK, QKV_Q_PROJ_OUT_CHUNK], dtype=pl.INT32)
            for h_inner in pl.range(QKV_Q_PROJ_GROUP):
                for qb in pl.pipeline(0, QKV_Q_PROJ_BLOCKS, stage=2):
                    qr_proj_col0 = qb * QKV_Q_PROJ_CHUNK
                    qr_i8_chunk = qr_tile[:, qr_proj_col0 : qr_proj_col0 + QKV_Q_PROJ_CHUNK]
                    wq_chunk = wq_b[
                        qr_proj_col0 : qr_proj_col0 + QKV_Q_PROJ_CHUNK,
                        (hg + h_inner) * QKV_Q_PROJ_OUT_CHUNK : (hg + h_inner) * QKV_Q_PROJ_OUT_CHUNK + QKV_Q_PROJ_OUT_CHUNK,
                    ]
                    if qr_proj_col0 == 0:
                        col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                    else:
                        col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
                col_acc_all[
                    (hg + h_inner) * QKV_PREFILL_QKV_TOKEN_CHUNK : (hg + h_inner) * QKV_PREFILL_QKV_TOKEN_CHUNK + QKV_PREFILL_QKV_TOKEN_CHUNK,
                    :,
                ] = col_acc

        for hbg_idx in pl.spmd(QKV_Q_PROJ_HEAD_BLOCKS // QKV_Q_PROJ_DEQUANT_GROUP, name_hint="prefill_qproj_dequant"):
            hbg = hbg_idx * QKV_Q_PROJ_DEQUANT_GROUP
            for h_inner in pl.range(QKV_Q_PROJ_DEQUANT_GROUP):
                col_acc_chunk = col_acc_all[
                    (hbg + h_inner) * QKV_PREFILL_QKV_TOKEN_CHUNK : (hbg + h_inner) * QKV_PREFILL_QKV_TOKEN_CHUNK + QKV_PREFILL_QKV_TOKEN_CHUNK,
                    :,
                ]
                col_fp32 = pl.cast(col_acc_chunk, target_type=pl.FP32, mode="none")
                w_col0 = (hbg + h_inner) * QKV_Q_PROJ_OUT_CHUNK
                w_scale = pl.reshape(wq_b_scale[w_col0 : w_col0 + QKV_Q_PROJ_OUT_CHUNK], [1, QKV_Q_PROJ_OUT_CHUNK])
                col_dequant = pl.col_expand_mul(pl.row_expand_mul(col_fp32, qr_scale_tile), w_scale)
                q_proj_fp32[
                    :,
                    (hbg + h_inner) * QKV_Q_PROJ_OUT_CHUNK : (hbg + h_inner) * QKV_Q_PROJ_OUT_CHUNK + QKV_Q_PROJ_OUT_CHUNK,
                ] = col_dequant

        q_flat = pl.reshape(q, [T, H * HEAD_DIM])
        # Stage 5.1: per-head RMSNorm for q_proj output. The noPE portion and
        # RoPE tail are both written directly to q_flat, mirroring decode's
        # fused q_head_rope path and avoiding a separate RoPE pair stage.
        for hg_idx in pl.spmd(H // QKV_Q_HEAD_RMS_GROUP, name_hint="prefill_q_head_rms_rope_fused"):
            hg = hg_idx * QKV_Q_HEAD_RMS_GROUP
            rope_cos_fp32 = pl.cast(rope_cos_tile[:, :ROPE_HALF], target_type=pl.FP32)
            rope_sin_fp32 = pl.cast(rope_sin_tile[:, :ROPE_HALF], target_type=pl.FP32)
            for h_inner in pl.range(QKV_Q_HEAD_RMS_GROUP):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_head_sq_sum = pl.full([1, QKV_PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
                for db in pl.pipeline(HEAD_DIM // QKV_HEAD_CHUNK, stage=2):
                    d0 = h0 + db * QKV_HEAD_CHUNK
                    q_head_chunk = q_proj_fp32[:, d0 : d0 + QKV_HEAD_CHUNK]
                    q_head_sq_sum = pl.add(
                        q_head_sq_sum,
                        pl.reshape(pl.row_sum(pl.mul(q_head_chunk, q_head_chunk)), [1, QKV_PREFILL_QKV_TOKEN_CHUNK]),
                    )
                # Match decode qkv: rsqrt is mathematically equivalent, but this
                # backend rounds pl.recip(pl.sqrt(...)) differently and that
                # precision matters after q-path quantization.
                q_head_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS)))
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [QKV_PREFILL_QKV_TOKEN_CHUNK, 1])

                for nb in pl.pipeline(NOPE_DIM // QKV_HEAD_CHUNK, stage=2):
                    n0 = nb * QKV_HEAD_CHUNK
                    q_nope_chunk = q_proj_fp32[:, h0 + n0 : h0 + n0 + QKV_HEAD_CHUNK]
                    q_normed = pl.row_expand_mul(q_nope_chunk, q_head_inv_rms_t)
                    q_flat[
                        t0 : t0 + QKV_PREFILL_QKV_TOKEN_CHUNK,
                        h0 + n0 : h0 + n0 + QKV_HEAD_CHUNK,
                    ] = pl.cast(q_normed, target_type=pl.BF16, mode="rint")

                q_rope = q_proj_fp32[
                    :,
                    h0 + NOPE_DIM : h0 + NOPE_DIM + ROPE_DIM,
                ]
                q_rope_norm = pl.row_expand_mul(q_rope, q_head_inv_rms_t)
                q_even = pl.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
                q_odd = pl.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
                q_rot_even = pl.sub(pl.mul(q_even, rope_cos_fp32), pl.mul(q_odd, rope_sin_fp32))
                q_rot_odd = pl.add(pl.mul(q_even, rope_sin_fp32), pl.mul(q_odd, rope_cos_fp32))
                q_rope_buf = pl.full([QKV_PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], dtype=pl.FP32, value=0.0)
                q_rope_buf = pl.tensor.scatter(
                    pl.cast(pl.cast(q_rot_even, target_type=pl.BF16, mode="rint"), target_type=pl.FP32),
                    mask_pattern=pl.tile.MaskPattern.P0101,
                    dst=q_rope_buf,
                )
                q_rope_buf = pl.tensor.scatter(
                    pl.cast(pl.cast(q_rot_odd, target_type=pl.BF16, mode="rint"), target_type=pl.FP32),
                    mask_pattern=pl.tile.MaskPattern.P1010,
                    dst=q_rope_buf,
                )
                q_flat[
                    t0 : t0 + QKV_PREFILL_QKV_TOKEN_CHUNK,
                    h0 + NOPE_DIM : h0 + NOPE_DIM + ROPE_DIM,
                ] = pl.cast(q_rope_buf, target_type=pl.BF16, mode="rint")

        q = pl.reshape(q_flat, [T, H, HEAD_DIM])
    return q, kv, qr, qr_scale


@pl.jit
def prefill_qkv_proj_rope_test(
    x:         pl.Tensor[[T, D],                 pl.BF16],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Out[pl.Tensor[[T, H, HEAD_DIM], pl.BF16]],
    kv:        pl.Out[pl.Tensor[[T, HEAD_DIM],    pl.BF16]],
    qr:        pl.Out[pl.Tensor[[T, Q_LORA],      pl.INT8]],
    qr_scale:  pl.Out[pl.Tensor[[T, 1],           pl.FP32]],
    rope_cos_t: pl.Out[pl.Tensor[[T, ROPE_DIM], pl.BF16]],
    rope_sin_t: pl.Out[pl.Tensor[[T, ROPE_DIM], pl.BF16]],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    return prefill_qkv_proj_rope_core(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        freqs_cos,
        freqs_sin,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        rope_cos_t,
        rope_sin_t,
        position_ids,
        num_tokens,
    )


def golden_prefill_qkv_proj_rope(tensors):
    import torch

    x = tensors["x"].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv = tensors["wkv"].float()
    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = freqs_cos.index_select(0, positions).contiguous()
    rope_sin_t = freqs_sin.index_select(0, positions).contiguous()

    def int8_quant_per_row(v):
        rows = v.reshape(-1, v.shape[-1]).float()
        amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = rows * scale_quant
        out_i32 = torch.round(scaled).to(torch.int32)
        out_half = out_i32.to(torch.float16)
        out_i8 = out_half.to(torch.int8)
        return out_i8.reshape_as(v), (1.0 / scale_quant).reshape(*v.shape[:-1], 1)

    def rms_norm(v, gamma, eps=EPS):
        inv = torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
        return v * inv * gamma

    def matmul_bf16_input_fp32(a, b):
        return torch.matmul(a.to(torch.bfloat16).float(), b.to(torch.bfloat16).float()).float()

    def apply_rope(x_rope, cos, sin):
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

    qr_out = rms_norm(matmul_bf16_input_fp32(x, wq_a), gamma_cq)
    qr_i8, qr_scale = int8_quant_per_row(qr_out.to(torch.bfloat16).float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(T, H, HEAD_DIM)
    q_full = q_full * torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos_t, rope_sin_t)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    kv_full = rms_norm(matmul_bf16_input_fp32(x, wkv), gamma_ckv)
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)
    kv_rope = apply_rope(kv_rope_in, rope_cos_t, rope_sin_t).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:] = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_i8
    tensors["qr_scale"][:] = qr_scale
    tensors["rope_cos_t"][:] = rope_cos_t
    tensors["rope_sin_t"][:] = rope_sin_t


def build_tensor_specs(start_pos: int = 0):
    import torch
    from golden import ScalarSpec, TensorSpec

    if start_pos < 0 or start_pos + T > MAX_SEQ_LEN:
        raise ValueError(f"start_pos must satisfy 0 <= start_pos <= {MAX_SEQ_LEN - T}, got {start_pos}")

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x():
        return torch.randn(T, D) - 0.5
    def init_wq_a():
        return (torch.randn(D, Q_LORA) - 0.5) / (D ** 0.5)
    def init_wq_b():
        return (torch.randn(Q_LORA, H * HEAD_DIM) - 0.5) / ((H * HEAD_DIM) ** 0.5)
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / (D ** 0.5)
    def init_freqs_cos():
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_DIM).reshape(MAX_SEQ_LEN, ROPE_DIM) * 1e-3)
    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_DIM).reshape(MAX_SEQ_LEN, ROPE_DIM) * 1e-3)
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + T, dtype=torch.int32)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wq_b_scale = wq_b_scale.view(H * HEAD_DIM)

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("kv", [T, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("qr", [T, Q_LORA], torch.int8, is_output=True),
        TensorSpec("qr_scale", [T, 1], torch.float32, is_output=True),
        TensorSpec("rope_cos_t", [T, ROPE_DIM], torch.bfloat16, is_output=True),
        TensorSpec("rope_sin_t", [T, ROPE_DIM], torch.bfloat16, is_output=True),
        TensorSpec("position_ids", [MAX_TOKENS], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, T),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(
        description=(
            "Standalone token-major DeepSeek V4 prefill QKV/RoPE validation. "
            "--start-pos is fixture-only and is lowered into position_ids before entering the JIT."
        )
    )
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=0,
                        help="Fixture-only absolute position offset used to generate position_ids.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_qkv_proj_rope_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_qkv_proj_rope,
        runtime_cfg=dict(platform=args.platform, device_id=args.device, enable_l2_swimlane=args.enable_l2_swimlane),
        rtol=5e-3,
        atol=5e-3,
        compare_fn={
            "q": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "qr": ratio_allclose(atol=1, rtol=0, max_error_ratio=0),
            "qr_scale": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "rope_cos_t": ratio_allclose(atol=0, rtol=0, max_error_ratio=0),
            "rope_sin_t": ratio_allclose(atol=0, rtol=0, max_error_ratio=0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
