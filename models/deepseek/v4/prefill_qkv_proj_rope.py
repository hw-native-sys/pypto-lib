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
from decode_qkv_proj_rope import build_tensor_specs as _build_qkv_tensor_specs


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
ROPE_CHUNK = 64
ROPE_PAIR_CHUNK = ROPE_CHUNK // 2
HEAD_CHUNK = 64
HEAD_GROUP = 8
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_CHUNK = 512
Q_PROJ_GROUP = 8
QR_NORM_GROUP = 8
ATTN_NORM_GROUP = 4
KV_PROJ_GROUP = 1
Q_PROJ_DEQUANT_GROUP = 32
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
assert (Q_LORA // Q_LORA_TILE) % QR_NORM_GROUP == 0, \
    "Q_BLOCKS must be divisible by QR_NORM_GROUP"
assert (D // D_CHUNK) % ATTN_NORM_GROUP == 0, \
    "D_BLOCKS must be divisible by ATTN_NORM_GROUP"
assert (HEAD_DIM // KV_CHUNK) % KV_PROJ_GROUP == 0, \
    "KV_BLOCKS must be divisible by KV_PROJ_GROUP"
assert ((H * HEAD_DIM) // Q_PROJ_OUT_CHUNK) % Q_PROJ_DEQUANT_GROUP == 0, \
    "Q_PROJ_HEAD_BLOCKS must be divisible by Q_PROJ_DEQUANT_GROUP"
assert (D // D_CHUNK) % ATTN_RMS_PARTIALS == 0, \
    "D_BLOCKS must be divisible by ATTN_RMS_PARTIALS"
assert (Q_LORA // Q_LORA_TILE) % QR_RMS_PARTIALS == 0, \
    "Q_BLOCKS must be divisible by QR_RMS_PARTIALS"
Q_BLOCKS = Q_LORA // Q_LORA_TILE
Q_PROJ_BLOCKS = Q_LORA // Q_PROJ_CHUNK
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK
D_BLOCKS = D // D_CHUNK
KV_BLOCKS = HEAD_DIM // KV_CHUNK

PREFILL_START_POS = 0
PREFILL_QKV_TOKEN_CHUNK = min(64, T)
PREFILL_QKV_CHUNKS = T // PREFILL_QKV_TOKEN_CHUNK
PREFILL_ROPE_BATCH_TILE = min(B, max(1, 256 // S))
assert T % PREFILL_QKV_TOKEN_CHUNK == 0, "prefill qkv per-call token count must be divisible by the internal token chunk"
assert B % PREFILL_ROPE_BATCH_TILE == 0, "B must be divisible by PREFILL_ROPE_BATCH_TILE"


@pl.jit.inline
def prefill_qkv_proj_rope_core(
    x:         pl.Tensor[[B, S, D],              pl.BF16],
    norm_w:    pl.Tensor[[D],                    pl.FP32],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t:  pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    kv:        pl.Tensor[[T, HEAD_DIM],    pl.BF16],
    qr:        pl.Tensor[[T, Q_LORA],      pl.INT8],
    qr_scale:  pl.Tensor[[T, 1],           pl.FP32],
    start_pos: pl.Scalar[pl.INT32],
):
    rope_cos_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)

    # Stage -1: materialize the absolute-position RoPE rows for this prefill
    # tile. The flattened token order is [batch, seq] -> [B*S], so every batch
    # row in this tile reuses the same contiguous [start_pos, start_pos + S)
    # frequency slice.
    for b0 in pl.range(0, B, PREFILL_ROPE_BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rope_batch_tile"):
            pos = pl.cast(start_pos, pl.INDEX)
            cos_rows = pl.slice(freqs_cos, [S, ROPE_DIM], [pos, 0])
            sin_rows = pl.slice(freqs_sin, [S, ROPE_DIM], [pos, 0])
            cos_tile = pl.full([PREFILL_ROPE_BATCH_TILE * S, ROPE_DIM], dtype=pl.BF16, value=0.0)
            sin_tile = pl.full([PREFILL_ROPE_BATCH_TILE * S, ROPE_DIM], dtype=pl.BF16, value=0.0)
            for bi in pl.range(PREFILL_ROPE_BATCH_TILE):
                tile_row = bi * S
                cos_tile = pl.assemble(cos_tile, cos_rows, [tile_row, 0])
                sin_tile = pl.assemble(sin_tile, sin_rows, [tile_row, 0])
            rope_offset = b0 * S
            rope_cos_t = pl.assemble(rope_cos_t, cos_tile, [rope_offset, 0])
            rope_sin_t = pl.assemble(rope_sin_t, sin_tile, [rope_offset, 0])

    x_flat = pl.reshape(x, [T, D])

    # Stage 0+: process this [B, S] QKV invocation in token chunks. The
    # internal chunk keeps the largest local tensors small enough for the QKV
    # projection, quantization, and RoPE scopes.
    for chunk_idx in pl.range(PREFILL_QKV_CHUNKS):
        t0 = chunk_idx * PREFILL_QKV_TOKEN_CHUNK
        x_tile = pl.slice(x_flat, [PREFILL_QKV_TOKEN_CHUNK, D], [t0, 0])

        # Stage 0.1: attention input RMSNorm reduction. Split the D reduction
        # into deterministic FP32 partial sums, matching the decode qkv core's
        # accuracy-sensitive accumulation pattern.
        D_BLOCKS_PER_PARTIAL = D_BLOCKS // ATTN_RMS_PARTIALS
        x_sq_partial = pl.create_tensor([ATTN_RMS_PARTIALS, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32)
        for wg in pl.parallel(0, ATTN_RMS_PARTIALS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_attn_norm_rms_partial"):
                rms_d_base = wg * D_BLOCKS_PER_PARTIAL * D_CHUNK
                local_sum = pl.full([1, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
                for rms_db in pl.range(D_BLOCKS_PER_PARTIAL):
                    rms_d0 = rms_d_base + rms_db * D_CHUNK
                    rms_x_chunk = pl.cast(x_tile[:, rms_d0 : rms_d0 + D_CHUNK], target_type=pl.FP32)
                    local_sum = pl.add(
                        local_sum,
                        pl.reshape(pl.row_sum(pl.mul(rms_x_chunk, rms_x_chunk)), [1, PREFILL_QKV_TOKEN_CHUNK]),
                    )
                x_sq_partial[wg : wg + 1, :] = local_sum

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attn_norm_rms_final"):
            x_sq_sum = pl.full([1, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
            for w in pl.range(ATTN_RMS_PARTIALS):
                x_sq_sum = pl.add(x_sq_sum, x_sq_partial[w : w + 1, :])
            x_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS)))

        # Stage 0.2: apply attention RMSNorm and cast the normalized activation
        # to BF16 before feeding the LoRA A and KV projections.
        x_inv_rms_t = pl.reshape(x_inv_rms, [PREFILL_QKV_TOKEN_CHUNK, 1])
        token_x_bf16 = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, D], dtype=pl.BF16)
        for dbg in pl.parallel(0, D_BLOCKS, ATTN_NORM_GROUP):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_attn_norm_apply"):
                for d_inner in pl.range(ATTN_NORM_GROUP):
                    apply_d0 = (dbg + d_inner) * D_CHUNK
                    apply_x_chunk = pl.cast(x_tile[:, apply_d0 : apply_d0 + D_CHUNK], target_type=pl.FP32)
                    norm_w_chunk = pl.reshape(norm_w[apply_d0 : apply_d0 + D_CHUNK], [1, D_CHUNK])
                    x_normed = pl.col_expand_mul(pl.row_expand_mul(apply_x_chunk, x_inv_rms_t), norm_w_chunk)
                    token_x_bf16[:, apply_d0 : apply_d0 + D_CHUNK] = pl.cast(x_normed, target_type=pl.BF16, mode="rint")

        kv_tile = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, HEAD_DIM], dtype=pl.BF16)
        qr_tile = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, Q_LORA], dtype=pl.INT8)
        qr_scale_tile = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, 1], dtype=pl.FP32)
        rope_cos_tile = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], dtype=pl.BF16)
        rope_sin_tile = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_rope_chunk_materialize"):
            rope_cos_tile[:, :] = pl.slice(rope_cos_t, [PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], [t0, 0])
            rope_sin_tile[:, :] = pl.slice(rope_sin_t, [PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], [t0, 0])

        # Stage 1: LoRA A projection for the query path, qr_fp32 =
        # RMSNorm(x) @ wq_a. Inputs are BF16, accumulation stays FP32.
        qr_fp32 = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, Q_LORA], dtype=pl.FP32)
        for qb in pl.parallel(0, Q_BLOCKS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_proj_matmul"):
                q_a_col0 = qb * Q_LORA_CHUNK
                q_acc = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, Q_LORA_CHUNK], dtype=pl.FP32)
                for db in pl.pipeline(0, D_BLOCKS, stage=4):
                    qr_d0 = db * D_CHUNK
                    q_x_chunk_bf16 = token_x_bf16[:, qr_d0 : qr_d0 + D_CHUNK]
                    w_chunk = wq_a[qr_d0 : qr_d0 + D_CHUNK, q_a_col0 : q_a_col0 + Q_LORA_CHUNK]
                    if qr_d0 == 0:
                        q_acc = pl.matmul(q_x_chunk_bf16, w_chunk, out_dtype=pl.FP32)
                    else:
                        q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk)
                qr_fp32[:, q_a_col0 : q_a_col0 + Q_LORA_CHUNK] = q_acc

        # Stage 2.1: RMSNorm reduction for qr. This is the DeepSeek q_a_layernorm
        # equivalent before quantizing qr for the W8A8C16 q_proj.
        Q_BLOCKS_PER_QR_PARTIAL = Q_BLOCKS // QR_RMS_PARTIALS
        qr_sq_partial = pl.create_tensor([QR_RMS_PARTIALS, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32)
        for wgr in pl.parallel(0, QR_RMS_PARTIALS, 1):
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="prefill_qr_rms_partial"):
                qr_rms_q_base = wgr * Q_BLOCKS_PER_QR_PARTIAL * Q_LORA_CHUNK
                qr_local_sum = pl.full([1, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
                for qr_rms_qb in pl.range(Q_BLOCKS_PER_QR_PARTIAL):
                    qr_rms_col0 = qr_rms_q_base + qr_rms_qb * Q_LORA_CHUNK
                    qr_rms_chunk = qr_fp32[:, qr_rms_col0 : qr_rms_col0 + Q_LORA_CHUNK]
                    qr_local_sum = pl.add(
                        qr_local_sum,
                        pl.reshape(pl.row_sum(pl.mul(qr_rms_chunk, qr_rms_chunk)), [1, PREFILL_QKV_TOKEN_CHUNK]),
                    )
                qr_sq_partial[wgr : wgr + 1, :] = qr_local_sum

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_rms_final"):
            qr_sq_sum = pl.full([1, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
            for w in pl.range(QR_RMS_PARTIALS):
                qr_sq_sum = pl.add(qr_sq_sum, qr_sq_partial[w : w + 1, :])
            qr_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)))

        # Stage 2.2: apply q_a_layernorm gamma, cast qr to BF16, and collect
        # per-row amax on the BF16 representation used by INT8 quantization.
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [PREFILL_QKV_TOKEN_CHUNK, 1])
        qr_bf16 = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, Q_LORA], dtype=pl.BF16)
        qr_amax_partial = pl.create_tensor([Q_BLOCKS, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32)
        for qbg in pl.parallel(0, Q_BLOCKS, QR_NORM_GROUP):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_norm_apply"):
                local_amax = pl.full([1, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for q_inner in pl.range(QR_NORM_GROUP):
                    qr_norm_col0 = (qbg + q_inner) * Q_LORA_CHUNK
                    qr_norm_chunk = qr_fp32[:, qr_norm_col0 : qr_norm_col0 + Q_LORA_CHUNK]
                    gamma_chunk = pl.reshape(
                        pl.cast(gamma_cq[qr_norm_col0 : qr_norm_col0 + Q_LORA_CHUNK], target_type=pl.FP32),
                        [1, Q_LORA_CHUNK],
                    )
                    qr_normed = pl.col_expand_mul(pl.row_expand_mul(qr_norm_chunk, qr_inv_rms_t), gamma_chunk)
                    qr_normed_bf16 = pl.cast(qr_normed, target_type=pl.BF16, mode="rint")
                    qr_bf16[:, qr_norm_col0 : qr_norm_col0 + Q_LORA_CHUNK] = qr_normed_bf16
                    qr_norm_amax_f32 = pl.cast(qr_normed_bf16, target_type=pl.FP32)
                    qr_norm_amax_abs = pl.maximum(qr_norm_amax_f32, pl.neg(qr_norm_amax_f32))
                    local_amax = pl.maximum(
                        local_amax,
                        pl.reshape(pl.row_max(qr_norm_amax_abs), [1, PREFILL_QKV_TOKEN_CHUNK]),
                    )
                qr_amax_partial[qbg : qbg + 1, :] = local_amax

        # Stage 2.3a: reduce per-row amax and produce the dequant scale stored
        # as qr_scale. qr_scale_quant_t is the reciprocal scale used only while
        # writing qr_tile below.
        qr_scale_quant_t = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, 1], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_quant_amax"):
            qr_amax = pl.full([1, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for w in pl.range(0, Q_BLOCKS, QR_NORM_GROUP):
                qr_amax = pl.maximum(qr_amax, qr_amax_partial[w : w + 1, :])
            qr_scale_quant_row = pl.div(pl.full([1, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=INT8_SCALE_MAX), qr_amax)
            qr_scale_tile[:, :] = pl.reshape(pl.recip(qr_scale_quant_row), [PREFILL_QKV_TOKEN_CHUNK, 1])
            qr_scale_quant_t[:, :] = pl.reshape(qr_scale_quant_row, [PREFILL_QKV_TOKEN_CHUNK, 1])

        # Stage 2.3b: quantize normalized qr to INT8. The round/cast sequence
        # mirrors the original qkv kernel so qr and qr_scale compare tightly.
        for qa in pl.parallel(0, Q_LORA, QUANT_APPLY_CHUNK):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qr_quant_apply"):
                for q1 in pl.range(0, QUANT_APPLY_CHUNK, QUANT_CHUNK):
                    qr_q_f32 = pl.cast(qr_bf16[:, qa + q1 : qa + q1 + QUANT_CHUNK], target_type=pl.FP32)
                    qr_q_scaled = pl.row_expand_mul(qr_q_f32, qr_scale_quant_t)
                    qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
                    qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
                    qr_tile[:, qa + q1 : qa + q1 + QUANT_CHUNK] = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")

        # Stage 3.1: KV projection from the same normalized activation,
        # kv_fp32 = RMSNorm(x) @ wkv.
        kv_fp32 = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, HEAD_DIM], dtype=pl.FP32)
        for kbg in pl.parallel(0, KV_BLOCKS, KV_PROJ_GROUP):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_proj_matmul"):
                kv_acc = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, KV_CHUNK], dtype=pl.FP32)
                for k_inner in pl.range(KV_PROJ_GROUP):
                    kv_col0 = (kbg + k_inner) * KV_CHUNK
                    for db in pl.pipeline(0, D_BLOCKS, stage=4):
                        d0 = db * D_CHUNK
                        kv_x_chunk_bf16 = token_x_bf16[:, d0 : d0 + D_CHUNK]
                        wkv_chunk = wkv[d0 : d0 + D_CHUNK, kv_col0 : kv_col0 + KV_CHUNK]
                        if d0 == 0:
                            kv_acc = pl.matmul(kv_x_chunk_bf16, wkv_chunk, out_dtype=pl.FP32)
                        else:
                            kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk)
                    kv_fp32[:, kv_col0 : kv_col0 + KV_CHUNK] = kv_acc

        # Stage 3.2: apply kv_a_layernorm over the full 512-dim KV vector.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_rms"):
            kv_sq_sum = pl.full([1, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
            for kb in pl.range(KV_BLOCKS):
                kv_sq_col0 = kb * KV_CHUNK
                kv_chunk = kv_fp32[:, kv_sq_col0 : kv_sq_col0 + KV_CHUNK]
                kv_sq_sum = pl.add(
                    kv_sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(kv_chunk, kv_chunk)), [1, PREFILL_QKV_TOKEN_CHUNK]),
                )
            kv_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)))

        # Stage 3.3: write the noPE part of KV after RMSNorm and gamma_ckv.
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [PREFILL_QKV_TOKEN_CHUNK, 1])
        for nb in pl.parallel(0, NOPE_DIM // KV_CHUNK, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_norm_nope"):
                n0 = nb * KV_CHUNK
                kv_chunk = kv_fp32[:, n0 : n0 + KV_CHUNK]
                gamma_kv_chunk = pl.reshape(
                    pl.cast(gamma_ckv[n0 : n0 + KV_CHUNK], target_type=pl.FP32),
                    [1, KV_CHUNK],
                )
                kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                kv_tile[:, n0 : n0 + KV_CHUNK] = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")

        # Stage 3.4: apply partial RoPE to the KV RoPE tail. The first scope
        # computes even/odd rotated pairs; the second scope reassembles them
        # back into the interleaved DeepSeek head layout.
        kv_rot_even_tmp = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, ROPE_HALF], dtype=pl.BF16)
        kv_rot_odd_tmp = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, ROPE_HALF], dtype=pl.BF16)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_rope_apply"):
            kv_rope = kv_fp32[:, NOPE_DIM : NOPE_DIM + ROPE_DIM]
            gamma_rope = pl.reshape(
                pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32),
                [1, ROPE_DIM],
            )
            kv_rope_norm = pl.col_expand_mul(pl.row_expand_mul(kv_rope, kv_inv_rms_t), gamma_rope)
            kv_even = pl.tensor.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
            kv_odd = pl.tensor.gather(kv_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
            cos = pl.cast(rope_cos_tile[:, :ROPE_HALF], target_type=pl.FP32)
            sin = pl.cast(rope_sin_tile[:, :ROPE_HALF], target_type=pl.FP32)
            kv_rot_even = pl.sub(pl.mul(kv_even, cos), pl.mul(kv_odd, sin))
            kv_rot_odd = pl.add(pl.mul(kv_even, sin), pl.mul(kv_odd, cos))
            kv_rot_even_tmp[:, :] = pl.cast(kv_rot_even, target_type=pl.BF16, mode="rint")
            kv_rot_odd_tmp[:, :] = pl.cast(kv_rot_odd, target_type=pl.BF16, mode="rint")

        kv_rope_full = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_rope_reassemble"):
            for rope_col in pl.range(0, ROPE_DIM, ROPE_CHUNK):
                pair_col = rope_col // 2
                kv_rot_even_chunk = kv_rot_even_tmp[:, pair_col : pair_col + ROPE_PAIR_CHUNK]
                kv_rot_odd_chunk = kv_rot_odd_tmp[:, pair_col : pair_col + ROPE_PAIR_CHUNK]
                kv_rot_chunk = pl.matmul(
                    kv_rot_even_chunk,
                    even_select_t[pair_col : pair_col + ROPE_PAIR_CHUNK, rope_col : rope_col + ROPE_CHUNK],
                    out_dtype=pl.FP32,
                )
                kv_rot_chunk = pl.matmul_acc(
                    kv_rot_chunk,
                    kv_rot_odd_chunk,
                    odd_select_t[pair_col : pair_col + ROPE_PAIR_CHUNK, rope_col : rope_col + ROPE_CHUNK],
                )
                kv_rope_full[:, rope_col : rope_col + ROPE_CHUNK] = kv_rot_chunk

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_rope_write"):
            kv_tile[:, NOPE_DIM : NOPE_DIM + ROPE_DIM] = pl.cast(kv_rope_full, target_type=pl.BF16, mode="rint")

        # Stage 3.5: publish KV, qr, and qr_scale for this token chunk before
        # the q_proj path consumes qr_tile locally.
        kv = pl.assemble(kv, kv_tile, [t0, 0])
        qr = pl.assemble(qr, qr_tile, [t0, 0])
        qr_scale = pl.assemble(qr_scale, qr_scale_tile, [t0, 0])

        # Stage 4: W8A8C16 q_proj. INT8 qr_tile multiplies INT8 wq_b into INT32
        # accumulators; a separate dequant pass applies qr_scale and per-output
        # wq_b_scale to recover FP32 q projection values.
        q_proj_fp32 = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, H * HEAD_DIM], dtype=pl.FP32)
        col_acc_all = pl.create_tensor([Q_PROJ_HEAD_BLOCKS * PREFILL_QKV_TOKEN_CHUNK, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)
        for hg in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, Q_PROJ_GROUP):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qproj_matmul"):
                col_acc = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, Q_PROJ_OUT_CHUNK], dtype=pl.INT32)
                for h_inner in pl.range(Q_PROJ_GROUP):
                    for qb in pl.pipeline(0, Q_PROJ_BLOCKS, stage=2):
                        qr_proj_col0 = qb * Q_PROJ_CHUNK
                        qr_i8_chunk = qr_tile[:, qr_proj_col0 : qr_proj_col0 + Q_PROJ_CHUNK]
                        wq_chunk = wq_b[
                            qr_proj_col0 : qr_proj_col0 + Q_PROJ_CHUNK,
                            (hg + h_inner) * Q_PROJ_OUT_CHUNK : (hg + h_inner) * Q_PROJ_OUT_CHUNK + Q_PROJ_OUT_CHUNK,
                        ]
                        if qr_proj_col0 == 0:
                            col_acc = pl.matmul(qr_i8_chunk, wq_chunk, out_dtype=pl.INT32)
                        else:
                            col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk)
                    col_acc_all[
                        (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK : (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK + PREFILL_QKV_TOKEN_CHUNK,
                        :,
                    ] = col_acc

        for hbg in pl.parallel(0, Q_PROJ_HEAD_BLOCKS, Q_PROJ_DEQUANT_GROUP):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_qproj_dequant"):
                for h_inner in pl.range(Q_PROJ_DEQUANT_GROUP):
                    col_acc_chunk = col_acc_all[
                        (hbg + h_inner) * PREFILL_QKV_TOKEN_CHUNK : (hbg + h_inner) * PREFILL_QKV_TOKEN_CHUNK + PREFILL_QKV_TOKEN_CHUNK,
                        :,
                    ]
                    col_fp32 = pl.cast(col_acc_chunk, target_type=pl.FP32, mode="none")
                    w_col0 = (hbg + h_inner) * Q_PROJ_OUT_CHUNK
                    w_scale = pl.reshape(wq_b_scale[w_col0 : w_col0 + Q_PROJ_OUT_CHUNK], [1, Q_PROJ_OUT_CHUNK])
                    col_dequant = pl.col_expand_mul(pl.row_expand_mul(col_fp32, qr_scale_tile), w_scale)
                    q_proj_fp32[
                        :,
                        (hbg + h_inner) * Q_PROJ_OUT_CHUNK : (hbg + h_inner) * Q_PROJ_OUT_CHUNK + Q_PROJ_OUT_CHUNK,
                    ] = col_dequant

        q_flat = pl.reshape(q, [T, H * HEAD_DIM])
        q_head_inv_rms_all = pl.create_tensor([H, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32)
        q_rope_pair_stage = pl.create_tensor([H * PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], dtype=pl.BF16)
        # Stage 5.1: per-head RMSNorm for q_proj output. The noPE portion can
        # be written directly to q; the RoPE portion keeps the per-head inverse
        # RMS for the next rotation stage.
        for h in pl.parallel(0, H, 1):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_q_head_rms_nope"):
                h0 = h * HEAD_DIM
                q_head_sq_sum = pl.full([1, PREFILL_QKV_TOKEN_CHUNK], dtype=pl.FP32, value=0.0)
                for db in pl.range(HEAD_DIM // HEAD_CHUNK):
                    d0 = h0 + db * HEAD_CHUNK
                    q_head_chunk = q_proj_fp32[:, d0 : d0 + HEAD_CHUNK]
                    q_head_sq_sum = pl.add(
                        q_head_sq_sum,
                        pl.reshape(pl.row_sum(pl.mul(q_head_chunk, q_head_chunk)), [1, PREFILL_QKV_TOKEN_CHUNK]),
                    )
                # Match decode qkv: rsqrt is mathematically equivalent, but this
                # backend rounds pl.recip(pl.sqrt(...)) differently and that
                # precision matters after q-path quantization.
                q_head_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM), EPS)))
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [PREFILL_QKV_TOKEN_CHUNK, 1])
                q_head_inv_rms_all[h : h + 1, :] = q_head_inv_rms

                for nb in pl.range(NOPE_DIM // HEAD_CHUNK):
                    n0 = nb * HEAD_CHUNK
                    q_nope_chunk = q_proj_fp32[:, h0 + n0 : h0 + n0 + HEAD_CHUNK]
                    q_normed = pl.row_expand_mul(q_nope_chunk, q_head_inv_rms_t)
                    q_flat[
                        t0 : t0 + PREFILL_QKV_TOKEN_CHUNK,
                        h0 + n0 : h0 + n0 + HEAD_CHUNK,
                    ] = pl.cast(q_normed, target_type=pl.BF16, mode="rint")

        # Stage 5.2: apply partial RoPE to q for each head group. Intermediate
        # even/odd rotated pairs are staged in grouped layout to reduce tiny
        # per-head tasks.
        for hg in pl.parallel(0, H, HEAD_GROUP):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_q_head_rope"):
                q_head_inv_rms_t = pl.create_tensor([PREFILL_QKV_TOKEN_CHUNK, 1], dtype=pl.FP32)
                rope_cos_fp32 = pl.cast(rope_cos_tile[:, :ROPE_HALF], target_type=pl.FP32)
                rope_sin_fp32 = pl.cast(rope_sin_tile[:, :ROPE_HALF], target_type=pl.FP32)
                for h_inner in pl.range(HEAD_GROUP):
                    q_head_inv_rms_t = pl.reshape(
                        q_head_inv_rms_all[hg + h_inner : hg + h_inner + 1, :],
                        [PREFILL_QKV_TOKEN_CHUNK, 1],
                    )
                    q_rope = q_proj_fp32[
                        :,
                        (hg + h_inner) * HEAD_DIM + NOPE_DIM : (hg + h_inner) * HEAD_DIM + NOPE_DIM + ROPE_DIM,
                    ]
                    q_rope_norm = pl.row_expand_mul(q_rope, q_head_inv_rms_t)
                    q_even = pl.tensor.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P0101)
                    q_odd = pl.tensor.gather(q_rope_norm, mask_pattern=pl.tile.MaskPattern.P1010)
                    q_rot_even = pl.sub(pl.mul(q_even, rope_cos_fp32), pl.mul(q_odd, rope_sin_fp32))
                    q_rot_odd = pl.add(pl.mul(q_even, rope_sin_fp32), pl.mul(q_odd, rope_cos_fp32))
                    q_rope_pair_stage[
                        (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK : (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK + PREFILL_QKV_TOKEN_CHUNK,
                        :ROPE_HALF,
                    ] = pl.cast(q_rot_even, target_type=pl.BF16, mode="rint")
                    q_rope_pair_stage[
                        (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK : (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK + PREFILL_QKV_TOKEN_CHUNK,
                        ROPE_HALF : ROPE_DIM,
                    ] = pl.cast(q_rot_odd, target_type=pl.BF16, mode="rint")

        # Stage 5.3: reassemble q RoPE even/odd pairs into interleaved head
        # layout and write the RoPE tail back into q_flat.
        for hg in pl.parallel(0, H, HEAD_GROUP):
            q_rope_grp_fp32 = pl.create_tensor([HEAD_GROUP * PREFILL_QKV_TOKEN_CHUNK, ROPE_DIM], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_q_rope_reassemble"):
                for h_inner in pl.range(HEAD_GROUP):
                    even_chunk = q_rope_pair_stage[
                        (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK : (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK + PREFILL_QKV_TOKEN_CHUNK,
                        :ROPE_HALF,
                    ]
                    odd_chunk = q_rope_pair_stage[
                        (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK : (hg + h_inner) * PREFILL_QKV_TOKEN_CHUNK + PREFILL_QKV_TOKEN_CHUNK,
                        ROPE_HALF : ROPE_DIM,
                    ]
                    rot = pl.matmul(even_chunk, even_select_t[:, :], out_dtype=pl.FP32)
                    rot = pl.matmul_acc(rot, odd_chunk, odd_select_t[:, :])
                    q_rope_grp_fp32[
                        h_inner * PREFILL_QKV_TOKEN_CHUNK : h_inner * PREFILL_QKV_TOKEN_CHUNK + PREFILL_QKV_TOKEN_CHUNK,
                        :,
                    ] = rot

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_q_rope_write"):
                for h_inner in pl.range(HEAD_GROUP):
                    rot_fp32 = q_rope_grp_fp32[
                        h_inner * PREFILL_QKV_TOKEN_CHUNK : h_inner * PREFILL_QKV_TOKEN_CHUNK + PREFILL_QKV_TOKEN_CHUNK,
                        :,
                    ]
                    q_flat[
                        t0 : t0 + PREFILL_QKV_TOKEN_CHUNK,
                        (hg + h_inner) * HEAD_DIM + NOPE_DIM : (hg + h_inner) * HEAD_DIM + NOPE_DIM + ROPE_DIM,
                    ] = pl.cast(rot_fp32, target_type=pl.BF16, mode="rint")
    return q, kv, qr, qr_scale


@pl.jit
def prefill_qkv_proj_rope(
    x:         pl.Tensor[[B, S, D],              pl.BF16],
    norm_w:    pl.Tensor[[D],                    pl.FP32],
    wq_a:      pl.Tensor[[D, Q_LORA],            pl.BF16],
    wq_b:      pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv:       pl.Tensor[[D, HEAD_DIM],          pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    odd_select_t:  pl.Tensor[[ROPE_HALF, ROPE_DIM], pl.BF16],
    gamma_cq:  pl.Tensor[[Q_LORA],               pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM],             pl.BF16],
    q:         pl.Out[pl.Tensor[[T, H, HEAD_DIM], pl.BF16]],
    kv:        pl.Out[pl.Tensor[[T, HEAD_DIM],    pl.BF16]],
    qr:        pl.Out[pl.Tensor[[T, Q_LORA],      pl.INT8]],
    qr_scale:  pl.Out[pl.Tensor[[T, 1],           pl.FP32]],
    start_pos: pl.Scalar[pl.INT32],
):
    q, kv, qr, qr_scale = prefill_qkv_proj_rope_core(
        x,
        norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        freqs_cos,
        freqs_sin,
        even_select_t,
        odd_select_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        start_pos,
    )
    return q, kv, qr, qr_scale


def golden_prefill_qkv_proj_rope(tensors):
    import torch

    start_pos = int(tensors["start_pos"])
    x = tensors["x"].float()
    norm_w = tensors["norm_w"].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv = tensors["wkv"].float()
    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()

    positions = torch.arange(start_pos, start_pos + S, device=freqs_cos.device)
    rope_cos_t = freqs_cos.index_select(0, positions).unsqueeze(0).expand(B, S, ROPE_DIM)
    rope_sin_t = freqs_sin.index_select(0, positions).unsqueeze(0).expand(B, S, ROPE_DIM)
    rope_cos_flat = rope_cos_t.reshape(T, ROPE_DIM).contiguous()
    rope_sin_flat = rope_sin_t.reshape(T, ROPE_DIM).contiguous()

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

    token_x = rms_norm(x.reshape(T, D), norm_w)

    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)
    qr_i8, qr_scale = int8_quant_per_row(qr_out.to(torch.bfloat16).float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(T, H, HEAD_DIM)
    q_full = q_full * torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos_flat, rope_sin_flat)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    kv_full = rms_norm(matmul_bf16_input_fp32(token_x, wkv), gamma_ckv)
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)
    kv_rope = apply_rope(kv_rope_in, rope_cos_flat, rope_sin_flat).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    tensors["q"][:] = q_out.to(torch.bfloat16)
    tensors["kv"][:] = kv_out.to(torch.bfloat16)
    tensors["qr"][:] = qr_i8
    tensors["qr_scale"][:] = qr_scale


def build_tensor_specs(start_pos: int = PREFILL_START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x():
        return torch.randn(B, S, D) - 0.5

    def init_freqs_cos():
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_DIM).reshape(MAX_SEQ_LEN, ROPE_DIM) * 1e-3)

    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_DIM).reshape(MAX_SEQ_LEN, ROPE_DIM) * 1e-3)

    specs = []
    for spec in _build_qkv_tensor_specs():
        if spec.name == "rope_cos":
            specs.append(TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos))
        elif spec.name == "rope_sin":
            specs.append(TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin))
        elif spec.name == "x":
            specs.append(TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x))
        elif spec.name == "q":
            specs.append(TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, is_output=True))
        elif spec.name == "kv":
            specs.append(TensorSpec("kv", [T, HEAD_DIM], torch.bfloat16, is_output=True))
        elif spec.name == "qr":
            specs.append(TensorSpec("qr", [T, Q_LORA], torch.int8, is_output=True))
        elif spec.name == "qr_scale":
            specs.append(TensorSpec("qr_scale", [T, 1], torch.float32, is_output=True))
        else:
            specs.append(spec)
    specs.append(ScalarSpec("start_pos", torch.int32, start_pos))
    return specs


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=PREFILL_START_POS)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_qkv_proj_rope,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_qkv_proj_rope,
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
