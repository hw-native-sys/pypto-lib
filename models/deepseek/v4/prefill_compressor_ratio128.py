# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill compressor, ratio=128 bring-up.

This standalone module mirrors the ratio-128 compressor used by
prefill_attention_hca.py: project KV/score, add APE, softmax-pool the
128-token prompt chunk, then apply the compressor RMSNorm/RoPE before
publishing one compressed KV row.
"""

import pypto.language as pl

from prefill_sparse_attn import CMP_MAX_BLOCKS as SPARSE_CMP_MAX_BLOCKS
from config import BLOCK_SIZE, FLASH as M, PREFILL_BATCH, PREFILL_SEQ


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
HEAD_DIM = M.head_dim
HEAD_DIM_INV = 1.0 / HEAD_DIM
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 128
OUT_DIM = HEAD_DIM
STATE_LEN = COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
START_POS = 0
ROTATE = False

K_CHUNK = 512
OUT_CHUNK = 32
HEAD_CHUNK = 64
RMS_TILE = 16
RMS_BLOCKS = (B + RMS_TILE - 1) // RMS_TILE
RMS_PAD_ROWS = RMS_BLOCKS * RMS_TILE
T_CHUNK = S
K_BLOCKS = D // K_CHUNK
OUT_BLOCKS = OUT_DIM // OUT_CHUNK
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK
PROJ_BLOCKS = B * OUT_BLOCKS
POOL_BLOCKS = B * HEAD_BLOCKS

assert S == COMPRESS_RATIO, "ratio128 prefill compressor bring-up expects one full compression chunk"
assert PREFILL_COMPRESSED_LEN == 1


@pl.jit.inline
def prefill_compressor_ratio128(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, PREFILL_COMPRESSED_LEN, HEAD_DIM], pl.FP32],
    kv_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HALF], pl.FP32],
    even_idx: pl.Tensor[[1, ROPE_HALF], pl.INT32],
    odd_idx: pl.Tensor[[1, ROPE_HALF], pl.INT32],
    kv_cache: pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([B * S, OUT_DIM], dtype=pl.FP32)
    kv_flat = pl.reshape(kv, [B * PREFILL_COMPRESSED_LEN, HEAD_DIM])
    kv_cache_flat = pl.reshape(kv_cache, [B * IDX_KV_LEN, HEAD_DIM])
    kv_state_flat = pl.reshape(kv_state, [B * STATE_LEN, OUT_DIM])
    score_state_flat = pl.reshape(score_state, [B * STATE_LEN, OUT_DIM])
    pooled_kv_pad = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    normed_kv_pad = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)

    for pad_block in pl.spmd(RMS_BLOCKS, name_hint="prefill_c128_norm_pad_init"):
        pad_row = pad_block * RMS_TILE
        for hb in pl.pipeline(HEAD_BLOCKS, stage=2):
            init_h0 = hb * HEAD_CHUNK
            pooled_kv_pad[pad_row : pad_row + RMS_TILE, init_h0 : init_h0 + HEAD_CHUNK] = pl.full(
                [RMS_TILE, HEAD_CHUNK],
                dtype=pl.FP32,
                value=0.0,
            )
            normed_kv_pad[pad_row : pad_row + RMS_TILE, init_h0 : init_h0 + HEAD_CHUNK] = pl.full(
                [RMS_TILE, HEAD_CHUNK],
                dtype=pl.FP32,
                value=0.0,
            )

    for proj_idx in pl.spmd(PROJ_BLOCKS, name_hint="prefill_c128_proj"):
        batch_idx = proj_idx // OUT_BLOCKS
        o0 = (proj_idx - batch_idx * OUT_BLOCKS) * OUT_CHUNK
        t0 = batch_idx * S
        kv_acc = pl.create_tensor([T_CHUNK, OUT_CHUNK], dtype=pl.FP32)
        score_acc = pl.create_tensor([T_CHUNK, OUT_CHUNK], dtype=pl.FP32)
        for kb in pl.pipeline(0, K_BLOCKS, stage=2):
            k0 = kb * K_CHUNK
            x_tile = x_flat[t0 : t0 + T_CHUNK, k0 : k0 + K_CHUNK]
            wkv_tile = wkv[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
            wgate_tile = wgate[k0 : k0 + K_CHUNK, o0 : o0 + OUT_CHUNK]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)
        score_acc = pl.add(score_acc, ape[0:T_CHUNK, o0 : o0 + OUT_CHUNK])
        kv_proj[t0 : t0 + T_CHUNK, o0 : o0 + OUT_CHUNK] = kv_acc
        score_proj[t0 : t0 + T_CHUNK, o0 : o0 + OUT_CHUNK] = score_acc
        kv_state_flat[t0 : t0 + T_CHUNK, o0 : o0 + OUT_CHUNK] = kv_acc
        score_state_flat[t0 : t0 + T_CHUNK, o0 : o0 + OUT_CHUNK] = score_acc

    for pool_idx in pl.spmd(POOL_BLOCKS, name_hint="prefill_c128_softmax_pool"):
        pool_b = pool_idx // HEAD_BLOCKS
        hb = pool_idx - pool_b * HEAD_BLOCKS
        pool_h0 = hb * HEAD_CHUNK
        t0 = pool_b * S
        score_tile = score_proj[t0 : t0 + S, pool_h0 : pool_h0 + HEAD_CHUNK]
        kv_tile = kv_proj[t0 : t0 + S, pool_h0 : pool_h0 + HEAD_CHUNK]
        score_t = pl.transpose(score_tile, axis1=0, axis2=1)
        kv_t = pl.transpose(kv_tile, axis1=0, axis2=1)
        score_max = pl.row_max(score_t)
        score_exp = pl.exp(pl.row_expand_sub(score_t, score_max))
        score_sum = pl.row_sum(score_exp)
        score_prob = pl.row_expand_div(score_exp, score_sum)
        pooled_t = pl.row_sum(pl.mul(kv_t, score_prob))
        pooled_chunk = pl.reshape(pooled_t, [1, HEAD_CHUNK])
        pooled_bf16 = pl.cast(pooled_chunk, target_type=pl.BF16, mode="rint")
        kv_row = pool_b * PREFILL_COMPRESSED_LEN
        kv_flat[kv_row : kv_row + 1, pool_h0 : pool_h0 + HEAD_CHUNK] = pl.cast(pooled_bf16, target_type=pl.FP32)
        pooled_kv_pad[pool_b : pool_b + 1, pool_h0 : pool_h0 + HEAD_CHUNK] = pl.cast(pooled_bf16, target_type=pl.FP32)

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    for norm_block in pl.spmd(RMS_BLOCKS, name_hint="prefill_c128_norm_rope"):
        batch_base = norm_block * RMS_TILE
        rope_row_ones = pl.full([RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=1.0)
        cos_b = pl.col_expand_mul(rope_row_ones, cos[0:1, 0:ROPE_HALF])
        sin_b = pl.col_expand_mul(rope_row_ones, sin[0:1, 0:ROPE_HALF])
        partial_sq = pl.full([1, RMS_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(HEAD_BLOCKS, stage=2):
            rms_h0 = rms_kb * HEAD_CHUNK
            kv_rms_chunk = pooled_kv_pad[batch_base : batch_base + RMS_TILE, rms_h0 : rms_h0 + HEAD_CHUNK]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for rms_kb in pl.pipeline(NOPE_HEAD_DIM // HEAD_CHUNK, stage=2):
            norm_h0 = rms_kb * HEAD_CHUNK
            kv_norm_chunk = pooled_kv_pad[batch_base : batch_base + RMS_TILE, norm_h0 : norm_h0 + HEAD_CHUNK]
            gamma = norm_w_2d[:, norm_h0 : norm_h0 + HEAD_CHUNK]
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_chunk_bf16 = pl.cast(
                normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )
            normed_kv_pad[batch_base : batch_base + RMS_TILE, norm_h0 : norm_h0 + HEAD_CHUNK] = pl.cast(
                normed_chunk_bf16,
                target_type=pl.FP32,
            )

        kv_rope_norm = pooled_kv_pad[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM]
        gamma_rope = norm_w_2d[:, NOPE_HEAD_DIM:HEAD_DIM]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_normed_bf16 = pl.cast(rope_normed, target_type=pl.BF16, mode="rint")
        rope_normed_fp32 = pl.cast(rope_normed_bf16, target_type=pl.FP32)
        rope_even = pl.gather(rope_normed_fp32, mask_pattern=pl.tile.MaskPattern.P0101)
        rope_odd = pl.gather(rope_normed_fp32, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_rot_even = pl.sub(pl.mul(rope_even, cos_b), pl.mul(rope_odd, sin_b))
        rope_rot_odd = pl.add(pl.mul(rope_even, sin_b), pl.mul(rope_odd, cos_b))
        rope_even_bf16 = pl.cast(rope_rot_even, target_type=pl.BF16, mode="rint")
        rope_odd_bf16 = pl.cast(rope_rot_odd, target_type=pl.BF16, mode="rint")
        rope_buf = pl.full([RMS_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        rope_buf = pl.tensor.scatter(
            pl.cast(rope_even_bf16, target_type=pl.FP32),
            mask_pattern=pl.tile.MaskPattern.P0101,
            dst=rope_buf,
        )
        rope_buf = pl.tensor.scatter(
            pl.cast(rope_odd_bf16, target_type=pl.FP32),
            mask_pattern=pl.tile.MaskPattern.P1010,
            dst=rope_buf,
        )
        normed_kv_pad[batch_base : batch_base + RMS_TILE, NOPE_HEAD_DIM:HEAD_DIM] = rope_buf

    for final_b in pl.parallel(0, B, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_c128_finalize"):
            kv_row = final_b * PREFILL_COMPRESSED_LEN
            cache_row = final_b * IDX_KV_LEN + pl.cast(start_pos // COMPRESS_RATIO, pl.INDEX)
            for hb in pl.range(HEAD_BLOCKS):
                final_h0 = hb * HEAD_CHUNK
                final_chunk = normed_kv_pad[final_b : final_b + 1, final_h0 : final_h0 + HEAD_CHUNK]
                final_bf16 = pl.cast(final_chunk, target_type=pl.BF16, mode="rint")
                kv_flat[kv_row : kv_row + 1, final_h0 : final_h0 + HEAD_CHUNK] = pl.cast(final_bf16, target_type=pl.FP32)
                kv_cache_flat[cache_row : cache_row + 1, final_h0 : final_h0 + HEAD_CHUNK] = final_bf16

    kv = pl.reshape(kv_flat, [B, PREFILL_COMPRESSED_LEN, HEAD_DIM])
    kv_cache = pl.reshape(kv_cache_flat, [B, IDX_KV_LEN, HEAD_DIM])
    kv_state = pl.reshape(kv_state_flat, [B, STATE_LEN, OUT_DIM])
    score_state = pl.reshape(score_state_flat, [B, STATE_LEN, OUT_DIM])
    return kv, kv_state, score_state, kv_cache


# Packed HCA ratio-128 adapter. The standalone compressor above remains
# rectangular; this variant consumes lowered request/token metadata.
MAX_REQS = 2
MAX_TOKENS = T
MAIN_OUT_DIM = OUT_DIM
MAIN_STATE_LEN = STATE_LEN
ROPE_DIM = ROPE_HEAD_DIM
NOPE_DIM = NOPE_HEAD_DIM
MAX_CMP_WRITES = MAX_REQS * max(1, MAX_TOKENS // COMPRESS_RATIO)
HCA_CMP_BLOCK_NUM = MAX_REQS * SPARSE_CMP_MAX_BLOCKS
CMP_K_CHUNK = K_CHUNK
CMP_OUT_CHUNK = OUT_CHUNK
CMP_HEAD_CHUNK = HEAD_CHUNK
CMP_K_BLOCKS = K_BLOCKS
CMP_OUT_BLOCKS = OUT_BLOCKS
CMP_HEAD_BLOCKS = HEAD_BLOCKS
HCA_KV_STORE_TILE = 16
HCA_C128_RMS_TILE = 8
HCA_C128_RMS_PAD_ROWS = HCA_C128_RMS_TILE

PACKED_C128_PROJ_BLOCKS = CMP_OUT_BLOCKS
PACKED_C128_POOL_BLOCKS = MAX_CMP_WRITES * CMP_HEAD_BLOCKS


@pl.jit.inline
def prefill_compressor_ratio128_packed(
    x: pl.Tensor[[MAX_TOKENS, D], pl.BF16],
    kv_state: pl.Tensor[[MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    cmp_kv: pl.Out[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    num_cmp_writes: pl.Scalar[pl.INT32],
    cmp_write_token_ids: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
):
    x_flat = x
    kv_proj = pl.create_tensor([MAX_TOKENS, MAIN_OUT_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([MAX_TOKENS, MAIN_OUT_DIM], dtype=pl.FP32)
    kv_state_flat = pl.reshape(kv_state, [MAX_REQS * MAIN_STATE_LEN, MAIN_OUT_DIM])
    score_state_flat = pl.reshape(score_state, [MAX_REQS * MAIN_STATE_LEN, MAIN_OUT_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    pooled_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    normed_kv_pad = pl.create_tensor([HCA_C128_RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_norm_pad_init"):
        for init_hb in pl.pipeline(CMP_HEAD_BLOCKS, stage=2):
            init_h0 = init_hb * CMP_HEAD_CHUNK
            zero_chunk = pl.full([HCA_C128_RMS_TILE, CMP_HEAD_CHUNK], dtype=pl.FP32, value=0.0)
            pooled_kv_pad[0:HCA_C128_RMS_TILE, init_h0 : init_h0 + CMP_HEAD_CHUNK] = zero_chunk
            normed_kv_pad[0:HCA_C128_RMS_TILE, init_h0 : init_h0 + CMP_HEAD_CHUNK] = zero_chunk

    for proj_idx in pl.spmd(PACKED_C128_PROJ_BLOCKS, name_hint="prefill_hca_c128_state_proj"):
        o0 = proj_idx * CMP_OUT_CHUNK
        kv_acc = pl.create_tensor([MAX_TOKENS, CMP_OUT_CHUNK], dtype=pl.FP32)
        score_acc = pl.create_tensor([MAX_TOKENS, CMP_OUT_CHUNK], dtype=pl.FP32)
        for kb in pl.pipeline(0, CMP_K_BLOCKS, stage=2):
            k0 = kb * CMP_K_CHUNK
            x_tile = x_flat[0:MAX_TOKENS, k0 : k0 + CMP_K_CHUNK]
            wkv_tile = wkv[k0 : k0 + CMP_K_CHUNK, o0 : o0 + CMP_OUT_CHUNK]
            wgate_tile = wgate[k0 : k0 + CMP_K_CHUNK, o0 : o0 + CMP_OUT_CHUNK]
            if k0 == 0:
                kv_acc = pl.matmul(x_tile, wkv_tile, out_dtype=pl.FP32)
                score_acc = pl.matmul(x_tile, wgate_tile, out_dtype=pl.FP32)
            else:
                kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile)
                score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile)
        kv_proj[0:MAX_TOKENS, o0 : o0 + CMP_OUT_CHUNK] = kv_acc
        score_proj[0:MAX_TOKENS, o0 : o0 + CMP_OUT_CHUNK] = score_acc

    for state_t0 in pl.parallel(0, MAX_TOKENS, HCA_KV_STORE_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_state_update"):
            for dt in pl.range(HCA_KV_STORE_TILE):
                t = state_t0 + dt
                if t < num_tokens:
                    req = pl.cast(pl.read(token_to_request, [t]), pl.INDEX)
                    pos = pl.read(position_ids, [t])
                    state_slot = pl.cast(pos % COMPRESS_RATIO, pl.INDEX)
                    state_row = req * MAIN_STATE_LEN + state_slot
                    for ob in pl.pipeline(0, CMP_OUT_BLOCKS, stage=4):
                        o0 = ob * CMP_OUT_CHUNK
                        ape_row = ape[state_slot : state_slot + 1, o0 : o0 + CMP_OUT_CHUNK]
                        score_row = pl.add(score_proj[t : t + 1, o0 : o0 + CMP_OUT_CHUNK], ape_row)
                        kv_state_flat[state_row : state_row + 1, o0 : o0 + CMP_OUT_CHUNK] = kv_proj[
                            t : t + 1,
                            o0 : o0 + CMP_OUT_CHUNK,
                        ]
                        score_state_flat[state_row : state_row + 1, o0 : o0 + CMP_OUT_CHUNK] = score_row
                else:
                    score_proj[t : t + 1, 0:CMP_OUT_CHUNK] = score_proj[t : t + 1, 0:CMP_OUT_CHUNK]

    for pool_idx in pl.spmd(PACKED_C128_POOL_BLOCKS, name_hint="prefill_hca_c128_softmax_pool"):
        write_i = pool_idx // CMP_HEAD_BLOCKS
        hb = pool_idx - write_i * CMP_HEAD_BLOCKS
        h0 = hb * CMP_HEAD_CHUNK
        if write_i < num_cmp_writes:
            write_token = pl.cast(pl.read(cmp_write_token_ids, [write_i]), pl.INDEX)
            req = pl.cast(pl.read(token_to_request, [write_token]), pl.INDEX)
            state_row0 = req * MAIN_STATE_LEN
            score_tile = score_state_flat[state_row0 : state_row0 + MAIN_STATE_LEN, h0 : h0 + CMP_HEAD_CHUNK]
            kv_tile = kv_state_flat[state_row0 : state_row0 + MAIN_STATE_LEN, h0 : h0 + CMP_HEAD_CHUNK]
            score_t = pl.transpose(score_tile, axis1=0, axis2=1)
            kv_t = pl.transpose(kv_tile, axis1=0, axis2=1)
            score_max = pl.row_max(score_t)
            score_exp = pl.exp(pl.row_expand_sub(score_t, score_max))
            score_sum = pl.row_sum(score_exp)
            score_prob = pl.row_expand_div(score_exp, score_sum)
            pooled_t = pl.row_sum(pl.mul(kv_t, score_prob))
            pooled_chunk = pl.reshape(pooled_t, [1, CMP_HEAD_CHUNK])
            pooled_bf16 = pl.cast(pooled_chunk, target_type=pl.BF16, mode="rint")
            pooled_kv_pad[write_i : write_i + 1, h0 : h0 + CMP_HEAD_CHUNK] = pl.cast(
                pooled_bf16,
                target_type=pl.FP32,
            )
        else:
            pooled_kv_pad[write_i : write_i + 1, h0 : h0 + CMP_HEAD_CHUNK] = pooled_kv_pad[
                write_i : write_i + 1,
                h0 : h0 + CMP_HEAD_CHUNK,
            ]

    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_norm_rope"):
        cos_b = pl.full([HCA_C128_RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([HCA_C128_RMS_TILE, ROPE_HALF], dtype=pl.FP32, value=0.0)
        for norm_i in pl.range(HCA_C128_RMS_TILE):
            if norm_i < num_cmp_writes:
                norm_write_token = pl.cast(pl.read(cmp_write_token_ids, [norm_i]), pl.INDEX)
                norm_cmp_pos = pl.cast(pl.read(position_ids, [norm_write_token]) + 1 - COMPRESS_RATIO, pl.INDEX)
                cos_row = pl.cast(freqs_cos[norm_cmp_pos : norm_cmp_pos + 1, 0:ROPE_HALF], target_type=pl.FP32)
                sin_row = pl.cast(freqs_sin[norm_cmp_pos : norm_cmp_pos + 1, 0:ROPE_HALF], target_type=pl.FP32)
                cos_b[norm_i : norm_i + 1, 0:ROPE_HALF] = cos_row
                sin_b[norm_i : norm_i + 1, 0:ROPE_HALF] = sin_row
        partial_sq = pl.full([1, HCA_C128_RMS_TILE], dtype=pl.FP32, value=0.0)
        for rms_kb in pl.pipeline(CMP_HEAD_BLOCKS, stage=2):
            rms_h0 = rms_kb * CMP_HEAD_CHUNK
            kv_rms_chunk = pooled_kv_pad[0:HCA_C128_RMS_TILE, rms_h0 : rms_h0 + CMP_HEAD_CHUNK]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(kv_rms_sq), [1, HCA_C128_RMS_TILE]))

        variance = pl.reshape(pl.add(pl.mul(partial_sq, 1.0 / HEAD_DIM), EPS), [HCA_C128_RMS_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for norm_kb in pl.pipeline(NOPE_DIM // CMP_HEAD_CHUNK, stage=2):
            norm_h0 = norm_kb * CMP_HEAD_CHUNK
            kv_norm_chunk = pooled_kv_pad[0:HCA_C128_RMS_TILE, norm_h0 : norm_h0 + CMP_HEAD_CHUNK]
            gamma = norm_w_2d[:, norm_h0 : norm_h0 + CMP_HEAD_CHUNK]
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_chunk_bf16 = pl.cast(normed_chunk, target_type=pl.BF16, mode="rint")
            normed_kv_pad[0:HCA_C128_RMS_TILE, norm_h0 : norm_h0 + CMP_HEAD_CHUNK] = pl.cast(
                normed_chunk_bf16,
                target_type=pl.FP32,
            )

        kv_rope = pooled_kv_pad[0:HCA_C128_RMS_TILE, NOPE_DIM:HEAD_DIM]
        gamma_rope = norm_w_2d[:, NOPE_DIM:HEAD_DIM]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope, inv_rms), gamma_rope)
        rope_normed_bf16 = pl.cast(rope_normed, target_type=pl.BF16, mode="rint")
        rope_normed_fp32 = pl.cast(rope_normed_bf16, target_type=pl.FP32)
        rope_even = pl.gather(rope_normed_fp32, mask_pattern=pl.tile.MaskPattern.P0101)
        rope_odd = pl.gather(rope_normed_fp32, mask_pattern=pl.tile.MaskPattern.P1010)
        rope_rot_even = pl.sub(pl.mul(rope_even, cos_b), pl.mul(rope_odd, sin_b))
        rope_rot_odd = pl.add(pl.mul(rope_even, sin_b), pl.mul(rope_odd, cos_b))
        rope_even_bf16 = pl.cast(rope_rot_even, target_type=pl.BF16, mode="rint")
        rope_odd_bf16 = pl.cast(rope_rot_odd, target_type=pl.BF16, mode="rint")
        rope_buf = pl.full([HCA_C128_RMS_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
        rope_buf = pl.tensor.scatter(
            pl.cast(rope_even_bf16, target_type=pl.FP32),
            mask_pattern=pl.tile.MaskPattern.P0101,
            dst=rope_buf,
        )
        rope_buf = pl.tensor.scatter(
            pl.cast(rope_odd_bf16, target_type=pl.FP32),
            mask_pattern=pl.tile.MaskPattern.P1010,
            dst=rope_buf,
        )
        normed_kv_pad[0:HCA_C128_RMS_TILE, NOPE_DIM:HEAD_DIM] = rope_buf

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_c128_finalize"):
        for final_i in pl.range(MAX_CMP_WRITES):
            if final_i < num_cmp_writes:
                final_cmp_row = pl.cast(pl.read(cmp_slot_mapping, [final_i]), pl.INDEX)
                for final_hb in pl.range(CMP_HEAD_BLOCKS):
                    final_h0 = final_hb * CMP_HEAD_CHUNK
                    final_chunk = normed_kv_pad[final_i : final_i + 1, final_h0 : final_h0 + CMP_HEAD_CHUNK]
                    cmp_kv_flat[final_cmp_row : final_cmp_row + 1, final_h0 : final_h0 + CMP_HEAD_CHUNK] = pl.cast(
                        final_chunk,
                        target_type=pl.BF16,
                        mode="rint",
                    )
            else:
                normed_kv_pad[final_i : final_i + 1, 0:CMP_HEAD_CHUNK] = normed_kv_pad[
                    final_i : final_i + 1,
                    0:CMP_HEAD_CHUNK,
                ]

    cmp_kv = pl.reshape(cmp_kv_flat, [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    kv_state = pl.reshape(kv_state_flat, [MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM])
    score_state = pl.reshape(score_state_flat, [MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM])
    return cmp_kv, kv_state, score_state


@pl.jit
def prefill_compressor_ratio128_test(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B, PREFILL_COMPRESSED_LEN, HEAD_DIM], pl.FP32]],
    kv_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    score_state: pl.Out[pl.Tensor[[B, STATE_LEN, OUT_DIM], pl.FP32]],
    wkv: pl.Tensor[[D, OUT_DIM], pl.BF16],
    wgate: pl.Tensor[[D, OUT_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cos: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], pl.FP32],
    even_idx: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.INT32],
    odd_idx: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.INT32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, HEAD_DIM], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
    rotate: pl.Scalar[pl.BOOL],
):
    return prefill_compressor_ratio128(
        x,
        kv,
        kv_state,
        score_state,
        wkv,
        wgate,
        ape,
        norm_w,
        cos,
        sin,
        even_idx,
        odd_idx,
        kv_cache,
        start_pos,
    )


def golden_prefill_compressor_ratio128(tensors):
    import torch

    start_pos = int(tensors["start_pos"])
    if start_pos % COMPRESS_RATIO != 0:
        raise ValueError("prefill_compressor_ratio128 expects start_pos aligned to COMPRESS_RATIO")
    cache_slot = start_pos // COMPRESS_RATIO
    if cache_slot >= IDX_KV_LEN:
        raise ValueError("prefill_compressor_ratio128 start_pos exceeds kv_cache length")

    kv_proj = tensors["x"].float() @ tensors["wkv"].float()
    score_proj = tensors["x"].float() @ tensors["wgate"].float()
    score_proj = score_proj + tensors["ape"][:S].view(1, S, OUT_DIM)

    tensors["kv_state"][:, :S, :] = kv_proj
    tensors["score_state"][:, :S, :] = score_proj

    pooled = (kv_proj * score_proj.softmax(dim=1)).sum(dim=1, keepdim=True).to(torch.bfloat16).float()
    norm_w = tensors["norm_w"].float()
    inv = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
    kv_norm = (pooled * inv * norm_w.view(1, 1, HEAD_DIM)).to(torch.bfloat16)
    rope = kv_norm[..., NOPE_HEAD_DIM:].float()
    rope_pair = rope.unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos = tensors["cos"].float().view(1, PREFILL_COMPRESSED_LEN, ROPE_HALF)
    sin = tensors["sin"].float().view(1, PREFILL_COMPRESSED_LEN, ROPE_HALF)
    rot_even = (rope_even * cos - rope_odd * sin).to(torch.bfloat16)
    rot_odd = (rope_even * sin + rope_odd * cos).to(torch.bfloat16)
    rope_full = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
    kv_bf16 = torch.cat([kv_norm[..., :NOPE_HEAD_DIM], rope_full], dim=-1).to(torch.bfloat16)
    tensors["kv"][:, 0:1, :] = kv_bf16.float()
    tensors["kv_cache"][:, cache_slot : cache_slot + 1, :] = kv_bf16


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale

    def init_x():
        return seeded_uniform((B, S, D), 1, 0.1)
    def init_kv():
        return torch.zeros(B, PREFILL_COMPRESSED_LEN, HEAD_DIM)
    def init_kv_state():
        return torch.zeros(B, STATE_LEN, OUT_DIM)
    def init_wkv():
        return seeded_uniform((D, OUT_DIM), 2, D ** -0.5)
    def init_wgate():
        return seeded_uniform((D, OUT_DIM), 3, D ** -0.5)
    def init_ape():
        return seeded_uniform((COMPRESS_RATIO, OUT_DIM), 4, 0.1)
    def init_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cos():
        return torch.cos(torch.arange(PREFILL_COMPRESSED_LEN * ROPE_HALF).reshape(PREFILL_COMPRESSED_LEN, ROPE_HALF) * 1e-3)
    def init_sin():
        return torch.sin(torch.arange(PREFILL_COMPRESSED_LEN * ROPE_HALF).reshape(PREFILL_COMPRESSED_LEN, ROPE_HALF) * 1e-3)
    def init_even_idx():
        return torch.arange(0, ROPE_HEAD_DIM, 2, dtype=torch.int32).unsqueeze(0)
    def init_odd_idx():
        return torch.arange(1, ROPE_HEAD_DIM, 2, dtype=torch.int32).unsqueeze(0)
    def init_hadamard():
        return torch.zeros(HEAD_DIM, HEAD_DIM)
    def init_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, HEAD_DIM)

    return [
        TensorSpec("x", [B, S, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv", [B, PREFILL_COMPRESSED_LEN, HEAD_DIM], torch.float32, init_value=init_kv, is_output=True),
        TensorSpec("kv_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("score_state", [B, STATE_LEN, OUT_DIM], torch.float32, init_value=init_kv_state, is_output=True),
        TensorSpec("wkv", [D, OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("cos", [PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_cos),
        TensorSpec("sin", [PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], torch.float32, init_value=init_sin),
        TensorSpec("even_idx", [1, ROPE_HEAD_DIM // 2], torch.int32, init_value=init_even_idx),
        TensorSpec("odd_idx", [1, ROPE_HEAD_DIM // 2], torch.int32, init_value=init_odd_idx),
        TensorSpec("hadamard", [HEAD_DIM, HEAD_DIM], torch.bfloat16, init_value=init_hadamard),
        TensorSpec("kv_cache", [B, IDX_KV_LEN, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        ScalarSpec("start_pos", torch.int32, start_pos),
        ScalarSpec("rotate", torch.bool, ROTATE),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(
        description=(
            "Standalone DeepSeek V4 prefill compressor ratio128 validation. "
            "This entry tests the rectangular compressor; packed HCA uses prefill_compressor_ratio128_packed "
            "with lowered token/request/write metadata."
        )
    )
    parser.add_argument(
        "-p", "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
        help="PyPTO compile/runtime backend for this standalone validation. Default: %(default)s.",
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=0,
        help="NPU device id passed to runtime_cfg.device_id. Under task-submit, '{}' is usually substituted here.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help="Fixture-only absolute compression slot offset; ratio128 standalone expects it aligned to 128.",
    )
    parser.add_argument(
        "--enable-l2-swimlane",
        action="store_true",
        default=False,
        help="Enable L2 swimlane profiling/report generation in runtime_cfg for this validation run.",
    )
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_compressor_ratio128_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_compressor_ratio128,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "kv_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "score_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
