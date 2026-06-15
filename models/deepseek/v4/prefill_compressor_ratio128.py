# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 token-major prefill compressor, ratio=128."""

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
START_POS = 0

K_CHUNK = 512
OUT_CHUNK = 32
HEAD_CHUNK = 64
K_BLOCKS = D // K_CHUNK
OUT_BLOCKS = OUT_DIM // OUT_CHUNK
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK

assert S == COMPRESS_RATIO, "ratio128 prefill compressor bring-up expects one full compression chunk"


MAX_REQS = 2
MAX_TOKENS = T
MAIN_OUT_DIM = OUT_DIM
MAIN_STATE_LEN = STATE_LEN
HCA_STATE_BLOCK_SIZE = 8
HCA_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + HCA_STATE_BLOCK_SIZE - 1) // HCA_STATE_BLOCK_SIZE
HCA_STATE_BLOCK_NUM = MAX_REQS * HCA_STATE_MAX_BLOCKS
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
def prefill_compressor_ratio128(
    x: pl.Tensor[[MAX_TOKENS, D], pl.BF16],
    kv_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[MAX_REQS, HCA_STATE_MAX_BLOCKS], pl.INT32],
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
    cmp_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
):
    x_flat = x
    kv_proj = pl.create_tensor([MAX_TOKENS, MAIN_OUT_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([MAX_TOKENS, MAIN_OUT_DIM], dtype=pl.FP32)
    kv_state_flat = pl.reshape(kv_state, [HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM])
    score_state_flat = pl.reshape(score_state, [HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM])
    state_block_table_flat = pl.reshape(compress_state_block_table, [MAX_REQS * HCA_STATE_MAX_BLOCKS])
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

    for pool_idx in pl.spmd(PACKED_C128_POOL_BLOCKS, name_hint="prefill_hca_c128_softmax_pool"):
        write_i = pool_idx // CMP_HEAD_BLOCKS
        hb = pool_idx - write_i * CMP_HEAD_BLOCKS
        h0 = hb * CMP_HEAD_CHUNK
        pool_kv_tile = pl.create_tensor([MAIN_STATE_LEN, CMP_HEAD_CHUNK], dtype=pl.FP32)
        pool_score_tile = pl.create_tensor([MAIN_STATE_LEN, CMP_HEAD_CHUNK], dtype=pl.FP32)
        write_token = 0
        write_slot_raw = pl.cast(-1, pl.INT64)
        write_seen = pl.cast(0, pl.INDEX)
        for scan_w in pl.range(MAX_TOKENS):
            if scan_w < num_tokens:
                scan_slot_raw = pl.read(cmp_slot_mapping, [scan_w])
                if scan_slot_raw >= 0:
                    if write_seen == write_i:
                        write_token = scan_w
                        write_slot_raw = scan_slot_raw
                    write_seen = write_seen + 1
        if write_slot_raw >= 0:
            write_pos = pl.read(position_ids, [write_token])
            req = pl.cast(pl.read(token_to_request, [write_token]), pl.INDEX)
            for pool_state_i in pl.range(MAIN_STATE_LEN):
                pool_kv_tile[pool_state_i : pool_state_i + 1, 0:CMP_HEAD_CHUNK] = pl.full(
                    [1, CMP_HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pool_score_tile[pool_state_i : pool_state_i + 1, 0:CMP_HEAD_CHUNK] = pl.full(
                    [1, CMP_HEAD_CHUNK],
                    dtype=pl.FP32,
                    value=0.0,
                )
                pool_abs = write_pos + 1 - COMPRESS_RATIO + pool_state_i
                pool_state_block = pl.cast(pool_abs // HCA_STATE_BLOCK_SIZE, pl.INDEX)
                pool_state_intra = pl.cast(pool_abs - pool_state_block * HCA_STATE_BLOCK_SIZE, pl.INDEX)
                pool_state_block_pos = req * HCA_STATE_MAX_BLOCKS + pool_state_block
                pool_phys_block_raw = pl.read(state_block_table_flat, [pool_state_block_pos])
                if pool_phys_block_raw >= 0:
                    pool_phys_block = pl.cast(pool_phys_block_raw, pl.INDEX)
                    pool_state_row = pool_phys_block * HCA_STATE_BLOCK_SIZE + pool_state_intra
                    pool_kv_tile[pool_state_i : pool_state_i + 1, 0:CMP_HEAD_CHUNK] = kv_state_flat[
                        pool_state_row : pool_state_row + 1,
                        h0 : h0 + CMP_HEAD_CHUNK,
                    ]
                    pool_score_tile[pool_state_i : pool_state_i + 1, 0:CMP_HEAD_CHUNK] = score_state_flat[
                        pool_state_row : pool_state_row + 1,
                        h0 : h0 + CMP_HEAD_CHUNK,
                    ]
            for pool_t in pl.range(MAX_TOKENS):
                if pool_t < num_tokens:
                    pool_req = pl.cast(pl.read(token_to_request, [pool_t]), pl.INDEX)
                    pool_pos = pl.read(position_ids, [pool_t])
                    if pool_req == req:
                        if pool_pos <= write_pos:
                            pool_slot = pl.cast(pool_pos % COMPRESS_RATIO, pl.INDEX)
                            pool_ape = ape[pool_slot : pool_slot + 1, h0 : h0 + CMP_HEAD_CHUNK]
                            pool_score = pl.add(score_proj[pool_t : pool_t + 1, h0 : h0 + CMP_HEAD_CHUNK], pool_ape)
                            pool_kv_tile[pool_slot : pool_slot + 1, 0:CMP_HEAD_CHUNK] = kv_proj[
                                pool_t : pool_t + 1,
                                h0 : h0 + CMP_HEAD_CHUNK,
                            ]
                            pool_score_tile[pool_slot : pool_slot + 1, 0:CMP_HEAD_CHUNK] = pool_score
            init_slot = MAIN_STATE_LEN - 1
            mi_buf = pl.create_tensor([1, CMP_HEAD_CHUNK], dtype=pl.FP32)
            li_buf = pl.create_tensor([1, CMP_HEAD_CHUNK], dtype=pl.FP32)
            oi_buf = pl.create_tensor([1, CMP_HEAD_CHUNK], dtype=pl.FP32)
            mi_buf[0:1, 0:CMP_HEAD_CHUNK] = pool_score_tile[init_slot : init_slot + 1, 0:CMP_HEAD_CHUNK]
            li_buf[0:1, 0:CMP_HEAD_CHUNK] = pl.exp(pl.sub(mi_buf[0:1, 0:CMP_HEAD_CHUNK], mi_buf[0:1, 0:CMP_HEAD_CHUNK]))
            oi_buf[0:1, 0:CMP_HEAD_CHUNK] = pool_kv_tile[init_slot : init_slot + 1, 0:CMP_HEAD_CHUNK]
            for pool_slot_i in pl.range(MAIN_STATE_LEN - 1):
                mi = mi_buf[0:1, 0:CMP_HEAD_CHUNK]
                li = li_buf[0:1, 0:CMP_HEAD_CHUNK]
                oi = oi_buf[0:1, 0:CMP_HEAD_CHUNK]
                slot_score = pool_score_tile[pool_slot_i : pool_slot_i + 1, 0:CMP_HEAD_CHUNK]
                slot_kv = pool_kv_tile[pool_slot_i : pool_slot_i + 1, 0:CMP_HEAD_CHUNK]
                mi_next = pl.maximum(mi, slot_score)
                alpha = pl.exp(pl.sub(mi, mi_next))
                beta = pl.exp(pl.sub(slot_score, mi_next))
                li_next = pl.add(pl.mul(alpha, li), beta)
                oi_next = pl.add(pl.mul(oi, alpha), pl.mul(slot_kv, beta))
                mi_buf[0:1, 0:CMP_HEAD_CHUNK] = mi_next
                li_buf[0:1, 0:CMP_HEAD_CHUNK] = li_next
                oi_buf[0:1, 0:CMP_HEAD_CHUNK] = oi_next
            pooled_chunk = pl.div(
                oi_buf[0:1, 0:CMP_HEAD_CHUNK],
                li_buf[0:1, 0:CMP_HEAD_CHUNK],
            )
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
            norm_write_token = 0
            norm_slot_raw = pl.cast(-1, pl.INT64)
            norm_seen = pl.cast(0, pl.INDEX)
            for scan_w in pl.range(MAX_TOKENS):
                if scan_w < num_tokens:
                    scan_slot_raw = pl.read(cmp_slot_mapping, [scan_w])
                    if scan_slot_raw >= 0:
                        if norm_seen == norm_i:
                            norm_write_token = scan_w
                            norm_slot_raw = scan_slot_raw
                        norm_seen = norm_seen + 1
            if norm_slot_raw >= 0:
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
            final_cmp_row_raw = pl.cast(-1, pl.INT64)
            final_seen = pl.cast(0, pl.INDEX)
            for scan_w in pl.range(MAX_TOKENS):
                if scan_w < num_tokens:
                    scan_slot_raw = pl.read(cmp_slot_mapping, [scan_w])
                    if scan_slot_raw >= 0:
                        if final_seen == final_i:
                            final_cmp_row_raw = scan_slot_raw
                        final_seen = final_seen + 1
            if final_cmp_row_raw >= 0:
                final_cmp_row = pl.cast(final_cmp_row_raw, pl.INDEX)
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

    for update_idx in pl.spmd(MAX_TOKENS * CMP_OUT_BLOCKS, name_hint="prefill_hca_c128_state_update"):
        update_ob = update_idx % CMP_OUT_BLOCKS
        update_t = update_idx // CMP_OUT_BLOCKS
        update_o0 = update_ob * CMP_OUT_CHUNK
        if update_t < num_tokens:
            state_row_raw = pl.read(state_slot_mapping, [update_t])
            if state_row_raw >= 0:
                state_row = pl.cast(state_row_raw, pl.INDEX)
                update_pos = pl.read(position_ids, [update_t])
                ape_slot = pl.cast(update_pos % COMPRESS_RATIO, pl.INDEX)
                ape_row = ape[ape_slot : ape_slot + 1, update_o0 : update_o0 + CMP_OUT_CHUNK]
                pool_dep = pl.mul(pooled_kv_pad[0:1, 0:CMP_OUT_CHUNK], 0.0)
                kv_state_flat[state_row : state_row + 1, update_o0 : update_o0 + CMP_OUT_CHUNK] = pl.add(
                    kv_proj[
                        update_t : update_t + 1,
                        update_o0 : update_o0 + CMP_OUT_CHUNK,
                    ],
                    pool_dep,
                )
                score_state_flat[state_row : state_row + 1, update_o0 : update_o0 + CMP_OUT_CHUNK] = pl.add(
                    pl.add(
                        score_proj[update_t : update_t + 1, update_o0 : update_o0 + CMP_OUT_CHUNK],
                        ape_row,
                    ),
                    pool_dep,
                )

    cmp_kv = pl.reshape(cmp_kv_flat, [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])
    kv_state = pl.reshape(kv_state_flat, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM])
    score_state = pl.reshape(score_state_flat, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM])
    return cmp_kv, kv_state, score_state


@pl.jit
def prefill_compressor_ratio128_test(
    x: pl.Tensor[[MAX_TOKENS, D], pl.BF16],
    kv_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    score_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[MAX_REQS, HCA_STATE_MAX_BLOCKS], pl.INT32],
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
    cmp_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
):
    return prefill_compressor_ratio128(
        x, kv_state, score_state, compress_state_block_table, wkv, wgate, ape, norm_w, freqs_cos, freqs_sin,
        cmp_kv, token_to_request, position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )


def golden_prefill_compressor_ratio128(tensors):
    import torch

    num_tokens = int(tensors["num_tokens"])
    kv_proj = tensors["x"].float() @ tensors["wkv"].float()
    score_proj = tensors["x"].float() @ tensors["wgate"].float()
    kv_state_flat = tensors["kv_state"].view(HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
    score_state_flat = tensors["score_state"].view(HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
    state_block_table = tensors["compress_state_block_table"]
    cmp_kv_flat = tensors["cmp_kv"].view(HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)

    def state_row(req, abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        phys_block = int(state_block_table[req, block].item())
        if phys_block < 0:
            return -1
        return phys_block * HCA_STATE_BLOCK_SIZE + intra

    for token_id in range(num_tokens):
        dst_row = int(tensors["cmp_slot_mapping"][token_id].item())
        if dst_row < 0:
            continue
        req = int(tensors["token_to_request"][token_id].item())
        write_pos = int(tensors["position_ids"][token_id].item())
        pool_kv_state = torch.zeros(MAIN_STATE_LEN, MAIN_OUT_DIM, dtype=torch.float32)
        pool_score_state = torch.zeros(MAIN_STATE_LEN, MAIN_OUT_DIM, dtype=torch.float32)
        for slot in range(MAIN_STATE_LEN):
            row = state_row(req, write_pos + 1 - COMPRESS_RATIO + slot)
            if row >= 0:
                pool_kv_state[slot] = kv_state_flat[row]
                pool_score_state[slot] = score_state_flat[row]
        for t in range(num_tokens):
            if int(tensors["token_to_request"][t].item()) != req:
                continue
            pos = int(tensors["position_ids"][t].item())
            if pos > write_pos:
                continue
            slot = pos % COMPRESS_RATIO
            pool_kv_state[slot] = kv_proj[t]
            pool_score_state[slot] = score_proj[t] + tensors["ape"][slot]
        pooled = (pool_kv_state * pool_score_state.softmax(dim=0)).sum(dim=0, keepdim=True)
        pooled = pooled.to(torch.bfloat16).float()
        inv = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
        normed = (pooled * inv * tensors["norm_w"].float().view(1, HEAD_DIM)).to(torch.bfloat16)
        rope_pair = normed[..., NOPE_DIM:].unflatten(-1, (-1, 2))
        even = rope_pair[..., 0].float()
        odd = rope_pair[..., 1].float()
        cmp_pos = write_pos + 1 - COMPRESS_RATIO
        cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        rot_even = (even * cos - odd * sin).to(torch.bfloat16)
        rot_odd = (even * sin + odd * cos).to(torch.bfloat16)
        normed[:, NOPE_DIM:] = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        cmp_kv_flat[dst_row] = normed[0]

    for t in range(num_tokens):
        pos = int(tensors["position_ids"][t].item())
        dst_row = int(tensors["state_slot_mapping"][t].item())
        if dst_row < 0:
            continue
        slot = pos % COMPRESS_RATIO
        kv_state_flat[dst_row] = kv_proj[t]
        score_state_flat[dst_row] = score_proj[t] + tensors["ape"][slot]


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    num_tokens = T
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if start_pos + num_tokens > MAX_SEQ_LEN:
        raise ValueError("start_pos + num_tokens exceeds max_position_embeddings")

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale
    def init_compress_state_block_table():
        table = torch.full((MAX_REQS, HCA_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for req in range(MAX_REQS):
            for block in range(HCA_STATE_MAX_BLOCKS):
                table[req, block] = req * HCA_STATE_MAX_BLOCKS + ((block * 17 + 3) % HCA_STATE_MAX_BLOCKS)
        return table
    def state_row(req, abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        return int(table[req, block].item()) * HCA_STATE_BLOCK_SIZE + intra
    def init_x():
        return seeded_uniform((MAX_TOKENS, D), 1, 0.1).to(torch.bfloat16)
    def init_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
        for abs_pos in range(max(0, start_pos - COMPRESS_RATIO), start_pos):
            row = state_row(0, abs_pos)
            if row >= 0:
                state.view(-1, MAIN_OUT_DIM)[row] = seeded_uniform((MAIN_OUT_DIM,), 1000 + abs_pos, 0.05)
        return state
    def init_wkv():
        return seeded_uniform((D, MAIN_OUT_DIM), 2, D ** -0.5).to(torch.bfloat16)
    def init_wgate():
        return seeded_uniform((D, MAIN_OUT_DIM), 3, D ** -0.5).to(torch.bfloat16)
    def init_ape():
        return seeded_uniform((COMPRESS_RATIO, MAIN_OUT_DIM), 4, 0.01)
    def init_norm_w():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_cmp_kv():
        return torch.zeros(HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    def init_token_to_request():
        return torch.zeros(MAX_TOKENS, dtype=torch.int32)
    def init_position_ids():
        return torch.arange(start_pos, start_pos + MAX_TOKENS, dtype=torch.int32)
    def init_cmp_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        for token_id in range(num_tokens):
            pos = start_pos + token_id
            if pos + 1 >= COMPRESS_RATIO and (pos + 1) % COMPRESS_RATIO == 0:
                mapping[token_id] = (pos + 1) // COMPRESS_RATIO - 1
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        for token_id in range(num_tokens):
            mapping[token_id] = state_row(0, start_pos + token_id)
        return mapping

    return [
        TensorSpec("x", [MAX_TOKENS, D], torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state", [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], torch.float32, init_value=init_state, is_output=True),
        TensorSpec("score_state", [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], torch.float32, init_value=init_state, is_output=True),
        TensorSpec("compress_state_block_table", [MAX_REQS, HCA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("wkv", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("wgate", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_wgate),
        TensorSpec("ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [HEAD_DIM], torch.float32, init_value=init_norm_w),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_kv", [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv, is_output=True),
        TensorSpec("token_to_request", [MAX_TOKENS], torch.int32, init_value=init_token_to_request),
        TensorSpec("position_ids", [MAX_TOKENS], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("cmp_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_state_slot_mapping),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone token-major DeepSeek V4 prefill compressor ratio128 validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only. This is also the implicit behavior on *sim platforms used by CI.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help=(
            "Fixture-only absolute position for token 0. It is lowered into position_ids and compressed write "
            "slot mapping; it is not a JIT kernel parameter."
        ),
    )
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_compressor_ratio128_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_compressor_ratio128,
        runtime_cfg=dict(platform=args.platform, device_id=args.device, enable_l2_swimlane=args.enable_l2_swimlane),
        rtol=1e-3,
        atol=1e-3,
        compile_only=args.compile_only or args.platform.endswith("sim"),
        compare_fn={
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
            "kv_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
            "score_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
