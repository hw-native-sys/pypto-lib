# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 CSA (Compressed Sparse Attention) decode orchestration.

This standalone harness targets the ratio-4 compression step used by the current
workflow checkpoint. It composes:

- hc_pre
- qkv_proj_rope
- main compressor (ratio=4, rotate=False)
- inner compressor (ratio=4, rotate=True)
- indexer
- sparse_attn (with fused grouped o_proj)
- hc_post

The helper stack in this repo has already moved to the refreshed v4 contracts:
q_proj runs through the W8A8 path, sparse_attn owns grouped o_proj, and the
indexer consumes a prepared `idx_kv_cache` instead of owning the inner
compressor itself. This file aligns to that stack instead of the older draft
surface.
"""


import pypto.language as pl

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from decode_compressor_ratio4 import compressor_ratio4
from hc_post import hc_post
from hc_pre import hc_pre
from decode_indexer import indexer
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import attn_norm
from decode_sparse_attn import sparse_attn

B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps

D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK
WIN = M.sliding_window

QR_D_CHUNK = 512
QR_D_BLOCKS = D // QR_D_CHUNK
QR_CHUNK = 128
QR_BLOCKS = Q_LORA // QR_CHUNK

HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim

IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
INDEXER_SCORE_LEN = M.max_position_embeddings // 4

INNER_HEAD_DIM_INV = 1.0 / IDX_HEAD_DIM
INNER_OUT_CHUNK = 64
INNER_HEAD_CHUNK = 64
INNER_HEAD_BLOCKS = IDX_HEAD_DIM // INNER_HEAD_CHUNK
INNER_ROPE_CHUNK = 32
INNER_NOPE_HEAD_DIM = IDX_HEAD_DIM - ROPE_HEAD_DIM

MAX_SEQ_LEN = M.max_position_embeddings
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)

MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
MAIN_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
MAIN_STATE_MAX_BLOCKS = 65
MAIN_STATE_BLOCK_NUM = B * MAIN_STATE_MAX_BLOCKS
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_LEN = COFF * COMPRESS_RATIO
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = 65
INNER_STATE_BLOCK_NUM = B * INNER_STATE_MAX_BLOCKS
INNER_STATE_DIM = 2 * INNER_OUT_DIM
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_CACHE_MAX_BLOCKS = 64
IDX_CACHE_BLOCK_NUM = B * IDX_CACHE_MAX_BLOCKS

SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK
SPARSE_TOPK = WIN + IDX_TOPK
CSA_TOPK_TOKEN_TILE = 2
CSA_TOPK_BLOCKS = (T + CSA_TOPK_TOKEN_TILE - 1) // CSA_TOPK_TOKEN_TILE
WRITEBACK_GUARD_TILE = 16

O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

ORI_MAX_BLOCKS = 1
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 64
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

@pl.jit.inline
def attention_csa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T, HC_MULT, D], pl.BF16],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    x_mixed = hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post_t,
        comb_t,
    )

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    step_cos = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    step_sin = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_step"):
        for b in pl.range(B):
            first_t = b * S
            first_pos_b = pl.read(position_ids, [first_t])
            step_pos_b = pl.cast(first_pos_b, pl.INDEX)
            for s in pl.range(S):
                t = b * S + s
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                cos_row = pl.cast(pl.slice(freqs_cos, [1, ROPE_HEAD_DIM], [pos_b, 0]), target_type=pl.FP32)
                sin_row = pl.cast(pl.slice(freqs_sin, [1, ROPE_HEAD_DIM], [pos_b, 0]), target_type=pl.FP32)
                rope_cos_t = pl.assemble(rope_cos_t, pl.cast(cos_row, target_type=pl.BF16), [t, 0])
                rope_sin_t = pl.assemble(rope_sin_t, pl.cast(sin_row, target_type=pl.BF16), [t, 0])
            step_cos = pl.assemble(step_cos, pl.cast(pl.slice(freqs_cos, [1, HALF_ROPE], [step_pos_b, 0]), target_type=pl.FP32), [b, 0])
            step_sin = pl.assemble(step_sin, pl.cast(pl.slice(freqs_sin, [1, HALF_ROPE], [step_pos_b, 0]), target_type=pl.FP32), [b, 0])

    cmp_cos = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_cmp_rope"):
        for b in pl.range(B):
            first_t = b * S
            first_pos_b = pl.read(position_ids, [first_t])
            cmp_offset_b = COMPRESS_RATIO - (first_pos_b % COMPRESS_RATIO)
            cmp_pos_b = pl.cast(first_pos_b + cmp_offset_b - COMPRESS_RATIO, pl.INDEX)
            cmp_cos = pl.assemble(cmp_cos, pl.cast(pl.slice(freqs_cos, [1, HALF_ROPE], [cmp_pos_b, 0]), target_type=pl.FP32), [b, 0])
            cmp_sin = pl.assemble(cmp_sin, pl.cast(pl.slice(freqs_sin, [1, HALF_ROPE], [cmp_pos_b, 0]), target_type=pl.FP32), [b, 0])

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    x_normed_t = pl.create_tensor([T, D], dtype=pl.BF16)
    x_normed_t = attn_norm(x_mixed, attn_norm_w, x_normed_t)
    q = qkv_proj_rope(
        x_normed_t,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos_t,
        rope_sin_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )

    x_normed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    x_normed = pl.reshape(x_normed_t, [B, S, D])
    cmp_out = pl.create_tensor([B, S, HEAD_DIM], dtype=pl.FP32)
    position_ids_bsd = pl.create_tensor([B, S], dtype=pl.INT32)
    position_ids_bsd = pl.reshape(position_ids, [B, S])
    cmp_slot_mapping_bsd = pl.create_tensor([B, S], dtype=pl.INT64)
    cmp_slot_mapping_bsd = pl.reshape(cmp_slot_mapping, [B, S])
    idx_slot_mapping_bsd = pl.create_tensor([B, S], dtype=pl.INT64)
    idx_slot_mapping_bsd = pl.reshape(idx_slot_mapping, [B, S])
    state_slot_mapping_bsd = pl.create_tensor([B, S], dtype=pl.INT64)
    state_slot_mapping_bsd = pl.reshape(state_slot_mapping, [B, S])
    inner_state_slot_mapping_bsd = pl.create_tensor([B, S], dtype=pl.INT64)
    inner_state_slot_mapping_bsd = pl.reshape(inner_state_slot_mapping, [B, S])
    cmp_out, compress_state, cmp_kv = compressor_ratio4(
        x_normed,
        cmp_out,
        compress_state,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_cos,
        cmp_sin,
        cmp_kv,
        position_ids_bsd,
        cmp_slot_mapping_bsd,
        state_slot_mapping_bsd,
    )

    idx_kv_unused = pl.create_tensor([B, S, IDX_HEAD_DIM], dtype=pl.FP32)
    idx_score_unused = pl.create_tensor([B, S, INDEXER_SCORE_LEN], dtype=pl.FP32)
    idx_topk_full = pl.create_tensor([B, S, INDEXER_SCORE_LEN], dtype=pl.INT32)
    idx_score_unused, idx_kv_cache, idx_topk_full = indexer(
        x_normed,
        qr,
        qr_scale,
        idx_wq_b,
        idx_wq_b_scale,
        weights_proj,
        step_cos,
        step_sin,
        hadamard_idx,
        idx_kv_unused,
        inner_compress_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        idx_block_table,
        idx_score_unused,
        idx_topk_full,
        position_ids_bsd,
        idx_slot_mapping_bsd,
        inner_state_slot_mapping_bsd,
        kv_seq_lens,
        WIN + S,
    )

    # Keep sparse indices as an explicit scratch tensor so sparse_attn sees
    # fixed metadata while the CSA path still composes indexer at runtime.
    cmp_sparse_indices = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    idx_topk_flat = pl.reshape(idx_topk_full, [T, INDEXER_SCORE_LEN])
    for topk_block in pl.spmd(CSA_TOPK_BLOCKS, name_hint="csa_sparse_idx_tile"):
        topk_t0 = topk_block * CSA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(CSA_TOPK_TOKEN_TILE):
            t_idx = topk_t0 + topk_dt
            if t_idx < T:
                topk_b = t_idx // S
                topk_s = t_idx - topk_b * S
                topk_abs_pos = pl.read(position_ids, [t_idx])

                if topk_abs_pos >= WIN - 1:
                    topk_win_start = (topk_abs_pos % WIN) + 1
                    for topk_k in pl.range(WIN):
                        topk_val = (topk_win_start + topk_k) % WIN
                        topk_out = topk_val
                        for topk_os in pl.range(S):
                            if topk_os <= topk_s:
                                topk_overlay_t = topk_b * S + topk_os
                                topk_overlay_pos = pl.read(position_ids, [topk_overlay_t])
                                if topk_val == topk_overlay_pos % WIN:
                                    topk_out = WIN + topk_os
                        pl.write(cmp_sparse_indices, [t_idx, topk_k], pl.cast(topk_out, pl.INT32))
                else:
                    for topk_k in pl.range(WIN):
                        if topk_k <= topk_abs_pos:
                            topk_out = topk_k
                            for topk_os in pl.range(S):
                                if topk_os <= topk_s:
                                    topk_overlay_t = topk_b * S + topk_os
                                    topk_overlay_pos = pl.read(position_ids, [topk_overlay_t])
                                    if topk_k == topk_overlay_pos % WIN:
                                        topk_out = WIN + topk_os
                            pl.write(cmp_sparse_indices, [t_idx, topk_k], pl.cast(topk_out, pl.INT32))
                        else:
                            pl.write(cmp_sparse_indices, [t_idx, topk_k], pl.cast(-1, pl.INT32))

                topk_cmp_valid = pl.min((topk_abs_pos + 1) // COMPRESS_RATIO, pl.read(kv_seq_lens, [topk_b]) // COMPRESS_RATIO)
                for topk_ck in pl.range(IDX_TOPK):
                    cmp_raw = pl.read(idx_topk_flat, [t_idx, topk_ck])
                    if cmp_raw >= WIN + S:
                        cmp_slot = cmp_raw - (WIN + S)
                        if cmp_slot < topk_cmp_valid:
                            pl.write(cmp_sparse_indices, [t_idx, WIN + topk_ck], pl.cast(cmp_raw, pl.INT32))
                        else:
                            pl.write(cmp_sparse_indices, [t_idx, WIN + topk_ck], pl.cast(-1, pl.INT32))
                    else:
                        pl.write(cmp_sparse_indices, [t_idx, WIN + topk_ck], pl.cast(-1, pl.INT32))

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = sparse_attn(
        q,
        kv_cache,
        ori_block_table,
        kv,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        attn_sink,
        rope_cos_t,
        rope_sin_t,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )

    # Commit current MTP tokens only after sparse_attn has gathered the old cache
    # plus the explicit overlay, matching the SWA/HCA MTP contract.
    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_cache_writeback"):
        kc_touch = kv_cache_flat[0 : 1, 0 : HEAD_DIM]
        kv_cache_flat[0 : 1, 0 : HEAD_DIM] = kc_touch
        for write_t in pl.range(T):
            write_row = pl.cast(pl.read(ori_slot_mapping, [write_t]), pl.INDEX)
            if write_row >= 0:
                write_guard = pl.cast(attn_out[write_t : write_t + 1, 0 : WRITEBACK_GUARD_TILE], target_type=pl.FP32)
                write_zero = pl.mul(write_guard, 0.0)
                write_kv_head = pl.cast(kv[write_t : write_t + 1, 0 : WRITEBACK_GUARD_TILE], target_type=pl.FP32)
                kv_cache_flat[write_row : write_row + 1, 0 : WRITEBACK_GUARD_TILE] = pl.cast(pl.add(write_kv_head, write_zero), target_type=pl.BF16)
                kv_cache_flat[write_row : write_row + 1, WRITEBACK_GUARD_TILE : HEAD_DIM] = kv[write_t : write_t + 1, WRITEBACK_GUARD_TILE : HEAD_DIM]

    x_out = hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def attention_csa_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
):
    x_out = attention_csa(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        freqs_cos,
        freqs_sin,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        compress_state,
        compress_state_block_table,
        idx_wq_b,
        idx_wq_b_scale,
        weights_proj,
        hadamard_idx,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        inner_compress_state,
        inner_compress_state_block_table,
        kv_cache,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        idx_kv_cache,
        idx_block_table,
        ori_slot_mapping,
        cmp_slot_mapping,
        idx_slot_mapping,
        state_slot_mapping,
        inner_state_slot_mapping,
        position_ids,
        kv_seq_lens,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        x_out,
    )
    return x_out


def golden_attention_csa(tensors):
    """Torch reference for the ratio-4 compression-step CSA orchestration."""
    import torch

    from decode_compressor_ratio4 import golden_compressor
    from hc_pre import golden_hc_pre
    from decode_indexer import golden_indexer
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_attn_norm
    from decode_sparse_attn import golden_sparse_attn
    from hc_post import golden_hc_post

    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_t = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb_t = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    position_ids = tensors["position_ids"].to(torch.int64)
    kv_seq_lens = tensors["kv_seq_lens"].to(torch.int64)
    position_ids_bsd = position_ids.reshape(B, S).to(torch.int32).contiguous()
    cmp_slot_mapping_bsd = tensors["cmp_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()
    idx_slot_mapping_bsd = tensors["idx_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()
    state_slot_mapping_bsd = tensors["state_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()
    inner_state_slot_mapping_bsd = tensors["inner_state_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_t = freqs_cos[position_ids].contiguous()
    rope_sin_t = freqs_sin[position_ids].contiguous()
    first_pos = position_ids.reshape(B, S)[:, 0]
    step_cos = freqs_cos[first_pos, :HALF_ROPE].float().contiguous()
    step_sin = freqs_sin[first_pos, :HALF_ROPE].float().contiguous()
    cmp_pos = first_pos + (COMPRESS_RATIO - (first_pos % COMPRESS_RATIO)) - COMPRESS_RATIO
    cmp_cos = freqs_cos[cmp_pos, :HALF_ROPE].float().contiguous()
    cmp_sin = freqs_sin[cmp_pos, :HALF_ROPE].float().contiguous()

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr_i8 = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_attn_norm(x_mixed, tensors["attn_norm_w"])
    golden_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_t,
        "rope_sin": rope_sin_t,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr_i8,
        "qr_scale": qr_scale,
    })

    kv_cache = tensors["kv_cache"]
    ori_block_table = tensors["ori_block_table"]
    cmp_kv = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]

    cmp_out = torch.zeros(B, S, HEAD_DIM, dtype=torch.float32)
    golden_compressor({
        "x": x_normed.reshape(B, S, D),
        "kv": cmp_out,
        "compress_state": tensors["compress_state"],
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "cos": cmp_cos,
        "sin": cmp_sin,
        "cmp_kv_cache": cmp_kv,
        "position_ids": position_ids_bsd,
        "cmp_slot_mapping": cmp_slot_mapping_bsd,
        "state_slot_mapping": state_slot_mapping_bsd,
    })

    idx_kv = torch.zeros(B, S, IDX_HEAD_DIM, dtype=torch.float32)
    idx_score = torch.zeros(B, S, INDEXER_SCORE_LEN, dtype=torch.float32)
    idx_topk_full = torch.full((B, S, INDEXER_SCORE_LEN), -1, dtype=torch.int32)
    golden_indexer({
        "x": x_normed.reshape(B, S, D),
        "qr": qr_i8,
        "qr_scale": qr_scale,
        "wq_b": tensors["idx_wq_b"],
        "wq_b_scale": tensors["idx_wq_b_scale"],
        "weights_proj": tensors["weights_proj"],
        "cos": step_cos,
        "sin": step_sin,
        "hadamard": tensors["hadamard_idx"],
        "inner_kv": idx_kv,
        "inner_compress_state": tensors["inner_compress_state"],
        "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
        "inner_wkv": tensors["inner_wkv"],
        "inner_wgate": tensors["inner_wgate"],
        "inner_ape": tensors["inner_ape"],
        "inner_norm_w": tensors["inner_norm_w"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_block_table": tensors["idx_block_table"],
        "score": idx_score,
        "topk_idxs": idx_topk_full,
        "position_ids": position_ids_bsd,
        "idx_slot_mapping": idx_slot_mapping_bsd,
        "inner_state_slot_mapping": inner_state_slot_mapping_bsd,
        "kv_seq_lens": tensors["kv_seq_lens"],
        "offset": torch.tensor(WIN + S, dtype=torch.int32),
    })

    sparse_topk = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
    for t in range(T):
        b = t // S
        s = t % S
        abs_pos = int(position_ids[t].item())
        overlay_slots = {
            int(position_ids[b * S + os].item()) % WIN: os
            for os in range(s + 1)
        }
        if abs_pos >= WIN - 1:
            win_start = (abs_pos % WIN) + 1
            vals = ((torch.arange(WIN, dtype=torch.int32) + win_start) % WIN).tolist()
        else:
            vals = list(range(abs_pos + 1))
        for k, raw in enumerate(vals):
            if raw in overlay_slots:
                sparse_topk[t, k] = WIN + overlay_slots[raw]
            else:
                sparse_topk[t, k] = raw
    idx_topk_flat = idx_topk_full.view(T, INDEXER_SCORE_LEN)
    for t in range(T):
        b = t // S
        abs_pos = int(position_ids[t].item())
        cmp_valid = min((abs_pos + 1) // COMPRESS_RATIO, int(kv_seq_lens[b].item()) // COMPRESS_RATIO)
        for ck, raw_t in enumerate(idx_topk_flat[t, :IDX_TOPK].tolist()):
            raw = int(raw_t)
            if raw >= WIN + S and raw - (WIN + S) < cmp_valid:
                sparse_topk[t, WIN + ck] = raw

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": ori_block_table,
        "mtp_kv_overlay": kv,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "cmp_sparse_indices": sparse_topk,
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    kv_cache_out = kv_cache.clone()
    ori_slot_mapping = tensors["ori_slot_mapping"].to(torch.int64)
    for t in range(T):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            blk_id = write_row // BLOCK_SIZE
            intra = write_row % BLOCK_SIZE
            kv_cache_out[blk_id, intra, 0] = kv[t]
    tensors["kv_cache"][:] = kv_cache_out

    y = torch.zeros(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tensor_specs(start_pos=None):
    import torch
    from golden import TensorSpec
    from hc_pre import golden_hc_pre
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)
    def round_half_away_from_zero(x):
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, w.shape[1])
        w_i32 = round_half_away_from_zero(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def quant_w_per_row(w):
        amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.unsqueeze(-1)
        w_i32 = round_half_away_from_zero(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x_hc():
        return torch.randn(T, HC_MULT, D) * 0.05

    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5

    def init_hc_attn_scale():
        return torch.ones(3) * 0.5

    def init_hc_attn_base():
        return torch.zeros(MIX_HC)

    def init_attn_norm_w():
        return torch.ones(D)

    def init_wq_a():
        return torch.randn(D, Q_LORA) / D ** 0.5

    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5

    def init_wkv():
        return torch.randn(D, HEAD_DIM) / D ** 0.5

    def init_gamma_cq():
        return torch.ones(Q_LORA)

    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)

    def init_freqs_cos():
        return shared_freqs_cos.clone()

    def init_freqs_sin():
        return shared_freqs_sin.clone()

    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    def init_cmp_wkv():
        return torch.randn(D, MAIN_OUT_DIM) / D ** 0.5

    def init_cmp_wgate():
        return torch.randn(D, MAIN_OUT_DIM) / D ** 0.5

    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.01

    def init_cmp_norm_w():
        return torch.ones(HEAD_DIM)

    def init_compress_state():
        state = torch.zeros(MAIN_STATE_BLOCK_NUM, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM)
        state[:, :, MAIN_OUT_DIM:] = float("-inf")
        starts = init_start_pos().to(torch.int64)
        gen = torch.Generator()
        gen.manual_seed(101)
        hist = torch.randn(MAIN_STATE_BLOCK_NUM, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM, generator=gen) * 0.05
        for b in range(B):
            for abs_pos in range(int(starts[b].item())):
                blk = b * MAIN_STATE_MAX_BLOCKS + abs_pos // MAIN_STATE_BLOCK_SIZE
                intra = abs_pos % MAIN_STATE_BLOCK_SIZE
                state[blk, intra] = hist[blk, intra]
        return state

    def init_compress_state_block_table():
        tbl = torch.full((B, MAIN_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(MAIN_STATE_MAX_BLOCKS):
                tbl[b, j] = b * MAIN_STATE_MAX_BLOCKS + j
        return tbl

    def init_idx_wq_b():
        return torch.randn(Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM) / Q_LORA ** 0.5

    def init_weights_proj():
        return torch.randn(D, IDX_N_HEADS) / D ** 0.5

    def init_hadamard_idx():
        h = torch.ones((1, 1))
        while h.shape[0] < IDX_HEAD_DIM:
            h = torch.cat([
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ], dim=0)
        return h / (IDX_HEAD_DIM ** 0.5)

    def init_inner_wkv():
        return torch.randn(D, INNER_OUT_DIM) / D ** 0.5

    def init_inner_wgate():
        return torch.randn(D, INNER_OUT_DIM) / D ** 0.5

    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.01

    def init_inner_norm_w():
        return torch.ones(IDX_HEAD_DIM)

    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM)
        state[:, :, INNER_OUT_DIM:] = float("-inf")
        starts = init_start_pos().to(torch.int64)
        gen = torch.Generator()
        gen.manual_seed(102)
        hist = torch.randn(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM, generator=gen) * 0.05
        for b in range(B):
            for abs_pos in range(int(starts[b].item())):
                blk = b * INNER_STATE_MAX_BLOCKS + abs_pos // INNER_STATE_BLOCK_SIZE
                intra = abs_pos % INNER_STATE_BLOCK_SIZE
                state[blk, intra] = hist[blk, intra]
        return state

    def init_inner_compress_state_block_table():
        tbl = torch.full((B, INNER_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(INNER_STATE_MAX_BLOCKS):
                tbl[b, j] = b * INNER_STATE_MAX_BLOCKS + j
        return tbl

    def init_kv_cache():
        return init_normalized_cache((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_ori_block_table():
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_cmp_kv():
        return init_normalized_cache((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_cmp_block_table():
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl

    def init_idx_kv_cache():
        return init_normalized_cache((IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM))

    def init_idx_block_table():
        tbl = torch.full((B, IDX_CACHE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(IDX_CACHE_MAX_BLOCKS):
                tbl[b, j] = b * IDX_CACHE_MAX_BLOCKS + j
        return tbl

    def init_attn_sink():
        return torch.ones(H) * 4.0

    def init_start_pos():
        if start_pos is not None:
            return torch.full((B,), start_pos, dtype=torch.int32)
        # Default per-batch pattern mirrors the HCA fixture style while keeping
        # CSA's separate ratio-4 compressor and sliding-window boundaries covered:
        #   10        : short-context compressed cache with multiple valid entries
        #   RATIO-S   : compress, boundary on 2nd token with one cache entry
        #   RATIO-1   : compress, boundary on 1st token with one cache entry
        #   RATIO     : no new boundary on 1st token; 2nd token advances the next window
        #   2*RATIO-S : compress aligned in the 2nd window with previous-window overlap
        #   3*RATIO-1 : compress crossing in the 3rd window with previous-window overlap
        #   WIN-S     : final token reaches the sliding-window boundary
        #   WIN-1     : 1st token reaches the sliding-window boundary; 2nd spills past it
        #   WIN       : post-window ring-cache path with valid compressed tail
        pattern = torch.tensor([
            10,
            COMPRESS_RATIO - S,
            COMPRESS_RATIO - 1,
            COMPRESS_RATIO,
            COMPRESS_RATIO * 2 - S,
            COMPRESS_RATIO * 3 - 1,
            WIN - S,
            WIN - 1,
            WIN,
        ], dtype=torch.int32)
        return pattern.repeat((B + pattern.numel() - 1) // pattern.numel())[:B].clone()

    def init_position_ids():
        starts = init_start_pos().to(torch.int64)
        positions = torch.empty((T,), dtype=torch.int32)
        for t in range(T):
            b = t // S
            s = t - b * S
            positions[t] = starts[b] + s
        return positions

    def init_kv_seq_lens():
        starts = init_start_pos().to(torch.int64)
        return (starts + S).to(torch.int32)

    def init_ori_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        block_table = init_ori_block_table().to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // S
            pos = int(positions[t].item())
            slot = pos % WIN
            blk = int(block_table[b, slot // BLOCK_SIZE].item())
            mapping[t] = blk * BLOCK_SIZE + slot % BLOCK_SIZE
        return mapping

    def init_cmp_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        block_table = init_cmp_block_table().to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // S
            pos = int(positions[t].item())
            if (pos + 1) % COMPRESS_RATIO == 0:
                cache_col = pos // COMPRESS_RATIO
                logical_blk = cache_col // BLOCK_SIZE
                intra = cache_col % BLOCK_SIZE
                blk = int(block_table[b, logical_blk].item())
                mapping[t] = blk * BLOCK_SIZE + intra
        return mapping

    def init_idx_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        block_table = init_idx_block_table().to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // S
            pos = int(positions[t].item())
            if (pos + 1) % COMPRESS_RATIO == 0:
                cache_col = pos // COMPRESS_RATIO
                logical_blk = cache_col // BLOCK_SIZE
                intra = cache_col % BLOCK_SIZE
                blk = int(block_table[b, logical_blk].item())
                mapping[t] = blk * BLOCK_SIZE + intra
        return mapping

    def init_state_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        block_table = init_compress_state_block_table().to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // S
            pos = int(positions[t].item())
            logical_blk = pos // MAIN_STATE_BLOCK_SIZE
            intra = pos % MAIN_STATE_BLOCK_SIZE
            blk = int(block_table[b, logical_blk].item())
            mapping[t] = blk * MAIN_STATE_BLOCK_SIZE + intra
        return mapping

    def init_inner_state_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        block_table = init_inner_compress_state_block_table().to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // S
            pos = int(positions[t].item())
            logical_blk = pos // INNER_STATE_BLOCK_SIZE
            intra = pos % INNER_STATE_BLOCK_SIZE
            blk = int(block_table[b, logical_blk].item())
            mapping[t] = blk * INNER_STATE_BLOCK_SIZE + intra
        return mapping

    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5

    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    shared_x_hc = init_x_hc().to(torch.bfloat16)
    shared_hc_attn_fn = init_hc_attn_fn().to(torch.float32)
    shared_hc_attn_scale = init_hc_attn_scale().to(torch.float32)
    shared_hc_attn_base = init_hc_attn_base().to(torch.float32)
    shared_attn_norm_w = init_attn_norm_w().to(torch.float32)
    shared_wq_a = init_wq_a().to(torch.bfloat16)
    shared_gamma_cq = init_gamma_cq().to(torch.bfloat16)

    shared_x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    shared_post = torch.zeros(T, HC_MULT, dtype=torch.float32)
    shared_comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": shared_x_hc,
        "hc_fn": shared_hc_attn_fn,
        "hc_scale": shared_hc_attn_scale,
        "hc_base": shared_hc_attn_base,
        "x_mixed": shared_x_mixed,
        "post": shared_post,
        "comb": shared_comb,
    })
    shared_idx_wq_b = init_idx_wq_b().to(torch.bfloat16)
    idx_wq_b_i8, idx_wq_b_scale = quant_w_per_output_channel(shared_idx_wq_b)
    shared_weights_proj = init_weights_proj().to(torch.bfloat16)
    shared_hadamard_idx = init_hadamard_idx().to(torch.bfloat16)
    shared_idx_kv_cache = init_idx_kv_cache().to(torch.bfloat16)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.bfloat16, init_value=lambda: shared_x_hc.clone()),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=lambda: shared_hc_attn_fn.clone()),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=lambda: shared_hc_attn_scale.clone()),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=lambda: shared_hc_attn_base.clone()),
        TensorSpec("attn_norm_w", [D], torch.float32, init_value=lambda: shared_attn_norm_w.clone()),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=lambda: shared_wq_a.clone()),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=lambda: shared_gamma_cq.clone()),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_freqs_cos.clone()),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_freqs_sin.clone()),
        TensorSpec("cmp_wkv", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.float32, init_value=init_cmp_norm_w),
        TensorSpec("compress_state", [MAIN_STATE_BLOCK_NUM, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("compress_state_block_table", [B, MAIN_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("idx_wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: idx_wq_b_i8),
        TensorSpec("idx_wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: idx_wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=lambda: shared_weights_proj.clone()),
        TensorSpec("hadamard_idx", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_hadamard_idx.clone()),
        TensorSpec("inner_wkv", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.float32, init_value=init_inner_norm_w),
        TensorSpec("inner_compress_state", [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], torch.float32, init_value=init_inner_compress_state),
        TensorSpec("inner_compress_state_block_table", [B, INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_kv_cache", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_idx_kv_cache.clone()),
        TensorSpec("idx_block_table", [B, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("ori_slot_mapping", [T], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("idx_slot_mapping", [T], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [T], torch.int64, init_value=init_inner_state_slot_mapping),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec("kv_seq_lens", [B], torch.int32, init_value=init_kv_seq_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [T, HC_MULT, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Fixture-only compatibility seed for position_ids and slot mappings; "
                             "otherwise use the default per-batch coverage pattern.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=attention_csa_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_attention_csa,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            # Precision reference: CANN model-level criterion (quantized rel < 1e-2) —
            # cann-recipes-infer/.agents/agents/model-infer-reviewer.md
            "x_out": ratio_reldiff(diff_thd=0.01, pct_thd=0.005, max_diff_hd=10),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
