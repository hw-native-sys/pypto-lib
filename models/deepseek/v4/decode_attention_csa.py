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
from decode_compressor_ratio4 import compressor
from hc_post import hc_post
from hc_pre import hc_pre
from decode_indexer import indexer
from decode_qkv_proj_rope import qkv_proj_rope
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
MAIN_STATE_MAX_BLOCKS = 64
MAIN_STATE_BLOCK_NUM = B * MAIN_STATE_MAX_BLOCKS
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_LEN = COFF * COMPRESS_RATIO
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_MAX_BLOCKS = 64
INNER_STATE_BLOCK_NUM = B * INNER_STATE_MAX_BLOCKS
INNER_STATE_DIM = 2 * INNER_OUT_DIM
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
IDX_CACHE_MAX_BLOCKS = 64
IDX_CACHE_BLOCK_NUM = B * IDX_CACHE_MAX_BLOCKS

SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK
SPARSE_TOPK = WIN + IDX_TOPK

O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

ORI_MAX_BLOCKS = 1
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 64
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

ROTATE_MAIN = False
ROTATE_INNER = True

# Keep the default fixture on a full-window compression step so sparse_attn
# exercises both the window and compressed paths. The --start-pos fixture option
# covers post-window no-compression/aligned/boundary positions.
START_POS = 126

@pl.jit.inline
def attention_csa(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
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
    even_select_t: pl.Tensor[[HALF_ROPE, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[HALF_ROPE, ROPE_HEAD_DIM], pl.BF16],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, HALF_ROPE], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, HALF_ROPE], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    cmp_even_idx: pl.Tensor[[1, HALF_ROPE], pl.INT32],
    cmp_odd_idx: pl.Tensor[[1, HALF_ROPE], pl.INT32],
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
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post_t = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
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
            start_pos_b = pl.read(start_pos, [b])
            step_pos_b = pl.cast(start_pos_b, pl.INDEX)
            for s in pl.range(S):
                t = b * S + s
                pos_b = pl.cast(start_pos_b + s, pl.INDEX)
                cos_row = pl.cast(pl.slice(freqs_cos, [1, ROPE_HEAD_DIM], [pos_b, 0]), target_type=pl.FP32)
                sin_row = pl.cast(pl.slice(freqs_sin, [1, ROPE_HEAD_DIM], [pos_b, 0]), target_type=pl.FP32)
                rope_cos_t = pl.assemble(rope_cos_t, pl.cast(cos_row, target_type=pl.BF16), [t, 0])
                rope_sin_t = pl.assemble(rope_sin_t, pl.cast(sin_row, target_type=pl.BF16), [t, 0])
            step_cos = pl.assemble(step_cos, pl.cast(pl.slice(freqs_cos, [1, HALF_ROPE], [step_pos_b, 0]), target_type=pl.FP32), [b, 0])
            step_sin = pl.assemble(step_sin, pl.cast(pl.slice(freqs_sin, [1, HALF_ROPE], [step_pos_b, 0]), target_type=pl.FP32), [b, 0])

    cmp_start_pos = pl.create_tensor([B], dtype=pl.INT32)
    cmp_cos = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_cmp_rope"):
        for b in pl.range(B):
            start_pos_b = pl.read(start_pos, [b])
            pl.write(cmp_start_pos, [b], pl.cast(start_pos_b, pl.INT32))
            cmp_offset_b = COMPRESS_RATIO - (start_pos_b % COMPRESS_RATIO)
            cmp_pos_b = pl.cast(start_pos_b + cmp_offset_b - COMPRESS_RATIO, pl.INDEX)
            cmp_cos = pl.assemble(cmp_cos, pl.cast(pl.slice(freqs_cos, [1, HALF_ROPE], [cmp_pos_b, 0]), target_type=pl.FP32), [b, 0])
            cmp_sin = pl.assemble(cmp_sin, pl.cast(pl.slice(freqs_sin, [1, HALF_ROPE], [cmp_pos_b, 0]), target_type=pl.FP32), [b, 0])

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    q = qkv_proj_rope(
        x_mixed,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos_t,
        rope_sin_t,
        even_select_t,
        odd_select_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )

    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [B * ORI_MAX_BLOCKS])
    # Per-batch per-token KV scatter: token s of batch b -> slot (start_pos + s) % WIN.
    for s_idx in pl.range(S):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_scatter_ori"):
            for b in pl.parallel(B):
                start_pos_b = pl.read(start_pos, [b])
                ori_slot = (start_pos_b + s_idx) % WIN
                blk_id = pl.cast(pl.read(ori_block_table_flat, [b * ORI_MAX_BLOCKS + ori_slot // BLOCK_SIZE]), pl.INDEX)
                dst_row = blk_id * BLOCK_SIZE + ori_slot % BLOCK_SIZE
                kv_cache_flat = pl.assemble(kv_cache_flat, kv[b * S + s_idx : b * S + s_idx + 1, 0:HEAD_DIM], [dst_row, 0])
    kv_cache = pl.reshape(kv_cache_flat, [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    cmp_out = pl.create_tensor([B, S, HEAD_DIM], dtype=pl.FP32)
    hadamard_main = pl.create_tensor([HEAD_DIM, HEAD_DIM], dtype=pl.BF16)
    cmp_out, compress_state, cmp_kv = compressor(
        x_mixed,
        cmp_out,
        compress_state,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_cos,
        cmp_sin,
        cmp_even_idx,
        cmp_odd_idx,
        hadamard_main,
        cmp_kv,
        cmp_block_table,
        cmp_start_pos,
        ROTATE_MAIN,
    )

    idx_kv_unused = pl.create_tensor([B, S, IDX_HEAD_DIM], dtype=pl.FP32)
    idx_score_unused = pl.create_tensor([B, S, INDEXER_SCORE_LEN], dtype=pl.FP32)
    idx_topk_full = pl.create_tensor([B, S, INDEXER_SCORE_LEN], dtype=pl.INT32)
    idx_score_unused, idx_kv_cache, idx_topk_full = indexer(
        x_mixed,
        qr,
        qr_scale,
        idx_wq_b,
        idx_wq_b_scale,
        weights_proj,
        step_cos,
        step_sin,
        even_select,
        odd_select,
        cmp_even_idx,
        cmp_odd_idx,
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
        cmp_start_pos,
        WIN,
        ROTATE_INNER,
    )

    # Keep sparse indices as an explicit scratch tensor so sparse_attn sees
    # fixed metadata while the CSA path still composes indexer at runtime.
    cmp_sparse_indices = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    idx_topk_flat = pl.reshape(idx_topk_full, [T, INDEXER_SCORE_LEN])
    for t_idx in pl.parallel(T):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_sparse_idx"):
            window_row = pl.tensor.arange(0, [1, WIN], dtype=pl.INT32)
            invalid_row = pl.full([1, SPARSE_TOPK], dtype=pl.INT32, value=-1)
            cmp_sparse_indices = pl.assemble(cmp_sparse_indices, invalid_row, [t_idx, 0])
            cmp_sparse_indices = pl.assemble(cmp_sparse_indices, window_row, [t_idx, 0])
            cmp_topk = pl.slice(idx_topk_flat, [1, IDX_TOPK], [t_idx, 0])
            cmp_sparse_indices = pl.assemble(cmp_sparse_indices, cmp_topk, [t_idx, WIN])

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = sparse_attn(
        q,
        kv_cache,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        attn_sink,
        seqused_kv,
        rope_cos_t,
        rope_sin_t,
        even_select_local,
        odd_select_local,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )

    attn_out_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    attn_out_3d = pl.reshape(attn_out, [B, S, D])
    x_out = hc_post(attn_out_3d, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def attention_csa_test_refresh(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
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
    even_select_t: pl.Tensor[[HALF_ROPE, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[HALF_ROPE, ROPE_HEAD_DIM], pl.BF16],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, HALF_ROPE], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, HALF_ROPE], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    cmp_even_idx: pl.Tensor[[1, HALF_ROPE], pl.INT32],
    cmp_odd_idx: pl.Tensor[[1, HALF_ROPE], pl.INT32],
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
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Tensor[[B], pl.INT32],
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
        even_select_t,
        odd_select_t,
        even_select,
        odd_select,
        even_select_local,
        odd_select_local,
        cmp_even_idx,
        cmp_odd_idx,
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
        attn_sink,
        seqused_kv,
        wo_a,
        wo_b,
        wo_b_scale,
        x_out,
        start_pos,
    )
    return x_out


def golden_attention_csa(tensors):
    """Torch reference for the ratio-4 compression-step CSA orchestration."""
    import torch

    from decode_compressor_ratio4 import golden_compressor
    from hc_pre import golden_hc_pre
    from decode_indexer import golden_indexer
    from decode_qkv_proj_rope import golden_qkv_proj_rope
    from decode_sparse_attn import golden_sparse_attn
    from hc_post import golden_hc_post

    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    start_pos_t = tensors["start_pos"].to(torch.int64)

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    token_pos = (start_pos_t[:, None] + torch.arange(S, dtype=torch.int64)).reshape(T)
    rope_cos_t = freqs_cos[token_pos].contiguous()
    rope_sin_t = freqs_sin[token_pos].contiguous()
    step_cos = freqs_cos[start_pos_t, :HALF_ROPE].float().contiguous()
    step_sin = freqs_sin[start_pos_t, :HALF_ROPE].float().contiguous()
    cmp_pos = start_pos_t + (COMPRESS_RATIO - (start_pos_t % COMPRESS_RATIO)) - COMPRESS_RATIO
    cmp_cos = freqs_cos[cmp_pos, :HALF_ROPE].float().contiguous()
    cmp_sin = freqs_sin[cmp_pos, :HALF_ROPE].float().contiguous()
    cmp_start_pos = start_pos_t.to(torch.int32).contiguous()

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr_i8 = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    golden_qkv_proj_rope({
        "x": x_mixed,
        "norm_w": tensors["attn_norm_w"],
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

    # Per-batch per-token ori_kv scatter: token s of batch b -> slot (start_pos + s) % WIN.
    for t in range(T):
        b = t // S
        s = t % S
        start_pos_b = int(start_pos_t[b].item())
        ori_slot = (start_pos_b + s) % WIN
        blk_id = int(ori_block_table[b, ori_slot // BLOCK_SIZE].item())
        intra = ori_slot % BLOCK_SIZE
        kv_cache[blk_id, intra, 0] = kv[t]

    cmp_out = torch.zeros(B, S, HEAD_DIM, dtype=torch.float32)
    golden_compressor({
        "x": x_mixed,
        "kv": cmp_out,
        "compress_state": tensors["compress_state"],
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "cos": cmp_cos,
        "sin": cmp_sin,
        "hadamard": torch.eye(HEAD_DIM, dtype=torch.bfloat16),
        "cmp_kv_cache": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "start_pos": cmp_start_pos,
        "rotate": False,
    })

    idx_kv = torch.zeros(B, S, IDX_HEAD_DIM, dtype=torch.float32)
    idx_score = torch.zeros(B, S, INDEXER_SCORE_LEN, dtype=torch.float32)
    idx_topk_full = torch.full((B, S, INDEXER_SCORE_LEN), -1, dtype=torch.int32)
    golden_indexer({
        "x": x_mixed,
        "qr": qr_i8,
        "qr_scale": qr_scale,
        "wq_b": tensors["idx_wq_b"],
        "wq_b_scale": tensors["idx_wq_b_scale"],
        "weights_proj": tensors["weights_proj"],
        "cos": step_cos,
        "sin": step_sin,
        "even_select": tensors["even_select"],
        "odd_select": tensors["odd_select"],
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
        "start_pos": cmp_start_pos,
        "offset": torch.tensor(WIN, dtype=torch.int32),
        "inner_rotate": torch.tensor(ROTATE_INNER, dtype=torch.bool),
    })

    sparse_topk = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
    sparse_topk[:, :WIN] = torch.arange(WIN, dtype=torch.int32)
    sparse_topk[:, WIN:WIN + IDX_TOPK] = idx_topk_full.view(T, INDEXER_SCORE_LEN)[:, :IDX_TOPK]

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": ori_block_table,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "cmp_sparse_indices": sparse_topk,
        "attn_sink": tensors["attn_sink"],
        "seqused_kv": tensors["seqused_kv"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "even_select_local": tensors["even_select_local"],
        "odd_select_local": tensors["odd_select_local"],
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tensor_specs(start_pos: int = START_POS, hetero_start_pos: bool = False):
    import torch
    from golden import TensorSpec
    from hc_pre import golden_hc_pre
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
        return torch.randn(B, S, HC_MULT, D) * 0.05

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
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)

    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)

    def init_even_select_t():
        m = torch.zeros((HALF_ROPE, ROPE_HEAD_DIM))
        for i in range(HALF_ROPE):
            m[i, 2 * i] = 1
        return m

    def init_odd_select_t():
        m = torch.zeros((HALF_ROPE, ROPE_HEAD_DIM))
        for i in range(HALF_ROPE):
            m[i, 2 * i + 1] = 1
        return m

    def init_even_select():
        m = torch.zeros((ROPE_HEAD_DIM, HALF_ROPE))
        for i in range(HALF_ROPE):
            m[2 * i, i] = 1
        return m

    def init_odd_select():
        m = torch.zeros((ROPE_HEAD_DIM, HALF_ROPE))
        for i in range(HALF_ROPE):
            m[2 * i + 1, i] = 1
        return m

    def init_even_select_local():
        m = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            m[2 * i, i] = 1
        return m

    def init_odd_select_local():
        m = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            m[2 * i + 1, i] = 1
        return m

    def init_cmp_even_idx():
        return torch.arange(0, ROPE_HEAD_DIM, 2, dtype=torch.int32).unsqueeze(0).contiguous()

    def init_cmp_odd_idx():
        return torch.arange(1, ROPE_HEAD_DIM, 2, dtype=torch.int32).unsqueeze(0).contiguous()

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
        return torch.zeros(H)

    def init_cmp_sparse_indices():
        return torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)

    def init_start_pos():
        if not hetero_start_pos:
            return torch.full((B,), start_pos, dtype=torch.int32)
        # Keep row 0 at the maximum position because the current indexer score
        # loop bounds are derived from row 0. Alternating with start_pos - 1
        # covers mixed compression/no-compression rows without relying on the
        # unrelated very-short-context sparse_attn fixture path.
        pattern = torch.tensor([start_pos, max(start_pos - 1, 0)], dtype=torch.int32)
        return pattern.repeat((B + pattern.numel() - 1) // pattern.numel())[:B].clone()

    def init_seqused_kv():
        seq = init_start_pos().to(torch.int64) + S
        sparse_len = torch.where(seq <= WIN, seq, WIN + seq // COMPRESS_RATIO)
        return sparse_len.to(torch.int32)


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

    shared_x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    shared_post = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
    shared_comb = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
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
    shared_freqs_cos = init_freqs_cos().to(torch.bfloat16)
    shared_freqs_sin = init_freqs_sin().to(torch.bfloat16)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [B, S, HC_MULT, D], torch.bfloat16, init_value=lambda: shared_x_hc.clone()),
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
        TensorSpec("even_select_t", [HALF_ROPE, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_even_select_t),
        TensorSpec("odd_select_t", [HALF_ROPE, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_odd_select_t),
        TensorSpec("even_select", [ROPE_HEAD_DIM, HALF_ROPE], torch.bfloat16, init_value=init_even_select),
        TensorSpec("odd_select", [ROPE_HEAD_DIM, HALF_ROPE], torch.bfloat16, init_value=init_odd_select),
        TensorSpec("even_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("cmp_even_idx", [1, HALF_ROPE], torch.int32, init_value=init_cmp_even_idx),
        TensorSpec("cmp_odd_idx", [1, HALF_ROPE], torch.int32, init_value=init_cmp_odd_idx),
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
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_kv_cache", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_idx_kv_cache.clone()),
        TensorSpec("idx_block_table", [B, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], torch.int32, init_value=init_seqused_kv),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
        TensorSpec("start_pos", [B], torch.int32, init_value=init_start_pos),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Decode start position for no-compression/aligned/crossing coverage.")
    parser.add_argument("--hetero-start-pos", action=argparse.BooleanOptionalAction, default=True,
                        help="Use per-row start_pos values in the standalone fixture.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()
    max_error_ratio = 0.01 if args.hetero_start_pos else 0.005

    result = run_jit(
        fn=attention_csa_test_refresh,
        specs=build_tensor_specs(args.start_pos, args.hetero_start_pos),
        golden_fn=golden_attention_csa,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=2/128,
        atol=3e-3,
        compare_fn={
            "x_out": ratio_allclose(atol=3e-3, rtol=2.0 / 128, max_error_ratio=max_error_ratio),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
