# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 HCA (Hierarchical Compressed Attention) decode orchestration — `compress_ratio == 128` path.
Active in layers 3/5 of the model (2 of the 8 layers in demo). Has the main compressor (ratio=128,
overlap=False) but NO indexer; the compressed-portion topk for sparse_attn comes from a deterministic
index computation, not from a learned indexer score.
Companion files: attention_swa.py (ratio=0)
                 attention_csa_draft.py (ratio=4)."""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, INT8_SCALE_MAX, INT8_AMAX_EPS
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from compressor_ratio128 import compressor
from sparse_attn import sparse_attn


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local (HCA: ratio-128 main compressor, no indexer)
COMPRESS_RATIO = 128  # HCA
ROTATE_MAIN = False
OVERLAP = COMPRESS_RATIO == 4   # always False for HCA
COFF = 1 + int(OVERLAP)         # always 1 for HCA
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
ORI_MAX_BLOCKS = 1                  # WIN==BLOCK_SIZE → 1 ori block per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS  # ori KV pool size (matches sparse_attn ORI_BLOCK_NUM)
CMP_MAX_BLOCKS = 64                 # matches sparse_attn CMP_MAX_BLOCKS (HCA writes 1 cmp slot; pool sized for the contract)
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS  # cmp KV pool size
CMP_TOPK = MAX_SEQ_LEN // COMPRESS_RATIO   # demo 32; flash/pro 8192 (= 1048576/128); max compressed positions
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO  # matches compressor_ratio128 kv_cache 2nd-dim contract
SPARSE_IDX_TOPK = M.index_topk             # sparse_attn module's IDX_TOPK (static shape contract)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK        # sparse_attn module's TOPK (= 640 for demo)
HCA_TOPK_CHUNK = 16 if DECODE_BATCH * DECODE_SEQ >= 64 else DECODE_BATCH * DECODE_SEQ
# Non-compression-step fixture: (START_POS + 1) % COMPRESS_RATIO != 0.
# Warmup positions before the first compressed slot are supported; seqused_kv
# bounds the valid compressed topk range.
START_POS = 128
# tiling
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK
SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK


@pl.jit.inline
def attention_hca(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    # main compressor (rotate=False, head_dim=HEAD_DIM, ratio=128, overlap=False)
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cmp_even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    cmp_odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    cmp_kv_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    # KV cache split into ori (sliding window) and cmp (compressed) pools to match sparse_attn's contract.
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B, S], pl.INT32],
    # o_proj (fused into sparse_attn)
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],  # decode step; MUST equal compile-time START_POS — see contract above
    cmp_rotate: pl.Scalar[pl.BOOL],  # always False on the ratio=128 path; threaded through for the compressor inline contract
):
    """HCA decode orchestration for compress_ratio=128. Caller contract:
    runtime ``start_pos`` MUST equal compile-time ``START_POS``.
    """
    compress_offset = COMPRESS_RATIO - (start_pos % COMPRESS_RATIO)

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
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_rope_step"):
        pos = pl.cast(start_pos, pl.INDEX)
        cos_row = pl.cast(pl.slice(freqs_cos, [1, ROPE_HEAD_DIM], [pos, 0]), target_type=pl.FP32)
        sin_row = pl.cast(pl.slice(freqs_sin, [1, ROPE_HEAD_DIM], [pos, 0]), target_type=pl.FP32)
        rope_cos_fp32 = pl.col_expand(
            pl.full([T, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
            cos_row,
        )
        rope_sin_fp32 = pl.col_expand(
            pl.full([T, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
            sin_row,
        )
        rope_cos_t = pl.cast(rope_cos_fp32, target_type=pl.BF16, mode="rint")
        rope_sin_t = pl.cast(rope_sin_fp32, target_type=pl.BF16, mode="rint")

    # Compressor RoPE is only consumed on compression steps. Warmup positions
    # before the first compressed slot keep a zero placeholder to avoid a
    # negative table row.
    cmp_cos = pl.create_tensor([1, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([1, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    if (start_pos % COMPRESS_RATIO) + S >= COMPRESS_RATIO:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_cmp_rope"):
            cmp_cos_base = pl.full([1, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            cmp_sin_base = pl.full([1, ROPE_HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            cmp_pos = pl.cast(start_pos + compress_offset - COMPRESS_RATIO, pl.INDEX)
            cmp_cos = pl.col_expand(
                cmp_cos_base,
                pl.cast(pl.slice(freqs_cos, [1, ROPE_HEAD_DIM // 2], [cmp_pos, 0]), target_type=pl.FP32),
            )
            cmp_sin = pl.col_expand(
                cmp_sin_base,
                pl.cast(pl.slice(freqs_sin, [1, ROPE_HEAD_DIM // 2], [cmp_pos, 0]), target_type=pl.FP32),
            )
    else:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_cmp_rope_zero"):
            zero_cos = pl.cast(pl.slice(freqs_cos, [1, ROPE_HEAD_DIM // 2], [0, 0]), target_type=pl.FP32)
            zero_sin = pl.cast(pl.slice(freqs_sin, [1, ROPE_HEAD_DIM // 2], [0, 0]), target_type=pl.FP32)
            cmp_cos = pl.sub(zero_cos, zero_cos)
            cmp_sin = pl.sub(zero_sin, zero_sin)

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)        # unused on HCA path
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

    # ori_kv scatter at slot = (start_pos + s) % WIN per token s of each batch.
    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [B * ORI_MAX_BLOCKS])
    for s_idx in pl.range(S):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_scatter_ori"):
            ori_slot = (start_pos + s_idx) % WIN
            for b in pl.parallel(0, B, 1):
                blk_id = pl.cast(pl.read(ori_block_table_flat, [b]), pl.INDEX)
                dst_row = blk_id * BLOCK_SIZE + ori_slot
                kv_cache_flat = pl.assemble(
                    kv_cache_flat,
                    kv[b * S + s_idx : b * S + s_idx + 1, 0:HEAD_DIM],
                    [dst_row, 0],
                )
    kv_cache = pl.reshape(kv_cache_flat, [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    # Main compressor (ratio=128, rotate=False).
    # New compressor signature returns (kv, kv_state, score_state, kv_cache); the BF16
    # compressed row we want is `cmp_kv_buf[:, start_pos // COMPRESS_RATIO, :]`.
    # `hadamard_identity` is a dead buffer on this path (rotate=False); pypto leaves
    # `pl.create_tensor` uninitialized so it does not consume Vec buffer budget.
    cmp_kv_proj = pl.create_tensor([B, S, HEAD_DIM], dtype=pl.FP32)
    cmp_kv_buf = pl.create_tensor([B, IDX_KV_LEN, HEAD_DIM], dtype=pl.BF16)
    hadamard_identity = pl.create_tensor([HEAD_DIM, HEAD_DIM], dtype=pl.BF16)
    cmp_kv_proj, cmp_kv_state, cmp_score_state, cmp_kv_buf = compressor(
        x_mixed,
        cmp_kv_proj,
        cmp_kv_state,
        cmp_score_state,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_cos,
        cmp_sin,
        cmp_even_select,
        cmp_odd_select,
        hadamard_identity,
        cmp_kv_buf,
        start_pos,
        cmp_rotate,
    )

    # cmp_kv scatter only happens on compression steps. Non-compression steps
    # update compressor state but keep the existing compressed cache intact.
    if (start_pos % COMPRESS_RATIO) + S >= COMPRESS_RATIO:
        cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
        cmp_block_table_flat = pl.reshape(cmp_block_table, [B * CMP_MAX_BLOCKS])
        cmp_kv_buf_flat = pl.reshape(cmp_kv_buf, [B * IDX_KV_LEN, HEAD_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_scatter_cmp"):
            cmp_slot_rel = start_pos // COMPRESS_RATIO
            cmp_intra = cmp_slot_rel % BLOCK_SIZE
            cmp_blk_off = cmp_slot_rel // BLOCK_SIZE
            for b in pl.parallel(0, B, 1):
                cmp_blk_id = pl.cast(pl.read(cmp_block_table_flat, [b * CMP_MAX_BLOCKS + cmp_blk_off]), pl.INDEX)
                cmp_dst_row = cmp_blk_id * BLOCK_SIZE + cmp_intra
                cmp_src_row = b * IDX_KV_LEN + cmp_slot_rel
                cmp_kv_flat = pl.assemble(
                    cmp_kv_flat,
                    cmp_kv_buf_flat[cmp_src_row:cmp_src_row + 1, 0:HEAD_DIM],
                    [cmp_dst_row, 0],
                )
        cmp_kv = pl.reshape(cmp_kv_flat, [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    # topk_idxs: [0..WIN) window plus deterministic compressed slots.
    # sparse_attn's static TOPK contract is SPARSE_TOPK (= WIN+IDX_TOPK = 640 in demo);
    # The actual valid count is bounded by seqused_kv inside sparse_attn.
    # Chunk over T so [T, SPARSE_TOPK] INT32 (T=128 * 640 * 4 = 320KB) doesn't blow Vec budget.
    topk_idxs = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    for t0 in pl.range(0, T, HCA_TOPK_CHUNK):
        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="hca_topk"):
            win_idx = pl.arange(0, [1, WIN], dtype=pl.INT32)
            cmp_idx = pl.add(
                pl.arange(0, [1, CMP_TOPK], dtype=pl.INT32),
                pl.full([1, CMP_TOPK], dtype=pl.INT32, value=WIN),
            )
            pad_idx = pl.full([1, SPARSE_IDX_TOPK - CMP_TOPK], dtype=pl.INT32, value=-1)
            topk_row = pl.concat(pl.concat(win_idx, cmp_idx), pad_idx)
            topk_tile = pl.col_expand(
                pl.full([HCA_TOPK_CHUNK, SPARSE_TOPK], dtype=pl.INT32, value=-1),
                topk_row,
            )
            topk_idxs = pl.assemble(topk_idxs, topk_tile, [t0, 0])

    # sparse_attn + fused o_proj.
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn(
        q,
        kv_cache,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        topk_idxs,
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
    x_out = hc_post(
        attn_out_3d,
        x_hc,
        post_t,
        comb_t,
        x_out,
    )
    return x_out


@pl.jit
def attention_hca_test(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cmp_even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    cmp_odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    cmp_kv_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B, S], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
    cmp_rotate: pl.Scalar[pl.BOOL],
):
    x_out = attention_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, even_select_t, odd_select_t,
        even_select_local, odd_select_local,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_even_select, cmp_odd_select,
        cmp_kv_state, cmp_score_state,
        kv_cache, ori_block_table, cmp_kv, cmp_block_table,
        attn_sink, seqused_kv,
        wo_a, wo_b, wo_b_scale,
        x_out,
        start_pos,
        cmp_rotate,
    )
    return x_out


def golden_attention_hca(tensors):
    """End-to-end orchestration for the ratio=128 (HCA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==128 path: main compressor only,
    no indexer, compress_topk_idxs computed deterministically) + Block.hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from compressor_ratio128 import golden_compressor
    from sparse_attn import golden_sparse_attn
    from hc_post import golden_hc_post

    # ---- Block.hc_pre ----
    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    # ===== Attention.forward, ratio==128 branch =====
    start_pos = int(tensors["start_pos"])
    bsz, seqlen, _ = x_mixed.shape
    win = WIN
    ratio = COMPRESS_RATIO
    rd = ROPE_HEAD_DIM
    compress_offset = ratio - (start_pos % ratio)
    should_compress = compress_offset <= S

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    step_cos = freqs_cos[start_pos:start_pos + 1]                            # [1, rd]
    step_sin = freqs_sin[start_pos:start_pos + 1]
    rope_cos_T = step_cos.expand(T, rd).contiguous()
    rope_sin_T = step_sin.expand(T, rd).contiguous()
    half_rd = rd // 2
    if should_compress:
        cmp_pos = start_pos + compress_offset - ratio
        cmp_cos = freqs_cos[cmp_pos:cmp_pos + 1, :half_rd]   # ratio128 compressor: half-vec [1, rd//2]
        cmp_sin = freqs_sin[cmp_pos:cmp_pos + 1, :half_rd]
    else:
        cmp_cos = torch.zeros(1, half_rd, dtype=torch.bfloat16)
        cmp_sin = torch.zeros(1, half_rd, dtype=torch.bfloat16)

    # q + win kv (W8A8 q_proj)
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    golden_qkv_proj_rope({
        "x": x_mixed,
        "norm_w": tensors["attn_norm_w"],
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_T,
        "rope_sin": rope_sin_T,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,                                                              # qr unused on HCA path
        "qr_scale": qr_scale,
    })

    # window topk + compress topk; HCA uses get_compress_topk_idxs.
    # Pad to sparse_attn's static SPARSE_TOPK; seqused_kv bounds the valid range.
    topk_idxs = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
    topk_idxs[:, :win] = torch.arange(win, dtype=torch.int32)
    offset = win
    cache_len = (start_pos + seqlen) // ratio
    if cache_len > 0:
        k = min(cache_len, CMP_TOPK)
        topk_idxs[:, win:win + k] = torch.arange(k, dtype=torch.int32) + offset
    topk_idxs = topk_idxs.int()

    # ori_kv scatter
    kv_cache = tensors["kv_cache"]
    ori_block_table = tensors["ori_block_table"]
    cmp_kv = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]
    # Per-batch per-token ori_kv scatter: token s of batch b -> slot (start_pos + s) % win.
    for t in range(T):
        b = t // S
        s = t % S
        ori_slot = (start_pos + s) % win
        blk_id = int(ori_block_table[b, ori_slot // BLOCK_SIZE].item())
        intra = ori_slot % BLOCK_SIZE
        kv_cache[blk_id, intra, 0] = kv[t]

    # main compressor (writes cmp_kv via the orchestration scatter below on should_compress)
    # New golden_compressor contract: requires `kv` / `kv_cache` / `rotate` / `even_select` / `odd_select`,
    # writes the BF16 compressed row into kv_cache[:bsz, start_pos // ratio].
    cmp_kv_proj = torch.zeros(B, S, HEAD_DIM, dtype=torch.float32)
    cmp_kv_cache_buf = torch.zeros(B, MAX_SEQ_LEN // ratio, HEAD_DIM, dtype=torch.bfloat16)
    golden_compressor({
        "x": x_mixed,
        "kv": cmp_kv_proj,
        "kv_state": tensors["cmp_kv_state"],
        "score_state": tensors["cmp_score_state"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "cos": cmp_cos,
        "sin": cmp_sin,
        "even_select": tensors["cmp_even_select"],
        "odd_select": tensors["cmp_odd_select"],
        "hadamard": torch.eye(HEAD_DIM, dtype=torch.bfloat16),                 # rotate=False; identity placeholder
        "kv_cache": cmp_kv_cache_buf,
        "start_pos": tensors["start_pos"],
        "rotate": torch.tensor(False),
    })
    if should_compress:
        cmp_slot_rel = start_pos // ratio
        for b in range(B):
            blk_id = int(cmp_block_table[b, cmp_slot_rel // BLOCK_SIZE].item())
            intra = cmp_slot_rel % BLOCK_SIZE
            cmp_kv[blk_id, intra, 0] = cmp_kv_cache_buf[b, cmp_slot_rel]

    # sparse_attn + fused o_proj
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": ori_block_table,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "cmp_sparse_indices": topk_idxs,
        "attn_sink": tensors["attn_sink"],
        "seqused_kv": tensors["seqused_kv"].view(B, S),
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "even_select_local": tensors["even_select_local"],
        "odd_select_local": tensors["odd_select_local"],
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # ===== Block.hc_post =====
    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })

    tensors["x_out"][:] = y


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def quant_w_per_row(w):
        amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.unsqueeze(-1)
        w_i32 = torch.round(scaled).to(torch.int32)
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
        m = torch.zeros((ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM))
        for i in range(ROPE_HEAD_DIM // 2):
            m[i, 2 * i] = 1
        return m
    def init_odd_select_t():
        m = torch.zeros((ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM))
        for i in range(ROPE_HEAD_DIM // 2):
            m[i, 2 * i + 1] = 1
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
    def init_cmp_even_select():
        m = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            m[2 * i, i] = 1
        return m
    def init_cmp_odd_select():
        m = torch.zeros((ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2))
        for i in range(ROPE_HEAD_DIM // 2):
            m[2 * i + 1, i] = 1
        return m
    def init_cmp_kv_state():
        return torch.zeros(B, MAIN_STATE_LEN, MAIN_OUT_DIM)
    def init_cmp_score_state():
        return torch.full((B, MAIN_STATE_LEN, MAIN_OUT_DIM), float("-inf"))
    def init_kv_cache():
        return init_normalized_cache((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))
    def init_cmp_kv():
        return init_normalized_cache((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_ori_block_table():
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_cmp_block_table():
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl

    def init_attn_sink():
        return torch.zeros(H)
    def init_seqused_kv():
        # sparse_attn uses: window_valid = min(WIN, seq_used); cmp_valid = seq_used - window_valid.
        s = torch.arange(1, S + 1, dtype=torch.int32) + START_POS
        seq = torch.where(s <= WIN, s, WIN + s // COMPRESS_RATIO)
        return seq.expand(B, S).clone()
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wq_b_scale = wq_b_scale.view(Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [B, S, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.float32, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("even_select_t", [ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_even_select_t),
        TensorSpec("odd_select_t", [ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_odd_select_t),
        TensorSpec("even_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("cmp_wkv", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.float32, init_value=init_cmp_norm_w),
        TensorSpec("cmp_even_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_cmp_even_select),
        TensorSpec("cmp_odd_select", [ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], torch.bfloat16, init_value=init_cmp_odd_select),
        TensorSpec("cmp_kv_state", [B, MAIN_STATE_LEN, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_kv_state),
        TensorSpec("cmp_score_state", [B, MAIN_STATE_LEN, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_score_state),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B, S], torch.int32, init_value=init_seqused_kv),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("start_pos", torch.int32, START_POS),
        ScalarSpec("cmp_rotate", torch.bool, ROTATE_MAIN),
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
        fn=attention_hca_test,
        specs=build_tensor_specs(),
        golden_fn=golden_attention_hca,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        atol=1e-2,
        compare_fn={
            "x_out": ratio_allclose(atol=3e-3, rtol=2.0 / 128),
        },
        # Random ori/cmp cache fixtures exercise non-zero history values
# instead of the previous all-zero cache.
rtol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
