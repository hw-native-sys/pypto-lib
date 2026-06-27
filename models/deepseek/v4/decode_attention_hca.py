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

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from decode_compressor_ratio128 import compressor_ratio128
from decode_sparse_attn_hca import sparse_attn_hca, PADDED_TOPK


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
OVERLAP = COMPRESS_RATIO == 4   # always False for HCA
COFF = 1 + int(OVERLAP)         # always 1 for HCA
MAIN_OUT_DIM = COFF * HEAD_DIM
ORI_MAX_BLOCKS = 1                  # WIN==BLOCK_SIZE → 1 ori block per batch
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS  # ori KV pool size (matches sparse_attn ORI_BLOCK_NUM)
CMP_MAX_BLOCKS = 8                  # matches sparse_attn CMP_MAX_BLOCKS
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS  # cmp KV pool size (shared with the compressor's cmp_kv_cache)
# Main compressor state pool (kv + score channels merged into one paged FP32 buffer).
COMPRESS_STATE_MAX_BLOCKS = 64
COMPRESS_STATE_BLOCK_NUM = B * COMPRESS_STATE_MAX_BLOCKS
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
CMP_TOPK = MAX_SEQ_LEN // COMPRESS_RATIO   # demo 32; flash/pro 8192 (= 1048576/128); max compressed positions
SPARSE_IDX_TOPK = M.index_topk             # sparse_attn module's IDX_TOPK (static shape contract)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK        # sparse_attn module's TOPK (= 640 for demo)
HCA_TOPK_LIMIT = min(CMP_TOPK, SPARSE_IDX_TOPK)

# ZERO-GATHER: block 0 = window page (slots [0,WIN)), block 1 = S relocated overlay
# slots + the compressed tail, padded to PADDED_TOPK (the kernel reads the full,
# 32-byte-aligned PADDED_TOPK width; the tail past WIN+S+compressed stays -1).
HCA_SPARSE_TOPK = PADDED_TOPK

# tiling
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE
HCA_TOPK_TOKEN_TILE = 8   # tokens per overlay-topk SPMD block
HCA_WB_TOKEN_TILE = 32  # tokens per cache-writeback SPMD block
WRITEBACK_GUARD_TILE = 16  # head cols folded with attn_out*0 to order the writeback after sparse_attn's gather


@pl.jit.inline
def attention_hca(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # main compressor (head_dim=HEAD_DIM, ratio=128, overlap=False)
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    # KV cache split into ori (sliding window) and cmp (compressed) pools to match sparse_attn's contract.
    # cmp_kv is shared with the compressor: it writes the compressed row directly into this pool.
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj (fused into sparse_attn)
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T, HC_MULT, D], pl.BF16],
):
    """HCA decode orchestration for compress_ratio=128."""
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    cmp_cos = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_rope"):
        for b in pl.range(B):
            first_t = b * S
            first_pos_b = pl.read(position_ids, [first_t])
            cmp_offset_b = COMPRESS_RATIO - (first_pos_b % COMPRESS_RATIO)
            cmp_pos_b = pl.cast(first_pos_b + cmp_offset_b - COMPRESS_RATIO, pl.INDEX)
            cmp_cos_row = freqs_cos[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_sin_row = freqs_sin[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_cos[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_cos_row, target_type=pl.FP32)
            cmp_sin[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_sin_row, target_type=pl.FP32)
            for s in pl.range(S):
                t = b * S + s
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                step_cos_row = pl.cast(freqs_cos[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                step_sin_row = pl.cast(freqs_sin[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                rope_cos_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(step_cos_row, target_type=pl.BF16, mode="rint")
                rope_sin_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(step_sin_row, target_type=pl.BF16, mode="rint")

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_norm(x_mixed, attn_norm_w, x_normed)
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)        # unused on HCA path
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale,
    )

    x_normed_bsd = pl.reshape(x_normed, [B, S, D])
    cmp_kv_proj = pl.create_tensor([B, S, HEAD_DIM], dtype=pl.FP32)
    position_ids_bsd = pl.reshape(position_ids, [B, S])
    cmp_slot_mapping_bsd = pl.reshape(cmp_slot_mapping, [B, S])
    state_slot_mapping_bsd = pl.reshape(state_slot_mapping, [B, S])
    compressor_ratio128(
        x_normed_bsd, cmp_kv_proj,
        compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_cos, cmp_sin, cmp_kv,
        position_ids_bsd, cmp_slot_mapping_bsd, state_slot_mapping_bsd,
    )

    # Sparse-index build fanned out over an SPMD (8 tokens/block) instead of one
    # serial CORE_GROUP loop. The two window-slot abs_pos branches collapse into
    # one: column k -> ring slot k, live iff k <= abs_pos. sparse_attn pairs each
    # K/V by its stored raw value (order-agnostic), so the full-ring rotation is
    # dead. The compressed-slot ramp is fused into the same block.
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    topk_all = pl.create_tensor([T, HCA_SPARSE_TOPK], dtype=pl.INT32)
    for topk_block in pl.spmd(T // HCA_TOPK_TOKEN_TILE, name_hint="hca_overlay_topk"):
        topk_t0 = topk_block * HCA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(HCA_TOPK_TOKEN_TILE):
            topk_t = topk_t0 + topk_dt
            if topk_t < T:
                topk_b = topk_t // S
                topk_s = topk_t - topk_b * S
                topk_abs_pos = pl.read(position_ids, [topk_t])

                # Block 0 (slots [0,WIN)) = PHYSICAL window page (slot k -> ring row k),
                # zero-gathered by the kernel. Valid iff k <= abs_pos AND k is NOT a
                # current-token ring slot (those hold stale KV and are attended via the
                # overlay in block 1). build_valid turns this into the page's bias.
                for topk_k in pl.range(WIN):
                    if topk_k <= topk_abs_pos:
                        topk_out = topk_k
                        for topk_os in pl.range(S):
                            if topk_os <= topk_s:
                                topk_overlay_t = topk_b * S + topk_os
                                topk_overlay_pos = pl.read(position_ids, [topk_overlay_t])
                                if topk_k == topk_overlay_pos % WIN:
                                    topk_out = -1
                        pl.write(topk_all, [topk_t, topk_k], pl.cast(topk_out, pl.INT32))
                    else:
                        pl.write(topk_all, [topk_t, topk_k], pl.cast(-1, pl.INT32))

                # Block 1 head: S relocated overlay slots (raw WIN+os), gathered.
                for topk_os in pl.range(S):
                    if topk_os <= topk_s:
                        pl.write(topk_all, [topk_t, WIN + topk_os], pl.cast(WIN + topk_os, pl.INT32))
                    else:
                        pl.write(topk_all, [topk_t, WIN + topk_os], pl.cast(-1, pl.INT32))

                # Block 1 tail: compressed slots (raw WIN+S+ck), gathered.
                topk_cmp_valid = pl.min(
                    HCA_TOPK_LIMIT,
                    pl.min((topk_abs_pos + 1) // COMPRESS_RATIO, pl.read(kv_seq_lens, [topk_b]) // COMPRESS_RATIO),
                )
                for topk_ck in pl.range(HCA_SPARSE_TOPK - WIN - S):
                    if topk_ck < topk_cmp_valid:
                        pl.write(topk_all, [topk_t, WIN + S + topk_ck], pl.cast(WIN + S + topk_ck, pl.INT32))
                    else:
                        pl.write(topk_all, [topk_t, WIN + S + topk_ck], pl.cast(-1, pl.INT32))

    sparse_attn_hca(
        q, kv_cache, ori_block_table, kv,
        cmp_kv, cmp_block_table, topk_all,
        attn_sink, rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )

    # Commit new tokens to the cache AFTER sparse_attn reads the pre-update
    # history (the current token reaches attention via the `kv` overlay).
    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for wb_blk in pl.spmd(T // HCA_WB_TOKEN_TILE, name_hint="hca_cache_writeback"):
        wb_t0 = wb_blk * HCA_WB_TOKEN_TILE
        # No-op self-copy marks kv_cache_flat add_inout for the WAR edge vs
        # sparse_attn's read (pypto-lib#481); row 0 is batch 0's intra-0 slot, only
        # ever written by batch-0 tokens, which all live in this same block 0, so
        # there is no cross-block race.
        if wb_blk == 0:
            kc_touch = kv_cache_flat[0 : 1, 0 : HEAD_DIM]
            kv_cache_flat[0 : 1, 0 : HEAD_DIM] = kc_touch
        for write_dt in pl.range(HCA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            write_row = pl.cast(pl.read(ori_slot_mapping, [write_t]), pl.INDEX)
            if write_row >= 0:
                # Fold attn_out*0 into the first WRITEBACK_GUARD_TILE cols so the
                # writeback data-depends on attn_out -> it cannot run until
                # sparse_attn has finished its in-qk_pv gather of the old cache.
                # Needed because the fused gather_row read is not a tracked tile
                # load, so the kv_touch marker alone does not order it (pypto-lib#481).
                write_guard = pl.cast(attn_out[write_t : write_t + 1, 0 : WRITEBACK_GUARD_TILE], target_type=pl.FP32)
                write_zero = pl.mul(write_guard, 0.0)
                write_kv_head = pl.cast(kv[write_t : write_t + 1, 0 : WRITEBACK_GUARD_TILE], target_type=pl.FP32)
                kv_cache_flat[write_row : write_row + 1, 0 : WRITEBACK_GUARD_TILE] = pl.cast(
                    pl.add(write_kv_head, write_zero), target_type=pl.BF16)
                kv_cache_flat[write_row : write_row + 1, WRITEBACK_GUARD_TILE : HEAD_DIM] = kv[write_t : write_t + 1, WRITEBACK_GUARD_TILE : HEAD_DIM]

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def attention_hca_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
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
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
):
    attention_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, ori_block_table, cmp_kv, cmp_block_table,
        ori_slot_mapping, cmp_slot_mapping, state_slot_mapping,
        position_ids, kv_seq_lens,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
    )
    return x_out


def golden_attention_hca(tensors):
    """End-to-end orchestration for the ratio=128 (HCA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==128 path: main compressor only,
    no indexer, compress_topk_idxs computed deterministically) + Block.hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm
    from decode_compressor_ratio128 import golden_compressor
    from decode_sparse_attn_hca import golden_sparse_attn
    from hc_post import golden_hc_post

    # ---- Block.hc_pre ----
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_t = torch.zeros(T, HC_MULT)
    comb_t = torch.zeros(T, HC_MULT * HC_MULT)
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
    position_ids = tensors["position_ids"].to(torch.int64)
    kv_seq_lens = tensors["kv_seq_lens"].to(torch.int64)
    win = WIN
    ratio = COMPRESS_RATIO
    rd = ROPE_HEAD_DIM

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_T = torch.empty(T, rd, dtype=freqs_cos.dtype)
    rope_sin_T = torch.empty(T, rd, dtype=freqs_sin.dtype)
    for t in range(T):
        pos = int(position_ids[t].item())
        rope_cos_T[t] = freqs_cos[pos]
        rope_sin_T[t] = freqs_sin[pos]

    # q + win kv (W8A8 q_proj)
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    golden_qkv_proj_rope({
        "x": x_normed,
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

    kv_cache = tensors["kv_cache"]
    ori_block_table = tensors["ori_block_table"]
    cmp_kv = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)

    half_rd = rd // 2
    cmp_cos = torch.empty(B, half_rd, dtype=torch.float32)
    cmp_sin = torch.empty(B, half_rd, dtype=torch.float32)
    for b in range(B):
        first_pos_b = int(position_ids[b * S].item())
        cmp_offset_b = ratio - (first_pos_b % ratio)
        cmp_pos_b = first_pos_b + cmp_offset_b - ratio
        cmp_cos[b] = freqs_cos[cmp_pos_b, :half_rd].float()
        cmp_sin[b] = freqs_sin[cmp_pos_b, :half_rd].float()

    cmp_kv_proj = torch.zeros(B, S, HEAD_DIM, dtype=torch.float32)
    position_ids_bsd = position_ids.reshape(B, S).to(torch.int32).contiguous()
    cmp_slot_mapping_bsd = tensors["cmp_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()
    state_slot_mapping_bsd = tensors["state_slot_mapping"].reshape(B, S).to(torch.int64).contiguous()
    golden_compressor({
        "x": x_normed.reshape(B, S, D),
        "kv": cmp_kv_proj,
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

    # ZERO-GATHER layout (must mirror the wrapper's hca_overlay_topk EXACTLY so the
    # golden's block-0 gather lands in the same PHYSICAL order as the kernel's page
    # read -- bf16 PV is not associative): block 0 [0,WIN) = physical window (slot k ->
    # ring row k, live iff k<=abs_pos AND k is not a current-token ring slot); block 1
    # head [WIN,WIN+S) = relocated overlay; block 1 tail [WIN+S,..) = compressed.
    topk_all = torch.full((T, HCA_SPARSE_TOPK), -1, dtype=torch.int32)
    for t in range(T):
        b = t // S
        s = t % S
        abs_pos = int(position_ids[t].item())
        overlay_slots = {
            int(position_ids[b * S + os].item()) % win: os
            for os in range(s + 1)
        }
        for k in range(win):
            if k <= abs_pos and k not in overlay_slots:
                topk_all[t, k] = k
        for os in range(s + 1):
            topk_all[t, win + os] = WIN + os
        cmp_valid = min(HCA_TOPK_LIMIT, (abs_pos + 1) // ratio, int(kv_seq_lens[b].item()) // ratio)
        if cmp_valid:
            topk_all[t, win + S:win + S + cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32) + win + S

    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": ori_block_table,
        "mtp_kv_overlay": kv,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "cmp_sparse_indices": topk_all,
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # In-place sliding-window KV-cache update (validated as an inout tensor).
    ori_slot_mapping = tensors["ori_slot_mapping"].to(torch.int64)
    for t in range(T):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]

    # ===== Block.hc_post =====
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
    import torch  # type: ignore[import]
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

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
        return torch.empty(T, HC_MULT, D).uniform_(-1, 1)
    # Real layer-9 (HCA, ratio-128) hc_attn scale/base (fn synthetic at real magnitude). A
    # synthetic scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out
    # and the hc residual to near-zero in x_out where W8A8 noise blows up the relative tail.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0495
    def init_hc_attn_scale():
        return torch.tensor([0.079046, 0.04213, 0.121901])
    def init_hc_attn_base():
        return torch.tensor([
            -3.3004, 2.5553, -2.2787, -3.4925,
            -3.8197, -3.4161, -2.7144, -2.9181,
            2.362, -2.4746, -2.1352, -3.2216,
            -4.474, 2.2488, -2.1053, -3.1675,
            -2.8362, -1.9042, 2.0432, -3.062,
            -2.7902, -3.0908, -3.002, 3.1161,
        ])
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

    # Main compressor fixtures calibrated to the real DeepSeek-V4-Flash HCA layers
    # (mean l7/l9 of extract_weights_flash): clean zero-mean Gaussian BF16 weights at the
    # measured std; the RMSNorm gamma centers near a measured mean (not ones).
    def init_cmp_wkv():
        return torch.randn(D, MAIN_OUT_DIM) * 0.0246
    def init_cmp_wgate():
        return torch.randn(D, MAIN_OUT_DIM) * 0.0316
    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.0340
    def init_cmp_norm_w():
        return 0.1001 + 0.0549 * torch.randn(HEAD_DIM)
    def init_compress_state():
        return torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
    def init_compress_state_block_table():
        tbl = torch.full((B, COMPRESS_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(COMPRESS_STATE_MAX_BLOCKS):
                tbl[b, j] = b * COMPRESS_STATE_MAX_BLOCKS + j
        return tbl
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
    def init_start_pos():
        if start_pos is not None:
            return torch.full((B,), start_pos, dtype=torch.int32)
        # Default per-batch pattern spans both HCA coverage axes at once:
        # sparse visibility and compressor branches.
        pattern = torch.tensor([
            10,
            COMPRESS_RATIO - S,
            COMPRESS_RATIO - 1,
            COMPRESS_RATIO,
            COMPRESS_RATIO * 2 - S,
            COMPRESS_RATIO * 3 - 1,
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
    def init_state_slot_mapping():
        positions = init_position_ids().to(torch.int64)
        block_table = init_compress_state_block_table().to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // S
            pos = int(positions[t].item())
            logical_blk = pos // COMPRESS_STATE_BLOCK_SIZE
            intra = pos % COMPRESS_STATE_BLOCK_SIZE
            blk = int(block_table[b, logical_blk].item())
            mapping[t] = blk * COMPRESS_STATE_BLOCK_SIZE + intra
        return mapping
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.bfloat16, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_wkv", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("compress_state_block_table", [B, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("ori_slot_mapping", [T], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
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
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Fixture-only compatibility seed for position_ids and slot mappings; "
                             "otherwise use the default per-batch coverage pattern.")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=attention_hca_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_attention_hca,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        atol=1e-2,
        compare_fn={
            # Tightened from CANN's 1e-2 bar: the realistic layer-9 hc_attn gates keep
            # x_out well-conditioned, so it holds 0% over 3e-3 (worst rdiff well under 1).
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.005, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
        rtol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
