# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 prefill attention_csa scaffold.

Kernel body is intentionally empty; golden follows the torch reference for this stage.
"""

import pypto.language as pl

from config import (
    FLASH as M,
    BLOCK_SIZE,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    PREFILL_BATCH,
    PREFILL_SEQ,
)

from decode_attention_csa import *  # noqa: F401,F403
from prefill_compressor_ratio4 import prefill_compressor_ratio4
from prefill_hc_post import prefill_hc_post
from prefill_hc_pre import prefill_hc_pre
from prefill_indexer import prefill_indexer
from prefill_qkv_proj_rope import prefill_attn_norm, prefill_qkv_proj_rope_core
from prefill_sparse_attn import prefill_sparse_attn

B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
COMPRESS_RATIO = 4
START_POS = 0
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
IDX_HEAD_DIM = M.index_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_TOPK = M.index_topk
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
SPARSE_TOPK = WIN + IDX_TOPK
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
COFF = 2
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_LEN = COFF * COMPRESS_RATIO
ORI_MAX_BLOCKS = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = max(1, (PREFILL_COMPRESSED_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE)
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS
SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK


@pl.jit.inline
def prefill_attention_csa(
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
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cmp_kv_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post_t = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_mixed, post_t, comb_t = prefill_hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post_t,
        comb_t,
    )

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    x_normed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    x_normed = prefill_attn_norm(x_mixed, attn_norm_w, x_normed)
    q, kv, qr, qr_scale = prefill_qkv_proj_rope_core(
        x_normed,
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
        start_pos,
    )

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    idx_cos = pl.create_tensor([S, HALF_ROPE], dtype=pl.FP32)
    idx_sin = pl.create_tensor([S, HALF_ROPE], dtype=pl.FP32)
    cmp_cos = pl.create_tensor([PREFILL_COMPRESSED_LEN, HALF_ROPE], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([PREFILL_COMPRESSED_LEN, HALF_ROPE], dtype=pl.FP32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_rope_rows"):
        pos = pl.cast(start_pos, pl.INDEX)
        cos_rows = pl.slice(freqs_cos, [S, ROPE_HEAD_DIM], [pos, 0])
        sin_rows = pl.slice(freqs_sin, [S, ROPE_HEAD_DIM], [pos, 0])
        rope_cos_t = pl.assemble(rope_cos_t, cos_rows, [0, 0])
        rope_sin_t = pl.assemble(rope_sin_t, sin_rows, [0, 0])
        idx_cos = pl.assemble(
            idx_cos,
            pl.cast(pl.slice(freqs_cos, [S, HALF_ROPE], [pos, 0]), target_type=pl.FP32),
            [0, 0],
        )
        idx_sin = pl.assemble(
            idx_sin,
            pl.cast(pl.slice(freqs_sin, [S, HALF_ROPE], [pos, 0]), target_type=pl.FP32),
            [0, 0],
        )

    for c in pl.range(PREFILL_COMPRESSED_LEN):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_cmp_rope"):
            pos = pl.cast(start_pos + c * COMPRESS_RATIO, pl.INDEX)
            cmp_cos = pl.assemble(
                cmp_cos,
                pl.cast(pl.slice(freqs_cos, [1, HALF_ROPE], [pos, 0]), target_type=pl.FP32),
                [c, 0],
            )
            cmp_sin = pl.assemble(
                cmp_sin,
                pl.cast(pl.slice(freqs_sin, [1, HALF_ROPE], [pos, 0]), target_type=pl.FP32),
                [c, 0],
            )

    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for s_idx in pl.range(S):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_scatter_ori"):
            blk_id = pl.cast(pl.read(ori_block_table, [0, s_idx // BLOCK_SIZE]), pl.INDEX)
            dst_row = blk_id * BLOCK_SIZE + s_idx % BLOCK_SIZE
            kv_cache_flat = pl.assemble(kv_cache_flat, kv[s_idx : s_idx + 1, 0:HEAD_DIM], [dst_row, 0])
    kv_cache = pl.reshape(kv_cache_flat, [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    cmp_out = pl.create_tensor([B, PREFILL_COMPRESSED_LEN, HEAD_DIM], dtype=pl.FP32)
    cmp_dense_cache = pl.create_tensor([B, IDX_KV_LEN, HEAD_DIM], dtype=pl.BF16)
    cmp_out, cmp_kv_state, cmp_score_state, cmp_dense_cache = prefill_compressor_ratio4(
        x_normed,
        cmp_out,
        cmp_kv_state,
        cmp_score_state,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_cos,
        cmp_sin,
        cmp_dense_cache,
        start_pos,
    )

    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for slot in pl.range(PREFILL_COMPRESSED_LEN):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_pack_cmp"):
            blk_id = pl.cast(pl.read(cmp_block_table, [0, slot // BLOCK_SIZE]), pl.INDEX)
            dst_row = blk_id * BLOCK_SIZE + slot % BLOCK_SIZE
            cmp_row = pl.reshape(cmp_out[0:1, slot : slot + 1, 0:HEAD_DIM], [1, HEAD_DIM])
            cmp_kv_flat = pl.assemble(cmp_kv_flat, pl.cast(cmp_row, target_type=pl.BF16), [dst_row, 0])
    cmp_kv = pl.reshape(cmp_kv_flat, [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    inner_kv = pl.create_tensor([B, PREFILL_COMPRESSED_LEN, IDX_HEAD_DIM], dtype=pl.FP32)
    idx_score = pl.create_tensor([B, S, IDX_KV_LEN], dtype=pl.FP32)
    idx_topk_full = pl.create_tensor([B, S, IDX_KV_LEN], dtype=pl.INT32)
    idx_score, idx_kv_cache, idx_topk_full = prefill_indexer(
        x_normed,
        qr,
        qr_scale,
        idx_wq_b,
        idx_wq_b_scale,
        weights_proj,
        idx_cos,
        idx_sin,
        hadamard_idx,
        inner_kv,
        inner_kv_state,
        inner_score_state,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        idx_score,
        idx_topk_full,
        start_pos,
        S,
    )

    sparse_topk = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    idx_topk_flat = pl.reshape(idx_topk_full, [T, IDX_KV_LEN])
    for t_idx in pl.parallel(T):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_sparse_idx_init"):
            invalid_row = pl.full([1, SPARSE_TOPK], dtype=pl.INT32, value=-1)
            sparse_topk = pl.assemble(sparse_topk, invalid_row, [t_idx, 0])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_sparse_idx_window"):
            for key in pl.range(WIN):
                if key <= t_idx:
                    pl.write(sparse_topk, [t_idx, key], pl.cast(key, pl.INT32))
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_sparse_idx_cmp"):
            cmp_topk = pl.slice(idx_topk_flat, [1, PREFILL_COMPRESSED_LEN], [t_idx, 0])
            sparse_topk = pl.assemble(sparse_topk, cmp_topk, [t_idx, WIN])

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = prefill_sparse_attn(
        q,
        kv_cache,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        sparse_topk,
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
    x_out = prefill_hc_post(
        attn_out_3d,
        x_hc,
        post_t,
        comb_t,
        x_out,
    )
    return x_out


@pl.jit
def prefill_attention_csa_test(
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
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cmp_kv_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
):
    x_out = prefill_attention_csa(
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
        even_select_local,
        odd_select_local,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_kv_state,
        cmp_score_state,
        idx_wq_b,
        idx_wq_b_scale,
        weights_proj,
        hadamard_idx,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        inner_kv_state,
        inner_score_state,
        kv_cache,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        idx_kv_cache,
        attn_sink,
        seqused_kv,
        wo_a,
        wo_b,
        wo_b_scale,
        x_out,
        start_pos,
    )
    return x_out


def golden_prefill_attention_csa(tensors):
    """Torch reference for the ratio-4 CSA prefill attention path."""
    import torch

    from prefill_compressor_ratio4 import golden_prefill_compressor_ratio4
    from prefill_hc_post import golden_prefill_hc_post
    from prefill_hc_pre import golden_prefill_hc_pre
    from prefill_indexer import golden_prefill_indexer
    from prefill_qkv_proj_rope import golden_prefill_attn_norm, golden_prefill_qkv_proj_rope
    from prefill_sparse_attn import golden_prefill_sparse_attn

    start_pos = int(tensors["start_pos"])
    if start_pos != 0:
        raise ValueError("golden_prefill_attention_csa only supports start_pos == 0")

    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
    golden_prefill_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_prefill_attn_norm(x_mixed, tensors["attn_norm_w"])
    golden_prefill_qkv_proj_rope({
        "x": x_normed,
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
        "start_pos": tensors["start_pos"],
    })

    positions = torch.arange(start_pos, start_pos + S, device=tensors["freqs_cos"].device)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).reshape(T, ROPE_HEAD_DIM).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).reshape(T, ROPE_HEAD_DIM).contiguous()
    cmp_cos = tensors["freqs_cos"].index_select(0, positions[::COMPRESS_RATIO])[:, :HALF_ROPE].float().contiguous()
    cmp_sin = tensors["freqs_sin"].index_select(0, positions[::COMPRESS_RATIO])[:, :HALF_ROPE].float().contiguous()

    kv_cache = tensors["kv_cache"]
    ori_block_table = tensors["ori_block_table"]
    kv_view = kv.view(B, S, HEAD_DIM)
    for b in range(B):
        for s in range(S):
            blk_id = int(ori_block_table[b, s // BLOCK_SIZE].item())
            kv_cache[blk_id, s % BLOCK_SIZE, 0] = kv_view[b, s]

    cmp_dense_cache = torch.zeros(B, IDX_KV_LEN, HEAD_DIM, dtype=torch.bfloat16, device=kv.device)
    cmp_out = torch.zeros(B, PREFILL_COMPRESSED_LEN, HEAD_DIM, dtype=torch.float32, device=kv.device)
    golden_prefill_compressor_ratio4({
        "x": x_normed,
        "kv": cmp_out,
        "kv_state": tensors["cmp_kv_state"],
        "score_state": tensors["cmp_score_state"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "cos": cmp_cos,
        "sin": cmp_sin,
        "kv_cache": cmp_dense_cache,
        "start_pos": tensors["start_pos"],
    })

    cmp_kv = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]
    for b in range(B):
        for slot in range(PREFILL_COMPRESSED_LEN):
            blk_id = int(cmp_block_table[b, slot // BLOCK_SIZE].item())
            cmp_kv[blk_id, slot % BLOCK_SIZE, 0] = cmp_out[b, slot].to(cmp_kv.dtype)

    inner_kv = torch.zeros(B, PREFILL_COMPRESSED_LEN, IDX_HEAD_DIM, dtype=torch.float32, device=kv.device)
    idx_score = torch.zeros(B, S, IDX_KV_LEN, dtype=torch.float32, device=kv.device)
    idx_topk_full = torch.full((B, S, IDX_KV_LEN), -1, dtype=torch.int32, device=kv.device)
    golden_prefill_indexer({
        "x": x_normed,
        "qr": qr,
        "qr_scale": qr_scale,
        "wq_b": tensors["idx_wq_b"],
        "wq_b_scale": tensors["idx_wq_b_scale"],
        "weights_proj": tensors["weights_proj"],
        "cos": rope_cos_t[:, :HALF_ROPE].float().contiguous(),
        "sin": rope_sin_t[:, :HALF_ROPE].float().contiguous(),
        "hadamard": tensors["hadamard_idx"],
        "inner_kv": inner_kv,
        "inner_kv_state": tensors["inner_kv_state"],
        "inner_score_state": tensors["inner_score_state"],
        "inner_wkv": tensors["inner_wkv"],
        "inner_wgate": tensors["inner_wgate"],
        "inner_ape": tensors["inner_ape"],
        "inner_norm_w": tensors["inner_norm_w"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "score": idx_score,
        "topk_idxs": idx_topk_full,
        "start_pos": tensors["start_pos"],
        "offset": torch.tensor(S, dtype=torch.int32),
    })

    sparse_topk = torch.full((T, WIN + IDX_TOPK), -1, dtype=torch.int32, device=kv.device)
    for t in range(T):
        s = t % S
        sparse_topk[t, :s + 1] = torch.arange(s + 1, dtype=torch.int32, device=kv.device)
    sparse_topk[:, WIN:WIN + PREFILL_COMPRESSED_LEN] = idx_topk_full.view(T, IDX_KV_LEN)[:, :PREFILL_COMPRESSED_LEN]

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16, device=kv.device)
    golden_prefill_sparse_attn({
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

    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16, device=kv.device)
    golden_prefill_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })
    tensors["x_out"][:] = y

def build_tensor_specs(start_pos: int = 0):
    import torch
    from golden import ScalarSpec, TensorSpec

    if start_pos != 0:
        raise ValueError("prefill_attention_csa_draft only supports start_pos == 0")

    def round_half_away_from_zero(x):
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, w.shape[1])
        w_i32 = round_half_away_from_zero(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        return w_i32.to(torch.float16).to(torch.int8), (1.0 / scale_quant).float()

    def quant_w_per_row(w):
        amax = w.float().abs().amax(dim=1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(-1, 1)
        w_i32 = round_half_away_from_zero(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        return w_i32.to(torch.float16).to(torch.int8), (1.0 / scale_quant).float()

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

    def init_even_select_local():
        matrix = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            matrix[2 * i, i] = 1
        return matrix

    def init_odd_select_local():
        matrix = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            matrix[2 * i + 1, i] = 1
        return matrix

    def init_cmp_wkv():
        return torch.randn(D, MAIN_OUT_DIM) / D ** 0.5

    def init_cmp_wgate():
        return torch.randn(D, MAIN_OUT_DIM) / D ** 0.5

    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.01

    def init_cmp_norm_w():
        return torch.ones(HEAD_DIM)

    def init_cmp_state():
        return torch.zeros(B, MAIN_STATE_LEN, MAIN_OUT_DIM)

    def init_cmp_score_state():
        return torch.zeros(B, MAIN_STATE_LEN, MAIN_OUT_DIM)

    def init_idx_wq_b():
        return torch.randn(Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM) / Q_LORA ** 0.5

    def init_weights_proj():
        return torch.randn(D, IDX_N_HEADS) / D ** 0.5

    def init_hadamard_idx():
        return torch.randn(IDX_HEAD_DIM, IDX_HEAD_DIM) * IDX_HEAD_DIM ** -0.5

    def init_inner_wkv():
        return torch.randn(D, INNER_OUT_DIM) / D ** 0.5

    def init_inner_wgate():
        return torch.randn(D, INNER_OUT_DIM) / D ** 0.5

    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.01

    def init_inner_norm_w():
        return torch.ones(IDX_HEAD_DIM)

    def init_inner_state():
        return torch.zeros(B, INNER_STATE_LEN, INNER_OUT_DIM)

    def init_inner_score_state():
        return torch.zeros(B, INNER_STATE_LEN, INNER_OUT_DIM)

    def init_kv_cache():
        return torch.zeros(ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)

    def init_ori_block_table():
        table = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                table[b, j] = b * ORI_MAX_BLOCKS + j
        return table

    def init_cmp_kv():
        return torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)

    def init_cmp_block_table():
        table = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                table[b, j] = b * CMP_MAX_BLOCKS + j
        return table

    def init_idx_kv_cache():
        return torch.zeros(B, IDX_KV_LEN, IDX_HEAD_DIM)

    def init_attn_sink():
        return torch.zeros(H)

    def init_seqused_kv():
        return torch.full((B,), S, dtype=torch.int32)

    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5

    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    idx_wq_b_bf16 = init_idx_wq_b().to(torch.bfloat16)
    idx_wq_b_i8, idx_wq_b_scale = quant_w_per_output_channel(idx_wq_b_bf16)
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
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("even_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("cmp_wkv", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.float32, init_value=init_cmp_norm_w),
        TensorSpec("cmp_kv_state", [B, MAIN_STATE_LEN, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_state),
        TensorSpec("cmp_score_state", [B, MAIN_STATE_LEN, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_score_state),
        TensorSpec("idx_wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: idx_wq_b_i8),
        TensorSpec("idx_wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: idx_wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=init_weights_proj),
        TensorSpec("hadamard_idx", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard_idx),
        TensorSpec("inner_wkv", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.float32, init_value=init_inner_norm_w),
        TensorSpec("inner_kv_state", [B, INNER_STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_state),
        TensorSpec("inner_score_state", [B, INNER_STATE_LEN, INNER_OUT_DIM], torch.float32, init_value=init_inner_score_state),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_kv_cache", [B, IDX_KV_LEN, IDX_HEAD_DIM], torch.bfloat16, init_value=init_idx_kv_cache),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], torch.int32, init_value=init_seqused_kv),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("start_pos", torch.int32, start_pos),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="Prefill CSA only supports start_pos=0.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_attention_csa_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_attention_csa,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=2 / 128,
        atol=3e-3,
        compare_fn={
            "x_out": ratio_allclose(atol=4e-3, rtol=2.0 / 128, max_error_ratio=0.015),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
