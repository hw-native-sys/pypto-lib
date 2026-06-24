# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill CSA attention.

This module wires HC pre/post, ratio-4 compressor, indexer, sparse attention,
and cache writeback for the compressed sparse attention path. Single-request
the layer owns the per-request loop and feeds this op one
contiguous run of <=T tokens.
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

from prefill_compressor_ratio4 import (
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    golden_prefill_compressor_ratio4,
    prefill_compressor_ratio4,
)
from hc_post import golden_hc_post, hc_post
from hc_pre import golden_hc_pre, hc_pre
from prefill_indexer import (
    IDX_CACHE_MAX_BLOCKS,
    INDEXER_SCORE_CAP,
    INDEXER_TOPK_CAP,
    gen_shared_weight,
    golden_prefill_indexer_core,
    prefill_indexer,
)
from prefill_indexer_compressor import (
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
)
from qkv_proj_rope import golden_qkv_proj_rope, materialize_rope_rows, qkv_proj_rope
from rmsnorm import golden_rms_norm, rms_norm
from prefill_sparse_attn import (
    _quant_w_per_channel,
    golden_prefill_sparse_attn,
    prefill_sparse_attn,
)

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
CSA_TOPK_TOKEN_TILE = 2


# prefill_sparse_attn cache/topk contract (mirrors prefill_sparse_attn).
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
SPARSE_ORI_MAX_BLOCKS = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
SPARSE_CMP_MAX_BLOCKS = max(1, (PREFILL_MAX_COMPRESSED + BLOCK_SIZE - 1) // BLOCK_SIZE)
PREFILL_SPARSE_TOPK = min(SPARSE_TOPK, min(WIN, S) + PREFILL_MAX_COMPRESSED)
PREFILL_ATTN_TILE = 128
PREFILL_ATTN_BLOCKS = (PREFILL_SPARSE_TOPK + PREFILL_ATTN_TILE - 1) // PREFILL_ATTN_TILE
SPARSE_PREFILL_SPARSE_PAD = PREFILL_ATTN_BLOCKS * PREFILL_ATTN_TILE

MAX_CMP_WRITES = max(1, T // COMPRESS_RATIO)
CSA_ORI_BLOCK_NUM = SPARSE_ORI_MAX_BLOCKS
CSA_CMP_BLOCK_NUM = SPARSE_CMP_MAX_BLOCKS
assert S == WIN, "packed CSA prefill currently assumes one static window page"


@pl.jit.inline
def prefill_attention_csa(
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
    cmp_kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_kv_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_norm(x_mixed, attn_norm_w, x_normed)

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        position_ids,
        num_tokens,
        rope_cos_t,
        rope_sin_t,
    )
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale,
    )

    prefill_compressor_ratio4(
        x_normed, cmp_kv_state, cmp_score_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        freqs_cos, freqs_sin, cmp_kv,
        position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )
    # Half-width FP32 cos/sin rows for the indexer Q-RoPE: gather freqs at each token's position
    # and take the first HALF_ROPE columns (matches the golden's materialize_half_rope_tables).
    idx_cos = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    idx_sin = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_idx_halfrope"):
        for idx_t in pl.range(T):
            idx_pos = pl.cast(pl.read(position_ids, [idx_t]), pl.INDEX)
            idx_cos = pl.assemble(idx_cos, pl.cast(pl.slice(freqs_cos, [1, HALF_ROPE], [idx_pos, 0]), target_type=pl.FP32), [idx_t, 0])
            idx_sin = pl.assemble(idx_sin, pl.cast(pl.slice(freqs_sin, [1, HALF_ROPE], [idx_pos, 0]), target_type=pl.FP32), [idx_t, 0])

    cmp_topk_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    idx_score_unused = pl.create_tensor([T, INDEXER_SCORE_CAP], dtype=pl.FP32)
    prefill_indexer(
        x_normed, qr, qr_scale,
        idx_wq_b, idx_wq_b_scale, idx_weights_proj,
        idx_cos, idx_sin, freqs_cos, freqs_sin, hadamard_idx,
        inner_kv_state, inner_score_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        idx_kv_cache, idx_block_table,
        idx_score_unused, cmp_topk_indices,
        position_ids, num_tokens,
        idx_slot_mapping, inner_state_slot_mapping,
    )

    # Assemble the packed sparse-index rows (window ring + overlay + compressed
    # slots) inline; padding rows stay all -1 via the pl.full row template.
    # Rows are fully -1 padded below, so expose the whole width to the shared kernel
    # via a constant lens == SPARSE_TOPK (the kernel's per-slot >= 0 check does the
    # masking). Build it through assemble (SSA-tracked) so the gather's read is ordered
    # after this write, not via in-place pl.write which can race the round-trip.
    # Build it 2D ([1, T], assemble-friendly) inside a scope, then reshape to the
    # [T] the kernel expects -- assemble + reshape is SSA-tracked so the gather's read
    # is ordered after this write (a 1D in-place pl.write races the round-trip).
    cmp_sparse_lens_2d = pl.create_tensor([1, T], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_sparse_lens"):
        cmp_sparse_lens_2d = pl.assemble(cmp_sparse_lens_2d, pl.full([1, T], dtype=pl.INT32, value=SPARSE_TOPK), [0, 0])
    cmp_sparse_lens = pl.reshape(cmp_sparse_lens_2d, [T])
    cmp_sparse_work = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    for topk_block in pl.spmd((T + CSA_TOPK_TOKEN_TILE - 1) // CSA_TOPK_TOKEN_TILE,
                              name_hint="prefill_csa_sparse_idx_tile"):
        topk_t0 = topk_block * CSA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(CSA_TOPK_TOKEN_TILE):
            t_idx = topk_t0 + topk_dt
            sparse_row = pl.full([1, SPARSE_TOPK], dtype=pl.INT32, value=-1)
            if t_idx < num_tokens:
                abs_pos = pl.read(position_ids, [t_idx])
                window_valid = pl.min(WIN, abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for sparse_col in pl.range(SPARSE_TOPK):
                    sparse_raw = pl.cast(-1, pl.INT32)
                    sparse_col_i32 = pl.cast(sparse_col, pl.INT32)
                    if sparse_col_i32 < window_valid:
                        key_abs = key_start_abs + sparse_col_i32
                        sparse_raw = pl.cast(key_abs % WIN, pl.INT32)
                        for scan_t in pl.range(T):
                            if scan_t < num_tokens:
                                if scan_t <= t_idx:
                                    scan_pos = pl.read(position_ids, [scan_t])
                                    if scan_pos == key_abs:
                                        sparse_raw = pl.cast(WIN + scan_t, pl.INT32)
                    else:
                        comp_start = window_valid
                        if sparse_col_i32 >= comp_start:
                            comp_col = sparse_col_i32 - comp_start
                            visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
                            if comp_col < visible_cmp:
                                if comp_col < pl.cast(INDEXER_TOPK_CAP, pl.INT32):
                                    topk_raw = pl.read(cmp_topk_indices, [t_idx, comp_col])
                                    if topk_raw >= WIN + T:
                                        sparse_raw = topk_raw
                    pl.write(sparse_row, [0, sparse_col], sparse_raw)
            cmp_sparse_work = pl.assemble(cmp_sparse_work, sparse_row, [t_idx, 0])
    # CSA builds every sparse row itself and pads unused entries with -1, so it
    # can safely expose the full row width to the shared sparse-attn kernel,
    # while external serving callers still use cmp_sparse_lens in
    # prefill_sparse_attn where lens == 0 means no valid sparse indices.

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    prefill_sparse_attn(
        q, kv_cache, ori_block_table, kv,
        cmp_kv, cmp_block_table,
        cmp_sparse_work, cmp_sparse_lens,
        attn_sink, num_tokens,
        rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )
    # Commit new tokens to the cache AFTER sparse_attn reads the pre-update
    # history (the current tokens reach attention via the `kv` overlay).
    kv_cache_flat = pl.reshape(kv_cache, [CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_cache_writeback"):
        # No-op self-copy: marks kv_cache add_inout so the runtime orders this
        # write after the gather's read (WAR); see pypto-lib#481.
        kc_touch = kv_cache_flat[0:1, 0:HEAD_DIM]
        kv_cache_flat[0:1, 0:HEAD_DIM] = kc_touch
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]

    hc_post(attn_out, x_hc, post, comb, x_out)
    return kv_cache, cmp_kv, cmp_kv_state, cmp_score_state, idx_kv_cache, inner_kv_state, inner_score_state, x_out


@pl.jit
def prefill_attention_csa_test(
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
    cmp_kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_kv_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    prefill_attention_csa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_kv_state, cmp_score_state, compress_state_block_table,
        hadamard_idx, idx_wq_b, idx_wq_b_scale, idx_weights_proj,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        inner_kv_state, inner_score_state, inner_compress_state_block_table,
        kv_cache, ori_block_table, ori_slot_mapping,
        cmp_kv, cmp_block_table, idx_kv_cache, idx_block_table,
        position_ids, cmp_slot_mapping, idx_slot_mapping,
        state_slot_mapping, inner_state_slot_mapping,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_out, num_tokens,
    )
    return kv_cache, cmp_kv, cmp_kv_state, cmp_score_state, idx_kv_cache, inner_kv_state, inner_score_state, x_out


def golden_prefill_attention_csa(tensors):
    """Torch reference for token-major packed CSA with overlay compressor/indexer."""
    import torch

    num_tokens = int(tensors["num_tokens"])
    x_hc_rect = tensors["x_hc"].view(B, S, HC_MULT, D)
    x_hc_flat = x_hc_rect.view(T, HC_MULT, D)
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": x_hc_flat,
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    rope_cos_t = torch.zeros(T, ROPE_HEAD_DIM, dtype=torch.bfloat16)
    rope_sin_t = torch.zeros(T, ROPE_HEAD_DIM, dtype=torch.bfloat16)
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).contiguous()
    golden_qkv_proj_rope({
        "x": x_normed.view(T, D),
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
        "qr": qr,
        "qr_scale": qr_scale,
    })

    golden_prefill_compressor_ratio4({
        "x": x_normed.view(T, D),
        "kv_state": tensors["cmp_kv_state"],
        "score_state": tensors["cmp_score_state"],
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "cmp_kv": tensors["cmp_kv"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "cmp_slot_mapping": tensors["cmp_slot_mapping"],
        "state_slot_mapping": tensors["state_slot_mapping"],
    })
    idx_cos = rope_cos_t[:, :HALF_ROPE].float().contiguous()
    idx_sin = rope_sin_t[:, :HALF_ROPE].float().contiguous()
    cmp_topk_indices, _idx_score = golden_prefill_indexer_core({
        "x": x_normed.view(T, D),
        "qr": qr,
        "qr_scale": qr_scale,
        "wq_b": tensors["idx_wq_b"],
        "wq_b_scale": tensors["idx_wq_b_scale"],
        "weights_proj": tensors["idx_weights_proj"],
        "cos": idx_cos,
        "sin": idx_sin,
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "hadamard": tensors["hadamard_idx"],
        "inner_kv_state": tensors["inner_kv_state"],
        "inner_score_state": tensors["inner_score_state"],
        "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
        "inner_wkv": tensors["inner_wkv"],
        "inner_wgate": tensors["inner_wgate"],
        "inner_ape": tensors["inner_ape"],
        "inner_norm_w": tensors["inner_norm_w"],
        "idx_kv_cache": tensors["idx_kv_cache"],
        "idx_block_table": tensors["idx_block_table"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    })

    def assemble_sparse_indices(cmp_topk):
        topk_idxs = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
        sparse_lens = torch.zeros(T, dtype=torch.int32)
        pos = tensors["position_ids"]
        current = {int(pos[t].item()): t for t in range(num_tokens)}
        compressed_cap = SPARSE_PREFILL_SPARSE_PAD - WIN
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                overlay_t = current.get(key_abs)
                if overlay_t is not None and overlay_t <= t:
                    topk_idxs[t, k] = WIN + overlay_t
                else:
                    topk_idxs[t, k] = key_abs % WIN
            cursor = window_valid
            visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
            for ck, raw_t in enumerate(cmp_topk[t, :compressed_cap].tolist()):
                if cursor >= SPARSE_PREFILL_SPARSE_PAD:
                    break
                raw = int(raw_t)
                if raw >= WIN + T and raw - (WIN + T) < visible_cmp:
                    topk_idxs[t, cursor] = raw
                    cursor += 1
            sparse_lens[t] = cursor
        return topk_idxs, sparse_lens

    cmp_sparse_indices, cmp_sparse_lens = assemble_sparse_indices(cmp_topk_indices)
    kv_cache_in = tensors["kv_cache"].clone()
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": kv_cache_in,
        "ori_block_table": tensors["ori_block_table"],
        "kv_overlay": kv,
        "cmp_kv": tensors["cmp_kv"],
        "cmp_block_table": tensors["cmp_block_table"],
        "cmp_sparse_indices": cmp_sparse_indices,
        "cmp_sparse_lens": cmp_sparse_lens,
        "attn_sink": tensors["attn_sink"],
        "num_tokens": tensors["num_tokens"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    kv_cache_out = kv_cache_in.clone()
    kv_cache_flat = kv_cache_out.view(CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(num_tokens):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]
    tensors["kv_cache"][:] = kv_cache_out

    y = torch.zeros(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post,
        "comb": comb,
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tensor_specs(
    start_pos: int = START_POS,
    num_tokens: int = T,
):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    # Single-request geometry: q_len = num_tokens (active prefix), context_len =
    # start_pos (absolute position base, a multiple of S=WIN under chunked prefill).
    context_len = start_pos
    q_len = num_tokens
    if num_tokens <= 0 or num_tokens > T:
        raise ValueError(f"num_tokens must be in [1, {T}], got {num_tokens}")
    if context_len < 0:
        raise ValueError(f"context length must be non-negative, got {context_len}")
    max_position = context_len + q_len - 1 if q_len > 0 else 0
    if max_position >= MAX_SEQ_LEN:
        raise ValueError(f"position id {max_position} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}")
    max_visible_cmp = (context_len + q_len) // COMPRESS_RATIO
    max_sparse_rows = WIN + max_visible_cmp
    if max_sparse_rows > SPARSE_PREFILL_SPARSE_PAD:
        raise ValueError(
            f"needs {max_sparse_rows} sparse rows; current packed sparse CSA cap is {SPARSE_PREFILL_SPARSE_PAD}"
        )
    if max_visible_cmp > SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE:
        raise ValueError(
            f"needs {max_visible_cmp} compressed slots; current cmp cache cap is "
            f"{SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE}"
        )
    # The real prefill indexer scores only PREFILL_COMPRESSED_LEN compressed slots (its inner
    # compressor handles only the current chunk, no historical compressed context), which is a
    # tighter bound than the sparse cmp-cache cap above. Reject suffix-prefill loudly here.
    if max_visible_cmp > PREFILL_COMPRESSED_LEN:
        raise ValueError(
            f"the real prefill indexer scores only PREFILL_COMPRESSED_LEN={PREFILL_COMPRESSED_LEN} "
            f"compressed slots, but context_len={context_len} + q_len={q_len} needs {max_visible_cmp}. "
            f"Suffix-prefill is not yet supported by the indexer (full prefill / start_pos=0 only)."
        )


    def token_pos():
        # Single-request absolute positions: pos[t] = context_len + local_idx
        # Padding rows keep their arange default; they are inactive.
        pos = torch.arange(T, dtype=torch.int32)
        for local_s in range(q_len):
            pos[local_s] = context_len + local_s
        return pos

    def cmp_write_records():
        pos = token_pos()
        records = []
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            if (abs_pos + 1) % COMPRESS_RATIO == 0:
                cmp_slot = (abs_pos + 1) // COMPRESS_RATIO - 1
                records.append((t, cmp_slot))
        if len(records) > MAX_CMP_WRITES:
            raise ValueError(f"CSA fixture generated {len(records)} compressed writes, cap is {MAX_CMP_WRITES}")
        return records

    def validate_overlay_topk(topk_idxs, pos):
        current = {int(pos[t].item()): t for t in range(num_tokens)}
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            seen_window_abs = set()
            seen_cmp = set()
            for raw_i in topk_idxs[t, :SPARSE_TOPK].tolist():
                raw = int(raw_i)
                if raw < 0:
                    continue
                if raw < WIN:
                    candidates = [
                        key_abs
                        for key_abs in range(key_start_abs, abs_pos + 1)
                        if key_abs % WIN == raw
                    ]
                    if len(candidates) != 1:
                        raise ValueError(f"ambiguous CSA ring raw={raw} for token {t}")
                    key_abs = candidates[0]
                    if key_abs in current:
                        raise ValueError(f"current suffix abs_pos={key_abs} must use CSA overlay for token {t}")
                    if key_abs in seen_window_abs:
                        raise ValueError(f"duplicate CSA window abs_pos={key_abs} for token {t}")
                    seen_window_abs.add(key_abs)
                elif raw < WIN + T:
                    overlay_t = raw - WIN
                    if overlay_t >= num_tokens:
                        raise ValueError(f"CSA overlay raw={raw} points past active tokens for token {t}")
                    key_abs = int(pos[overlay_t].item())
                    if key_abs > abs_pos:
                        raise ValueError(f"CSA overlay raw={raw} is future key abs_pos={key_abs} for token {t}")
                    if key_abs in seen_window_abs:
                        raise ValueError(f"duplicate CSA overlay abs_pos={key_abs} for token {t}")
                    seen_window_abs.add(key_abs)
                else:
                    cmp_slot = raw - (WIN + T)
                    visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
                    if cmp_slot < 0 or cmp_slot >= visible_cmp:
                        raise ValueError(f"CSA compressed slot={cmp_slot} is not visible for token {t}")
                    if cmp_slot in seen_cmp:
                        raise ValueError(f"duplicate CSA compressed slot={cmp_slot} for token {t}")
                    seen_cmp.add(cmp_slot)

    def init_x_hc():
        x = torch.empty(T, HC_MULT, D).uniform_(-1, 1)
        x[num_tokens:] = 0
        return x
    # Real layer-8 (CSA, ratio-4) hc_attn scale/base (fn synthetic at real magnitude). A
    # synthetic scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out
    # and the hc residual to near-zero in x_out where W8A8 noise blows up the relative tail.
    # Mirrors decode_attention_csa.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0519
    def init_hc_attn_scale():
        return torch.tensor([0.076099, 0.032597, 0.226994])
    def init_hc_attn_base():
        return torch.tensor([
            5.9166, -3.6223, -2.9324, -3.3124,
            -3.9100, -0.9384, -3.3256, -2.5240,
            2.0706, -2.5728, 0.1424, -3.9453,
            -3.8859, 3.4634, -3.3799, -2.6077,
            -2.7191, -2.4846, 2.0395, -0.5010,
            -3.5992, -2.7520, -3.3493, 3.1587,
        ])
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return (torch.rand(D, Q_LORA) - 0.5) * D ** -0.5
    def init_wq_b():
        return (torch.rand(Q_LORA, H * HEAD_DIM) - 0.5) * Q_LORA ** -0.5
    def init_wkv():
        return (torch.rand(D, HEAD_DIM) - 0.5) * D ** -0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    # Quant-faithful CSA (ratio-4) main compressor fixtures (mean l8/l32 of extract_weights_flash):
    # zero-mean Gaussian BF16 weights at the measured std; RMSNorm gamma near the measured mean.
    # Mirrors decode_attention_csa / decode_compressor_ratio4.
    def init_cmp_wkv():
        return torch.randn(D, MAIN_OUT_DIM) * 0.0245
    def init_cmp_wgate():
        return torch.randn(D, MAIN_OUT_DIM) * 0.0388
    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.1243
    def init_cmp_norm_w():
        return 0.9666 + torch.randn(HEAD_DIM,) * 0.1929
    def init_compress_state_block_table():
        table = torch.full((CSA_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(CSA_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % CSA_STATE_MAX_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        return int(table[block].item()) * CSA_STATE_BLOCK_SIZE + intra
    def init_cmp_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
        flat = state.view(-1, MAIN_OUT_DIM)
        for abs_pos in range(max(0, context_len - MAIN_STATE_LEN), context_len):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(MAIN_OUT_DIM,) - 0.5) * 0.05
        return state
    def init_cmp_score_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
        flat = state.view(-1, MAIN_OUT_DIM)
        for abs_pos in range(max(0, context_len - MAIN_STATE_LEN), context_len):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(MAIN_OUT_DIM,) - 0.5) * 0.05
        return state
    def init_hadamard_idx():
        h = torch.ones((1, 1))
        while h.shape[0] < IDX_HEAD_DIM:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        return h * (IDX_HEAD_DIM ** -0.5)
    # Quant-faithful indexer inner compressor fixtures (mean l8/l32 of extract_weights_flash):
    # zero-mean Gaussian BF16 weights at the measured std; RMSNorm gamma near the measured mean.
    # Mirrors decode_attention_csa / decode_indexer.
    def init_inner_wkv():
        return torch.randn(D, INNER_OUT_DIM) * 0.0293
    def init_inner_wgate():
        return torch.randn(D, INNER_OUT_DIM) * 0.0512
    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1528
    def init_inner_norm_w():
        return 0.6850 + torch.randn(IDX_HEAD_DIM,) * 0.2610
    def init_inner_compress_state_block_table():
        table = torch.full((INNER_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(INNER_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % INNER_STATE_MAX_BLOCKS
        return table
    def inner_state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_inner_compress_state_block_table()
        block = abs_pos // INNER_STATE_BLOCK_SIZE
        intra = abs_pos % INNER_STATE_BLOCK_SIZE
        return int(table[block].item()) * INNER_STATE_BLOCK_SIZE + intra
    def init_inner_kv_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM)
        flat = state.view(-1, INNER_OUT_DIM)
        for abs_pos in range(max(0, context_len - INNER_STATE_LEN), context_len):
            row = inner_state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(INNER_OUT_DIM,) - 0.5) * 0.05
        return state
    def init_inner_score_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM)
        flat = state.view(-1, INNER_OUT_DIM)
        for abs_pos in range(max(0, context_len - INNER_STATE_LEN), context_len):
            row = inner_state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(INNER_OUT_DIM,) - 0.5) * 0.05
        return state
    def init_idx_kv_cache():
        cache = torch.zeros(CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM)
        cache_flat = cache.view(CSA_CMP_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
        table = init_idx_block_table()
        completed = context_len // COMPRESS_RATIO
        for cmp_slot in range(completed):
            if cmp_slot >= SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE:
                break
            row = cache_row_from_table(table, cmp_slot)
            if row >= 0:
                cache_flat[row] = ((torch.rand(IDX_HEAD_DIM,) - 0.5) * 0.05).to(torch.bfloat16)
        return cache
    def init_kv_cache():
        cache = torch.zeros(CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_ori_block_table()
        start = max(0, context_len - WIN)
        for abs_pos in range(start, context_len):
            row = cache_row_from_table(table, abs_pos % WIN)
            value = (torch.rand(HEAD_DIM,) - 0.5) * 0.1
            if row >= 0:
                cache_flat[row] = value.to(torch.bfloat16)
        return cache
    def init_ori_block_table():
        table = torch.full((SPARSE_ORI_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(SPARSE_ORI_MAX_BLOCKS):
            table[block] = block
        return table
    def init_ori_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        pos = token_pos()
        table = init_ori_block_table()
        for t in range(num_tokens):
            mapping[t] = cache_row_from_table(table, int(pos[t].item()) % WIN)
        return mapping
    def init_cmp_kv():
        cache = torch.zeros(CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(CSA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_cmp_block_table()
        completed = context_len // COMPRESS_RATIO
        for cmp_slot in range(completed):
            if cmp_slot >= SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE:
                break
            row = cache_row_from_table(table, cmp_slot)
            value = (torch.rand(HEAD_DIM,) - 0.5) * 0.1
            if row >= 0:
                cache_flat[row] = value.to(torch.bfloat16)
        return cache
    def init_cmp_block_table():
        table = torch.full((SPARSE_CMP_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(SPARSE_CMP_MAX_BLOCKS):
            table[block] = block
        return table
    def init_idx_block_table():
        table = torch.full((IDX_CACHE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(IDX_CACHE_MAX_BLOCKS):
            table[block] = block
        return table
    def cache_row_from_table(table, slot):
        block = slot // BLOCK_SIZE
        intra = slot % BLOCK_SIZE
        phys_block = int(table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * BLOCK_SIZE + intra
    def init_cmp_sparse_indices():
        topk_idxs = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
        pos = token_pos()
        current = {int(pos[t].item()): t for t in range(num_tokens)}
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            cursor = 0
            for key_abs in range(key_start_abs, abs_pos + 1):
                overlay_t = current.get(key_abs)
                if overlay_t is not None and overlay_t <= t:
                    topk_idxs[t, cursor] = WIN + overlay_t
                else:
                    topk_idxs[t, cursor] = key_abs % WIN
                cursor += 1
            visible_cmp = min((abs_pos + 1) // COMPRESS_RATIO, SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE)
            for cmp_slot in range(visible_cmp):
                if cursor >= SPARSE_PREFILL_SPARSE_PAD:
                    break
                topk_idxs[t, cursor] = WIN + T + cmp_slot
                cursor += 1
        validate_overlay_topk(topk_idxs, pos)
        return topk_idxs
    def init_position_ids():
        return token_pos()
    def init_cmp_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        table = init_cmp_block_table()
        records = cmp_write_records()
        for token_id, cmp_slot in records:
            mapping[token_id] = cache_row_from_table(table, cmp_slot)
        return mapping
    def init_idx_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        table = init_idx_block_table()
        records = cmp_write_records()
        for token_id, cmp_slot in records:
            mapping[token_id] = cache_row_from_table(table, cmp_slot)
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        pos = token_pos()
        for t in range(num_tokens):
            mapping[t] = state_row(int(pos[t].item()))
        return mapping
    def init_inner_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        pos = token_pos()
        for t in range(num_tokens):
            mapping[t] = inner_state_row(int(pos[t].item()))
        return mapping
    def init_attn_sink():
        return torch.zeros(H)
    def init_wo_a():
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN ** -0.5
    def init_wo_b():
        return (torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel_local(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)
    # Indexer Q up-proj + weights projection (mirrors the standalone prefill_indexer fixtures).
    idx_wq_b_i8_T, idx_wq_b_scale = gen_shared_weight((IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56)
    idx_wq_b_i8 = idx_wq_b_i8_T.t().contiguous()

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
        TensorSpec(
            "cmp_kv_state",
            [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM],
            torch.float32,
            init_value=init_cmp_state,
        ),
        TensorSpec(
            "cmp_score_state",
            [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM],
            torch.float32,
            init_value=init_cmp_score_state,
        ),
        TensorSpec("compress_state_block_table", [CSA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("hadamard_idx", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard_idx),
        TensorSpec("idx_wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: idx_wq_b_i8),
        TensorSpec("idx_wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: idx_wq_b_scale),
        TensorSpec("idx_weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=lambda: (torch.randn(D, IDX_N_HEADS) * 0.2313).to(torch.bfloat16)),
        TensorSpec("inner_wkv", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec(
            "inner_kv_state",
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM],
            torch.float32,
            init_value=init_inner_kv_state,
        ),
        TensorSpec(
            "inner_score_state",
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM],
            torch.float32,
            init_value=init_inner_score_state,
        ),
        TensorSpec("inner_compress_state_block_table", [INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("kv_cache", [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_kv_cache, is_output=True),
        TensorSpec("ori_block_table", [SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("ori_slot_mapping", [T], torch.int64, init_value=init_ori_slot_mapping),
        # Compressor / indexer caches are written in-place but not validated here
        # (decode parity); the dedicated prefill_compressor_ratio4 and
        # prefill_indexer tests cover them.
        TensorSpec(
            "cmp_kv",
            [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
        ),
        TensorSpec("cmp_block_table", [SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec(
            "idx_kv_cache",
            [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM],
            torch.bfloat16,
            init_value=init_idx_kv_cache,
        ),
        TensorSpec("idx_block_table", [IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("idx_slot_mapping", [T], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [T], torch.int64, init_value=init_inner_state_slot_mapping),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [T, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


def valid_ratio_reldiff(
    num_tokens: int,
    diff_thd: float,
    pct_thd: float,
    max_diff_hd: float,
):
    """Relative-diff comparator restricted to the valid (active) token rows.

    Mirrors decode_attention_csa's ``ratio_reldiff`` bar and prefill_layer's
    ``valid_ratio_reldiff`` pattern: the packed buffer carries up to
    ``T`` rows but only the leading ``num_tokens`` are active, so the trailing
    padding rows (whose device scratch is undefined) are sliced off before the
    relative-diff check.
    """
    from golden import ratio_reldiff

    base_cmp = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd, max_diff_hd=max_diff_hd)

    def cmp(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        return base_cmp(
            actual[:num_tokens],
            expected[:num_tokens],
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )

    cmp.__name__ = f"valid_ratio_reldiff(num_tokens={num_tokens})"
    return cmp


def _quant_w_per_output_channel_local(w):
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return w_i32.to(torch.float16).to(torch.int8), (1.0 / scale_quant).float()


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 packed prefill CSA correctness test.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="context_len (multiple of S=WIN); fixture-only, lowered into token metadata.")
    parser.add_argument("--num-tokens", type=int, default=T,
                        help="Active token count (q_len), capped by T; passed to the kernel as num_tokens.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    args = parser.parse_args()
    compare_tokens = args.num_tokens

    result = run_jit(
        fn=prefill_attention_csa_test,
        specs=build_tensor_specs(
            args.start_pos,
            args.num_tokens,
        ),
        golden_fn=golden_prefill_attention_csa,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        rtol=1e-2,
        atol=1e-2,
        compile_only=args.compile_only,
        compare_fn={
            "x_out": valid_ratio_reldiff(compare_tokens, diff_thd=5e-3, pct_thd=0.005, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
