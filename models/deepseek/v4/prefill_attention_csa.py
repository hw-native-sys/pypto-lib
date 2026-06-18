# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 packed prefill CSA attention.

This module wires HC pre/post, ratio-4 compressor, indexer, sparse attention,
and cache writeback for the compressed sparse attention path.
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
from prefill_compressor_ratio4 import (
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    golden_prefill_compressor_ratio4,
    prefill_compressor_ratio4,
)
from hc_post import golden_hc_post, hc_post
from hc_pre import golden_hc_pre, hc_pre
from prefill_indexer import IDX_CACHE_MAX_BLOCKS, INDEXER_TOPK_CAP, golden_prefill_indexer_core, prefill_indexer
from prefill_indexer_compressor import (
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
)
from qkv_proj_rope import golden_qkv_proj_rope, materialize_rope_rows, qkv_proj_rope
from rmsnorm import golden_attn_norm, attn_norm
from prefill_sparse_attn import (
    HCA_CMP_BLOCK_NUM as SPARSE_HCA_CMP_BLOCK_NUM,
    HCA_ORI_BLOCK_NUM as SPARSE_HCA_ORI_BLOCK_NUM,
    CMP_MAX_BLOCKS as SPARSE_CMP_MAX_BLOCKS,
    ORI_MAX_BLOCKS as SPARSE_ORI_MAX_BLOCKS,
    PREFILL_SPARSE_PAD as SPARSE_PREFILL_SPARSE_PAD,
    _quant_w_per_channel,
    golden_prefill_sparse_attn,
    prefill_sparse_attn_padded_indices,
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


MAX_REQS = 2
MAX_TOKENS = T
MAX_CMP_WRITES = MAX_REQS * max(1, MAX_TOKENS // COMPRESS_RATIO)
CSA_ORI_BLOCK_NUM = SPARSE_HCA_ORI_BLOCK_NUM
CSA_CMP_BLOCK_NUM = SPARSE_HCA_CMP_BLOCK_NUM
CSA_CASES = (
    "custom",
    "basic1",
    "basic17",
    "basic128",
    "suffix3_1",
    "suffix2_2",
    "suffix5_7",
    "suffix64_16",
    "suffix96_32",
    "suffix100_50",
    "suffix128_17",
    "hetero_smoke",
    "hetero_boundary",
    "hetero_long_suffix_overlay_cmp",
    "hetero_full_capacity_overlay_cmp",
    "hetero_single_long_mix_overlay_cmp",
    "cmp_sparse_lens_boundary",
    "cmp_sparse_lens_zero_garbage",
    "issue511_mapping_boundary",
    "hetero_second_only",
    "issue511_active_no_write_slot",
    "issue511_ori_block_table_permutation",
    "issue511_cmp_block_table_permutation",
    "issue511_idx_slot_distinct",
    "issue511_idx_block_table_permutation",
)
CSA_WRITEBACK_DEP_COLS = 16
assert S == WIN, "packed CSA prefill currently assumes one static window page"
assert CSA_WRITEBACK_DEP_COLS < HEAD_DIM


def _resolve_csa_case(
    start_pos: int = START_POS,
    num_tokens: int = MAX_TOKENS,
    csa_case: str = "custom",
    hetero_smoke: bool = False,
    hetero_boundary: bool = False,
):
    alias_count = int(hetero_smoke) + int(hetero_boundary)
    if alias_count > 1:
        raise ValueError("--hetero-smoke and --hetero-boundary are mutually exclusive")
    if csa_case != "custom" and alias_count:
        raise ValueError("--csa-case cannot be combined with --hetero-* aliases")
    if hetero_smoke:
        csa_case = "hetero_smoke"
    elif hetero_boundary:
        csa_case = "hetero_boundary"

    if csa_case == "custom":
        q_lens_values = [num_tokens, 0]
        context_lens_values = [start_pos, 0]
    elif csa_case == "basic1":
        q_lens_values = [1, 0]
        context_lens_values = [0, 0]
    elif csa_case == "basic17":
        q_lens_values = [17, 0]
        context_lens_values = [0, 0]
    elif csa_case == "basic128":
        q_lens_values = [128, 0]
        context_lens_values = [0, 0]
    elif csa_case == "suffix3_1":
        q_lens_values = [1, 0]
        context_lens_values = [3, 0]
    elif csa_case == "suffix2_2":
        q_lens_values = [2, 0]
        context_lens_values = [2, 0]
    elif csa_case == "suffix5_7":
        q_lens_values = [7, 0]
        context_lens_values = [5, 0]
    elif csa_case == "suffix64_16":
        q_lens_values = [16, 0]
        context_lens_values = [64, 0]
    elif csa_case == "suffix96_32":
        q_lens_values = [32, 0]
        context_lens_values = [96, 0]
    elif csa_case == "suffix100_50":
        q_lens_values = [50, 0]
        context_lens_values = [100, 0]
    elif csa_case == "suffix128_17":
        q_lens_values = [17, 0]
        context_lens_values = [128, 0]
    elif csa_case == "hetero_smoke":
        q_lens_values = [32, 32]
        context_lens_values = [64, 120]
    elif csa_case == "hetero_boundary":
        q_lens_values = [50, 40]
        context_lens_values = [96, 230]
    elif csa_case == "hetero_long_suffix_overlay_cmp":
        q_lens_values = [30, 20]
        context_lens_values = [200, 500]
    elif csa_case == "hetero_full_capacity_overlay_cmp":
        q_lens_values = [96, 32]
        context_lens_values = [256, 384]
    elif csa_case == "hetero_single_long_mix_overlay_cmp":
        q_lens_values = [1, 127]
        context_lens_values = [255, 129]
    elif csa_case == "cmp_sparse_lens_boundary":
        q_lens_values = [7, 0]
        context_lens_values = [5, 0]
    elif csa_case == "cmp_sparse_lens_zero_garbage":
        q_lens_values = [7, 0]
        context_lens_values = [5, 0]
    elif csa_case == "issue511_mapping_boundary":
        q_lens_values = [5, 5]
        context_lens_values = [2, 6]
    elif csa_case == "hetero_second_only":
        q_lens_values = [0, 17]
        context_lens_values = [0, 100]
    elif csa_case == "issue511_active_no_write_slot":
        q_lens_values = [32, 32]
        context_lens_values = [96, 230]
    elif csa_case == "issue511_ori_block_table_permutation":
        q_lens_values = [32, 32]
        context_lens_values = [64, 120]
    elif csa_case == "issue511_cmp_block_table_permutation":
        q_lens_values = [50, 40]
        context_lens_values = [96, 230]
    elif csa_case == "issue511_idx_slot_distinct":
        q_lens_values = [50, 40]
        context_lens_values = [96, 230]
    elif csa_case == "issue511_idx_block_table_permutation":
        q_lens_values = [50, 40]
        context_lens_values = [96, 230]
    else:
        raise ValueError(f"unknown --csa-case {csa_case!r}; expected one of {CSA_CASES}")

    active_tokens = sum(q_lens_values)
    if active_tokens <= 0 or active_tokens > MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [1, {MAX_TOKENS}], got {active_tokens}")
    max_position = max(
        (ctx + q_len - 1 for ctx, q_len in zip(context_lens_values, q_lens_values) if q_len > 0),
        default=0,
    )
    if max_position >= MAX_SEQ_LEN:
        raise ValueError(f"position id {max_position} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}")
    max_visible_cmp = max((ctx + q_len) // COMPRESS_RATIO for ctx, q_len in zip(context_lens_values, q_lens_values))
    max_sparse_rows = WIN + max_visible_cmp
    if max_sparse_rows > SPARSE_PREFILL_SPARSE_PAD:
        raise ValueError(
            f"{csa_case} needs {max_sparse_rows} sparse rows; current packed sparse CSA cap is "
            f"{SPARSE_PREFILL_SPARSE_PAD}"
        )
    if max_visible_cmp > SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE:
        raise ValueError(
            f"{csa_case} needs {max_visible_cmp} compressed slots; current cmp cache cap is "
            f"{SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE}"
        )
    return csa_case, q_lens_values, context_lens_values, active_tokens


@pl.jit.inline
def _prefill_csa_cache_writeback_overlay(
    kv: pl.Tensor[[MAX_TOKENS, HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    attn_out: pl.Tensor[[MAX_TOKENS, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    kv_cache_flat = pl.reshape(kv_cache, [CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for t0 in pl.parallel(0, MAX_TOKENS, 16):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_cache_writeback_overlay"):
            for dt in pl.range(16):
                t = t0 + dt
                if t < num_tokens:
                    dst_row_raw = pl.read(ori_slot_mapping, [t])
                    if dst_row_raw >= 0:
                        dst_row = pl.cast(dst_row_raw, pl.INDEX)
                        dep_guard = pl.cast(attn_out[t : t + 1, 0:CSA_WRITEBACK_DEP_COLS], target_type=pl.FP32)
                        dep_zero = pl.mul(dep_guard, 0.0)
                        kv_head = pl.cast(kv[t : t + 1, 0:CSA_WRITEBACK_DEP_COLS], target_type=pl.FP32)
                        kv_head_dep = pl.cast(pl.add(kv_head, dep_zero), target_type=pl.BF16)
                        kv_cache_flat[dst_row : dst_row + 1, 0:CSA_WRITEBACK_DEP_COLS] = kv_head_dep
                        kv_cache_flat[dst_row : dst_row + 1, CSA_WRITEBACK_DEP_COLS:HEAD_DIM] = kv[
                            t : t + 1,
                            CSA_WRITEBACK_DEP_COLS:HEAD_DIM,
                        ]
    return pl.reshape(kv_cache_flat, [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])


@pl.jit.inline
def _prefill_csa_assemble_sparse_indices(
    cmp_topk_indices: pl.Tensor[[MAX_TOKENS, IDX_TOPK], pl.INT32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cmp_sparse_indices: pl.Tensor[[MAX_TOKENS, SPARSE_TOPK], pl.INT32],
):
    for topk_block in pl.spmd((MAX_TOKENS + CSA_TOPK_TOKEN_TILE - 1) // CSA_TOPK_TOKEN_TILE,
                              name_hint="prefill_csa_sparse_idx_tile"):
        topk_t0 = topk_block * CSA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(CSA_TOPK_TOKEN_TILE):
            t_idx = topk_t0 + topk_dt
            sparse_row = pl.full([1, SPARSE_TOPK], dtype=pl.INT32, value=-1)
            if t_idx < num_tokens:
                req = pl.read(token_to_request, [t_idx])
                abs_pos = pl.read(position_ids, [t_idx])
                window_valid = pl.min(WIN, abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for sparse_col in pl.range(SPARSE_TOPK):
                    sparse_raw = pl.cast(-1, pl.INT32)
                    sparse_col_i32 = pl.cast(sparse_col, pl.INT32)
                    if sparse_col_i32 < window_valid:
                        key_abs = key_start_abs + sparse_col_i32
                        sparse_raw = pl.cast(key_abs % WIN, pl.INT32)
                        for scan_t in pl.range(MAX_TOKENS):
                            if scan_t < num_tokens:
                                if scan_t <= t_idx:
                                    scan_req = pl.read(token_to_request, [scan_t])
                                    scan_pos = pl.read(position_ids, [scan_t])
                                    if scan_req == req:
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
                                    if topk_raw >= WIN + MAX_TOKENS:
                                        sparse_raw = topk_raw
                    pl.write(sparse_row, [0, sparse_col], sparse_raw)
            cmp_sparse_indices = pl.assemble(cmp_sparse_indices, sparse_row, [t_idx, 0])
    return cmp_sparse_indices


@pl.jit.inline
def prefill_attention_csa(
    x_hc: pl.Tensor[[MAX_TOKENS, HC_MULT, D], pl.BF16],
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
    cmp_kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[MAX_REQS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[MAX_REQS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[MAX_REQS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    idx_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[MAX_TOKENS, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    x_mixed = hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    x_normed = attn_norm(x_mixed, attn_norm_w, x_normed)

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
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
    q = qkv_proj_rope(
        x_normed,
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

    cmp_kv, cmp_kv_state, cmp_score_state = prefill_compressor_ratio4(
        x_normed,
        cmp_kv_state,
        cmp_score_state,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        freqs_cos,
        freqs_sin,
        cmp_kv,
        token_to_request,
        position_ids,
        num_tokens,
        cmp_slot_mapping,
        state_slot_mapping,
    )
    cmp_topk_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    idx_kv_cache, inner_kv_state, inner_score_state, cmp_topk_indices = prefill_indexer(
        x_normed,
        freqs_cos,
        freqs_sin,
        hadamard_idx,
        inner_kv_state,
        inner_score_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        idx_block_table,
        cmp_topk_indices,
        token_to_request,
        position_ids,
        num_tokens,
        idx_slot_mapping,
        inner_state_slot_mapping,
    )

    cmp_sparse_work = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    cmp_sparse_work = _prefill_csa_assemble_sparse_indices(
        cmp_topk_indices,
        token_to_request,
        position_ids,
        num_tokens,
        cmp_sparse_work,
    )
    # CSA builds every sparse row itself and pads unused entries with -1, so it
    # can safely expose the full row width to the shared sparse-attn kernel,
    # while external serving callers still use cmp_sparse_lens in
    # prefill_sparse_attn where lens == 0 means no valid sparse indices.

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = prefill_sparse_attn_padded_indices(
        q,
        kv_cache,
        ori_block_table,
        kv,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_work,
        attn_sink,
        token_to_request,
        num_tokens,
        rope_cos_t,
        rope_sin_t,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    kv_cache = _prefill_csa_cache_writeback_overlay(kv, kv_cache, ori_slot_mapping, attn_out, num_tokens)

    x_out = hc_post(attn_out, x_hc, post, comb, x_out)
    return kv_cache, cmp_kv, cmp_kv_state, cmp_score_state, idx_kv_cache, inner_kv_state, inner_score_state, x_out


@pl.jit
def prefill_attention_csa_test(
    x_hc: pl.Tensor[[MAX_TOKENS, HC_MULT, D], pl.BF16],
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
    cmp_kv_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[MAX_REQS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[MAX_REQS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Out[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    cmp_kv: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Out[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.BF16]],
    idx_block_table: pl.Tensor[[MAX_REQS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    idx_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[MAX_TOKENS, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    (
        kv_cache,
        cmp_kv,
        cmp_kv_state,
        cmp_score_state,
        idx_kv_cache,
        inner_kv_state,
        inner_score_state,
        x_out,
    ) = prefill_attention_csa(
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
        cmp_kv_state,
        cmp_score_state,
        compress_state_block_table,
        hadamard_idx,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        inner_kv_state,
        inner_score_state,
        inner_compress_state_block_table,
        kv_cache,
        ori_block_table,
        ori_slot_mapping,
        cmp_kv,
        cmp_block_table,
        idx_kv_cache,
        idx_block_table,
        token_to_request,
        position_ids,
        cmp_slot_mapping,
        idx_slot_mapping,
        state_slot_mapping,
        inner_state_slot_mapping,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        x_out,
        num_tokens,
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
    x_normed = golden_attn_norm(x_mixed, tensors["attn_norm_w"])
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
        "token_to_request": tensors["token_to_request"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "cmp_slot_mapping": tensors["cmp_slot_mapping"],
        "state_slot_mapping": tensors["state_slot_mapping"],
    })
    cmp_topk_indices = golden_prefill_indexer_core({
        "x": x_normed.view(T, D),
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
        "token_to_request": tensors["token_to_request"],
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "idx_slot_mapping": tensors["idx_slot_mapping"],
        "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
    })

    def assemble_sparse_indices(cmp_topk):
        topk_idxs = torch.full((MAX_TOKENS, SPARSE_TOPK), -1, dtype=torch.int32)
        sparse_lens = torch.zeros(MAX_TOKENS, dtype=torch.int32)
        token_to_req = tensors["token_to_request"]
        pos = tensors["position_ids"]
        current_by_req = [dict() for _ in range(MAX_REQS)]
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            current_by_req[req][int(pos[t].item())] = t
        compressed_cap = SPARSE_PREFILL_SPARSE_PAD - WIN
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                overlay_t = current_by_req[req].get(key_abs)
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
                if raw >= WIN + MAX_TOKENS and raw - (WIN + MAX_TOKENS) < visible_cmp:
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
        "token_to_request": tensors["token_to_request"],
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
    num_tokens: int = MAX_TOKENS,
    csa_case: str = "custom",
    hetero_smoke: bool = False,
    hetero_boundary: bool = False,
):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    _, q_lens_values, context_lens_values, num_tokens = _resolve_csa_case(
        start_pos,
        num_tokens,
        csa_case,
        hetero_smoke,
        hetero_boundary,
    )

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale
    def seeded_normal(shape, seed, std=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randn(*shape, generator=generator) * std

    def token_meta():
        token_to_req = torch.zeros(MAX_TOKENS, dtype=torch.int32)
        local_pos = torch.zeros(MAX_TOKENS, dtype=torch.int32)
        pos = torch.arange(MAX_TOKENS, dtype=torch.int32)
        cursor = 0
        for req, q_len in enumerate(q_lens_values):
            ctx = context_lens_values[req]
            for local_s in range(q_len):
                t = cursor + local_s
                token_to_req[t] = req
                local_pos[t] = local_s
                pos[t] = ctx + local_s
            cursor += q_len
        return token_to_req, local_pos, pos

    def cmp_write_records():
        token_to_req, _, pos = token_meta()
        records = []
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            if (abs_pos + 1) % COMPRESS_RATIO == 0:
                req = int(token_to_req[t].item())
                cmp_slot = (abs_pos + 1) // COMPRESS_RATIO - 1
                records.append((t, req, cmp_slot))
        if len(records) > MAX_CMP_WRITES:
            raise ValueError(f"CSA fixture generated {len(records)} compressed writes, cap is {MAX_CMP_WRITES}")
        return records

    def validate_overlay_topk(topk_idxs, token_to_req, pos):
        current_by_req = [dict() for _ in range(MAX_REQS)]
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            current_by_req[req][int(pos[t].item())] = t
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
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
                    if key_abs in current_by_req[req]:
                        raise ValueError(f"current suffix abs_pos={key_abs} must use CSA overlay for token {t}")
                    if key_abs in seen_window_abs:
                        raise ValueError(f"duplicate CSA window abs_pos={key_abs} for token {t}")
                    seen_window_abs.add(key_abs)
                elif raw < WIN + MAX_TOKENS:
                    overlay_t = raw - WIN
                    if overlay_t >= num_tokens:
                        raise ValueError(f"CSA overlay raw={raw} points past active tokens for token {t}")
                    overlay_req = int(token_to_req[overlay_t].item())
                    if overlay_req != req:
                        raise ValueError(f"CSA overlay raw={raw} crosses request {overlay_req}->{req}")
                    key_abs = int(pos[overlay_t].item())
                    if key_abs > abs_pos:
                        raise ValueError(f"CSA overlay raw={raw} is future key abs_pos={key_abs} for token {t}")
                    if key_abs in seen_window_abs:
                        raise ValueError(f"duplicate CSA overlay abs_pos={key_abs} for token {t}")
                    seen_window_abs.add(key_abs)
                else:
                    cmp_slot = raw - (WIN + MAX_TOKENS)
                    visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
                    if cmp_slot < 0 or cmp_slot >= visible_cmp:
                        raise ValueError(f"CSA compressed slot={cmp_slot} is not visible for token {t}")
                    if cmp_slot in seen_cmp:
                        raise ValueError(f"duplicate CSA compressed slot={cmp_slot} for token {t}")
                    seen_cmp.add(cmp_slot)

    def init_x_hc():
        x = seeded_normal((MAX_TOKENS, HC_MULT, D), 1, 0.05)
        x[num_tokens:] = 0
        return x
    # Real layer-8 (CSA, ratio-4) hc_attn scale/base (fn synthetic at real magnitude). A
    # synthetic scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out
    # and the hc residual to near-zero in x_out where W8A8 noise blows up the relative tail.
    # Mirrors decode_attention_csa.
    def init_hc_attn_fn():
        return seeded_normal((MIX_HC, HC_DIM), 2, 0.0519)
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
        return seeded_uniform((D, Q_LORA), 3, D ** -0.5)
    def init_wq_b():
        return seeded_uniform((Q_LORA, H * HEAD_DIM), 4, Q_LORA ** -0.5)
    def init_wkv():
        return seeded_uniform((D, HEAD_DIM), 5, D ** -0.5)
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
        return seeded_normal((D, MAIN_OUT_DIM), 11, 0.0245)
    def init_cmp_wgate():
        return seeded_normal((D, MAIN_OUT_DIM), 12, 0.0388)
    def init_cmp_ape():
        return seeded_normal((COMPRESS_RATIO, MAIN_OUT_DIM), 13, 0.1243)
    def init_cmp_norm_w():
        return 0.9666 + seeded_normal((HEAD_DIM,), 14, 0.1929)
    def init_compress_state_block_table():
        table = torch.full((MAX_REQS, CSA_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for req in range(MAX_REQS):
            for block in range(CSA_STATE_MAX_BLOCKS):
                table[req, block] = req * CSA_STATE_MAX_BLOCKS + ((block * 17 + 3) % CSA_STATE_MAX_BLOCKS)
        return table
    def state_row(req, abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // CSA_STATE_BLOCK_SIZE
        intra = abs_pos % CSA_STATE_BLOCK_SIZE
        return int(table[req, block].item()) * CSA_STATE_BLOCK_SIZE + intra
    def init_cmp_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
        flat = state.view(-1, MAIN_OUT_DIM)
        for req, ctx in enumerate(context_lens_values):
            for abs_pos in range(max(0, ctx - MAIN_STATE_LEN), ctx):
                row = state_row(req, abs_pos)
                if row >= 0:
                    flat[row] = seeded_uniform(
                    (MAIN_OUT_DIM,),
                    3000 + req * 65536 + abs_pos,
                    0.05,
                )
        return state
    def init_cmp_score_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
        flat = state.view(-1, MAIN_OUT_DIM)
        for req, ctx in enumerate(context_lens_values):
            for abs_pos in range(max(0, ctx - MAIN_STATE_LEN), ctx):
                row = state_row(req, abs_pos)
                if row >= 0:
                    flat[row] = seeded_uniform(
                    (MAIN_OUT_DIM,),
                    4000 + req * 65536 + abs_pos,
                    0.05,
                )
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
        return seeded_normal((D, INNER_OUT_DIM), 16, 0.0293)
    def init_inner_wgate():
        return seeded_normal((D, INNER_OUT_DIM), 17, 0.0512)
    def init_inner_ape():
        return seeded_normal((COMPRESS_RATIO, INNER_OUT_DIM), 18, 0.1528)
    def init_inner_norm_w():
        return 0.6850 + seeded_normal((IDX_HEAD_DIM,), 19, 0.2610)
    def init_inner_compress_state_block_table():
        table = torch.full((MAX_REQS, INNER_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for req in range(MAX_REQS):
            for block in range(INNER_STATE_MAX_BLOCKS):
                table[req, block] = req * INNER_STATE_MAX_BLOCKS + ((block * 17 + 3) % INNER_STATE_MAX_BLOCKS)
        return table
    def inner_state_row(req, abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_inner_compress_state_block_table()
        block = abs_pos // INNER_STATE_BLOCK_SIZE
        intra = abs_pos % INNER_STATE_BLOCK_SIZE
        return int(table[req, block].item()) * INNER_STATE_BLOCK_SIZE + intra
    def init_inner_kv_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM)
        flat = state.view(-1, INNER_OUT_DIM)
        for req, ctx in enumerate(context_lens_values):
            for abs_pos in range(max(0, ctx - INNER_STATE_LEN), ctx):
                row = inner_state_row(req, abs_pos)
                if row >= 0:
                    flat[row] = seeded_uniform(
                    (INNER_OUT_DIM,),
                    5000 + req * 65536 + abs_pos,
                    0.05,
                )
        return state
    def init_inner_score_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_OUT_DIM)
        flat = state.view(-1, INNER_OUT_DIM)
        for req, ctx in enumerate(context_lens_values):
            for abs_pos in range(max(0, ctx - INNER_STATE_LEN), ctx):
                row = inner_state_row(req, abs_pos)
                if row >= 0:
                    flat[row] = seeded_uniform(
                    (INNER_OUT_DIM,),
                    6000 + req * 65536 + abs_pos,
                    0.05,
                )
        return state
    def init_idx_kv_cache():
        cache = torch.zeros(CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM)
        cache_flat = cache.view(CSA_CMP_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
        table = init_idx_block_table()
        for req, ctx in enumerate(context_lens_values):
            completed = ctx // COMPRESS_RATIO
            for cmp_slot in range(completed):
                if cmp_slot >= SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE:
                    break
                row = cache_row_from_table(table, req, cmp_slot)
                if row >= 0:
                    cache_flat[row] = seeded_uniform((IDX_HEAD_DIM,), 7000 + req * 65536 + cmp_slot, 0.05).to(torch.bfloat16)
        return cache
    def init_kv_cache():
        cache = torch.zeros(CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_ori_block_table()
        for req, ctx in enumerate(context_lens_values):
            start = max(0, ctx - WIN)
            for abs_pos in range(start, ctx):
                row = cache_row_from_table(table, req, abs_pos % WIN)
                value = seeded_uniform((HEAD_DIM,), 1000 + req * 4096 + abs_pos, 0.1)
                if row >= 0:
                    cache_flat[row] = value.to(torch.bfloat16)
        return cache
    def init_ori_block_table():
        table = torch.full((MAX_REQS, SPARSE_ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for req in range(MAX_REQS):
            for block in range(SPARSE_ORI_MAX_BLOCKS):
                phys_req = req
                if csa_case == "issue511_ori_block_table_permutation":
                    phys_req = (req + 1) % MAX_REQS
                table[req, block] = phys_req * SPARSE_ORI_MAX_BLOCKS + block
        return table
    def init_ori_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        token_to_req, _, pos = token_meta()
        table = init_ori_block_table()
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            mapping[t] = cache_row_from_table(table, req, int(pos[t].item()) % WIN)
        if csa_case == "issue511_active_no_write_slot":
            mapping[0] = -1
            mapping[num_tokens - 1] = -1
        return mapping
    def init_cmp_kv():
        cache = torch.zeros(CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(CSA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_cmp_block_table()
        for req, ctx in enumerate(context_lens_values):
            completed = ctx // COMPRESS_RATIO
            for cmp_slot in range(completed):
                if cmp_slot >= SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE:
                    break
                row = cache_row_from_table(table, req, cmp_slot)
                value = seeded_uniform((HEAD_DIM,), 2000 + req * 4096 + cmp_slot, 0.1)
                if row >= 0:
                    cache_flat[row] = value.to(torch.bfloat16)
        return cache
    def init_cmp_block_table():
        table = torch.full((MAX_REQS, SPARSE_CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for req in range(MAX_REQS):
            for block in range(SPARSE_CMP_MAX_BLOCKS):
                phys_req = req
                if csa_case == "issue511_cmp_block_table_permutation":
                    phys_req = (req + 1) % MAX_REQS
                table[req, block] = phys_req * SPARSE_CMP_MAX_BLOCKS + block
        return table
    def init_idx_block_table():
        table = torch.full((MAX_REQS, IDX_CACHE_MAX_BLOCKS), -1, dtype=torch.int32)
        for req in range(MAX_REQS):
            for block in range(IDX_CACHE_MAX_BLOCKS):
                phys = block
                if csa_case in ("issue511_idx_slot_distinct", "issue511_idx_block_table_permutation"):
                    if IDX_CACHE_MAX_BLOCKS > 1:
                        phys = (block * 5 + 1) % IDX_CACHE_MAX_BLOCKS
                table[req, block] = req * IDX_CACHE_MAX_BLOCKS + phys
        return table
    def cache_row_from_table(table, req, slot):
        block = slot // BLOCK_SIZE
        intra = slot % BLOCK_SIZE
        phys_block = int(table[req, block].item())
        if phys_block < 0:
            return -1
        return phys_block * BLOCK_SIZE + intra
    def init_cmp_sparse_indices():
        topk_idxs = torch.full((MAX_TOKENS, SPARSE_TOPK), -1, dtype=torch.int32)
        token_to_req, _, pos = token_meta()
        current_by_req = [dict() for _ in range(MAX_REQS)]
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            current_by_req[req][int(pos[t].item())] = t
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            cursor = 0
            for key_abs in range(key_start_abs, abs_pos + 1):
                overlay_t = current_by_req[req].get(key_abs)
                if overlay_t is not None and overlay_t <= t:
                    topk_idxs[t, cursor] = WIN + overlay_t
                else:
                    topk_idxs[t, cursor] = key_abs % WIN
                cursor += 1
            visible_cmp = min((abs_pos + 1) // COMPRESS_RATIO, SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE)
            for cmp_slot in range(visible_cmp):
                if cursor >= SPARSE_PREFILL_SPARSE_PAD:
                    break
                topk_idxs[t, cursor] = WIN + MAX_TOKENS + cmp_slot
                cursor += 1
        validate_overlay_topk(topk_idxs, token_to_req, pos)
        return topk_idxs
    def init_token_to_request():
        return token_meta()[0]
    def init_position_ids():
        return token_meta()[2]
    def init_cmp_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        table = init_cmp_block_table()
        records = cmp_write_records()
        for token_id, req, cmp_slot in records:
            mapping[token_id] = cache_row_from_table(table, req, cmp_slot)
        if csa_case == "issue511_active_no_write_slot" and records:
            token_id, _, _ = records[0]
            mapping[token_id] = -1
        return mapping
    def init_idx_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        table = init_idx_block_table()
        records = cmp_write_records()
        for token_id, req, cmp_slot in records:
            mapping[token_id] = cache_row_from_table(table, req, cmp_slot)
        if csa_case == "issue511_active_no_write_slot" and records:
            token_id, _, _ = records[0]
            mapping[token_id] = -1
        return mapping
    def init_state_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        token_to_req, _, pos = token_meta()
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            mapping[t] = state_row(req, int(pos[t].item()))
        if csa_case == "issue511_active_no_write_slot":
            mapping[0] = -1
            mapping[num_tokens - 1] = -1
        return mapping
    def init_inner_state_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        token_to_req, _, pos = token_meta()
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            mapping[t] = inner_state_row(req, int(pos[t].item()))
        if csa_case == "issue511_active_no_write_slot":
            mapping[0] = -1
            mapping[num_tokens - 1] = -1
        return mapping
    def init_attn_sink():
        return torch.zeros(H)
    def init_wo_a():
        return seeded_uniform((O_GROUPS, O_LORA, O_GROUP_IN), 9, O_GROUP_IN ** -0.5)
    def init_wo_b():
        return seeded_uniform((D, O_GROUPS * O_LORA), 10, (O_GROUPS * O_LORA) ** -0.5)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel_local(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)

    return [
        TensorSpec("x_hc", [MAX_TOKENS, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
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
        TensorSpec("cmp_wkv", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.float32, init_value=init_cmp_norm_w),
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
        TensorSpec("compress_state_block_table", [MAX_REQS, CSA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("hadamard_idx", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard_idx),
        TensorSpec("inner_wkv", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [D, INNER_OUT_DIM], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.float32, init_value=init_inner_norm_w),
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
        TensorSpec("inner_compress_state_block_table", [MAX_REQS, INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("kv_cache", [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_kv_cache, is_output=True),
        TensorSpec("ori_block_table", [MAX_REQS, SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("ori_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec(
            "cmp_kv",
            [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
            is_output=True,
        ),
        TensorSpec("cmp_block_table", [MAX_REQS, SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec(
            "idx_kv_cache",
            [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM],
            torch.bfloat16,
            init_value=init_idx_kv_cache,
            is_output=True,
        ),
        TensorSpec("idx_block_table", [MAX_REQS, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("token_to_request", [MAX_TOKENS], torch.int32, init_value=init_token_to_request),
        TensorSpec("position_ids", [MAX_TOKENS], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("idx_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("state_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_inner_state_slot_mapping),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [MAX_TOKENS, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


def active_x_out_compare(num_tokens: int):
    from golden import ratio_allclose

    base_cmp = ratio_allclose(atol=4e-3, rtol=2.0 / 128, max_error_ratio=0.015)

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

    cmp.__name__ = f"active_x_out_compare(num_tokens={num_tokens})"
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

    parser = argparse.ArgumentParser(
        description=(
            "Standalone DeepSeek V4 packed prefill CSA sparse-attention consumer test. "
            "CLI cases generate lowered token metadata and deterministic ratio4 compressed KV/topk fixtures."
        )
    )
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only; enabled only when this flag is explicitly passed.",
    )
    parser.add_argument(
        "--csa-case",
        type=str,
        default="custom",
        choices=CSA_CASES,
        help="Standalone fixture scenario; non-custom values override --start-pos/--num-tokens.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help="Fixture-only context length for request 0 when --csa-case=custom; not a JIT argument.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=MAX_TOKENS,
        help="Fixture active token count for --csa-case=custom; passed to the JIT as num_tokens.",
    )
    parser.add_argument("--hetero-smoke", action="store_true", default=False)
    parser.add_argument("--hetero-boundary", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument(
        "--enable-dep-gen",
        action="store_true",
        default=False,
        help="Capture PTO2 dependency edges (deps.json). Required to recover function names in the "
        "L2 swimlane for dynamic-shape kernels whose AICore records carry func_id=-1.",
    )
    args = parser.parse_args()
    try:
        _, _, _, compare_tokens = _resolve_csa_case(
            args.start_pos,
            args.num_tokens,
            args.csa_case,
            args.hetero_smoke,
            args.hetero_boundary,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    result = run_jit(
        fn=prefill_attention_csa_test,
        specs=build_tensor_specs(
            args.start_pos,
            args.num_tokens,
            args.csa_case,
            args.hetero_smoke,
            args.hetero_boundary,
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
            "x_out": active_x_out_compare(compare_tokens),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1e-2),
            "cmp_kv": ratio_allclose(atol=5e-3, rtol=1e-2),
            "idx_kv_cache": ratio_allclose(atol=5e-3, rtol=1e-2),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
