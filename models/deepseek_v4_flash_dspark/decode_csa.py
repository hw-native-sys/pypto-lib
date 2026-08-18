# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 CSA decode orchestration with configurable TP output."""


import sys

import config


# Sub-kernels freeze TP-derived shapes at import time, so select the standalone
# program's TP world before importing them below.
_TP_CHOICES = (1, 2, 4)
_TP_DEFAULT = 2


def _parse_tp_argv():
    for index, arg in enumerate(sys.argv):
        if arg == "--tp" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--tp="):
            return int(arg.split("=", 1)[1])
    return _TP_DEFAULT


TP_SIZE = _parse_tp_argv()
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")
config.TP = TP_SIZE

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    FLASH as M,
    DECODE_BATCH,
    DECODE_SEQ,
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    KV_CMP_BLOCK_NUM,
    IDX_CACHE_BLOCK_NUM,
    KV_ORI_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from decode_compressor_ratio4 import compressor_ratio4
from hc_post import hc_post
from hc_pre import hc_pre
from decode_indexer import indexer
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from rope_interleave import rope_interleave
from decode_o_proj import (
    ATTENTION_WINDOW_ROWS,
    GROUP_T_PAD,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    LOCAL_T,
    LOCAL_T_PAD,
    O_WINDOW_ROWS,
    attention_token_head_all_to_all_step,
    decode_sharded_o_projection_reduce_scatter,
)
from decode_sparse_attn_csa import (
    ATTN_K_TILE,
    CMP_TOPK,
    NOPE_DIM,
    ROPE_DIM,
    ROPE_CS_T_TILE,
    SOFTMAX_SCALE,
    T_PAD,
    sparse_attn_csa,
    sparse_attn_csa_heads,
)

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
B = DECODE_BATCH // TP_SIZE
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
HALF_ROPE = ROPE_HEAD_DIM // 2
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
INDEXER_SCORE_LEN = MAX_SEQ_LEN // 4
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local
COMPRESS_RATIO = 4
OVERLAP = COMPRESS_RATIO == 4
COFF = 1 + int(OVERLAP)
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_STATE_DIM = 2 * MAIN_OUT_DIM
MAIN_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
MAIN_STATE_PHYSICAL_BLOCKS = CSA_STATE_PHYSICAL_BLOCKS
MAIN_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
MAIN_STATE_BLOCK_NUM = MAIN_STATE_PHYSICAL_BLOCKS
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic("CSA_STATE_BLOCK_NUM_DYN")
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_PHYSICAL_BLOCKS = CSA_INNER_STATE_PHYSICAL_BLOCKS
INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
INNER_STATE_BLOCK_NUM = INNER_STATE_PHYSICAL_BLOCKS
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = KV_CMP_BLOCK_NUM
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")

# tiling
CSA_WB_TOKEN_TILE = 8

# fixture
FIXTURE_CMP_ENTRIES = (
    (ATTN_K_TILE - 1, BLOCK_SIZE // 4 - 1),
    (2 * ATTN_K_TILE - 1, BLOCK_SIZE),
    (3 * ATTN_K_TILE - 1, 2 * BLOCK_SIZE),
    (4 * ATTN_K_TILE - 1, 4 * BLOCK_SIZE - 3),
)
FIXTURE_FILTERED_POSITIONS = (1, ATTN_K_TILE + 1, 2 * ATTN_K_TILE + 1, 3 * ATTN_K_TILE + 1)
FIXTURE_CMP_SLOTS = tuple(slot for _, slot in FIXTURE_CMP_ENTRIES)
FIXTURE_CMP_LOGICAL_BLOCKS = (max(FIXTURE_CMP_SLOTS) + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE
FIXTURE_CMP_BOUND = 4 * BLOCK_SIZE - 2
FIXTURE_FILTERED_SLOT = FIXTURE_CMP_BOUND
FIXTURE_POSITION_ID = COMPRESS_RATIO * FIXTURE_CMP_BOUND - 1
FIXTURE_WINDOW_HEAD = (FIXTURE_POSITION_ID - WIN + 1) % BLOCK_SIZE
FIXTURE_WINDOW_BLOCKS = (FIXTURE_WINDOW_HEAD + WIN + BLOCK_SIZE - 1) // BLOCK_SIZE
FIXTURE_CMP_BLOCKS = (LOCAL_T // S) * FIXTURE_CMP_LOGICAL_BLOCKS
FIXTURE_OUTPUT_SENTINEL = -7.0

if T != LOCAL_T:
    raise ValueError(f"CSA token capacity {T} must equal TP local token capacity {LOCAL_T}")
if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"CSA token capacity {T_PAD} must equal TP local token capacity {LOCAL_T_PAD}")
if LOCAL_T < 2 * ROPE_CS_T_TILE:
    raise ValueError("CSA fixture token capacity must leave one RoPE row tile for subcapacity")
if LOCAL_T % ROPE_CS_T_TILE != 0:
    raise ValueError("CSA fixture token capacity must align to the RoPE row tile")
if (LOCAL_T - ROPE_CS_T_TILE) % S != 0:
    raise ValueError("CSA fixture subcapacity must preserve whole decode requests")
if max(position for position, _ in FIXTURE_CMP_ENTRIES) >= INDEXER_SCORE_LEN:
    raise ValueError("CSA fixture compressed positions exceed the indexer score row")
if max(position for position, _ in FIXTURE_CMP_ENTRIES) >= 4 * ATTN_K_TILE:
    raise ValueError("CSA fixture compressed positions exceed the four sparse tiles")
if max(FIXTURE_FILTERED_POSITIONS) >= CMP_TOPK:
    raise ValueError("CSA fixture filtered positions exceed the compressed top-k row")
if FIXTURE_FILTERED_SLOT >= FIXTURE_CMP_LOGICAL_BLOCKS * BLOCK_SIZE:
    raise ValueError("CSA filtered slot must remain inside the fixture block table")
if FIXTURE_WINDOW_BLOCKS > ORI_BLOCK_NUM:
    raise ValueError("CSA fixture window exceeds the original KV cache capacity")
if FIXTURE_CMP_LOGICAL_BLOCKS > CMP_MAX_BLOCKS:
    raise ValueError("CSA fixture compressed slots exceed the compressed block table")
if FIXTURE_CMP_BLOCKS > CMP_BLOCK_NUM:
    raise ValueError("CSA fixture compressed-cache pool exceeds its physical capacity")
if O_LORA < 4 * HEADS_PER_GROUP:
    raise ValueError("CSA fixture output projection needs four observable rows per local head")


@pl.jit.inline
def attention_csa(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B_DYN, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[B_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    t_dim = pl.tensor.dim(x_hc, 0)
    b_dim = t_dim // S
    wb_blocks = t_dim // CSA_WB_TOKEN_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    rope_cos_t = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    step_cos = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    step_sin = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    # Interleave-duplicated / sign-folded step rope rows for the indexer subsystem.
    # The indexer's qr_rope re-ran the j>>1 dup-gather on each of its 16 spmd blocks
    # (32 rows each) and its compressor once more; pl.gather lowers to a per-row
    # TGATHER loop, so that was ~1056 row-gathers per layer to rebuild one small
    # position-invariant table. Built once here instead (B rows, off the critical
    # path -- this scope has no producer) and read as a plain load downstream.
    step_cos_il = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    step_sin_signed = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_step"):
        for b in pl.range(b_dim):
            first_t = b * S
            first_pos_b = pl.read(position_ids, [first_t])
            step_pos_b = pl.cast(first_pos_b, pl.INDEX)
            for s in pl.range(S):
                t = b * S + s
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                cos_row = pl.cast(freqs_cos[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                sin_row = pl.cast(freqs_sin[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                rope_cos_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(cos_row, target_type=pl.BF16)
                rope_sin_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(sin_row, target_type=pl.BF16)
            step_cos[b : b + 1, 0 : HALF_ROPE] = pl.cast(freqs_cos[step_pos_b : step_pos_b + 1, 0 : HALF_ROPE], target_type=pl.FP32)
            step_sin[b : b + 1, 0 : HALF_ROPE] = pl.cast(freqs_sin[step_pos_b : step_pos_b + 1, 0 : HALF_ROPE], target_type=pl.FP32)

    rope_interleave(step_cos, step_sin, step_cos_il, step_sin_signed)

    cmp_cos = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([B, HALF_ROPE], dtype=pl.FP32)
    # Same hoist as step_cos_il above, for the main compressor's rmsnorm_rope.
    cmp_cos_il = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_cmp_rope"):
        for b in pl.range(b_dim):
            first_t = b * S
            first_pos_b = pl.read(position_ids, [first_t])
            cmp_offset_b = COMPRESS_RATIO - (first_pos_b % COMPRESS_RATIO)
            cmp_pos_b = pl.cast(first_pos_b + cmp_offset_b - COMPRESS_RATIO, pl.INDEX)
            cmp_cos[b : b + 1, 0 : HALF_ROPE] = pl.cast(freqs_cos[cmp_pos_b : cmp_pos_b + 1, 0 : HALF_ROPE], target_type=pl.FP32)
            cmp_sin[b : b + 1, 0 : HALF_ROPE] = pl.cast(freqs_sin[cmp_pos_b : cmp_pos_b + 1, 0 : HALF_ROPE], target_type=pl.FP32)

    rope_interleave(cmp_cos, cmp_sin, cmp_cos_il, cmp_sin_signed)

    x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_t)
    # rms_norm fans out to qr_proj_matmul (critical path), kv_proj_matmul, kv_score_proj
    # and weights_proj. The latter three take this barrier instead of racing the first:
    # the dummy resolves one hop after rms_norm, so qr_proj_matmul is dispatched first.
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed_t, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, late_dep,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    for wb_blk in pl.spmd(wb_blocks, name_hint="csa_cache_writeback"):
        wb_t0 = wb_blk * CSA_WB_TOKEN_TILE
        for write_dt in pl.range(CSA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            write_row_i64 = pl.read(ori_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]

    cmp_out = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    compressor_ratio4(
        x_normed_t, cmp_out,
        compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_cos_il, cmp_sin_signed, cmp_kv,
        position_ids, cmp_slot_mapping, state_slot_mapping,
        late_dep,
    )

    idx_kv_unused = pl.create_tensor([t_dim, IDX_HEAD_DIM], dtype=pl.FP32)
    idx_score_unused = pl.create_tensor([t_dim, INDEXER_SCORE_LEN], dtype=pl.FP32)
    idx_topk_full = pl.create_tensor([t_dim, INDEXER_SCORE_LEN], dtype=pl.INT32)
    indexer(
        x_normed_t, qr, qr_scale, idx_wq_b, idx_wq_b_scale,
        weights_proj, step_cos_il, step_sin_signed, hadamard_idx,
        idx_kv_unused, inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        idx_score_unused, idx_topk_full,
        position_ids, idx_slot_mapping, inner_state_slot_mapping,
        kv_seq_lens, 0, late_dep,
    )

    # sparse_attn_csa now folds the compressed-slot masking + valid-block flags in from
    # the raw indexer topk + position, so pass those directly.

    position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    sparse_attn_csa(
        q, kv_cache, window_swa_indices,
        cmp_kv, cmp_block_table, idx_topk_full, position_ids_t1,
        attn_sink, rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def attention_csa_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B_DYN, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[B_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x_hc.bind_dynamic(0, T_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    idx_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    inner_compress_state_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    idx_block_table.bind_dynamic(0, B_DYN)
    x_out.bind_dynamic(0, T_DYN)

    attention_csa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        inner_compress_state, inner_compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, idx_slot_mapping,
        state_slot_mapping, inner_state_slot_mapping,
        position_ids, kv_seq_lens,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_out,
    )
    return x_out


@pl.jit
def decode_csa_output(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[T_DYN, INDEXER_SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    o_local: pl.InOut[pl.Tensor[[LOCAL_T_PAD, D], pl.BF16]],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Run rank-local CSA heads, output A2A, sharded O projection, and RS."""
    q.bind_dynamic(0, T_DYN)
    ori_kv.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    idx_topk.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)

    attention_grouped = pl.create_tensor([O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], dtype=pl.BF16)
    attention_grouped, _ = sparse_attn_csa_heads(
        q, ori_kv, window_swa_indices,
        cmp_kv, cmp_block_table, idx_topk,
        position_ids, attn_sink, freqs_cos, freqs_sin,
        attention_grouped,
    )

    attention_local_flat = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_local_flat, attention_signal = attention_token_head_all_to_all_step(
        attention_grouped, attention_local_flat,
        attention_window, attention_signal,
        group_base, tp_rank, local_t,
    )

    attention_local_groups = pl.reshape(attention_local_flat, [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN])
    o_local, o_signal = decode_sharded_o_projection_reduce_scatter(
        attention_local_groups,
        wo_a, wo_b, wo_b_scale,
        local_t, o_local,
        o_window, o_signal,
        group_base, tp_rank,
    )
    return o_local, attention_signal, o_signal


@pl.jit.host
def l3_decode_csa_output(
    q: pl.Tensor[[TP_SIZE, T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[TP_SIZE, T_DYN, WIN], pl.INT32],
    cmp_kv: pl.Tensor[[TP_SIZE, CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[TP_SIZE, B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_topk: pl.Tensor[[TP_SIZE, T_DYN, INDEXER_SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[TP_SIZE, T_DYN, 1], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    freqs_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    o_local: pl.InOut[pl.Tensor[[TP_SIZE, LOCAL_T_PAD, D], pl.BF16]],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch the CSA output path on one TP group."""
    q.bind_dynamic(1, T_DYN)
    ori_kv.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    window_swa_indices.bind_dynamic(1, T_DYN)
    cmp_kv.bind_dynamic(1, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(1, B_DYN)
    idx_topk.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        decode_csa_output(
            q[rank],
            ori_kv[rank], window_swa_indices[rank],
            cmp_kv[rank], cmp_block_table[rank], idx_topk[rank],
            position_ids[rank], attn_sink[rank],
            freqs_cos[rank], freqs_sin[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            o_local[rank],
            attention_window, attention_signal,
            o_window, o_signal,
            0, rank, local_t,
            device=rank,
        )



def golden_attention_csa(tensors):
    """Torch reference for the ratio-4 compression-step CSA orchestration."""
    import torch

    from decode_compressor_ratio4 import golden_compressor
    from hc_pre import golden_hc_pre
    from decode_indexer import golden_indexer
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm
    from decode_sparse_attn_csa import golden_sparse_attn

    tokens = tensors["x_hc"].shape[0]
    batch = tokens // S
    from hc_post import golden_hc_post

    x_mixed = torch.zeros(tokens, D, dtype=torch.bfloat16)
    post_t = torch.zeros(tokens, HC_MULT, dtype=torch.float32)
    comb_t = torch.zeros(tokens, HC_MULT * HC_MULT, dtype=torch.float32)
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
    position_ids_bsd = position_ids.reshape(batch, S).to(torch.int32).contiguous()
    cmp_slot_mapping_bsd = tensors["cmp_slot_mapping"].reshape(batch, S).to(torch.int64).contiguous()
    idx_slot_mapping_bsd = tensors["idx_slot_mapping"].reshape(batch, S).to(torch.int64).contiguous()
    state_slot_mapping_bsd = tensors["state_slot_mapping"].reshape(batch, S).to(torch.int64).contiguous()
    inner_state_slot_mapping_bsd = tensors["inner_state_slot_mapping"].reshape(batch, S).to(torch.int64).contiguous()

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_t = freqs_cos[position_ids].contiguous()
    rope_sin_t = freqs_sin[position_ids].contiguous()
    first_pos = position_ids.reshape(batch, S)[:, 0]
    step_cos = freqs_cos[first_pos, :HALF_ROPE].float().contiguous()
    step_sin = freqs_sin[first_pos, :HALF_ROPE].float().contiguous()
    cmp_pos = first_pos + (COMPRESS_RATIO - (first_pos % COMPRESS_RATIO)) - COMPRESS_RATIO
    cmp_cos = freqs_cos[cmp_pos, :HALF_ROPE].float().contiguous()
    cmp_sin = freqs_sin[cmp_pos, :HALF_ROPE].float().contiguous()

    q = torch.zeros(tokens, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
    qr_i8 = torch.zeros(tokens, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(tokens, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
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
    window_swa_indices = tensors["window_swa_indices"]
    cmp_kv = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]

    cmp_out = torch.zeros(tokens, HEAD_DIM, dtype=torch.float32)
    golden_compressor({
        "x": x_normed,
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
        "position_ids": position_ids_bsd.reshape(-1),
        "cmp_slot_mapping": cmp_slot_mapping_bsd.reshape(-1),
        "state_slot_mapping": state_slot_mapping_bsd.reshape(-1),
    })

    idx_kv = torch.zeros(tokens, IDX_HEAD_DIM, dtype=torch.float32)
    idx_score = torch.zeros(tokens, INDEXER_SCORE_LEN, dtype=torch.float32)
    idx_topk_full = torch.full((tokens, INDEXER_SCORE_LEN), -1, dtype=torch.int32)
    golden_indexer({
        "x": x_normed,
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
        "idx_kv_scale": tensors["idx_kv_scale"],
        "idx_block_table": tensors["idx_block_table"],
        "score": idx_score,
        "topk_idxs": idx_topk_full,
        "position_ids": position_ids_bsd.reshape(-1),
        "idx_slot_mapping": idx_slot_mapping_bsd.reshape(-1),
        "inner_state_slot_mapping": inner_state_slot_mapping_bsd.reshape(-1),
        "kv_seq_lens": tensors["kv_seq_lens"],
        "offset": torch.tensor(0, dtype=torch.int32),
    })

    ori_slot_mapping = tensors["ori_slot_mapping"].to(torch.int64)
    for t in range(tokens):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            blk_id = write_row // BLOCK_SIZE
            intra = write_row % BLOCK_SIZE
            kv_cache[blk_id, intra, 0] = kv[t]

    idx_topk_flat = idx_topk_full.view(tokens, INDEXER_SCORE_LEN)

    attn_out = torch.zeros(tokens, D, dtype=torch.bfloat16)
    # sparse_attn_csa folds the compressed-slot masking in (0 <= raw < floor((pos+1)/
    # COMPRESS_RATIO)); pass raw idx_topk + position so the golden masks the same way.
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "window_swa_indices": window_swa_indices,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "idx_topk": idx_topk_flat,
        "position_ids": position_ids.view(tokens, 1),
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    y = torch.zeros(tokens, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tensor_specs(start_pos=None, batch=B):
    tokens = batch * S
    import torch
    from utils import (
        block_table,
        compressed_slot_mapping,
        csa_decode_start_set,
        kv_seq_lens_from_starts,
        ori_slot_mapping,
        position_ids_from_starts,
        resolve_start_positions,
        state_slot_mapping,
        swa_indices_and_lens,
    )
    from golden import TensorSpec
    from hc_pre import golden_hc_pre
    from utils import build_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)
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
        return torch.empty(tokens, HC_MULT, D).uniform_(-1, 1)

    # Real layer-8 (CSA, ratio-4) hc_attn scale/base (fn synthetic at real magnitude). A
    # synthetic scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out
    # and the hc residual to near-zero in x_out where W8A8 noise blows up the relative tail.
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
        return torch.randn(D, Q_LORA) / D ** 0.5

    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5

    def init_wkv():
        return torch.randn(D, HEAD_DIM) / D ** 0.5

    def init_gamma_cq():
        return torch.ones(Q_LORA)

    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)

    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    # BF16 weight std and RMSNorm gamma mean/std, averaged over DeepSeek-V4-Flash-0731
    # layers 8/32 (the CSA main and inner compressors). idx_wq_b is the only quantized
    # one and goes through the MXFP8 grid below, not a randn INT8.
    def init_cmp_wkv():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0240

    def init_cmp_wgate():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0381

    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.1226

    def init_cmp_norm_w():
        return 0.9569 + 0.1916 * torch.randn(HEAD_DIM)

    def init_compress_state():
        state = torch.zeros(MAIN_STATE_BLOCK_NUM, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM)
        state[:, :, MAIN_OUT_DIM:] = float("-inf")
        starts = init_start_pos().to(torch.int64)
        hist = torch.randn(MAIN_STATE_BLOCK_NUM, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM) * 0.05
        state_table = init_compress_state_block_table().to(torch.int64)
        for b in range(batch):
            for abs_pos in range(int(starts[b].item())):
                logical_blk = abs_pos // MAIN_STATE_BLOCK_SIZE
                blk = int(state_table[b, logical_blk].item())
                intra = abs_pos % MAIN_STATE_BLOCK_SIZE
                state[blk, intra] = hist[blk, intra]
        return state

    def init_compress_state_block_table():
        return block_table(
            batch=batch,
            table_blocks=MAIN_STATE_MAX_BLOCKS,
            physical_blocks=MAIN_STATE_PHYSICAL_BLOCKS,
        )

    def init_weights_proj():
        return torch.randn(D, IDX_N_HEADS) * 0.2218

    def init_hadamard_idx():
        h = torch.ones((1, 1))
        while h.shape[0] < IDX_HEAD_DIM:
            h = torch.cat([
                torch.cat([h, h], dim=1),
                torch.cat([h, -h], dim=1),
            ], dim=0)
        return h / (IDX_HEAD_DIM ** 0.5)

    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0270

    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0513

    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1524

    def init_inner_norm_w():
        return 0.6903 + 0.2663 * torch.randn(IDX_HEAD_DIM)

    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM)
        state[:, :, INNER_OUT_DIM:] = float("-inf")
        starts = init_start_pos().to(torch.int64)
        hist = torch.randn(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM) * 0.05
        state_table = init_inner_compress_state_block_table().to(torch.int64)
        for b in range(batch):
            for abs_pos in range(int(starts[b].item())):
                logical_blk = abs_pos // INNER_STATE_BLOCK_SIZE
                blk = int(state_table[b, logical_blk].item())
                intra = abs_pos % INNER_STATE_BLOCK_SIZE
                state[blk, intra] = hist[blk, intra]
        return state

    def init_inner_compress_state_block_table():
        return block_table(
            batch=batch,
            table_blocks=INNER_STATE_MAX_BLOCKS,
            physical_blocks=INNER_STATE_PHYSICAL_BLOCKS,
        )

    def init_kv_cache():
        return init_normalized_cache((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_window_block_table():
        return block_table(batch=batch, table_blocks=ORI_MAX_BLOCKS, physical_blocks=ORI_BLOCK_NUM)

    def init_cmp_kv():
        return init_normalized_cache((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_cmp_block_table():
        return block_table(
            batch=batch,
            table_blocks=CMP_MAX_BLOCKS,
            physical_blocks=CMP_BLOCK_NUM,
        )

    def init_idx_kv_cache():
        return init_normalized_cache((IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM))

    def init_idx_block_table():
        return block_table(
            batch=batch,
            table_blocks=IDX_CACHE_MAX_BLOCKS,
            physical_blocks=IDX_CACHE_BLOCK_NUM,
        )

    def init_attn_sink():
        return torch.ones(H) * 4.0

    def init_default_start_pos():
        # Canonical CSA start-position set (ratio-4 compressor + indexer + sliding-window + 8k).
        return csa_decode_start_set(
            batch=batch, seq=S, compress_ratio=COMPRESS_RATIO,
            state_block_size=INNER_STATE_BLOCK_SIZE, window=WIN)
    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=batch,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=init_default_start_pos,
        )

    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S).reshape(-1).contiguous()

    def init_kv_seq_lens():
        return kv_seq_lens_from_starts(init_start_pos(), seq=S)

    def init_ori_slot_mapping():
        return ori_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_window_block_table(),
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()

    def init_window_swa_metadata():
        return swa_indices_and_lens(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_window_block_table(),
            block_size=BLOCK_SIZE,
            window=WIN,
        )

    def init_window_swa_indices():
        return init_window_swa_metadata()[0].contiguous()

    def init_window_swa_lens():
        return init_window_swa_metadata()[1].contiguous()

    def init_cmp_slot_mapping():
        positions = position_ids_from_starts(init_start_pos(), seq=S)
        return compressed_slot_mapping(
            positions,
            init_cmp_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()

    def init_idx_slot_mapping():
        positions = position_ids_from_starts(init_start_pos(), seq=S)
        return compressed_slot_mapping(
            positions,
            init_idx_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()

    def init_state_slot_mapping():
        return state_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_compress_state_block_table(),
            state_block_size=MAIN_STATE_BLOCK_SIZE,
        ).reshape(-1).contiguous()

    def init_inner_state_slot_mapping():
        return state_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_inner_compress_state_block_table(),
            state_block_size=INNER_STATE_BLOCK_SIZE,
        ).reshape(-1).contiguous()

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

    shared_x_mixed = torch.zeros(tokens, D, dtype=torch.bfloat16)
    shared_post = torch.zeros(tokens, HC_MULT, dtype=torch.float32)
    shared_comb = torch.zeros(tokens, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": shared_x_hc,
        "hc_fn": shared_hc_attn_fn,
        "hc_scale": shared_hc_attn_scale,
        "hc_base": shared_hc_attn_base,
        "x_mixed": shared_x_mixed,
        "post": shared_post,
        "comb": shared_comb,
    })
    # idx_wq_b is the only quantized indexer weight: simulate the real MXFP8 (e4m3 +
    # 128x128-block E8M0) grid like the shared experts (199 levels, scaleCV ~0.61, ~1.1% zero
    # spike) instead of a benign randn INT8. gen_shared_weight reduces over the last (in) dim
    # and yields scale per output channel, so build [out, in] then transpose to [Q_LORA, out].
    from decode_indexer import gen_shared_weight
    idx_wq_b_i8_T, idx_wq_b_scale = gen_shared_weight(
        (IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56)
    idx_wq_b_i8 = idx_wq_b_i8_T.t().contiguous()
    shared_weights_proj = init_weights_proj().to(torch.bfloat16)
    shared_hadamard_idx = init_hadamard_idx().to(torch.bfloat16)
    shared_idx_kv_cache = init_idx_kv_cache().to(torch.bfloat16)
    # C8 indexer cache: INT8 + per-position scale from the bf16-rounded draw
    from utils import int8_quant_per_row
    _idx_kv_i8, _idx_kv_sc = int8_quant_per_row(
        shared_idx_kv_cache.float().reshape(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM))
    shared_idx_kv_cache_i8 = _idx_kv_i8.view(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM)
    shared_idx_kv_scale = _idx_kv_sc.view(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=lambda: shared_x_hc.clone()),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=lambda: shared_hc_attn_fn.clone()),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=lambda: shared_hc_attn_scale.clone()),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=lambda: shared_hc_attn_base.clone()),
        TensorSpec("attn_norm_w", [D], torch.bfloat16, init_value=lambda: shared_attn_norm_w.clone()),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=lambda: shared_wq_a.clone()),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=lambda: shared_gamma_cq.clone()),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_freqs_cos.clone()),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_freqs_sin.clone()),
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec("compress_state", [MAIN_STATE_BLOCK_NUM, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("compress_state_block_table", [batch, MAIN_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("idx_wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: idx_wq_b_i8),
        TensorSpec("idx_wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: idx_wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=lambda: shared_weights_proj.clone()),
        TensorSpec("hadamard_idx", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_hadamard_idx.clone()),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec("inner_compress_state", [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], torch.float32, init_value=init_inner_compress_state),
        TensorSpec("inner_compress_state_block_table", [batch, INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [batch, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_kv_cache", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], torch.int8, init_value=lambda: shared_idx_kv_cache_i8.clone()),
        TensorSpec("idx_kv_scale", [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], torch.float32, init_value=lambda: shared_idx_kv_scale.clone()),
        TensorSpec("idx_block_table", [batch, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("ori_slot_mapping", [tokens], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("window_swa_indices", [tokens, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("window_swa_lens", [tokens], torch.int32, init_value=init_window_swa_lens),
        TensorSpec("cmp_slot_mapping", [tokens], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("idx_slot_mapping", [tokens], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("state_slot_mapping", [tokens], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("inner_state_slot_mapping", [tokens], torch.int64, init_value=init_inner_state_slot_mapping),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=init_position_ids),
        TensorSpec("kv_seq_lens", [batch], torch.int32, init_value=init_kv_seq_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [tokens, HC_MULT, D], torch.float32, is_output=True),
    ]


def build_tp_tensor_specs(local_t):
    """Build deterministic tensor-parallel CSA output inputs."""
    import torch

    from golden import ScalarSpec, TensorSpec

    if (local_t < ROPE_CS_T_TILE or local_t > LOCAL_T
            or local_t % ROPE_CS_T_TILE != 0 or local_t % S != 0):
        raise ValueError(
            f"local_t must be a multiple of {ROPE_CS_T_TILE} "
            f"in [{ROPE_CS_T_TILE}, {LOCAL_T}], got {local_t}"
        )
    local_batch = local_t // S
    cmp_block_table_shape = [TP_SIZE, local_batch, CMP_MAX_BLOCKS]

    def init_q():
        q = torch.zeros(TP_SIZE, local_t, H, HEAD_DIM, dtype=torch.bfloat16)
        rank = torch.arange(TP_SIZE, dtype=torch.float32).reshape(TP_SIZE, 1, 1)
        token = torch.arange(local_t, dtype=torch.float32).reshape(1, local_t, 1)
        head = torch.arange(H, dtype=torch.float32).reshape(1, 1, H)
        q[..., 0] = (rank + token * 0.25 + head * 0.0625).to(torch.bfloat16)
        return q

    def init_ori_kv():
        shape = (TP_SIZE, FIXTURE_WINDOW_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM)
        ori_kv = torch.zeros(shape, dtype=torch.bfloat16)
        rank = torch.arange(TP_SIZE, dtype=torch.float32).reshape(TP_SIZE, 1, 1)
        block = torch.arange(FIXTURE_WINDOW_BLOCKS, dtype=torch.float32).reshape(1, FIXTURE_WINDOW_BLOCKS, 1)
        row = torch.arange(BLOCK_SIZE, dtype=torch.float32).reshape(1, 1, BLOCK_SIZE)
        ori_kv[:, :, :, 0, 0] = 0.25
        ori_kv[:, :, :, 0, 1] = (
            (rank + 1.0) * 0.0625
            + (block + 1.0) * 0.015625
            + (row + 1.0) * 0.0001220703125
        ).to(torch.bfloat16)
        return ori_kv

    def init_window_swa_indices():
        logical_row = FIXTURE_WINDOW_HEAD + torch.arange(WIN, dtype=torch.int32)
        logical_block = logical_row // BLOCK_SIZE
        physical_block = FIXTURE_WINDOW_BLOCKS - 1 - logical_block
        physical_row = logical_row % BLOCK_SIZE
        window_row = physical_block * BLOCK_SIZE + physical_row
        return window_row.reshape(1, 1, WIN).expand(TP_SIZE, local_t, WIN).clone()

    def init_cmp_kv():
        shape = (TP_SIZE, FIXTURE_CMP_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM)
        cmp_kv = torch.full(shape, float("nan"), dtype=torch.bfloat16)
        for rank in range(TP_SIZE):
            for request in range(local_batch):
                request_block_base = request * FIXTURE_CMP_LOGICAL_BLOCKS
                for entry_index, cmp_slot in enumerate(FIXTURE_CMP_SLOTS):
                    logical_block = cmp_slot // BLOCK_SIZE
                    physical_block = request_block_base + FIXTURE_CMP_LOGICAL_BLOCKS - 1 - logical_block
                    row = cmp_slot % BLOCK_SIZE
                    cmp_kv[rank, physical_block, row, 0, :] = 0.0
                    cmp_kv[rank, physical_block, row, 0, 0] = 0.25
                    cmp_kv[rank, physical_block, row, 0, 1] = (
                        (rank + 1) * 0.03125
                        + (request + 1) * 0.0078125
                        + (entry_index + 1) * 0.001953125
                    )
                    cmp_kv[rank, physical_block, row, 0, NOPE_DIM] = (
                        (rank + 1) * 0.0625
                        + (request + 1) * 0.015625
                        + (entry_index + 1) * 0.00390625
                    )
                    cmp_kv[rank, physical_block, row, 0, NOPE_DIM + 1] = (
                        -(logical_block + 1) * 0.0625
                        + (rank + 1) * 0.0078125
                        + (request + 1) * 0.001953125
                    )
        return cmp_kv

    def init_cmp_block_table():
        table = torch.full(cmp_block_table_shape, -1, dtype=torch.int32)
        for rank in range(TP_SIZE):
            for request in range(local_batch):
                request_block_base = request * FIXTURE_CMP_LOGICAL_BLOCKS
                for logical_block in range(FIXTURE_CMP_LOGICAL_BLOCKS):
                    table[rank, request, logical_block] = (
                        request_block_base + FIXTURE_CMP_LOGICAL_BLOCKS - 1 - logical_block
                    )
        return table

    def init_idx_topk():
        topk = torch.full((TP_SIZE, local_t, INDEXER_SCORE_LEN), -1, dtype=torch.int32)
        for position, cmp_slot in FIXTURE_CMP_ENTRIES:
            topk[:, :, position] = cmp_slot
        for position in FIXTURE_FILTERED_POSITIONS:
            topk[:, :, position] = FIXTURE_FILTERED_SLOT
        return topk

    def init_position_ids():
        return torch.full((TP_SIZE, local_t, 1), FIXTURE_POSITION_ID, dtype=torch.int32)

    def init_attn_sink():
        head = torch.arange(H, dtype=torch.int32).remainder(HEADS_PER_GROUP).to(torch.float32)
        sink = 4.0 + head * 0.125
        return sink.reshape(1, H).expand(TP_SIZE, H).clone()

    def init_freqs_cos():
        rank = torch.arange(TP_SIZE, dtype=torch.int32).reshape(TP_SIZE, 1, 1)
        token = torch.arange(local_t, dtype=torch.int32).reshape(1, local_t, 1)
        column = torch.arange(ROPE_DIM, dtype=torch.int32).reshape(1, 1, ROPE_DIM)
        phase = (rank + token + column).remainder(4)
        phase[:, :, HALF_ROPE:] = (phase[:, :, HALF_ROPE:] + 1).remainder(4)
        values = torch.tensor((1.0, 0.0, -1.0, 0.0), dtype=torch.bfloat16)
        return values[phase]

    def init_freqs_sin():
        rank = torch.arange(TP_SIZE, dtype=torch.int32).reshape(TP_SIZE, 1, 1)
        token = torch.arange(local_t, dtype=torch.int32).reshape(1, local_t, 1)
        column = torch.arange(ROPE_DIM, dtype=torch.int32).reshape(1, 1, ROPE_DIM)
        phase = (rank + token + column).remainder(4)
        phase[:, :, HALF_ROPE:] = (phase[:, :, HALF_ROPE:] + 1).remainder(4)
        values = torch.tensor((0.0, 1.0, 0.0, -1.0), dtype=torch.bfloat16)
        return values[phase]

    def init_wo_a():
        wo_a = torch.zeros(TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16)
        for rank in range(TP_SIZE):
            for local_group in range(LOCAL_O_GROUPS):
                shard_scale = (rank + 1) * (local_group + 1) * 0.125
                for head in range(HEADS_PER_GROUP):
                    wo_a[rank, local_group, head, head * HEAD_DIM] = shard_scale * (head + 1)
                    wo_a[
                        rank, local_group, HEADS_PER_GROUP + head,
                        head * HEAD_DIM + 1,
                    ] = shard_scale * (head + 1) * 0.75
                    wo_a[
                        rank, local_group, 2 * HEADS_PER_GROUP + head,
                        head * HEAD_DIM + NOPE_DIM,
                    ] = -shard_scale * (head + 1) * 0.5
                    wo_a[
                        rank, local_group, 3 * HEADS_PER_GROUP + head,
                        head * HEAD_DIM + NOPE_DIM + 1,
                    ] = shard_scale * (head + 1) * 0.25
        return wo_a

    def init_wo_b():
        base = torch.arange(D * LOCAL_O_WIDTH, dtype=torch.int32).reshape(D, LOCAL_O_WIDTH)
        rank = torch.arange(TP_SIZE, dtype=torch.int32).reshape(TP_SIZE, 1, 1)
        return (base.unsqueeze(0) + rank).remainder(7).sub(3).to(torch.int8)

    def init_wo_b_scale():
        channel_scale = torch.arange(D, dtype=torch.int32).remainder(4).to(torch.float32) * 0.25 + 0.5
        return channel_scale.reshape(1, D).expand(TP_SIZE, D).clone()

    return [
        TensorSpec("q", [TP_SIZE, local_t, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec(
            "ori_kv", [TP_SIZE, FIXTURE_WINDOW_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16, init_value=init_ori_kv,
        ),
        TensorSpec("window_swa_indices", [TP_SIZE, local_t, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec(
            "cmp_kv", [TP_SIZE, FIXTURE_CMP_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16, init_value=init_cmp_kv,
        ),
        TensorSpec("cmp_block_table", cmp_block_table_shape, torch.int32, init_value=init_cmp_block_table),
        TensorSpec("idx_topk", [TP_SIZE, local_t, INDEXER_SCORE_LEN], torch.int32, init_value=init_idx_topk),
        TensorSpec("position_ids", [TP_SIZE, local_t, 1], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [TP_SIZE, H], torch.float32, init_value=init_attn_sink),
        TensorSpec("freqs_cos", [TP_SIZE, local_t, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [TP_SIZE, local_t, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("wo_a", [TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [TP_SIZE, D, LOCAL_O_WIDTH], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [TP_SIZE, D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec(
            "o_local", [TP_SIZE, LOCAL_T_PAD, D], torch.bfloat16,
            init_value=FIXTURE_OUTPUT_SENTINEL, is_output=True,
        ),
        ScalarSpec("local_t", torch.int32, local_t),
    ]


def golden_decode_csa_output(tensors):
    """Compute the controlled CSA heads, sharded O projection, and reduced rows."""
    import torch

    local_t = int(tensors["local_t"])
    group_t = TP_SIZE * local_t
    observed_dims = torch.tensor((0, 1, NOPE_DIM, NOPE_DIM + 1), dtype=torch.int64)
    q = tensors["q"].float()

    window_indices = tensors["window_swa_indices"].long()
    window_valid = window_indices >= 0
    safe_window_indices = window_indices.clamp_min(0)
    ori_rows = tensors["ori_kv"][:, :, :, 0].reshape(TP_SIZE, -1, HEAD_DIM)
    ori_values = ori_rows[..., observed_dims].float()
    window_sum = torch.zeros(TP_SIZE, local_t, len(observed_dims), dtype=torch.float32)
    for rank in range(TP_SIZE):
        gathered = ori_values[rank][safe_window_indices[rank]]
        window_sum[rank] = (
            gathered * window_valid[rank].unsqueeze(-1)
        ).sum(dim=1)
    window_count = window_valid.sum(dim=-1).to(torch.float32)

    raw = tensors["idx_topk"][:, :, :CMP_TOPK].to(torch.int64)
    bound = ((tensors["position_ids"][:, :, 0].to(torch.int64) + 1) // COMPRESS_RATIO).unsqueeze(-1)
    keep = (raw >= 0) & (raw < bound)
    cmp_rows = tensors["cmp_kv"][:, :, :, 0]
    cmp_sum = torch.zeros_like(window_sum)
    cmp_count = torch.zeros(TP_SIZE, local_t, dtype=torch.float32)
    for rank in range(TP_SIZE):
        for token in range(local_t):
            request = token // S
            valid_slots = raw[rank, token][keep[rank, token]]
            cmp_count[rank, token] = valid_slots.numel()
            for cmp_slot_tensor in valid_slots:
                cmp_slot = int(cmp_slot_tensor.item())
                logical_block = cmp_slot // BLOCK_SIZE
                physical_block = int(tensors["cmp_block_table"][rank, request, logical_block].item())
                cmp_sum[rank, token] += cmp_rows[
                    rank, physical_block, cmp_slot % BLOCK_SIZE, observed_dims
                ].float()

    kv_sum = window_sum + cmp_sum
    valid_count = window_count + cmp_count
    scores = q[..., 0] * 0.25 * SOFTMAX_SCALE
    sink = tensors["attn_sink"].float().reshape(TP_SIZE, 1, H)
    denominator = valid_count.unsqueeze(-1) + torch.exp(sink - scores)
    head_observed = kv_sum.unsqueeze(2) / denominator.unsqueeze(-1)
    head_value = torch.zeros(TP_SIZE, local_t, H, HEAD_DIM, dtype=torch.float32)
    head_value[..., observed_dims] = head_observed

    rope_pair = head_value[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = tensors["freqs_cos"][..., :HALF_ROPE].float().unsqueeze(2)
    sin_half = tensors["freqs_sin"][..., :HALF_ROPE].float().unsqueeze(2)
    inverse_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inverse_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    inverse_rope = torch.stack((inverse_even, inverse_odd), dim=-1).flatten(-2)
    head_value = torch.cat((head_value[..., :NOPE_DIM], inverse_rope), dim=-1).to(torch.bfloat16)

    attention_grouped = head_value.reshape(
        TP_SIZE, local_t, O_GROUPS, HEADS_PER_GROUP, HEAD_DIM,
    )
    attention_grouped = attention_grouped.permute(0, 2, 1, 3, 4)
    attention_grouped = attention_grouped.reshape(
        TP_SIZE, O_GROUPS, local_t, O_GROUP_IN,
    )

    attention_local = torch.zeros(TP_SIZE, LOCAL_O_GROUPS, group_t, O_GROUP_IN, dtype=torch.bfloat16)
    for destination_rank in range(TP_SIZE):
        for local_group in range(LOCAL_O_GROUPS):
            global_group = destination_rank * LOCAL_O_GROUPS + local_group
            group_rows = attention_grouped[:, global_group].reshape(group_t, O_GROUP_IN)
            attention_local[destination_rank, local_group] = group_rows

    partials = torch.zeros(TP_SIZE, group_t, D, dtype=torch.float32)
    for rank in range(TP_SIZE):
        attention = attention_local[rank].float()
        o_a = torch.einsum("gti,gri->gtr", attention, tensors["wo_a"][rank].float())
        row_amax = o_a.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_q = INT8_SCALE_MAX / row_amax
        o_a_i8 = torch.round(o_a * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
        scale_dq = 1.0 / scale_q
        wo_b = tensors["wo_b"][rank].reshape(D, LOCAL_O_GROUPS, O_LORA)
        for local_group in range(LOCAL_O_GROUPS):
            group_i32 = o_a_i8[local_group].to(torch.int32)
            weight_i32 = wo_b[:, local_group].to(torch.int32)
            group_partial = group_i32 @ weight_i32.T
            partials[rank] += group_partial.float() * scale_dq[local_group]
        partials[rank] *= tensors["wo_b_scale"][rank].float().reshape(1, D)

    reduced = partials.sum(dim=0)
    tensors["o_local"].fill_(FIXTURE_OUTPUT_SENTINEL)
    for rank in range(TP_SIZE):
        row_start = rank * local_t
        tensors["o_local"][rank, :local_t] = reduced[row_start : row_start + local_t].to(torch.bfloat16)


def build_o_local_compare(local_t):
    """Compare valid token rows and require the poisoned capacity tail to survive."""
    import torch

    from golden import ratio_allclose

    prefix_compare = ratio_allclose(
        atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005,
        valid_rows=local_t, valid_axis=1,
    )

    def compare(actual, expected, **kwargs):
        actual_tail = actual[:, local_t:]
        expected_tail = expected[:, local_t:]
        if not torch.equal(actual_tail, expected_tail):
            mismatch_count = int((actual_tail != expected_tail).sum().item())
            return False, f"    inactive token tail mismatch count={mismatch_count}"
        return prefix_compare(actual, expected, **kwargs)

    compare.__name__ = f"csa_output_prefix_and_tail(local_t={local_t})"
    return compare


if __name__ == "__main__":
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES),
                        help="tensor-parallel world size")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(rank) for rank in range(TP_SIZE)),
                        help=f"comma-separated device ids; need exactly {TP_SIZE}")
    parser.add_argument("--case", choices=("all", "max", "subcapacity"), default="all")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument(
        "--save-data",
        action="store_true",
        default=False,
        help="persist inputs and golden outputs for replay",
    )
    parser.add_argument(
        "--golden-data",
        type=str,
        default=None,
        help="directory containing cached in/ and out/ tensors",
    )
    args = parser.parse_args()

    if args.case == "all" and args.golden_data is not None:
        parser.error("--golden-data requires --case max or --case subcapacity")
    if args.tp != TP_SIZE:
        parser.error(f"--tp must remain {TP_SIZE} after import-time specialization")
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(f"--device must be a comma-separated integer list, got {args.device!r}")
    if len(device_ids) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {device_ids}")
    if any(device < 0 for device in device_ids):
        parser.error(f"--device IDs must be non-negative, got {device_ids}")
    if len(set(device_ids)) != TP_SIZE:
        parser.error(f"need {TP_SIZE} distinct devices, got {device_ids}")

    case_local_t = {"max": LOCAL_T, "subcapacity": LOCAL_T - ROPE_CS_T_TILE}
    selected_cases = tuple(case_local_t) if args.case == "all" else (args.case,)
    for case in selected_cases:
        local_t = case_local_t[case]
        result = run_jit(
            fn=l3_decode_csa_output,
            specs=build_tp_tensor_specs(local_t),
            golden_fn=golden_decode_csa_output,
            golden_data=args.golden_data,
            save_data=args.save_data,
            compile_only=args.compile_only,
            compile_cfg=dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            ),
            runtime_cfg=dict(platform=args.platform),
            compare_fn={"o_local": build_o_local_compare(local_t)},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
