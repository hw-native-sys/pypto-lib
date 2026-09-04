# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 packed prefill CSA attention with compression, indexing, and cache writeback."""

import functools

import pypto.language as pl

from config import (
    FLASH as M,
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    KV_ORI_BLOCK_NUM,
    PREFILL_SEQ,
)

from prefill_compressor_ratio4 import (
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    compressor_ratio4,
    golden_prefill_compressor_ratio4,
)
from hc_post import golden_hc_post, hc_post
from hc_pre import golden_hc_pre, hc_pre
from prefill_indexer import (
    COMPRESS_RATIO as INDEXER_COMPRESS_RATIO,
    IDX_CACHE_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS,
    gen_shared_weight,
    golden_prefill_indexer_core,
    prefill_indexer,
    prefill_indexer_query,
    topk_prefix_contract_error,
)
from prefill_indexer_compressor import (
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    prefill_indexer_compressor,
)
from prefill_metadata import QUERY_START_LOC_DYN, REQUESTS_DYN
from qkv_proj_rope import golden_qkv_proj_rope, qkv_proj_rope
from rmsnorm import golden_rms_norm, rms_norm
from prefill_sparse_attn import (
    PREFILL_ATTN_BLOCKS,
    PREFILL_ATTN_TILE,
    PREFILL_RING_HEAP,
    PREFILL_SPARSE_PAD as SPARSE_PREFILL_SPARSE_PAD,
    SPARSE_CMP_BIAS_COLS,
    VALID_BLOCK_MASK_COLS,
    golden_prefill_sparse_attn,
    sparse_attn_physical,
)

import pypto.language.distributed as pld

from prefill_cp_token_allgather import (
    PREFILL_GROUP_CAP,
    TP_SIZE,
    prefill_cp_token_allgather_step,
)
from prefill_o_proj import (
    O_PROJ_LOCAL_COLS,
    O_PROJ_LOCAL_GROUPS,
    O_PROJ_SCRATCH_COLS,
    O_PROJ_SCRATCH_D,
    O_PROJ_SCRATCH_GROUPS,
    O_PROJ_SCRATCH_INPUT,
    O_PROJ_SCRATCH_RANK,
    O_PROJ_WO_A_WINDOW_COLS,
    O_PROJ_WO_A_WINDOW_ROWS,
    O_PROJ_WO_B_WINDOW_COLS,
    O_PROJ_WO_B_WINDOW_ROWS,
    gather_o_proj_full_weights,
)
from qkv_proj_rope import kv_proj_rope, q_proj_rope, rope_prepare


# Largest runtime token count the prefill path accepts (Issue #905 P4).
PREFILL_MAX_TOKENS = 8192

# Dynamic shape variables.
T_DYN = pl.dynamic("PREFILL_CSA_T_DYN")
CP_Q_T_DYN = pl.dynamic("PREFILL_CSA_CP_Q_T_DYN")
CP_KV_T_DYN = pl.dynamic("PREFILL_CSA_CP_KV_T_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_CMP_BLOCK_NUM_DYN")
IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")

# model config
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
IDX_HEAD_DIM = M.index_head_dim
IDX_N_HEADS = M.index_n_heads
IDX_TOPK = M.index_topk
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS
COFF = 2
MAIN_OUT_DIM = COFF * HEAD_DIM
MAIN_COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
INNER_STATE_LEN = COFF * COMPRESS_RATIO
MAX_CMP_WRITES = max(1, PREFILL_MAX_TOKENS // COMPRESS_RATIO)

# paged KV cache
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
CMP_BLOCK_NUM = CMP_MAX_BLOCKS
SPARSE_ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
SPARSE_CMP_MAX_BLOCKS = CMP_MAX_BLOCKS
CSA_ORI_BLOCK_NUM = ORI_BLOCK_NUM
CSA_CMP_BLOCK_NUM = CMP_BLOCK_NUM

# tiling
CSA_TOPK_TOKEN_TILE = 2

assert COMPRESS_RATIO == INDEXER_COMPRESS_RATIO
assert PREFILL_ATTN_BLOCKS <= VALID_BLOCK_MASK_COLS
assert IDX_TOPK <= SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE


@pl.jit.inline
def prefill_attention_csa(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[
        [MAIN_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    """Run CSA over one packed ragged prefill stream."""
    t_dim = pl.tensor.dim(x_hc, 0)
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)

    x_normed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    # Defers kv_proj_matmul one hop behind rms_norm so qr_proj_matmul dispatches first.
    late_dep = pl.system.task_dummy(deps=[rms_tid])

    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(
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
        late_dep,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    ori_cache_rows = ori_block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [ori_cache_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_cache_write"):
        for write_t in pl.range(t_dim):
            write_row_raw = pl.read(ori_slot_mapping, [write_t])
            if write_row_raw >= 0:
                write_row = pl.cast(write_row_raw, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, :] = kv[write_t : write_t + 1, :]

    compressor_ratio4(
        x_normed,
        query_start_loc,
        compress_state,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_freqs_cos,
        cmp_freqs_sin,
        cmp_kv,
        position_ids,
        cmp_slot_mapping,
        state_slot_mapping,
    )
    # Half-width FP32 current-token rows for the indexer Q-RoPE.
    idx_cos = pl.create_tensor([t_dim, HALF_ROPE], dtype=pl.FP32)
    idx_sin = pl.create_tensor([t_dim, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_idx_halfrope"):
        for idx_t in pl.range(t_dim):
            idx_cos[idx_t : idx_t + 1, 0:HALF_ROPE] = pl.cast(
                freqs_cos[idx_t : idx_t + 1, 0:HALF_ROPE], target_type=pl.FP32
            )
            idx_sin[idx_t : idx_t + 1, 0:HALF_ROPE] = pl.cast(
                freqs_sin[idx_t : idx_t + 1, 0:HALF_ROPE], target_type=pl.FP32
            )

    cmp_topk_indices = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.INT32)
    idx_kv_cache_out, idx_kv_scale_out, cmp_topk_indices = prefill_indexer(
        x_normed,
        query_start_loc,
        qr,
        qr_scale,
        idx_wq_b,
        idx_wq_b_scale,
        idx_weights_proj,
        idx_cos,
        idx_sin,
        cmp_freqs_cos,
        cmp_freqs_sin,
        hadamard_idx,
        inner_compress_state,
        inner_compress_state_block_table,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        idx_kv_cache,
        idx_kv_scale,
        idx_block_table,
        cmp_topk_indices,
        position_ids,
        local_request_ids,
        idx_slot_mapping,
        inner_state_slot_mapping,
    )

    swa_indices = pl.create_tensor([t_dim, WIN], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([t_dim, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    csa_topk_blocks = (t_dim + CSA_TOPK_TOKEN_TILE - 1) // CSA_TOPK_TOKEN_TILE
    for topk_block in pl.spmd(csa_topk_blocks, name_hint="prefill_csa_sparse_idx_tile"):
        topk_t0 = topk_block * CSA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(CSA_TOPK_TOKEN_TILE):
            t_idx = topk_t0 + topk_dt
            swa_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            mask_row = pl.full([1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
            if t_idx < t_dim:
                request_id = pl.read(local_request_ids, [t_idx])
                if request_id >= 0:
                    abs_pos = pl.read(position_ids, [t_idx])
                    # Sparse-block liveness from the dense TopK prefix.
                    visible_cmp = pl.min((abs_pos + 1) // COMPRESS_RATIO, pl.cast(IDX_TOPK, pl.INT32))
                    for mask_sb in pl.unroll(PREFILL_ATTN_BLOCKS):
                        cmp_lo = pl.max(mask_sb * PREFILL_ATTN_TILE - WIN, pl.cast(0, pl.INT32))
                        cmp_hi = pl.min(
                            (mask_sb + 1) * PREFILL_ATTN_TILE - WIN,
                            pl.cast(SPARSE_CMP_BIAS_COLS, pl.INT32),
                        )
                        if cmp_lo < cmp_hi:
                            if visible_cmp > cmp_lo:
                                pl.write(mask_row, [0, mask_sb], pl.cast(1, pl.INT32))
                    window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                    key_start_abs = abs_pos + 1 - window_valid
                    for win_col in pl.range(WIN):
                        win_col_i32 = pl.cast(win_col, pl.INT32)
                        if win_col_i32 < window_valid:
                            key_abs = key_start_abs + win_col_i32
                            blk_slot = key_abs // BLOCK_SIZE
                            blk = pl.read(ori_block_table, [request_id, pl.cast(blk_slot, pl.INDEX)])
                            if blk >= 0:
                                row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                                pl.write(swa_row, [0, win_col], row)
                                pl.write(mask_row, [0, win_col // PREFILL_ATTN_TILE], pl.cast(1, pl.INT32))
                swa_indices[t_idx : t_idx + 1, 0:WIN] = swa_row
                valid_block_mask[t_idx : t_idx + 1, 0:VALID_BLOCK_MASK_COLS] = mask_row

    o_proj_weight_dep = pl.system.task_dummy(deps=[])
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    attn_out = sparse_attn_physical(
        q,
        kv_cache, swa_indices,
        cmp_kv, cmp_block_table, local_request_ids, cmp_topk_indices,
        valid_block_mask, attn_sink,
        freqs_cos, freqs_sin,
        wo_a, wo_b, wo_b_scale,
        attn_out,
        o_proj_weight_dep,
    )

    hc_post(attn_out, x_hc, post, comb, x_out)
    return (
        kv_cache,
        cmp_kv,
        compress_state,
        idx_kv_cache,
        idx_kv_scale,
        inner_compress_state,
        x_out,
    )


@pl.jit
def prefill_attention_csa_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32]
    ],
    inner_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x_hc.bind_dynamic(0, T_DYN)
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    inner_compress_state.bind_dynamic(0, INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    ori_block_table.bind_dynamic(0, REQUESTS_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    idx_kv_cache.bind_dynamic(0, IDX_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(0, IDX_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(0, REQUESTS_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    local_request_ids.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    idx_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)

    prefill_attention_csa(
        x_hc,
        query_start_loc,
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
        cmp_freqs_cos,
        cmp_freqs_sin,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        compress_state,
        compress_state_block_table,
        hadamard_idx,
        idx_wq_b,
        idx_wq_b_scale,
        idx_weights_proj,
        inner_wkv,
        inner_wgate,
        inner_ape,
        inner_norm_w,
        inner_compress_state,
        inner_compress_state_block_table,
        kv_cache,
        ori_block_table,
        ori_slot_mapping,
        cmp_kv,
        cmp_block_table,
        idx_kv_cache,
        idx_kv_scale,
        idx_block_table,
        position_ids,
        local_request_ids,
        cmp_slot_mapping,
        idx_slot_mapping,
        state_slot_mapping,
        inner_state_slot_mapping,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        x_out,
    )
    return (
        kv_cache,
        cmp_kv,
        compress_state,
        idx_kv_cache,
        idx_kv_scale,
        inner_compress_state,
        x_out,
    )


def golden_prefill_attention_csa(tensors):
    """Torch reference for token-major packed CSA with overlay compressor/indexer."""
    import torch

    from utils import cache_row_from_table

    token_count = int(tensors["x_hc"].shape[0])
    x_hc_flat = tensors["x_hc"].view(token_count, HC_MULT, D)
    x_mixed = torch.zeros(token_count, D, dtype=torch.bfloat16)
    post = torch.zeros(token_count, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(token_count, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre(
        {
            "x": x_hc_flat,
            "hc_fn": tensors["hc_attn_fn"],
            "hc_scale": tensors["hc_attn_scale"],
            "hc_base": tensors["hc_attn_base"],
            "x_mixed": x_mixed,
            "post": post,
            "comb": comb,
        }
    )

    q = torch.zeros(token_count, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(token_count, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(token_count, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(token_count, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    rope_cos_t = tensors["freqs_cos"].contiguous()
    rope_sin_t = tensors["freqs_sin"].contiguous()
    golden_qkv_proj_rope(
        {
            "x": x_normed.view(token_count, D),
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
        }
    )

    query_start_loc = tensors["query_start_loc"]
    for request in range(query_start_loc.numel() - 1):
        request_start = int(query_start_loc[request].item())
        request_end = int(query_start_loc[request + 1].item())
        if request_end <= request_start:
            continue
        request_rows = slice(request_start, request_end)
        golden_prefill_compressor_ratio4(
            {
                "x": x_normed[request_rows].view(request_end - request_start, D),
                "compress_state": tensors["compress_state"],
                "compress_state_block_table": tensors["compress_state_block_table"][request : request + 1],
                "wkv": tensors["cmp_wkv"],
                "wgate": tensors["cmp_wgate"],
                "ape": tensors["cmp_ape"],
                "norm_w": tensors["cmp_norm_w"],
                "cmp_freqs_cos": tensors["cmp_freqs_cos"][request_rows],
                "cmp_freqs_sin": tensors["cmp_freqs_sin"][request_rows],
                "cmp_kv": tensors["cmp_kv"],
                "position_ids": tensors["position_ids"][request_rows],
                "cmp_slot_mapping": tensors["cmp_slot_mapping"][request_rows],
                "state_slot_mapping": tensors["state_slot_mapping"][request_rows],
            }
        )
    idx_cos = rope_cos_t[:, :HALF_ROPE].float().contiguous()
    idx_sin = rope_sin_t[:, :HALF_ROPE].float().contiguous()
    cmp_topk_indices = golden_prefill_indexer_core(
        {
            "x": x_normed.view(token_count, D),
            "query_start_loc": tensors["query_start_loc"],
            "qr": qr,
            "qr_scale": qr_scale,
            "wq_b": tensors["idx_wq_b"],
            "wq_b_scale": tensors["idx_wq_b_scale"],
            "weights_proj": tensors["idx_weights_proj"],
            "cos": idx_cos,
            "sin": idx_sin,
            "cmp_freqs_cos": tensors["cmp_freqs_cos"],
            "cmp_freqs_sin": tensors["cmp_freqs_sin"],
            "hadamard": tensors["hadamard_idx"],
            "inner_compress_state": tensors["inner_compress_state"],
            "inner_compress_state_block_table": tensors["inner_compress_state_block_table"],
            "inner_wkv": tensors["inner_wkv"],
            "inner_wgate": tensors["inner_wgate"],
            "inner_ape": tensors["inner_ape"],
            "inner_norm_w": tensors["inner_norm_w"],
            "idx_kv_cache": tensors["idx_kv_cache"],
            "idx_kv_scale": tensors["idx_kv_scale"],
            "idx_block_table": tensors["idx_block_table"],
            "position_ids": tensors["position_ids"],
            "local_request_ids": tensors["local_request_ids"],
            "idx_slot_mapping": tensors["idx_slot_mapping"],
            "inner_state_slot_mapping": tensors["inner_state_slot_mapping"],
        }
    )

    kv_cache_in = tensors["kv_cache"].clone()
    kv_cache_flat = kv_cache_in.view(CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(token_count):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]

    def assemble_swa_indices():
        swa_idx = torch.full((token_count, WIN), -1, dtype=torch.int32)
        pos = tensors["position_ids"]
        ori_table = tensors["ori_block_table"]
        request_ids = tensors["local_request_ids"]
        for t in range(token_count):
            request_id = int(request_ids[t].item())
            if request_id < 0:
                continue
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                row = cache_row_from_table(ori_table[request_id], key_abs)
                if row >= 0:
                    swa_idx[t, k] = row
        return swa_idx

    contract_error = topk_prefix_contract_error(
        cmp_topk_indices,
        tensors["position_ids"],
    )
    if contract_error:
        raise AssertionError(f"prefill indexer top-k contract failed: {contract_error}")
    swa_indices = assemble_swa_indices()
    cmp_indices = cmp_topk_indices.clone()
    attn_out = torch.zeros(token_count, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn(
        {
            "q": q,
            "ori_kv": kv_cache_in,
            "swa_indices": swa_indices,
            "cmp_kv": tensors["cmp_kv"],
            "cmp_block_table": tensors["cmp_block_table"],
            "local_request_ids": tensors["local_request_ids"],
            "cmp_indices": cmp_indices,
            "attn_sink": tensors["attn_sink"],
            "freqs_cos": rope_cos_t,
            "freqs_sin": rope_sin_t,
            "wo_a": tensors["wo_a"],
            "wo_b": tensors["wo_b"],
            "wo_b_scale": tensors["wo_b_scale"],
            "attn_out": attn_out,
        }
    )

    tensors["kv_cache"][:] = kv_cache_in

    y = torch.zeros(token_count, HC_MULT, D, dtype=torch.float32)
    golden_hc_post(
        {
            "x": attn_out,
            "residual": tensors["x_hc"],
            "post": post,
            "comb": comb,
            "y": y,
        }
    )
    tensors["x_out"][:] = y


@functools.lru_cache(maxsize=None)
def _state_block_table(max_blocks, physical_blocks):
    """Constant scrambled state block table [max_blocks]."""
    import torch

    blocks = torch.arange(max_blocks, dtype=torch.int32)
    return (blocks * 17 + 3) % physical_blocks


def build_tensor_specs(
    start_pos: int = START_POS,
    token_count: int = PREFILL_SEQ,
):
    import torch
    from golden import TensorSpec
    from utils import (
        int8_quant_per_row,
        quant_w_per_channel,
        token_local_rope,
    )

    # Single-request geometry: the physical token dimension is q_len.
    context_len = start_pos
    q_len = token_count
    if token_count <= 0 or token_count > PREFILL_MAX_TOKENS:
        raise ValueError(f"token_count must be in [1, {PREFILL_MAX_TOKENS}], got {token_count}")
    if context_len < 0:
        raise ValueError(f"context length must be non-negative, got {context_len}")
    max_position = context_len + q_len - 1
    if max_position >= MAX_SEQ_LEN:
        raise ValueError(f"position id {max_position} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}")
    max_visible_cmp = (context_len + q_len) // COMPRESS_RATIO
    # The sparse rows are the window plus what the indexer actually emits, which
    # is its top-k, not every visible compressed position.
    max_sparse_rows = WIN + min(max_visible_cmp, IDX_TOPK)
    if max_sparse_rows > SPARSE_PREFILL_SPARSE_PAD:
        raise ValueError(
            f"needs {max_sparse_rows} sparse rows; current packed sparse CSA cap is {SPARSE_PREFILL_SPARSE_PAD}"
        )
    if max_visible_cmp > SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE:
        raise ValueError(
            f"needs {max_visible_cmp} compressed slots; current cmp cache cap is "
            f"{SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE}"
        )

    history_chunk_rows = 4096
    ori_block_table = (torch.arange(SPARSE_ORI_MAX_BLOCKS, dtype=torch.int64) % CSA_ORI_BLOCK_NUM).to(torch.int32)
    cmp_block_table = torch.arange(SPARSE_CMP_MAX_BLOCKS, dtype=torch.int32)
    idx_block_table = torch.arange(IDX_CACHE_MAX_BLOCKS, dtype=torch.int32)
    if torch.unique(ori_block_table).numel() != CSA_ORI_BLOCK_NUM:
        raise ValueError("raw fixture block table does not cover the physical pool")
    if (
        CSA_CMP_BLOCK_NUM != SPARSE_CMP_MAX_BLOCKS
        or torch.unique(cmp_block_table).numel() != SPARSE_CMP_MAX_BLOCKS
    ):
        raise ValueError("CSA fixture block table must map every logical page injectively")
    if (
        IDX_CACHE_BLOCK_NUM != IDX_CACHE_MAX_BLOCKS
        or torch.unique(idx_block_table).numel() != IDX_CACHE_MAX_BLOCKS
    ):
        raise ValueError("index fixture block table must map every logical page injectively")

    def paged_rows(block_table, logical_slots, block_size=BLOCK_SIZE):
        slots = logical_slots.to(torch.int64)
        logical_blocks = torch.div(slots, block_size, rounding_mode="floor")
        physical_blocks = block_table.index_select(0, logical_blocks).to(torch.int64)
        if torch.any(physical_blocks < 0):
            raise ValueError("fixture history references an unmapped logical page")
        return physical_blocks * block_size + slots.remainder(block_size)

    def scatter_bf16_history(cache_flat, block_table, slot_start, slot_end, value_scale):
        for chunk_start in range(slot_start, slot_end, history_chunk_rows):
            chunk_end = min(chunk_start + history_chunk_rows, slot_end)
            logical_slots = torch.arange(chunk_start, chunk_end, dtype=torch.int64)
            physical_rows = paged_rows(block_table, logical_slots)
            values = (torch.rand(chunk_end - chunk_start, cache_flat.shape[1]) - 0.5) * value_scale
            cache_flat.index_copy_(0, physical_rows, values.to(cache_flat.dtype))

    def token_pos():
        # Single-request absolute positions: pos[t] = context_len + local_idx
        return torch.arange(context_len, context_len + q_len, dtype=torch.int32)

    rope_positions = token_pos()
    shared_freqs_cos, shared_freqs_sin = token_local_rope(
        M, COMPRESS_RATIO, rope_positions,
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )
    cmp_positions = torch.where(
        (rope_positions + 1) % COMPRESS_RATIO == 0,
        rope_positions - (COMPRESS_RATIO - 1), torch.zeros_like(rope_positions),
    )
    shared_cmp_freqs_cos, shared_cmp_freqs_sin = token_local_rope(
        M, COMPRESS_RATIO, cmp_positions,
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )

    def cmp_write_records():
        positions = token_pos().to(torch.int64)
        token_ids = torch.nonzero((positions + 1) % COMPRESS_RATIO == 0, as_tuple=False).flatten()
        cmp_slots = (positions.index_select(0, token_ids) + 1) // COMPRESS_RATIO - 1
        if token_ids.numel() > MAX_CMP_WRITES:
            raise ValueError(f"CSA fixture generated {token_ids.numel()} compressed writes, cap is {MAX_CMP_WRITES}")
        return token_ids, cmp_slots

    def init_x_hc():
        return torch.empty(token_count, HC_MULT, D).uniform_(-1, 1)

    # Real layer-8 (CSA, ratio-4) hc_attn scale/base (fn synthetic at real magnitude). A
    # synthetic scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out
    # and the hc residual to near-zero in x_out where W8A8 noise blows up the relative tail.
    # Mirrors decode_csa.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0519

    def init_hc_attn_scale():
        return torch.tensor([0.076099, 0.032597, 0.226994])

    def init_hc_attn_base():
        return torch.tensor(
            [
                5.9166,
                -3.6223,
                -2.9324,
                -3.3124,
                -3.9100,
                -0.9384,
                -3.3256,
                -2.5240,
                2.0706,
                -2.5728,
                0.1424,
                -3.9453,
                -3.8859,
                3.4634,
                -3.3799,
                -2.6077,
                -2.7191,
                -2.4846,
                2.0395,
                -0.5010,
                -3.5992,
                -2.7520,
                -3.3493,
                3.1587,
            ]
        )

    def init_attn_norm_w():
        return torch.ones(D)

    def init_wq_a():
        return (torch.rand(D, Q_LORA) - 0.5) * D**-0.5

    def init_wq_b():
        return (torch.rand(Q_LORA, H * HEAD_DIM) - 0.5) * Q_LORA**-0.5

    def init_wkv():
        return (torch.rand(D, HEAD_DIM) - 0.5) * D**-0.5

    def init_gamma_cq():
        return torch.ones(Q_LORA)

    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)

    def init_freqs_cos():
        return shared_freqs_cos.clone()

    def init_freqs_sin():
        return shared_freqs_sin.clone()

    def init_cmp_freqs_cos():
        return shared_cmp_freqs_cos.clone()

    def init_cmp_freqs_sin():
        return shared_cmp_freqs_sin.clone()

    # Quant-faithful CSA (ratio-4) main compressor fixtures (mean l8/l32 of extract_weights_flash):
    # zero-mean Gaussian BF16 weights at the measured std; RMSNorm gamma near the measured mean.
    # Mirrors decode_csa / decode_compressor_ratio4.
    def init_cmp_wkv():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0240

    def init_cmp_wgate():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0381

    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.1226

    def init_cmp_norm_w():
        return (
            0.9569
            + torch.randn(
                HEAD_DIM,
            )
            * 0.1916
        )

    state_table = _state_block_table(CSA_STATE_MAX_BLOCKS, CSA_STATE_PHYSICAL_BLOCKS)

    def init_compress_state_block_table():
        return state_table.clone().unsqueeze(0)

    def init_compress_state():
        state = torch.zeros(CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM)
        flat = state.view(-1, MAIN_COMPRESS_STATE_DIM)
        history_positions = torch.arange(max(0, context_len - MAIN_STATE_LEN), context_len, dtype=torch.int64)
        history_rows = paged_rows(state_table, history_positions, CSA_STATE_BLOCK_SIZE)
        history = (torch.rand(history_rows.numel(), MAIN_COMPRESS_STATE_DIM) - 0.5) * 0.05
        flat.index_copy_(0, history_rows, history)
        return state

    def init_hadamard_idx():
        h = torch.ones((1, 1))
        while h.shape[0] < IDX_HEAD_DIM:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        return h * (IDX_HEAD_DIM**-0.5)

    # Quant-faithful indexer inner compressor fixtures (mean l8/l32 of extract_weights_flash):
    # zero-mean Gaussian BF16 weights at the measured std; RMSNorm gamma near the measured mean.
    # Mirrors decode_csa / decode_indexer.
    def init_inner_wkv():
        return torch.randn(INNER_OUT_DIM, D) * 0.0270

    def init_inner_wgate():
        return torch.randn(INNER_OUT_DIM, D) * 0.0513

    def init_inner_ape():
        return torch.randn(COMPRESS_RATIO, INNER_OUT_DIM) * 0.1524

    def init_inner_norm_w():
        return (
            0.6903
            + torch.randn(
                IDX_HEAD_DIM,
            )
            * 0.2663
        )

    inner_state_table = _state_block_table(
        INNER_STATE_MAX_BLOCKS,
        CSA_INNER_STATE_PHYSICAL_BLOCKS,
    )

    def init_inner_compress_state_block_table():
        return inner_state_table.clone().unsqueeze(0)

    def init_inner_compress_state():
        state = torch.zeros(INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM)
        flat = state.view(-1, INNER_COMPRESS_STATE_DIM)
        history_positions = torch.arange(max(0, context_len - INNER_STATE_LEN), context_len, dtype=torch.int64)
        history_rows = paged_rows(inner_state_table, history_positions, INNER_STATE_BLOCK_SIZE)
        history = (torch.rand(history_rows.numel(), INNER_COMPRESS_STATE_DIM) - 0.5) * 0.05
        flat.index_copy_(0, history_rows, history)
        return state

    # C8 historical index cache: completed compressed slots hold INT8 + a per-position dequant scale.
    # Build both from one bf16-rounded random draw so cache and scale stay consistent.
    _idx_hist = {}

    def _build_idx_hist():
        if "cache" in _idx_hist:
            return
        cache_i8 = torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM, dtype=torch.int8)
        scale = torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1)
        c_flat = cache_i8.view(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, IDX_HEAD_DIM)
        s_flat = scale.view(IDX_CACHE_BLOCK_NUM * BLOCK_SIZE, 1)
        completed = context_len // COMPRESS_RATIO
        for chunk_start in range(0, completed, history_chunk_rows):
            chunk_end = min(chunk_start + history_chunk_rows, completed)
            logical_slots = torch.arange(chunk_start, chunk_end, dtype=torch.int64)
            physical_rows = paged_rows(idx_block_table, logical_slots)
            history = ((torch.rand(chunk_end - chunk_start, IDX_HEAD_DIM) - 0.5) * 0.05).to(torch.bfloat16)
            history_i8, history_scale = int8_quant_per_row(history)
            c_flat.index_copy_(0, physical_rows, history_i8)
            s_flat.index_copy_(0, physical_rows, history_scale)
        _idx_hist["cache"] = cache_i8
        _idx_hist["scale"] = scale

    def init_idx_kv_cache():
        _build_idx_hist()
        return _idx_hist["cache"].clone()

    def init_idx_kv_scale():
        _build_idx_hist()
        return _idx_hist["scale"].clone()

    def init_kv_cache():
        cache = torch.zeros(CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
        cache_flat = cache.view(CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        start = max(0, context_len - WIN)
        scatter_bf16_history(cache_flat, ori_block_table, start, context_len, 0.1)
        return cache

    def init_ori_block_table():
        return ori_block_table.clone().unsqueeze(0)

    def init_ori_slot_mapping():
        return paged_rows(ori_block_table, token_pos().to(torch.int64))

    def init_cmp_kv():
        cache = torch.zeros(CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
        cache_flat = cache.view(CSA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        completed = context_len // COMPRESS_RATIO
        scatter_bf16_history(cache_flat, cmp_block_table, 0, completed, 0.1)
        return cache

    def init_cmp_block_table():
        return cmp_block_table.clone().unsqueeze(0)

    def init_idx_block_table():
        return idx_block_table.clone().unsqueeze(0)

    def init_position_ids():
        return token_pos()

    def init_local_request_ids():
        return torch.zeros(token_count, dtype=torch.int32)

    def init_cmp_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        token_ids, cmp_slots = cmp_write_records()
        mapping.index_copy_(0, token_ids, paged_rows(cmp_block_table, cmp_slots))
        return mapping

    def init_idx_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        token_ids, cmp_slots = cmp_write_records()
        mapping.index_copy_(0, token_ids, paged_rows(idx_block_table, cmp_slots))
        return mapping

    def init_state_slot_mapping():
        return paged_rows(state_table, token_pos().to(torch.int64), CSA_STATE_BLOCK_SIZE)

    def init_inner_state_slot_mapping():
        return paged_rows(inner_state_table, token_pos().to(torch.int64), INNER_STATE_BLOCK_SIZE)

    def init_attn_sink():
        return torch.zeros(H)

    def init_wo_a():
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN**-0.5

    def init_wo_b():
        return (torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel_local(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)
    # Indexer Q up-proj + weights projection (mirrors the standalone prefill_indexer fixtures).
    idx_wq_b_i8_T, idx_wq_b_scale = gen_shared_weight(
        (IDX_N_HEADS * IDX_HEAD_DIM, Q_LORA), dequant_std=0.108, chan_cv=0.56
    )
    idx_wq_b_i8 = idx_wq_b_i8_T.t().contiguous()

    return [
        TensorSpec("x_hc", [token_count, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("query_start_loc", [2], torch.int32, init_value=torch.tensor([0, token_count], dtype=torch.int32)),
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
        TensorSpec("freqs_cos", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_freqs_cos", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_cos),
        TensorSpec("cmp_freqs_sin", [token_count, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cmp_freqs_sin),
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec(
            "compress_state",
            [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [1, CSA_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec(
            "hadamard_idx", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard_idx
        ),
        TensorSpec(
            "idx_wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: idx_wq_b_i8
        ),
        TensorSpec(
            "idx_wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: idx_wq_b_scale
        ),
        TensorSpec(
            "idx_weights_proj",
            [D, IDX_N_HEADS],
            torch.bfloat16,
            init_value=lambda: (torch.randn(D, IDX_N_HEADS) * 0.2218).to(torch.bfloat16),
        ),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec(
            "inner_compress_state",
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_inner_compress_state,
        ),
        TensorSpec(
            "inner_compress_state_block_table",
            [1, INNER_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_inner_compress_state_block_table,
        ),
        TensorSpec(
            "kv_cache",
            [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_kv_cache,
        ),
        TensorSpec("ori_block_table", [1, SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("ori_slot_mapping", [token_count], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec(
            "cmp_kv",
            [CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
        ),
        TensorSpec("cmp_block_table", [1, SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec(
            "idx_kv_cache",
            [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM],
            torch.int8,
            init_value=init_idx_kv_cache,
        ),
        TensorSpec(
            "idx_kv_scale",
            [IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1],
            torch.float32,
            init_value=init_idx_kv_scale,
        ),
        TensorSpec("idx_block_table", [1, IDX_CACHE_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
        TensorSpec("position_ids", [token_count], torch.int32, init_value=init_position_ids),
        TensorSpec("local_request_ids", [token_count], torch.int32, init_value=init_local_request_ids),
        TensorSpec("cmp_slot_mapping", [token_count], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("idx_slot_mapping", [token_count], torch.int64, init_value=init_idx_slot_mapping),
        TensorSpec("state_slot_mapping", [token_count], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec(
            "inner_state_slot_mapping",
            [token_count],
            torch.int64,
            init_value=init_inner_state_slot_mapping,
        ),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [token_count, HC_MULT, D], torch.float32),
    ]


def _quant_w_per_output_channel_local(w):
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return w_i32.to(torch.float16).to(torch.int8), (1.0 / scale_quant).float()


@pl.jit.inline
def prefill_attention_csa_cp_core(
    x_normed_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    x_normed_full: pl.Tensor[[CP_KV_T_DYN, D], pl.BF16],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos_local: pl.Tensor[[CP_Q_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_local: pl.Tensor[[CP_Q_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_cos_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[
        [MAIN_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    idx_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    inner_state_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
    o_proj_weight_dep: pl.Scalar[pl.TASK_ID],
):
    """CSA attention body: local queries/indexer and replicated full KV state."""
    q_dim = pl.tensor.dim(x_normed_local, 0)
    kv_dim = pl.tensor.dim(x_normed_full, 0)

    q_cos_il = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    q_sin_signed = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    q_swap_idx = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_prepare(freqs_cos_local, freqs_sin_local, q_cos_il, q_sin_signed, q_swap_idx)

    q = pl.create_tensor([q_dim, H, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([q_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([q_dim, 1], dtype=pl.FP32)
    q_proj_rope(x_normed_local, wq_a, wq_b, wq_b_scale, gamma_cq, q_cos_il, q_sin_signed, q_swap_idx, q, qr, qr_scale)

    kv_cos_il = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    kv_sin_signed = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    kv_swap_idx = pl.create_tensor([kv_dim, ROPE_HEAD_DIM], dtype=pl.INT32)
    rope_prepare(freqs_cos_full, freqs_sin_full, kv_cos_il, kv_sin_signed, kv_swap_idx)

    kv_full = pl.create_tensor([kv_dim, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x_normed_full, wkv, gamma_ckv, kv_cos_il, kv_sin_signed, kv_swap_idx, kv_full, late_dep)

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    ori_cache_rows = ori_block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [ori_cache_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_cp_cache_write"):
        for write_t in pl.range(kv_dim):
            write_row_raw = pl.read(ori_slot_mapping_full, [write_t])
            if write_row_raw >= 0:
                write_row = pl.cast(write_row_raw, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, :] = kv_full[write_t : write_t + 1, :]

    compressor_ratio4(
        x_normed_full,
        query_start_loc,
        compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_freqs_cos_full, cmp_freqs_sin_full,
        cmp_kv,
        position_ids_full, cmp_slot_mapping_full, state_slot_mapping_full,
    )

    # Gathered-stream indexer cache with global positions and mappings.
    indexer_cache_ready = pl.array.create(1, pl.TASK_ID)
    prefill_indexer_compressor(
        x_normed_full,
        query_start_loc,
        inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        cmp_freqs_cos_full, cmp_freqs_sin_full,
        hadamard_idx,
        idx_kv_cache, idx_kv_scale,
        position_ids_full, idx_slot_mapping_full, inner_state_slot_mapping_full,
        indexer_cache_ready,
    )

    # Local indexer queries with half-width FP32 RoPE.
    idx_cos = pl.create_tensor([q_dim, HALF_ROPE], dtype=pl.FP32)
    idx_sin = pl.create_tensor([q_dim, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_cp_idx_halfrope"):
        for idx_t in pl.range(q_dim):
            idx_cos_bf16 = freqs_cos_local[idx_t : idx_t + 1, 0:HALF_ROPE]
            idx_cos_fp32 = pl.cast(idx_cos_bf16, target_type=pl.FP32)
            idx_cos[idx_t : idx_t + 1, 0:HALF_ROPE] = idx_cos_fp32
            idx_sin_bf16 = freqs_sin_local[idx_t : idx_t + 1, 0:HALF_ROPE]
            idx_sin_fp32 = pl.cast(idx_sin_bf16, target_type=pl.FP32)
            idx_sin[idx_t : idx_t + 1, 0:HALF_ROPE] = idx_sin_fp32

    cmp_topk_indices = pl.create_tensor([q_dim, IDX_TOPK], dtype=pl.INT32)
    cmp_topk_indices = prefill_indexer_query(
        x_normed_local, qr, qr_scale,
        idx_wq_b, idx_wq_b_scale, idx_weights_proj,
        idx_cos, idx_sin,
        hadamard_idx,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        cmp_topk_indices,
        position_ids_local, local_request_ids,
        indexer_cache_ready,
    )

    swa_indices = pl.create_tensor([q_dim, WIN], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([q_dim, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    csa_topk_blocks = (q_dim + CSA_TOPK_TOKEN_TILE - 1) // CSA_TOPK_TOKEN_TILE
    for topk_block in pl.spmd(csa_topk_blocks, name_hint="prefill_csa_cp_sparse_idx_tile"):
        topk_t0 = topk_block * CSA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(CSA_TOPK_TOKEN_TILE):
            t_idx = topk_t0 + topk_dt
            swa_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            mask_row = pl.full([1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
            if t_idx < q_dim:
                request_id = pl.read(local_request_ids, [t_idx])
                if request_id >= 0:
                    abs_pos = pl.read(position_ids_local, [t_idx])
                    visible_cmp = pl.min((abs_pos + 1) // COMPRESS_RATIO, pl.cast(IDX_TOPK, pl.INT32))
                    for mask_sb in pl.unroll(PREFILL_ATTN_BLOCKS):
                        cmp_lo = pl.max(mask_sb * PREFILL_ATTN_TILE - WIN, pl.cast(0, pl.INT32))
                        cmp_hi_unclamped = (mask_sb + 1) * PREFILL_ATTN_TILE - WIN
                        cmp_hi_cap = pl.cast(SPARSE_CMP_BIAS_COLS, pl.INT32)
                        cmp_hi = pl.min(cmp_hi_unclamped, cmp_hi_cap)
                        if cmp_lo < cmp_hi:
                            if visible_cmp > cmp_lo:
                                pl.write(mask_row, [0, mask_sb], pl.cast(1, pl.INT32))
                    window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                    key_start_abs = abs_pos + 1 - window_valid
                    for win_col in pl.range(WIN):
                        win_col_i32 = pl.cast(win_col, pl.INT32)
                        if win_col_i32 < window_valid:
                            key_abs = key_start_abs + win_col_i32
                            blk_slot = key_abs // BLOCK_SIZE
                            blk = pl.read(ori_block_table, [request_id, pl.cast(blk_slot, pl.INDEX)])
                            if blk >= 0:
                                row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                                pl.write(swa_row, [0, win_col], row)
                                pl.write(mask_row, [0, win_col // PREFILL_ATTN_TILE], pl.cast(1, pl.INT32))
                swa_indices[t_idx : t_idx + 1, 0:WIN] = swa_row
                valid_block_mask[t_idx : t_idx + 1, 0:VALID_BLOCK_MASK_COLS] = mask_row

    attn_out_local = sparse_attn_physical(
        q,
        kv_cache, swa_indices,
        cmp_kv, cmp_block_table, local_request_ids, cmp_topk_indices,
        valid_block_mask, attn_sink,
        freqs_cos_local, freqs_sin_local,
        wo_a, wo_b, wo_b_scale,
        attn_out_local,
        o_proj_weight_dep,
    )
    return attn_out_local


@pl.jit.inline
def prefill_attention_csa_cp(
    x_hc_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
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
    freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    idx_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    inner_state_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_local: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    x_out_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_order_fence: pl.Tensor[[1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    weight_epoch: pl.Scalar[pl.INT32],
):
    """DSA-CP CSA with replicated HC state at both layer boundaries."""
    o_proj_weight_dep = gather_o_proj_full_weights(
        wo_a_local, wo_b_local,
        wo_a_full, wo_b_full,
        o_proj_wo_a_window, o_proj_wo_b_window,
        o_proj_weight_ready, o_proj_weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, weight_epoch,
    )
    q_dim = pl.tensor.dim(position_ids_local, 0)
    kv_dim = pl.tensor.dim(x_hc_full, 0)

    x_mixed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    post_full = pl.create_tensor([kv_dim, HC_MULT], dtype=pl.FP32)
    comb_full = pl.create_tensor([kv_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc_full, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed_full, post_full, comb_full)

    x_normed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed_full, attn_norm_w, x_normed_full)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    local_base = tp_rank * q_dim
    x_normed_local = pl.create_tensor([q_dim, D], dtype=pl.BF16)
    freqs_cos_local = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    freqs_sin_local = pl.create_tensor([q_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    for local_row in pl.spmd(q_dim, name_hint="prefill_csa_cp_query_slice"):
        full_row = local_base + local_row
        query_row = pl.load(x_normed_full, [full_row, 0], [1, D], target_memory=pl.MemorySpace.Vec)
        pl.store(query_row, [local_row, 0], x_normed_local)
        cos_row = pl.load(freqs_cos, [full_row, 0], [1, ROPE_HEAD_DIM], target_memory=pl.MemorySpace.Vec)
        sin_row = pl.load(freqs_sin, [full_row, 0], [1, ROPE_HEAD_DIM], target_memory=pl.MemorySpace.Vec)
        pl.store(cos_row, [local_row, 0], freqs_cos_local)
        pl.store(sin_row, [local_row, 0], freqs_sin_local)

    attn_out_local = pl.create_tensor([q_dim, D], dtype=pl.BF16)
    attn_out_local = prefill_attention_csa_cp_core(
        x_normed_local, x_normed_full, query_start_loc,
        wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv,
        freqs_cos_local, freqs_sin_local,
        freqs_cos, freqs_sin,
        cmp_freqs_cos, cmp_freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        hadamard_idx,
        idx_wq_b, idx_wq_b_scale, idx_weights_proj,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        inner_compress_state, inner_compress_state_block_table,
        kv_cache, ori_block_table, ori_slot_mapping_full,
        cmp_kv, cmp_block_table,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        position_ids_local, position_ids_full, local_request_ids,
        cmp_slot_mapping_full, idx_slot_mapping_full,
        state_slot_mapping_full, inner_state_slot_mapping_full,
        attn_sink,
        wo_a_full, wo_b_full, wo_b_scale,
        attn_out_local,
        late_dep, o_proj_weight_dep,
    )

    attn_out_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    attn_out_full, gather_signal = prefill_cp_token_allgather_step(
        attn_out_local, attn_out_full,
        gather_window, gather_signal,
        group_base, tp_rank,
    )

    hc_post(attn_out_full, x_hc_full, post_full, comb_full, x_out_full)
    return x_out_full, gather_signal


@pl.jit
def prefill_attention_csa_cp_test(
    x_hc_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
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
    freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32]
    ],
    inner_compress_state_block_table: pl.Tensor[[REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    idx_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    inner_state_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out_full: pl.Out[pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Run one CP rank's query share with replicated CSA layer boundaries."""
    x_hc_full.bind_dynamic(0, CP_KV_T_DYN)
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    freqs_cos.bind_dynamic(0, CP_KV_T_DYN)
    freqs_sin.bind_dynamic(0, CP_KV_T_DYN)
    cmp_freqs_cos.bind_dynamic(0, CP_KV_T_DYN)
    cmp_freqs_sin.bind_dynamic(0, CP_KV_T_DYN)
    compress_state.bind_dynamic(0, MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    inner_compress_state.bind_dynamic(0, INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    ori_block_table.bind_dynamic(0, REQUESTS_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    idx_kv_cache.bind_dynamic(0, IDX_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(0, IDX_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(0, REQUESTS_DYN)
    ori_slot_mapping_full.bind_dynamic(0, CP_KV_T_DYN)
    position_ids_local.bind_dynamic(0, CP_Q_T_DYN)
    local_request_ids.bind_dynamic(0, CP_Q_T_DYN)
    position_ids_full.bind_dynamic(0, CP_KV_T_DYN)
    cmp_slot_mapping_full.bind_dynamic(0, CP_KV_T_DYN)
    idx_slot_mapping_full.bind_dynamic(0, CP_KV_T_DYN)
    state_slot_mapping_full.bind_dynamic(0, CP_KV_T_DYN)
    inner_state_slot_mapping_full.bind_dynamic(0, CP_KV_T_DYN)
    x_out_full.bind_dynamic(0, CP_KV_T_DYN)
    wo_a_full = pl.create_tensor([O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], dtype=pl.BF16)
    wo_b_full = pl.create_tensor([O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], dtype=pl.INT8)
    o_proj_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_csa_o_proj_order_init"):
        pl.write(o_proj_order_fence, [0], pl.cast(0, pl.INT32))
    x_out_full, gather_signal = prefill_attention_csa_cp(
        x_hc_full,
        query_start_loc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w,
        wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_freqs_cos, cmp_freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        hadamard_idx,
        idx_wq_b, idx_wq_b_scale,
        idx_weights_proj,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        inner_compress_state, inner_compress_state_block_table,
        kv_cache, ori_block_table, ori_slot_mapping_full,
        cmp_kv, cmp_block_table,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        position_ids_local, position_ids_full, local_request_ids,
        cmp_slot_mapping_full, idx_slot_mapping_full, state_slot_mapping_full,
        inner_state_slot_mapping_full,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        wo_a_full, wo_b_full,
        x_out_full,
        gather_window, gather_signal,
        o_proj_wo_a_window, o_proj_wo_b_window,
        o_proj_weight_ready, o_proj_weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, pl.const(1, pl.INT32),
    )
    return x_out_full


@pl.jit.host
def l3_prefill_attention_csa_cp(
    x_hc_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[TP_SIZE, QUERY_START_LOC_DYN], pl.INT32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[TP_SIZE, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[TP_SIZE, COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[TP_SIZE, MAIN_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[TP_SIZE, REQUESTS_DYN, CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[TP_SIZE, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[TP_SIZE, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[TP_SIZE, D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[TP_SIZE, INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[TP_SIZE, INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[TP_SIZE, COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[TP_SIZE, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [TP_SIZE, INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM], pl.FP32
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[[TP_SIZE, REQUESTS_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[TP_SIZE, REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT64],
    cmp_kv: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[TP_SIZE, REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[TP_SIZE, IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[TP_SIZE, IDX_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[TP_SIZE, REQUESTS_DYN, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[TP_SIZE, CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[TP_SIZE, CP_Q_T_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT64],
    idx_slot_mapping_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT64],
    inner_state_slot_mapping_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    x_out_full: pl.Out[pl.Tensor[[TP_SIZE, CP_KV_T_DYN, HC_MULT, D], pl.FP32]],
):
    """Launch one CP group's CSA block, one child per rank."""
    x_hc_full.bind_dynamic(1, CP_KV_T_DYN)
    query_start_loc.bind_dynamic(1, QUERY_START_LOC_DYN)
    freqs_cos.bind_dynamic(1, CP_KV_T_DYN)
    freqs_sin.bind_dynamic(1, CP_KV_T_DYN)
    cmp_freqs_cos.bind_dynamic(1, CP_KV_T_DYN)
    cmp_freqs_sin.bind_dynamic(1, CP_KV_T_DYN)
    compress_state.bind_dynamic(1, MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    inner_compress_state.bind_dynamic(1, INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    ori_block_table.bind_dynamic(1, REQUESTS_DYN)
    cmp_kv.bind_dynamic(1, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    idx_kv_cache.bind_dynamic(1, IDX_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(1, IDX_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(1, REQUESTS_DYN)
    ori_slot_mapping_full.bind_dynamic(1, CP_KV_T_DYN)
    position_ids_local.bind_dynamic(1, CP_Q_T_DYN)
    local_request_ids.bind_dynamic(1, CP_Q_T_DYN)
    position_ids_full.bind_dynamic(1, CP_KV_T_DYN)
    cmp_slot_mapping_full.bind_dynamic(1, CP_KV_T_DYN)
    idx_slot_mapping_full.bind_dynamic(1, CP_KV_T_DYN)
    state_slot_mapping_full.bind_dynamic(1, CP_KV_T_DYN)
    inner_state_slot_mapping_full.bind_dynamic(1, CP_KV_T_DYN)
    x_out_full.bind_dynamic(1, CP_KV_T_DYN)
    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_wo_a_window_buf = pld.alloc_window_buffer([O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16)
    o_proj_wo_b_window_buf = pld.alloc_window_buffer([O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8)
    o_proj_weight_ready_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_weight_consumed_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_wo_a_window = pld.window(
            o_proj_wo_a_window_buf, [O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16
        )
        o_proj_wo_b_window = pld.window(
            o_proj_wo_b_window_buf, [O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8
        )
        o_proj_weight_ready = pld.window(o_proj_weight_ready_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_weight_consumed = pld.window(o_proj_weight_consumed_buf, [TP_SIZE, 1], dtype=pl.INT32)
        prefill_attention_csa_cp_test(
            x_hc_full[rank],
            query_start_loc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            cmp_freqs_cos[rank], cmp_freqs_sin[rank],
            cmp_wkv[rank], cmp_wgate[rank], cmp_ape[rank], cmp_norm_w[rank],
            compress_state[rank], compress_state_block_table[rank],
            hadamard_idx[rank],
            idx_wq_b[rank], idx_wq_b_scale[rank],
            idx_weights_proj[rank],
            inner_wkv[rank], inner_wgate[rank], inner_ape[rank], inner_norm_w[rank],
            inner_compress_state[rank], inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping_full[rank],
            cmp_kv[rank], cmp_block_table[rank],
            idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank],
            position_ids_local[rank], position_ids_full[rank], local_request_ids[rank],
            cmp_slot_mapping_full[rank], idx_slot_mapping_full[rank], state_slot_mapping_full[rank],
            inner_state_slot_mapping_full[rank],
            attn_sink[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            x_out_full[rank],
            gather_window, gather_signal,
            o_proj_wo_a_window, o_proj_wo_b_window,
            o_proj_weight_ready, o_proj_weight_consumed,
            0, rank,
            device=rank,
        )



def build_cp_tensor_specs(
    start_pos: int = START_POS,
    token_count: int = PREFILL_SEQ,
    tp_size: int = TP_SIZE,
):
    """Replicate CSA layer boundaries and split query positions across the CP group."""
    import torch

    from golden import TensorSpec
    from prefill_cp_token_allgather import cp_stack, materialize_spec

    if tp_size != TP_SIZE:
        raise ValueError(f"tp_size={tp_size} must match import-time TP_SIZE={TP_SIZE}")
    if token_count % tp_size != 0:
        raise ValueError(f"token_count={token_count} must be a multiple of tp_size={tp_size}")
    local_t = token_count // tp_size

    full_names = (
        "ori_slot_mapping",
        "cmp_slot_mapping",
        "idx_slot_mapping",
        "state_slot_mapping",
        "inner_state_slot_mapping",
    )

    specs = []
    for spec in build_tensor_specs(start_pos, token_count):
        value = materialize_spec(spec)
        if spec.name == "x_hc":
            specs.append(TensorSpec(
                "x_hc_full", [tp_size, token_count, HC_MULT, D], spec.dtype,
                init_value=cp_stack(value, tp_size),
            ))
        elif spec.name in full_names:
            specs.append(TensorSpec(
                f"{spec.name}_full", [tp_size, token_count], spec.dtype,
                init_value=cp_stack(value, tp_size),
            ))
        elif spec.name == "position_ids":
            specs.append(TensorSpec(
                "position_ids_local", [tp_size, local_t], spec.dtype,
                init_value=value.reshape(tp_size, local_t).contiguous(),
            ))
            specs.append(TensorSpec(
                "position_ids_full", [tp_size, token_count], spec.dtype,
                init_value=cp_stack(value, tp_size),
            ))
        elif spec.name == "local_request_ids":
            specs.append(TensorSpec(
                "local_request_ids", [tp_size, local_t], spec.dtype,
                init_value=value.reshape(tp_size, local_t).contiguous(),
            ))
        elif spec.name == "wo_a":
            shards = [value[rank * O_PROJ_LOCAL_GROUPS : (rank + 1) * O_PROJ_LOCAL_GROUPS] for rank in range(tp_size)]
            specs.append(TensorSpec(
                "wo_a", [tp_size, O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], spec.dtype,
                init_value=torch.stack(shards).contiguous(),
            ))
        elif spec.name == "wo_b":
            shards = [value[:, rank * O_PROJ_LOCAL_COLS : (rank + 1) * O_PROJ_LOCAL_COLS] for rank in range(tp_size)]
            specs.append(TensorSpec(
                "wo_b", [tp_size, D, O_PROJ_LOCAL_COLS], spec.dtype,
                init_value=torch.stack(shards).contiguous(),
            ))
        elif spec.name == "x_out":
            specs.append(TensorSpec("x_out_full", [tp_size, token_count, HC_MULT, D], spec.dtype))
        else:
            specs.append(TensorSpec(
                spec.name, [tp_size, *spec.shape], spec.dtype,
                init_value=cp_stack(value, tp_size), 
            ))
    return specs


def build_ragged2_cp_tensor_specs(tp_size: int = TP_SIZE):
    """Build the two-request rank-crossing CSA fixture from the B1 CP specs."""
    import torch

    from golden import TensorSpec
    from prefill_cp_token_allgather import cp_stack
    from utils import (
        block_table as make_block_table,
        cache_row_from_table,
        compressed_slot_mapping,
        ori_slot_mapping as make_ori_slot_mapping,
        state_slot_mapping as make_state_slot_mapping,
        token_local_rope,
    )

    if tp_size != 2:
        raise ValueError(f"ragged2 requires tp_size=2, got {tp_size}")

    token_count = 8
    request_starts = (126, 30)
    request_positions = (
        torch.tensor([126, 127, 128], dtype=torch.int32),
        torch.tensor([30, 31, 32, 33], dtype=torch.int32),
    )
    position_ids = torch.cat((*request_positions, torch.zeros(1, dtype=torch.int32)))
    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
    request_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1, -1], dtype=torch.int32)

    ori_block_table = make_block_table(batch=2, table_blocks=SPARSE_ORI_MAX_BLOCKS, physical_blocks=CSA_ORI_BLOCK_NUM)
    cmp_block_table = make_block_table(batch=2, table_blocks=SPARSE_CMP_MAX_BLOCKS, physical_blocks=CSA_CMP_BLOCK_NUM)
    idx_block_table = make_block_table(batch=2, table_blocks=IDX_CACHE_MAX_BLOCKS, physical_blocks=IDX_CACHE_BLOCK_NUM)
    compress_state_block_table = make_block_table(
        batch=2, table_blocks=CSA_STATE_MAX_BLOCKS,
        physical_blocks=CSA_STATE_BLOCK_NUM,
    )
    inner_compress_state_block_table = make_block_table(
        batch=2, table_blocks=INNER_STATE_MAX_BLOCKS,
        physical_blocks=INNER_STATE_BLOCK_NUM,
    )

    ori_mappings = []
    cmp_mappings = []
    idx_mappings = []
    state_mappings = []
    inner_state_mappings = []
    state_size = CSA_STATE_BLOCK_SIZE
    inner_state_size = INNER_STATE_BLOCK_SIZE
    for request, positions in enumerate(request_positions):
        positions_2d = positions.unsqueeze(0)
        request_ori_table = ori_block_table[request : request + 1]
        request_cmp_table = cmp_block_table[request : request + 1]
        request_idx_table = idx_block_table[request : request + 1]
        request_state_table = compress_state_block_table[request : request + 1]
        inner_table = inner_compress_state_block_table[request : request + 1]
        ori_mapping = make_ori_slot_mapping(positions_2d, request_ori_table)
        cmp_mapping = compressed_slot_mapping(positions_2d, request_cmp_table, compress_ratio=COMPRESS_RATIO)
        idx_mapping = compressed_slot_mapping(positions_2d, request_idx_table, compress_ratio=COMPRESS_RATIO)
        state_mapping = make_state_slot_mapping(positions_2d, request_state_table, state_block_size=state_size)
        inner_state_mapping = make_state_slot_mapping(positions_2d, inner_table, state_block_size=inner_state_size)
        ori_mappings.append(ori_mapping.reshape(-1))
        cmp_mappings.append(cmp_mapping.reshape(-1))
        idx_mappings.append(idx_mapping.reshape(-1))
        state_mappings.append(state_mapping.reshape(-1))
        inner_state_mappings.append(inner_state_mapping.reshape(-1))
    pad_mapping = torch.full((1,), -1, dtype=torch.int64)
    ori_slot_mapping = torch.cat((*ori_mappings, pad_mapping))
    cmp_slot_mapping = torch.cat((*cmp_mappings, pad_mapping))
    idx_slot_mapping = torch.cat((*idx_mappings, pad_mapping))
    state_slot_mapping = torch.cat((*state_mappings, pad_mapping))
    inner_state_slot_mapping = torch.cat((*inner_state_mappings, pad_mapping))

    kv_cache = torch.zeros(CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    kv_cache_flat = kv_cache.view(CSA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for request, start_pos in enumerate(request_starts):
        for position in range(max(0, start_pos - WIN), start_pos):
            row = cache_row_from_table(ori_block_table[request], position)
            kv_cache_flat[row] = ((torch.rand(HEAD_DIM) - 0.5) * 0.1).to(torch.bfloat16)

    cmp_kv = torch.zeros(CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    idx_kv_cache = torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM, dtype=torch.int8)
    idx_kv_scale = torch.zeros(IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1, dtype=torch.float32)
    compress_state_shape = (CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM)
    inner_state_shape = (INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM)
    compress_state = torch.zeros(compress_state_shape, dtype=torch.float32)
    inner_compress_state = torch.zeros(inner_state_shape, dtype=torch.float32)
    compress_state_flat = compress_state.view(-1, MAIN_COMPRESS_STATE_DIM)
    inner_compress_state_flat = inner_compress_state.view(-1, INNER_COMPRESS_STATE_DIM)
    for request, start_pos in enumerate(request_starts):
        request_state_table = compress_state_block_table[request]
        inner_table = inner_compress_state_block_table[request]
        for position in range(max(0, start_pos - MAIN_STATE_LEN), start_pos):
            row = cache_row_from_table(request_state_table, position, block_size=state_size)
            compress_state_flat[row] = (torch.rand(MAIN_COMPRESS_STATE_DIM) - 0.5) * 0.05
        for position in range(max(0, start_pos - INNER_STATE_LEN), start_pos):
            row = cache_row_from_table(inner_table, position, block_size=inner_state_size)
            inner_compress_state_flat[row] = (torch.rand(INNER_COMPRESS_STATE_DIM) - 0.5) * 0.05

    freqs_cos, freqs_sin = token_local_rope(
        M, COMPRESS_RATIO, position_ids,
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )
    cmp_positions = torch.where(
        (position_ids + 1) % COMPRESS_RATIO == 0,
        position_ids - (COMPRESS_RATIO - 1),
        torch.zeros_like(position_ids),
    )
    cmp_freqs_cos, cmp_freqs_sin = token_local_rope(
        M, COMPRESS_RATIO, cmp_positions,
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )

    replacements = {
        "query_start_loc": cp_stack(query_start_loc, tp_size),
        "local_request_ids": request_ids.reshape(tp_size, token_count // tp_size).contiguous(),
        "freqs_cos": cp_stack(freqs_cos, tp_size),
        "freqs_sin": cp_stack(freqs_sin, tp_size),
        "cmp_freqs_cos": cp_stack(cmp_freqs_cos, tp_size),
        "cmp_freqs_sin": cp_stack(cmp_freqs_sin, tp_size),
        "compress_state": cp_stack(compress_state, tp_size),
        "compress_state_block_table": cp_stack(compress_state_block_table, tp_size),
        "inner_compress_state": cp_stack(inner_compress_state, tp_size),
        "inner_compress_state_block_table": cp_stack(inner_compress_state_block_table, tp_size),
        "kv_cache": cp_stack(kv_cache, tp_size),
        "ori_block_table": cp_stack(ori_block_table, tp_size),
        "ori_slot_mapping_full": cp_stack(ori_slot_mapping, tp_size),
        "cmp_kv": cp_stack(cmp_kv, tp_size),
        "cmp_block_table": cp_stack(cmp_block_table, tp_size),
        "idx_kv_cache": cp_stack(idx_kv_cache, tp_size),
        "idx_kv_scale": cp_stack(idx_kv_scale, tp_size),
        "idx_block_table": cp_stack(idx_block_table, tp_size),
        "position_ids_local": position_ids.reshape(tp_size, token_count // tp_size).contiguous(),
        "position_ids_full": cp_stack(position_ids, tp_size),
        "cmp_slot_mapping_full": cp_stack(cmp_slot_mapping, tp_size),
        "idx_slot_mapping_full": cp_stack(idx_slot_mapping, tp_size),
        "state_slot_mapping_full": cp_stack(state_slot_mapping, tp_size),
        "inner_state_slot_mapping_full": cp_stack(inner_state_slot_mapping, tp_size),
    }

    specs = []
    for spec in build_cp_tensor_specs(start_pos=0, token_count=token_count, tp_size=tp_size):
        value = replacements.get(spec.name)
        if value is None:
            specs.append(spec)
            continue
        replacement_spec = TensorSpec(
            spec.name, list(value.shape), spec.dtype, init_value=value,
            resident=spec.resident,
        )
        specs.append(replacement_spec)
    return specs


def golden_prefill_attention_csa_cp(tensors):
    """Run the single-die reference and replicate full outputs and caches across CP ranks."""
    import torch

    tp_size, token_count = tensors["x_hc_full"].shape[:2]

    shared = (
        "query_start_loc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin",
        "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
        "compress_state_block_table", "hadamard_idx", "idx_wq_b", "idx_wq_b_scale",
        "idx_weights_proj", "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
        "inner_compress_state_block_table", "ori_block_table", "cmp_block_table",
        "idx_block_table", "attn_sink", "wo_b_scale",
    )
    full = {name: tensors[name][0] for name in shared}
    full["wo_a"] = torch.cat([tensors["wo_a"][rank] for rank in range(tp_size)], dim=0)
    full["wo_b"] = torch.cat([tensors["wo_b"][rank] for rank in range(tp_size)], dim=1)
    full["x_hc"] = tensors["x_hc_full"][0]
    for name in ("compress_state", "inner_compress_state", "kv_cache", "cmp_kv",
                 "idx_kv_cache", "idx_kv_scale"):
        full[name] = tensors[name][0].clone()
    for name in ("ori_slot_mapping", "cmp_slot_mapping", "idx_slot_mapping",
                 "state_slot_mapping", "inner_state_slot_mapping"):
        full[name] = tensors[f"{name}_full"][0]
    full["position_ids"] = tensors["position_ids_full"][0]
    full["local_request_ids"] = tensors["local_request_ids"].reshape(token_count)
    full["x_out"] = torch.zeros(token_count, HC_MULT, D, dtype=torch.float32)

    golden_prefill_attention_csa(full)

    tensors["x_out_full"][:] = full["x_out"].unsqueeze(0).expand(tp_size, *full["x_out"].shape)
    for name in ("compress_state", "inner_compress_state", "kv_cache", "cmp_kv",
                 "idx_kv_cache", "idx_kv_scale"):
        tensors[name][:] = full[name].unsqueeze(0).expand(tp_size, *full[name].shape)


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, ratio_reldiff, run

    parser = argparse.ArgumentParser(
        description="Standalone DeepSeek V4 packed prefill CSA correctness test."
    )
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument(
        "-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)),
        help="comma-separated rank group; one ID at --tp 1",
    )
    parser.add_argument(
        "--tp", type=int, default=TP_SIZE, help="rank-group size; must match the import-time --tp"
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--start-pos", type=int, default=START_POS)
    parser.add_argument(
        "--token-count", "--num-tokens", dest="token_count", type=int, default=None,
        help=f"B1 physical query-token extent across the group; defaults to {PREFILL_SEQ}. ragged2 is fixed at 8.",
    )
    parser.add_argument(
        "--case", choices=["b1", "ragged2"], default="b1",
        help="Fixture case; ragged2 is the fixed two-request TP2 boundary case.",
    )
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    # High-prefix sparse-attention tolerance.
    x_out_diff_thd, x_out_max_diff = (8e-3, 2) if args.start_pos or args.case == "ragged2" else (5e-3, 1)
    cache_compare = {
        "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
        "inner_compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
        # INT8 quant-on-write: one LSB of rounding drift on a bounded row fraction.
        "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
        "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
    }

    if args.tp != TP_SIZE:
        raise SystemExit(f"--tp={args.tp} does not match import-time TP_SIZE={TP_SIZE}")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {device_ids}")
    if args.case == "ragged2" and TP_SIZE != 2:
        parser.error("--case ragged2 requires --tp 2")
    if args.case == "ragged2" and args.start_pos != 0:
        parser.error("--case ragged2 has fixed request starts and requires --start-pos 0")
    if args.token_count is None:
        args.token_count = 8 if args.case == "ragged2" else PREFILL_SEQ
    if args.case == "ragged2" and args.token_count != 8:
        parser.error("--case ragged2 has a fixed physical extent and requires --token-count 8")
    if args.case == "b1" and args.token_count % TP_SIZE != 0:
        parser.error(f"--token-count must be a multiple of --tp={TP_SIZE}, got {args.token_count}")

    if TP_SIZE == 1:
        result = run(
            fn=prefill_attention_csa_test,
            specs=build_tensor_specs(args.start_pos, args.token_count),
            golden_fn=golden_prefill_attention_csa,
            config=dict(
                dump_passes=args.dump_passes,
                platform=args.platform,
                device_id=device_ids[0],
                enable_chip_swimlane=args.enable_chip_swimlane,
                enable_dep_gen=args.enable_dep_gen,
            ),
            compile_only=args.compile_only,
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                "x_out": ratio_reldiff(diff_thd=x_out_diff_thd, pct_thd=0.005, max_diff_hd=x_out_max_diff),
                "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
                "inner_compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
                # INT8 quant-on-write: one LSB of rounding drift on a bounded row fraction.
                "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
                "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
            },
        )
    else:
        from pypto.ir.distributed_compiled_program import DistributedConfig

        specs = (
            build_ragged2_cp_tensor_specs(TP_SIZE)
            if args.case == "ragged2"
            else build_cp_tensor_specs(args.start_pos, args.token_count, TP_SIZE)
        )
        result = run(
            fn=l3_prefill_attention_csa_cp,
            specs=specs,
            golden_fn=golden_prefill_attention_csa_cp,
            config=dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
                platform=args.platform,
                ring_heap=PREFILL_RING_HEAP,
            ),
            compile_only=args.compile_only,
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                "x_out_full": ratio_reldiff(diff_thd=x_out_diff_thd, pct_thd=0.005, max_diff_hd=x_out_max_diff),
                **cache_compare,
            },
        )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
