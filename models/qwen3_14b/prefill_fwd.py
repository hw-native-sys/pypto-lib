# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim    # CI marker: full multi-layer forward — device-only, skip on *sim
"""Qwen3-14B full-layer prefill forward.

Each transformer layer runs the same fused prefill body: input RMSNorm,
Q/K/V projection, RoPE, KV cache update, causal attention, output projection,
post-attention RMSNorm, SwiGLU MLP, and the final residual path. The top-level
JIT loops over all layer rows in the flattened weight tensors, matching the
decode_fwd.py full-layer structure.

Dynamic batch design
--------------------
Every batch-dependent kernel signature dim is a `pl.dynamic(...)` variable
(`BATCH_DYN` / `PREFILL_TOKENS_DYN` / `KV_CACHE_ROWS_DYN` /
`BLOCK_TABLE_FLAT_DYN`), so a single compiled program serves any
`batch <= host KV-cache capacity`. Host allocates the input/output
token ids and slot mapping as packed token-major tensors with leading dim
`T = sum(chunk_lens)` (no `[batch, max_seq]` padding). The embedding table is
gathered on device before the first transformer layer. `seq_lens`
stores the absolute sequence length after the current chunk; `chunk_lens`
and `chunk_offsets` identify each batch row's slice inside the packed chunk.

Unlike the decode path, prefill bounds hidden-state lifetime by processing
128-token windows. Batch-1 uses a compact `[128, HIDDEN]` window so it does
not pay for fake batch rows; larger batches use one fixed `[BATCH_PAD * 128,
HIDDEN]` packed group per window to preserve the batch-16 prefill schedule.
The per-token `valid_tok` + `valid_shape` pattern still handles sequence-length
variation inside each window.
"""
import pypto.language as pl

from config import (
    QWEN3_14B_DIMS as D,
    QWEN3_14B_TILING as T,
    QWEN3_14B as M,
)
from rms_lm_head import rms_lm_head

BATCH_DYN = D.batch
KV_CACHE_ROWS_DYN = D.kv_cache_rows
BLOCK_TABLE_FLAT_DYN = D.block_table_flat
LAYER_DYN = D.layer
LAYER_HIDDEN_ROWS_DYN = D.layer_hidden_rows
LAYER_INTER_ROWS_DYN = D.layer_inter_rows
PREFILL_TOKENS_DYN = pl.dynamic("PREFILL_TOKENS_DYN")

BATCH_PAD = M.batch_pad
MODEL_MAX_SEQ = M.max_seq
NUM_HEADS = M.num_heads
NUM_KV_HEADS = M.num_kv_heads
HEAD_DIM = M.head_dim
HIDDEN = M.hidden
INTERMEDIATE = M.intermediate
KV_HIDDEN = M.kv_hidden
VOCAB = M.vocab
NUM_LAYERS = M.num_layers
EPS = M.eps
HIDDEN_INV = M.hidden_inv
HEAD_DIM_INV = M.head_dim_inv
ATTN_SCALE = M.attn_scale
HALF_DIM = M.half_dim
Q_PER_KV = M.q_per_kv
Q_HEAD_BATCH = M.q_head_batch
Q_HEAD_PAD = M.q_head_pad
Q_GROUPS = M.q_groups
TOTAL_Q_GROUPS = M.total_q_groups

# Single-layer prefill constants. Keep these local because config.py is shared
# with the decode kernels and uses decode-tuned tiling constants.
MAX_SEQ = MODEL_MAX_SEQ
DEFAULT_TEST_MAX_SEQ = 128
BATCH_TILE = 16
# Cube K/N tiles sized to fill the 512B L2 cache line on every GM->L1 (MTE2)
# load: bf16 => 256 elems * 2B = 512B. Below this the MTE2 DMA under-fills the
# line (128 elems = 256B -> 2x over-fetch, 64 elems = 128B -> 4x), which made the
# proj matmuls ~72% MTE2-bound. K_CHUNK is the activation inner span (q/k/v/gate/
# up) AND the down_proj weight-output inner span; the *_OUT_CHUNK are the weight
# N inner spans. TN=256/TK=256 clears the cache-line floor and fits Mat/L1
# (~384KB<512KB) and Acc/L0C (128KB==limit) at M=TOK_TILE=128
# (see cube-tile-tuning).
K_CHUNK = 256
Q_OUT_CHUNK = 256
KV_OUT_CHUNK = 256
TOK_TILE = 128
X_GAMMA_HIDDEN_CHUNK = 128
OUT_RESID_CHUNK = 64
SEQ_TILE = T.seq_tile
BLOCK_SIZE = T.block_size
EMBED_HIDDEN_CHUNK = K_CHUNK
TOKEN_EMBED_SPMD_BLOCKS = 48
ROPE_SPMD_BLOCKS = 32
ROPE_T_GROUP = 16
ROPE_Q_ROWS = ROPE_T_GROUP * Q_HEAD_BATCH
ROPE_READY_DEP_COUNT = 1
ROPE_INPUT_DEP_COUNT = 5
ATTN_TOK_GROUP = 8
ATTN_GI_GROUP = 1
FINALIZE_SPMD_BLOCKS = 48
FINALIZE_TOK_GROUP = TOK_TILE
Q_HEAD_BATCH_PAD = 16
ATTN_GI_SCORE_ROWS = ATTN_TOK_GROUP * ATTN_GI_GROUP * Q_HEAD_PAD
ATTN_GI_STAT_ROWS = ATTN_TOK_GROUP * ATTN_GI_GROUP * Q_HEAD_BATCH_PAD
ATTN_PHASE_MICRO_GROUPS = (FINALIZE_TOK_GROUP + ATTN_TOK_GROUP - 1) // ATTN_TOK_GROUP
ATTN_GI_BLOCKS = (TOTAL_Q_GROUPS + ATTN_GI_GROUP - 1) // ATTN_GI_GROUP
ATTN_PHASE_WORK_ITEMS = ATTN_PHASE_MICRO_GROUPS * ATTN_GI_BLOCKS
ATTN_PHASE_SPMD_BLOCKS = 24
ATTN_PHASE_SCORE_ROWS = ATTN_PHASE_WORK_ITEMS * ATTN_GI_SCORE_ROWS
ATTN_PHASE_STAT_ROWS = ATTN_PHASE_WORK_ITEMS * ATTN_GI_STAT_ROWS
ATTN_PHASE_ACC_SCORE_ROWS = ATTN_PHASE_MICRO_GROUPS * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_PAD
ATTN_PHASE_ACC_STAT_ROWS = ATTN_PHASE_MICRO_GROUPS * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_BATCH_PAD
ATTN_PHASE_FINALIZE_WORK_ITEMS = ATTN_PHASE_MICRO_GROUPS * ATTN_TOK_GROUP * TOTAL_Q_GROUPS
QKPV_PIPELINE_TOK_STEP = 1
QKPV_PLAN_SB_BATCH = 1
QKPV_PLAN_WORK_ITEMS = QKPV_PLAN_SB_BATCH * ATTN_PHASE_WORK_ITEMS
QKPV_PARTIAL_SCORE_ROWS = QKPV_PLAN_SB_BATCH * ATTN_PHASE_ACC_SCORE_ROWS
QKPV_PARTIAL_STAT_ROWS = QKPV_PLAN_SB_BATCH * ATTN_PHASE_ACC_STAT_ROWS
MLP_OUT_CHUNK = 256  # 512B cache line: gate/up weight-N inner + down_proj activation inner
LM_HEAD_K_CHUNK = 128
VOCAB_CHUNK = 64
DOWN_K_PARTS = 3
HIDDEN_BLOCKS = HIDDEN // K_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK
KV_OUT_BLOCKS = KV_HIDDEN // KV_OUT_CHUNK
X_GAMMA_HIDDEN_BLOCKS = HIDDEN // X_GAMMA_HIDDEN_CHUNK
OUT_RESID_BLOCKS = HIDDEN // OUT_RESID_CHUNK
MLP_OUT_BLOCKS = INTERMEDIATE // MLP_OUT_CHUNK
# Vector epilogues (silu, down+residual) are UB-bound, not cube-bound: a 256-wide
# fp32 frag overflows the 184KB Vec buffer. Decouple their frag from the 256 cube
# tiles (K_CHUNK / MLP_OUT_CHUNK) — they read/write GM intermediates column-wise,
# so a finer 128 frag is free of the cube's cache-line concern.
# Route B: the phase-major MLP weight matmuls tile M at 128 so each
# w_gate/w_up/w_down slab streams from HBM once and is reused across 128 rows in
# L1/L0. L0C caps M*N*4 <= 128KB, so N stays 256 (=MLP_OUT_CHUNK, keeps the
# 512B weight cache line) exactly at the wall. The
# vector epilogues (silu, down+resid) are UB-bound, so their frag drops to 64 to
# fit the 184KB Vec buffer at M=128.
MLP_M_TILE = 128
SILU_OUT_CHUNK = 64
SILU_OUT_BLOCKS = INTERMEDIATE // SILU_OUT_CHUNK
DOWN_RESID_CHUNK = 64
DOWN_RESID_BLOCKS = HIDDEN // DOWN_RESID_CHUNK
MLP_PROJ_BANDS = 4
MLP_BAND_BLOCKS = MLP_OUT_BLOCKS // MLP_PROJ_BANDS
MLP_BAND_WIDTH = MLP_BAND_BLOCKS * MLP_OUT_CHUNK
DOWN_PART_BLOCKS = (MLP_OUT_BLOCKS + DOWN_K_PARTS - 1) // DOWN_K_PARTS
DOWN_PART_WORK_ITEMS = HIDDEN_BLOCKS * DOWN_K_PARTS
RMSNORM_TOK_GROUP = 8
RMSNORM_TOK_GROUPS = (TOK_TILE + RMSNORM_TOK_GROUP - 1) // RMSNORM_TOK_GROUP
RMSNORM_WORK_ITEMS = RMSNORM_TOK_GROUPS
RMSNORM_SPMD_BLOCKS = 8
Q_PROJ_SPMD_BLOCKS = Q_OUT_BLOCKS
KV_PROJ_WORK_ITEMS = 2 * KV_OUT_BLOCKS
KV_PROJ_SPMD_BLOCKS = 8
QK_NORM_SPMD_BLOCKS = NUM_KV_HEADS
POST_RMSNORM_SPMD_BLOCKS = 8
DOWN_RESID_SPMD_BLOCKS = 20
OUT_PROJ_SPMD_BLOCKS = 20
OUT_RESID_SPMD_BLOCKS = 20
SILU_SPMD_BLOCKS = 24
# gate and up share ONE spmd so a core can pick up work from both projections.
# Each projection has MLP_BAND_BLOCKS (17) N-tiles over GATE_UP_SPMD_BLOCKS (24)
# cores. Within one band, shifting up by 17 maps up's active cores away from
# gate's first 17 active cores. Across the four bands, rotate the physical core
# start by 0/6/12/18 so the "2 matmul tiles" lanes move around: per MLP M-tile,
# cores carry either 5 or 6 gate/up matmuls instead of repeating the same 10
# heavy lanes every band.
GATE_UP_SPMD_BLOCKS = 24
UP_PROJ_CORE_SHIFT = MLP_BAND_BLOCKS % GATE_UP_SPMD_BLOCKS
MLP_BAND_CORE_ROT_STEP = GATE_UP_SPMD_BLOCKS // MLP_PROJ_BANDS
DOWN_PROJ_SPMD_BLOCKS = HIDDEN_BLOCKS

assert HIDDEN % EMBED_HIDDEN_CHUNK == 0
assert MLP_M_TILE == TOK_TILE
TOKEN_EMBED_HIDDEN_BLOCKS = HIDDEN // EMBED_HIDDEN_CHUNK


@pl.jit.inline(auto_scope=False)
def _attention_phase_window(
    attn_tile: pl.Tensor[[TOK_TILE, HIDDEN], pl.BF16],
    all_q_padded_tile: pl.Tensor[[TOK_TILE * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    cur_li_phase: pl.Tensor[[ATTN_PHASE_ACC_STAT_ROWS, 1], pl.FP32],
    oi_tmp_phase: pl.Tensor[[ATTN_PHASE_ACC_SCORE_ROWS, HEAD_DIM], pl.FP32],
    b: pl.Scalar[pl.INT32],
    max_blocks_per_seq: pl.Scalar[pl.INT32],
    layer_cache_base: pl.Scalar[pl.INT32],
    chunk_start: pl.Scalar[pl.INT32],
    p0: pl.Scalar[pl.INT32],
    final_ti0: pl.Scalar[pl.INT32],
    finalize_tok: pl.Scalar[pl.INT32],
    rope_ready_deps: pl.Array[ROPE_READY_DEP_COUNT, pl.TASK_ID],
) -> tuple[
    pl.Tensor[[TOK_TILE, HIDDEN], pl.BF16],
    pl.Tensor[[ATTN_PHASE_ACC_STAT_ROWS, 1], pl.FP32],
    pl.Tensor[[ATTN_PHASE_ACC_SCORE_ROWS, HEAD_DIM], pl.FP32],
]:
    cache_token_rows = pl.tensor.dim(k_cache, 0) // NUM_KV_HEADS
    k_cache_bsnd = pl.reshape(k_cache, [cache_token_rows, KV_HIDDEN])
    v_cache_bsnd = pl.reshape(v_cache, [cache_token_rows, KV_HIDDEN])
    cur_mi_phase = pl.create_tensor([ATTN_PHASE_ACC_STAT_ROWS, 1], dtype=pl.FP32)
    if finalize_tok > 0:
        block_ctx_len = chunk_start + p0 + final_ti0 + finalize_tok
        block_ctx_blocks = (block_ctx_len + SEQ_TILE - 1) // SEQ_TILE
        qkpv_order = pl.create_tensor([QKPV_PLAN_WORK_ITEMS], dtype=pl.INT32)
        qkpv_wcur = pl.create_tensor([1], dtype=pl.INT32)
        partial_mi_phase = pl.create_tensor([QKPV_PARTIAL_STAT_ROWS, 1], dtype=pl.FP32)
        partial_li_phase = pl.create_tensor([QKPV_PARTIAL_STAT_ROWS, 1], dtype=pl.FP32)
        partial_oi_phase = pl.create_tensor([QKPV_PARTIAL_SCORE_ROWS, HEAD_DIM], dtype=pl.FP32)
        for sb_chunk in pl.range(0, block_ctx_blocks, QKPV_PLAN_SB_BATCH):
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="qkpv_plan",
                allow_early_resolve=True,
            ):
                pl.write(qkpv_wcur, [0], pl.cast(0, pl.INT32))
                for plan_si in pl.unroll(QKPV_PLAN_SB_BATCH):
                    plan_sb = sb_chunk + plan_si
                    if plan_sb < block_ctx_blocks:
                        for plan_work_id in pl.unroll(ATTN_PHASE_WORK_ITEMS):
                            plan_micro_id = plan_work_id // ATTN_GI_BLOCKS
                            plan_attn_dt0 = plan_micro_id * ATTN_TOK_GROUP
                            if plan_attn_dt0 < finalize_tok:
                                plan_w = pl.read(qkpv_wcur, [0])
                                pl.write(
                                    qkpv_order,
                                    [plan_w],
                                    pl.cast(plan_si * ATTN_PHASE_WORK_ITEMS + plan_work_id, pl.INT32),
                                )
                                pl.write(qkpv_wcur, [0], pl.cast(plan_w + 1, pl.INT32))
                for plan_si in pl.unroll(QKPV_PLAN_SB_BATCH):
                    plan_sb = sb_chunk + plan_si
                    for plan_work_id in pl.unroll(ATTN_PHASE_WORK_ITEMS):
                        plan_micro_id = plan_work_id // ATTN_GI_BLOCKS
                        plan_attn_dt0 = plan_micro_id * ATTN_TOK_GROUP
                        if plan_sb >= block_ctx_blocks:
                            plan_w = pl.read(qkpv_wcur, [0])
                            pl.write(
                                qkpv_order,
                                [plan_w],
                                pl.cast(plan_si * ATTN_PHASE_WORK_ITEMS + plan_work_id, pl.INT32),
                            )
                            pl.write(qkpv_wcur, [0], pl.cast(plan_w + 1, pl.INT32))
                        else:
                            if plan_attn_dt0 >= finalize_tok:
                                plan_w = pl.read(qkpv_wcur, [0])
                                pl.write(
                                    qkpv_order,
                                    [plan_w],
                                    pl.cast(plan_si * ATTN_PHASE_WORK_ITEMS + plan_work_id, pl.INT32),
                                )
                                pl.write(qkpv_wcur, [0], pl.cast(plan_w + 1, pl.INT32))

            with pl.spmd(
                ATTN_PHASE_SPMD_BLOCKS,
                name_hint="qk_pv_partial_plan_spmd",
                deps=[rope_ready_deps[0]],
                sync_start=True,
            ) as qkpv_tid:
                phase_core = pl.tile.get_block_idx()
                qkpv_lane_iters = (
                    QKPV_PLAN_WORK_ITEMS - phase_core + ATTN_PHASE_SPMD_BLOCKS - 1
                ) // ATTN_PHASE_SPMD_BLOCKS
                for qkpv_it in pl.range(qkpv_lane_iters):
                    qkpv_flat = phase_core + qkpv_it * ATTN_PHASE_SPMD_BLOCKS
                    qkpv_item = pl.cast(pl.read(qkpv_order, [qkpv_flat]), pl.INDEX)
                    si = qkpv_item // ATTN_PHASE_WORK_ITEMS
                    work_id = qkpv_item - si * ATTN_PHASE_WORK_ITEMS
                    sb = sb_chunk + si
                    if sb < block_ctx_blocks:
                        micro_id = work_id // ATTN_GI_BLOCKS
                        gi_block = work_id - micro_id * ATTN_GI_BLOCKS
                        gi0 = gi_block * ATTN_GI_GROUP
                        attn_dt0 = micro_id * ATTN_TOK_GROUP
                        if attn_dt0 < finalize_tok:
                            attn_ti0 = final_ti0 + attn_dt0
                            attn_tok = pl.min(ATTN_TOK_GROUP, finalize_tok - attn_dt0)
                            for gg in pl.range(ATTN_GI_GROUP):
                                gi = gi0 + gg
                                if gi < TOTAL_Q_GROUPS:
                                    kvh = gi // Q_GROUPS
                                    block_table_idx = b * max_blocks_per_seq + sb
                                    pbid = pl.cast(
                                        pl.tensor.read(block_table, [block_table_idx]),
                                        pl.INDEX,
                                    )
                                    cache_row0 = layer_cache_base + pbid * BLOCK_SIZE
                                    cache_col = kvh * HEAD_DIM
                                    k_tile = pl.slice(k_cache_bsnd, [SEQ_TILE, HEAD_DIM], [cache_row0, cache_col])
                                    v_tile = pl.slice(v_cache_bsnd, [SEQ_TILE, HEAD_DIM], [cache_row0, cache_col])
                                    for dd in pl.pipeline(ATTN_TOK_GROUP, stage=3):
                                        if dd < attn_tok:
                                            ti = attn_ti0 + dd
                                            chunk_pos = p0 + ti
                                            pos = chunk_start + chunk_pos
                                            ctx_len = pos + 1
                                            ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                                            if sb < ctx_blocks:
                                                q_row0 = ti * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                                                q_padded = pl.slice(
                                                    all_q_padded_tile,
                                                    [Q_HEAD_PAD, HEAD_DIM],
                                                    [q_row0, 0],
                                                )
                                                raw_scores = pl.matmul(
                                                    q_padded,
                                                    k_tile,
                                                    b_trans=True,
                                                    out_dtype=pl.FP32,
                                                )
                                                s0 = sb * SEQ_TILE
                                                valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                                                scores = pl.fillpad(
                                                    pl.set_validshape(
                                                        pl.mul(raw_scores, ATTN_SCALE),
                                                        Q_HEAD_BATCH,
                                                        valid_len,
                                                    ),
                                                    pad_value=pl.PadValue.min,
                                                )
                                                cur_mi = pl.row_max(scores)
                                                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                                                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                                                cur_li = pl.row_sum(
                                                    pl.cast(exp_scores_bf16, target_type=pl.FP32),
                                                )
                                                oi_tmp = pl.matmul(
                                                    exp_scores_bf16,
                                                    v_tile,
                                                    out_dtype=pl.FP32,
                                                )
                                                acc_exp_row0 = (
                                                    si * ATTN_PHASE_ACC_SCORE_ROWS
                                                    + micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_PAD
                                                    + gi * ATTN_TOK_GROUP * Q_HEAD_PAD
                                                    + dd * Q_HEAD_PAD
                                                )
                                                acc_li_row0 = (
                                                    si * ATTN_PHASE_ACC_STAT_ROWS
                                                    + micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_BATCH_PAD
                                                    + gi * ATTN_TOK_GROUP * Q_HEAD_BATCH_PAD
                                                    + dd * Q_HEAD_BATCH_PAD
                                                )
                                                partial_oi_phase = pl.assemble(
                                                    partial_oi_phase,
                                                    pl.slice(oi_tmp, [Q_HEAD_BATCH_PAD, HEAD_DIM], [0, 0]),
                                                    [acc_exp_row0, 0],
                                                )
                                                partial_mi_phase = pl.assemble(
                                                    partial_mi_phase,
                                                    pl.slice(cur_mi, [Q_HEAD_BATCH_PAD, 1], [0, 0]),
                                                    [acc_li_row0, 0],
                                                )
                                                partial_li_phase = pl.assemble(
                                                    partial_li_phase,
                                                    pl.slice(cur_li, [Q_HEAD_BATCH_PAD, 1], [0, 0]),
                                                    [acc_li_row0, 0],
                                                )

            with pl.spmd(
                FINALIZE_SPMD_BLOCKS,
                name_hint="qk_pv_partial_merge_spmd",
            ) as merge_tid:
                merge_core = pl.tile.get_block_idx()
                for merge_work_id in pl.range(
                    merge_core,
                    ATTN_PHASE_FINALIZE_WORK_ITEMS,
                    FINALIZE_SPMD_BLOCKS,
                ):
                    merge_micro_id = merge_work_id // (ATTN_TOK_GROUP * TOTAL_Q_GROUPS)
                    merge_rem = merge_work_id - merge_micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS
                    merge_dd = merge_rem // TOTAL_Q_GROUPS
                    merge_gi = merge_rem - merge_dd * TOTAL_Q_GROUPS
                    merge_dt = merge_micro_id * ATTN_TOK_GROUP + merge_dd
                    if merge_dt < finalize_tok:
                        merge_ti = final_ti0 + merge_dt
                        merge_chunk_pos = p0 + merge_ti
                        merge_pos = chunk_start + merge_chunk_pos
                        merge_ctx_len = merge_pos + 1
                        merge_ctx_blocks = (merge_ctx_len + SEQ_TILE - 1) // SEQ_TILE
                        acc_exp_row0 = (
                            merge_micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_PAD
                            + merge_gi * ATTN_TOK_GROUP * Q_HEAD_PAD
                            + merge_dd * Q_HEAD_PAD
                        )
                        acc_li_row0 = (
                            merge_micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_BATCH_PAD
                            + merge_gi * ATTN_TOK_GROUP * Q_HEAD_BATCH_PAD
                            + merge_dd * Q_HEAD_BATCH_PAD
                        )
                        for merge_si in pl.range(QKPV_PLAN_SB_BATCH):
                            merge_sb = sb_chunk + merge_si
                            if merge_sb < block_ctx_blocks and merge_sb < merge_ctx_blocks:
                                part_exp_row0 = merge_si * ATTN_PHASE_ACC_SCORE_ROWS + acc_exp_row0
                                part_li_row0 = merge_si * ATTN_PHASE_ACC_STAT_ROWS + acc_li_row0
                                oi_tmp_sb = pl.slice(
                                    partial_oi_phase,
                                    [Q_HEAD_BATCH_PAD, HEAD_DIM],
                                    [part_exp_row0, 0],
                                )
                                cur_mi_acc = pl.slice(
                                    partial_mi_phase,
                                    [Q_HEAD_BATCH_PAD, 1],
                                    [part_li_row0, 0],
                                )
                                cur_li_acc = pl.slice(
                                    partial_li_phase,
                                    [Q_HEAD_BATCH_PAD, 1],
                                    [part_li_row0, 0],
                                )
                                if merge_sb == 0:
                                    oi_tmp_phase = pl.assemble(
                                        oi_tmp_phase,
                                        oi_tmp_sb,
                                        [acc_exp_row0, 0],
                                    )
                                    cur_li_phase = pl.assemble(
                                        cur_li_phase,
                                        cur_li_acc,
                                        [acc_li_row0, 0],
                                    )
                                    cur_mi_phase = pl.assemble(
                                        cur_mi_phase,
                                        cur_mi_acc,
                                        [acc_li_row0, 0],
                                    )
                                else:
                                    prev_oi = pl.slice(
                                        oi_tmp_phase,
                                        [Q_HEAD_BATCH_PAD, HEAD_DIM],
                                        [acc_exp_row0, 0],
                                    )
                                    prev_li = pl.slice(
                                        cur_li_phase,
                                        [Q_HEAD_BATCH_PAD, 1],
                                        [acc_li_row0, 0],
                                    )
                                    prev_mi = pl.slice(
                                        cur_mi_phase,
                                        [Q_HEAD_BATCH_PAD, 1],
                                        [acc_li_row0, 0],
                                    )
                                    mi_new = pl.maximum(prev_mi, cur_mi_acc)
                                    alpha = pl.exp(pl.sub(prev_mi, mi_new))
                                    beta = pl.exp(pl.sub(cur_mi_acc, mi_new))
                                    li_new = pl.add(
                                        pl.mul(alpha, prev_li),
                                        pl.mul(beta, cur_li_acc),
                                    )
                                    oi_new = pl.add(
                                        pl.row_expand_mul(prev_oi, alpha),
                                        pl.row_expand_mul(oi_tmp_sb, beta),
                                    )
                                    oi_tmp_phase = pl.assemble(
                                        oi_tmp_phase,
                                        oi_new,
                                        [acc_exp_row0, 0],
                                    )
                                    cur_li_phase = pl.assemble(
                                        cur_li_phase,
                                        li_new,
                                        [acc_li_row0, 0],
                                    )
                                    cur_mi_phase = pl.assemble(
                                        cur_mi_phase,
                                        mi_new,
                                        [acc_li_row0, 0],
                                    )
            pl.array.update_element(rope_ready_deps, 0, merge_tid)
        with pl.spmd(
            FINALIZE_SPMD_BLOCKS,
            name_hint="attention_finalize_phase_spmd",
        ) as finalize_tid:
            final_core = pl.tile.get_block_idx()
            for final_work_id in pl.range(
                final_core,
                ATTN_PHASE_FINALIZE_WORK_ITEMS,
                FINALIZE_SPMD_BLOCKS,
            ):
                final_micro_id = final_work_id // (ATTN_TOK_GROUP * TOTAL_Q_GROUPS)
                final_rem = final_work_id - final_micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS
                final_dd = final_rem // TOTAL_Q_GROUPS
                final_gi = final_rem - final_dd * TOTAL_Q_GROUPS
                final_dt = final_micro_id * ATTN_TOK_GROUP + final_dd
                if final_dt < finalize_tok:
                    ti = final_ti0 + final_dt
                    kvh = final_gi // Q_GROUPS
                    qg = final_gi - kvh * Q_GROUPS
                    q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH
                    acc_exp_row0 = (
                        final_micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_PAD
                        + final_gi * ATTN_TOK_GROUP * Q_HEAD_PAD
                        + final_dd * Q_HEAD_PAD
                    )
                    acc_li_row0 = (
                        final_micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_BATCH_PAD
                        + final_gi * ATTN_TOK_GROUP * Q_HEAD_BATCH_PAD
                        + final_dd * Q_HEAD_BATCH_PAD
                    )
                    oi = pl.slice(
                        oi_tmp_phase,
                        [Q_HEAD_BATCH_PAD, HEAD_DIM],
                        [acc_exp_row0, 0],
                    )
                    li = pl.slice(
                        cur_li_phase,
                        [Q_HEAD_BATCH_PAD, 1],
                        [acc_li_row0, 0],
                    )
                    ctx = pl.row_expand_div(oi, li)
                    ctx_bf16 = pl.cast(ctx, target_type=pl.BF16)
                    ctx_row = pl.reshape(
                        pl.slice(ctx_bf16, [Q_HEAD_BATCH, HEAD_DIM], [0, 0]),
                        [1, Q_HEAD_BATCH * HEAD_DIM],
                    )
                    attn_tile = pl.assemble(
                        attn_tile,
                        ctx_row,
                        [ti, q_base * HEAD_DIM],
                    )
        pl.array.update_element(rope_ready_deps, 0, finalize_tid)
    return attn_tile, cur_li_phase, oi_tmp_phase


@pl.jit.inline(auto_scope=False)
def _attention_phase_window_full_single_block(
    attn_tile: pl.Tensor[[TOK_TILE, HIDDEN], pl.BF16],
    all_q_padded_tile: pl.Tensor[[TOK_TILE * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    cur_li_phase: pl.Tensor[[ATTN_PHASE_ACC_STAT_ROWS, 1], pl.FP32],
    oi_tmp_phase: pl.Tensor[[ATTN_PHASE_ACC_SCORE_ROWS, HEAD_DIM], pl.FP32],
    b: pl.Scalar[pl.INT32],
    max_blocks_per_seq: pl.Scalar[pl.INT32],
    layer_cache_base: pl.Scalar[pl.INT32],
    chunk_start: pl.Scalar[pl.INT32],
    p0: pl.Scalar[pl.INT32],
    final_ti0: pl.Scalar[pl.INT32],
    rope_ready_deps: pl.Array[ROPE_READY_DEP_COUNT, pl.TASK_ID],
) -> tuple[
    pl.Tensor[[TOK_TILE, HIDDEN], pl.BF16],
    pl.Tensor[[ATTN_PHASE_ACC_STAT_ROWS, 1], pl.FP32],
    pl.Tensor[[ATTN_PHASE_ACC_SCORE_ROWS, HEAD_DIM], pl.FP32],
]:
    cache_token_rows = pl.tensor.dim(k_cache, 0) // NUM_KV_HEADS
    k_cache_bsnd = pl.reshape(k_cache, [cache_token_rows, KV_HIDDEN])
    v_cache_bsnd = pl.reshape(v_cache, [cache_token_rows, KV_HIDDEN])
    with pl.spmd(
        ATTN_PHASE_SPMD_BLOCKS,
        name_hint="qk_pv_single_block_spmd",
        deps=[rope_ready_deps[0]],
        sync_start=True,
    ) as attn_tid:
        phase_core = pl.tile.get_block_idx()
        for work_id in pl.range(phase_core, ATTN_PHASE_WORK_ITEMS, ATTN_PHASE_SPMD_BLOCKS):
            micro_id = work_id // ATTN_GI_BLOCKS
            gi_block = work_id - micro_id * ATTN_GI_BLOCKS
            gi = gi_block * ATTN_GI_GROUP
            attn_ti0 = final_ti0 + micro_id * ATTN_TOK_GROUP
            kvh = gi // Q_GROUPS
            block_table_idx = b * max_blocks_per_seq
            pbid = pl.cast(
                pl.tensor.read(block_table, [block_table_idx]),
                pl.INDEX,
            )
            cache_row0 = layer_cache_base + pbid * BLOCK_SIZE
            cache_col = kvh * HEAD_DIM
            k_tile = pl.slice(k_cache_bsnd, [SEQ_TILE, HEAD_DIM], [cache_row0, cache_col])
            v_tile = pl.slice(v_cache_bsnd, [SEQ_TILE, HEAD_DIM], [cache_row0, cache_col])
            for dd0 in pl.pipeline(0, ATTN_TOK_GROUP, QKPV_PIPELINE_TOK_STEP, stage=3):
                ti0 = attn_ti0 + dd0
                q_row0 = ti0 * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                q0 = pl.slice(
                    all_q_padded_tile,
                    [Q_HEAD_PAD, HEAD_DIM],
                    [q_row0, 0],
                )
                raw_scores0 = pl.matmul(
                    q0,
                    k_tile,
                    b_trans=True,
                    out_dtype=pl.FP32,
                )

                chunk_pos0 = p0 + ti0
                ctx_len0 = chunk_start + chunk_pos0 + 1
                scores0 = pl.fillpad(
                    pl.set_validshape(
                        pl.mul(raw_scores0, ATTN_SCALE),
                        Q_HEAD_BATCH,
                        ctx_len0,
                    ),
                    pad_value=pl.PadValue.min,
                )
                cur_mi0 = pl.row_max(scores0)
                exp_scores0 = pl.exp(pl.row_expand_sub(scores0, cur_mi0))
                exp_scores_bf16_0 = pl.cast(exp_scores0, target_type=pl.BF16)
                cur_li0 = pl.row_sum(
                    pl.cast(exp_scores_bf16_0, target_type=pl.FP32),
                )
                oi_tmp0 = pl.matmul(
                    exp_scores_bf16_0,
                    v_tile,
                    out_dtype=pl.FP32,
                )
                acc_exp_row0 = (
                    micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_PAD
                    + gi * ATTN_TOK_GROUP * Q_HEAD_PAD
                    + dd0 * Q_HEAD_PAD
                )
                acc_li_row0 = (
                    micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_BATCH_PAD
                    + gi * ATTN_TOK_GROUP * Q_HEAD_BATCH_PAD
                    + dd0 * Q_HEAD_BATCH_PAD
                )
                oi_tmp_phase = pl.assemble(
                    oi_tmp_phase,
                    pl.slice(oi_tmp0, [Q_HEAD_BATCH_PAD, HEAD_DIM], [0, 0]),
                    [acc_exp_row0, 0],
                )
                cur_li_phase = pl.assemble(
                    cur_li_phase,
                    pl.slice(cur_li0, [Q_HEAD_BATCH_PAD, 1], [0, 0]),
                    [acc_li_row0, 0],
                )
        pl.system.syncall(core_type=pl.KernelType.MIX)

        for final_work_id in pl.range(
            phase_core,
            ATTN_PHASE_FINALIZE_WORK_ITEMS,
            ATTN_PHASE_SPMD_BLOCKS,
        ):
            final_micro_id = final_work_id // (ATTN_TOK_GROUP * TOTAL_Q_GROUPS)
            final_rem = final_work_id - final_micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS
            final_dd = final_rem // TOTAL_Q_GROUPS
            final_gi = final_rem - final_dd * TOTAL_Q_GROUPS
            final_dt = final_micro_id * ATTN_TOK_GROUP + final_dd
            ti = final_ti0 + final_dt
            kvh = final_gi // Q_GROUPS
            qg = final_gi - kvh * Q_GROUPS
            q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH
            acc_exp_row0 = (
                final_micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_PAD
                + final_gi * ATTN_TOK_GROUP * Q_HEAD_PAD
                + final_dd * Q_HEAD_PAD
            )
            acc_li_row0 = (
                final_micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_BATCH_PAD
                + final_gi * ATTN_TOK_GROUP * Q_HEAD_BATCH_PAD
                + final_dd * Q_HEAD_BATCH_PAD
            )
            oi = pl.slice(
                oi_tmp_phase,
                [Q_HEAD_BATCH_PAD, HEAD_DIM],
                [acc_exp_row0, 0],
            )
            li = pl.slice(
                cur_li_phase,
                [Q_HEAD_BATCH_PAD, 1],
                [acc_li_row0, 0],
            )
            ctx = pl.row_expand_div(oi, li)
            ctx_bf16 = pl.cast(ctx, target_type=pl.BF16)
            ctx_row = pl.reshape(
                pl.slice(ctx_bf16, [Q_HEAD_BATCH, HEAD_DIM], [0, 0]),
                [1, Q_HEAD_BATCH * HEAD_DIM],
            )
            attn_tile = pl.assemble(
                attn_tile,
                ctx_row,
                [ti, q_base * HEAD_DIM],
            )
    pl.array.update_element(rope_ready_deps, 0, attn_tid)
    return attn_tile, cur_li_phase, oi_tmp_phase


@pl.jit.inline(auto_scope=False)
def _rope_kv_cache_phase(
    all_q_padded_tile: pl.Tensor[[TOK_TILE * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], pl.BF16],
    input_inv_rms_tile: pl.Tensor[[TOK_TILE, 1], pl.FP32],
    q_proj_tile: pl.Tensor[[TOK_TILE, HIDDEN], pl.FP32],
    k_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    v_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    layer_cache_base: pl.Scalar[pl.INT32],
    chunk_start: pl.Scalar[pl.INT32],
    p0: pl.Scalar[pl.INT32],
    final_ti0: pl.Scalar[pl.INT32],
    finalize_tok: pl.Scalar[pl.INT32],
    slot_token_p0: pl.Scalar[pl.INDEX],
    layer_idx: pl.Scalar[pl.INT32],
    p0_idx: pl.Scalar[pl.INDEX],
    rope_input_deps: pl.Array[ROPE_INPUT_DEP_COUNT, pl.TASK_ID],
    rope_ready_deps: pl.Array[ROPE_READY_DEP_COUNT, pl.TASK_ID],
) -> pl.Tensor[[TOK_TILE * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], pl.BF16]:
    cache_token_rows = pl.tensor.dim(k_cache, 0) // NUM_KV_HEADS
    k_cache_bsnd = pl.reshape(k_cache, [cache_token_rows, KV_HIDDEN])
    v_cache_bsnd = pl.reshape(v_cache, [cache_token_rows, KV_HIDDEN])
    if p0_idx == 0:
        with pl.spmd(
            ROPE_SPMD_BLOCKS,
            name_hint="rope_kv_cache",
            deps=[
                rope_input_deps[0],
                rope_input_deps[1],
                rope_input_deps[2],
                rope_input_deps[3],
                rope_input_deps[4],
            ],
        ) as rope_tid:
            rope_core = pl.tile.get_block_idx()
            rope_group_count = (finalize_tok + ROPE_T_GROUP - 1) // ROPE_T_GROUP
            rope_work_count = rope_group_count * NUM_KV_HEADS
            for rope_work in pl.range(rope_core, rope_work_count, ROPE_SPMD_BLOCKS):
                rope_group = rope_work // NUM_KV_HEADS
                ki = rope_work - rope_group * NUM_KV_HEADS
                rel_ti0 = rope_group * ROPE_T_GROUP
                ti0 = final_ti0 + rel_ti0
                group_tok = pl.min(ROPE_T_GROUP, finalize_tok - rel_ti0)
                pos0 = chunk_start + p0 + ti0
                cos_lo_g = pl.slice(
                    rope_cos,
                    [ROPE_T_GROUP, HALF_DIM],
                    [pos0, 0],
                    valid_shape=[group_tok, HALF_DIM],
                )
                cos_hi_g = pl.slice(
                    rope_cos,
                    [ROPE_T_GROUP, HALF_DIM],
                    [pos0, HALF_DIM],
                    valid_shape=[group_tok, HALF_DIM],
                )
                sin_lo_g = pl.slice(
                    rope_sin,
                    [ROPE_T_GROUP, HALF_DIM],
                    [pos0, 0],
                    valid_shape=[group_tok, HALF_DIM],
                )
                sin_hi_g = pl.slice(
                    rope_sin,
                    [ROPE_T_GROUP, HALF_DIM],
                    [pos0, HALF_DIM],
                    valid_shape=[group_tok, HALF_DIM],
                )
                input_inv_rms_g = pl.slice(input_inv_rms_tile, [ROPE_T_GROUP, 1], [ti0, 0])
                k_norm_g = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
                q_norm_g = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
                kv_col = ki * HEAD_DIM
                k_raw = pl.row_expand_mul(
                    pl.slice(
                        k_proj_tile,
                        [ROPE_T_GROUP, HEAD_DIM],
                        [ti0, kv_col],
                    ),
                    input_inv_rms_g,
                )
                k_sq = pl.reshape(pl.row_sum(pl.mul(k_raw, k_raw)), [ROPE_T_GROUP, 1])
                k_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), EPS)))
                k_normed = pl.col_expand_mul(
                    pl.row_expand_mul(k_raw, k_inv_rms),
                    k_norm_g,
                )
                k_lo_g = pl.slice(k_normed, [ROPE_T_GROUP, HALF_DIM], [0, 0])
                k_hi_g = pl.slice(k_normed, [ROPE_T_GROUP, HALF_DIM], [0, HALF_DIM])
                k_rot_lo_g = pl.sub(
                    pl.mul(k_lo_g, cos_lo_g),
                    pl.mul(k_hi_g, sin_lo_g),
                )
                k_rot_hi_g = pl.add(
                    pl.mul(k_hi_g, cos_hi_g),
                    pl.mul(k_lo_g, sin_hi_g),
                )
                cache_col = ki * HEAD_DIM
                v_scaled_g = pl.row_expand_mul(
                    pl.slice(
                        v_proj_tile,
                        [ROPE_T_GROUP, HEAD_DIM],
                        [ti0, cache_col],
                    ),
                    input_inv_rms_g,
                )
                q_base = ki * Q_PER_KV
                q_raw_g = pl.row_expand_mul(
                    pl.slice(
                        q_proj_tile,
                        [ROPE_T_GROUP, Q_HEAD_BATCH * HEAD_DIM],
                        [ti0, q_base * HEAD_DIM],
                    ),
                    input_inv_rms_g,
                )
                q_flat = pl.reshape(q_raw_g, [ROPE_Q_ROWS, HEAD_DIM])
                q_sq = pl.reshape(pl.row_sum(pl.mul(q_flat, q_flat)), [ROPE_Q_ROWS, 1])
                q_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), EPS)))
                q_normed = pl.col_expand_mul(
                    pl.row_expand_mul(q_flat, q_inv_rms),
                    q_norm_g,
                )
                for dt in pl.range(ROPE_T_GROUP):
                    if dt < group_tok:
                        ti = ti0 + dt
                        cache_slot = pl.cast(pl.tensor.read(slot_mapping, [slot_token_p0 + ti]), pl.INDEX)
                        cache_slot_block = cache_slot // BLOCK_SIZE
                        cache_slot_offset = cache_slot - cache_slot_block * BLOCK_SIZE
                        cache_row = layer_cache_base + cache_slot_block * BLOCK_SIZE + cache_slot_offset
                        k_cache_bsnd = pl.assemble(
                            k_cache_bsnd,
                            pl.cast(pl.slice(k_rot_lo_g, [1, HALF_DIM], [dt, 0]), target_type=pl.BF16),
                            [cache_row, cache_col],
                        )
                        k_cache_bsnd = pl.assemble(
                            k_cache_bsnd,
                            pl.cast(pl.slice(k_rot_hi_g, [1, HALF_DIM], [dt, 0]), target_type=pl.BF16),
                            [cache_row, cache_col + HALF_DIM],
                        )
                        v_cache_bsnd = pl.assemble(
                            v_cache_bsnd,
                            pl.cast(pl.slice(v_scaled_g, [1, HEAD_DIM], [dt, 0]), target_type=pl.BF16),
                            [cache_row, cache_col],
                        )
                        q_row0 = dt * Q_HEAD_BATCH
                        q_block = pl.slice(q_normed, [Q_HEAD_BATCH, HEAD_DIM], [q_row0, 0])
                        q_lo = pl.slice(q_block, [Q_HEAD_BATCH, HALF_DIM], [0, 0])
                        q_hi = pl.slice(q_block, [Q_HEAD_BATCH, HALF_DIM], [0, HALF_DIM])
                        cos_lo = pl.slice(cos_lo_g, [1, HALF_DIM], [dt, 0])
                        cos_hi = pl.slice(cos_hi_g, [1, HALF_DIM], [dt, 0])
                        sin_lo = pl.slice(sin_lo_g, [1, HALF_DIM], [dt, 0])
                        sin_hi = pl.slice(sin_hi_g, [1, HALF_DIM], [dt, 0])
                        q_rot_lo = pl.sub(
                            pl.col_expand_mul(q_lo, cos_lo),
                            pl.col_expand_mul(q_hi, sin_lo),
                        )
                        q_rot_hi = pl.add(
                            pl.col_expand_mul(q_hi, cos_hi),
                            pl.col_expand_mul(q_lo, sin_hi),
                        )
                        q_pad_row0 = ti * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD
                        all_q_padded_tile = pl.assemble(
                            all_q_padded_tile,
                            pl.cast(q_rot_lo, target_type=pl.BF16),
                            [q_pad_row0, 0],
                        )
                        all_q_padded_tile = pl.assemble(
                            all_q_padded_tile,
                            pl.cast(q_rot_hi, target_type=pl.BF16),
                            [q_pad_row0, HALF_DIM],
                        )
                        all_q_padded_tile = pl.assemble(
                            all_q_padded_tile,
                            pl.cast(
                                pl.full(
                                    [Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM],
                                    dtype=pl.FP32,
                                    value=0.0,
                                ),
                                target_type=pl.BF16,
                            ),
                            [q_pad_row0 + Q_HEAD_BATCH, 0],
                        )
        pl.array.update_element(rope_ready_deps, 0, rope_tid)
    return all_q_padded_tile


@pl.jit.inline(auto_scope=False)
def _manual_rope_kv_cache_phase(
    all_q_padded_tile: pl.Tensor[[TOK_TILE * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], pl.BF16],
    input_inv_rms_tile: pl.Tensor[[TOK_TILE, 1], pl.FP32],
    q_proj_tile: pl.Tensor[[TOK_TILE, HIDDEN], pl.FP32],
    k_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    v_proj_tile: pl.Tensor[[TOK_TILE, KV_HIDDEN], pl.FP32],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    layer_cache_base: pl.Scalar[pl.INT32],
    chunk_start: pl.Scalar[pl.INT32],
    p0: pl.Scalar[pl.INT32],
    final_ti0: pl.Scalar[pl.INT32],
    finalize_tok: pl.Scalar[pl.INT32],
    slot_token_p0: pl.Scalar[pl.INDEX],
    layer_idx: pl.Scalar[pl.INT32],
    p0_idx: pl.Scalar[pl.INDEX],
    rope_input_deps: pl.Array[ROPE_INPUT_DEP_COUNT, pl.TASK_ID],
    rope_ready_deps: pl.Array[ROPE_READY_DEP_COUNT, pl.TASK_ID],
) -> pl.Tensor[[TOK_TILE * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], pl.BF16]:
    with pl.manual_scope():
        all_q_padded_tile = _rope_kv_cache_phase(
            all_q_padded_tile,
            input_inv_rms_tile,
            q_proj_tile,
            k_proj_tile,
            v_proj_tile,
            q_norm_weight,
            k_norm_weight,
            rope_cos,
            rope_sin,
            slot_mapping,
            k_cache,
            v_cache,
            layer_cache_base,
            chunk_start,
            p0,
            final_ti0,
            finalize_tok,
            slot_token_p0,
            layer_idx,
            p0_idx,
            rope_input_deps,
            rope_ready_deps,
        )
    return all_q_padded_tile


@pl.jit.inline(auto_scope=False)
def _out_proj_aic_phase(
    out_proj_tile: pl.Tensor[[TOK_TILE, HIDDEN], pl.FP32],
    attn_tile: pl.Tensor[[TOK_TILE, HIDDEN], pl.BF16],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    layer_hidden_base: pl.Scalar[pl.INT32],
) -> pl.Tensor[[TOK_TILE, HIDDEN], pl.FP32]:
    with pl.spmd(
        OUT_PROJ_SPMD_BLOCKS,
        name_hint="out_proj_aic_spmd",
    ):
        out_core = pl.tile.get_block_idx()
        for ob in pl.range(out_core, Q_OUT_BLOCKS, OUT_PROJ_SPMD_BLOCKS):
            o0 = ob * Q_OUT_CHUNK
            tile_a = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, 0])
            tile_w = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, o0])
            o_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                k0 = kb * K_CHUNK
                tile_a_i = pl.slice(attn_tile, [TOK_TILE, K_CHUNK], [0, k0])
                tile_w_i = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + k0, o0])
                o_acc = pl.matmul_acc(o_acc, tile_a_i, tile_w_i)
            out_proj_tile = pl.assemble(out_proj_tile, o_acc, [0, o0])
    return out_proj_tile


@pl.jit.inline(auto_scope=False)
def prefill_layer(
    hidden_states: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[BATCH_DYN], pl.INT32],
    group_p0: pl.Scalar[pl.INDEX],
    active_prefill_tokens: pl.Scalar[pl.INDEX],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    out: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
    window_done: pl.Tensor[[BATCH_PAD, 8], pl.FP32],
    layer_idx: pl.Scalar[pl.INT32],
    kv_tail: pl.Array[NUM_LAYERS * BATCH_PAD, pl.TASK_ID],
) -> pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16]:
    hidden_states.bind_dynamic(0, PREFILL_TOKENS_DYN)
    out.bind_dynamic(0, PREFILL_TOKENS_DYN)

    # Runtime batch (host-visible batch). Outer batch loop
    # iterates with step 1 so every matmul tile's M dim is fully
    # determined by TOK_TILE (no batch-axis pad / trim needed).
    batch = pl.tensor.dim(seq_lens, 0)
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    cache_token_rows = pl.tensor.dim(k_cache, 0) // NUM_KV_HEADS
    layer_cache_rows = cache_token_rows // num_layers_actual
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    layer_cache_base = layer_idx * layer_cache_rows
    max_blocks_per_seq = pl.tensor.dim(block_table, 0) // batch

    # Consume each 128-token m-tile immediately after its attention/post-norm
    # tile finishes. Keep the temporary post-norm/residual storage local to
    # that m-tile so gate/up only depends on this tile's post-rmsnorm writer,
    # instead of every writer into a layer-wide packed tensor.
    prefill_tokens = active_prefill_tokens
    down_n_blocks = HIDDEN // K_CHUNK
    band_k_chunks = MLP_BAND_WIDTH // MLP_OUT_CHUNK
    silu_band_blocks = MLP_BAND_WIDTH // SILU_OUT_CHUNK

    for b in pl.parallel(0, batch, 1):
        post_norm_mtile = pl.create_tensor([MLP_M_TILE, HIDDEN], dtype=pl.BF16, manual_dep=True)
        resid1_mtile = pl.create_tensor([MLP_M_TILE, HIDDEN], dtype=pl.FP32)
        window_done_row = pl.slice(window_done, [1, 8], [b, 0])
        post_norm_tids = pl.array.create(MLP_M_TILE // TOK_TILE, pl.TASK_ID)
        original_token_base = pl.cast(pl.tensor.read(chunk_offsets, [b]), pl.INDEX) + group_p0
        token_base = pl.cast(b * MLP_M_TILE, pl.INDEX)
        seq_len_b = pl.tensor.read(seq_lens, [b])
        full_chunk_len_b = pl.tensor.read(chunk_lens, [b])
        group_p0_i32 = pl.cast(group_p0, pl.INT32)
        chunk_start = seq_len_b - full_chunk_len_b
        remaining_tok = full_chunk_len_b - group_p0_i32
        chunk_len_b = pl.min(MLP_M_TILE, pl.max(remaining_tok, 0))
        tok_blocks = (chunk_len_b + TOK_TILE - 1) // TOK_TILE
        kv_tail_idx = pl.cast(layer_idx, pl.INDEX) * BATCH_PAD + b
        for p0_idx in pl.unroll(MLP_M_TILE // TOK_TILE):
            post_norm_tid_for_ready = pl.system.task_invalid()
            if p0_idx < tok_blocks:
                with pl.scope():
                    p0 = p0_idx * TOK_TILE
                    token_p0 = token_base + p0
                    slot_token_p0 = original_token_base + p0
                    valid_tok = pl.min(TOK_TILE, chunk_len_b - p0)

                    x_gamma_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)
                    input_inv_rms_tile = pl.create_tensor([TOK_TILE, 1], dtype=pl.FP32)

                    with pl.spmd(
                        RMSNORM_SPMD_BLOCKS,
                        name_hint="x_gamma_spmd",
                        allow_early_resolve=True,
                    ) as x_gamma_tid:
                        xg_core = pl.tile.get_block_idx()
                        for work_id in pl.range(xg_core, X_GAMMA_HIDDEN_BLOCKS, RMSNORM_SPMD_BLOCKS):
                            k0 = work_id * X_GAMMA_HIDDEN_CHUNK
                            xg_chunk = pl.cast(
                                pl.slice(
                                    hidden_states,
                                    [TOK_TILE, X_GAMMA_HIDDEN_CHUNK],
                                    [token_p0, k0],
                                    valid_shape=[valid_tok, X_GAMMA_HIDDEN_CHUNK],
                                ),
                                target_type=pl.FP32,
                            )
                            input_gamma = pl.slice(
                                input_rms_weight,
                                [1, X_GAMMA_HIDDEN_CHUNK],
                                [layer_idx, k0],
                            )
                            x_gamma = pl.col_expand_mul(xg_chunk, input_gamma)
                            x_gamma_tile = pl.assemble(
                                x_gamma_tile,
                                pl.cast(x_gamma, target_type=pl.BF16),
                                [0, k0],
                            )

                    # The dummy gate biases x_gamma to dispatch first while leaving
                    # q/kv and rms_recip unordered after x_gamma's early resolve.
                    rms_recip_late = pl.system.task_dummy(deps=[x_gamma_tid])
                    with pl.spmd(
                        RMSNORM_SPMD_BLOCKS,
                        name_hint="rms_recip_spmd",
                        deps=[rms_recip_late],
                    ) as rms_recip_tid:
                        rms_core = pl.tile.get_block_idx()
                        for work_id in pl.range(rms_core, RMSNORM_WORK_ITEMS, RMSNORM_SPMD_BLOCKS):
                            ti0 = work_id * RMSNORM_TOK_GROUP
                            if ti0 < valid_tok:
                                rms_tok = pl.min(RMSNORM_TOK_GROUP, valid_tok - ti0)
                                sq_sum = pl.full([1, RMSNORM_TOK_GROUP], dtype=pl.FP32, value=0.0)
                                for rb in pl.range(HIDDEN_BLOCKS):
                                    k0 = rb * K_CHUNK
                                    rms_x_chunk = pl.cast(
                                        pl.slice(
                                            hidden_states,
                                            [RMSNORM_TOK_GROUP, K_CHUNK],
                                            [token_p0 + ti0, k0],
                                            valid_shape=[rms_tok, K_CHUNK],
                                        ),
                                        target_type=pl.FP32,
                                    )
                                    sq_part = pl.reshape(
                                        pl.row_sum(pl.mul(rms_x_chunk, rms_x_chunk)),
                                        [1, RMSNORM_TOK_GROUP],
                                    )
                                    sq_sum = pl.add(sq_sum, sq_part)
                                inv_rms = pl.reshape(
                                    pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))),
                                    [RMSNORM_TOK_GROUP, 1],
                                )
                                input_inv_rms_tile = pl.assemble(input_inv_rms_tile, inv_rms, [ti0, 0])

                    q_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
                    k_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.FP32)
                    v_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.FP32)
                    fused_qkpv_deps = pl.array.create(ROPE_INPUT_DEP_COUNT, pl.TASK_ID)
                    if p0_idx == 0:
                        with pl.spmd(
                            Q_PROJ_SPMD_BLOCKS,
                            name_hint="q_proj_spmd",
                        ) as q_proj_tid:
                            q_core = pl.tile.get_block_idx()
                            for ob in pl.range(q_core, Q_OUT_BLOCKS, Q_PROJ_SPMD_BLOCKS):
                                q0 = ob * Q_OUT_CHUNK
                                tile_a = pl.slice(x_gamma_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                tile_w = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, q0])
                                q_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                                for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                                    k0 = kb * K_CHUNK
                                    tile_a_i = pl.slice(x_gamma_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                    tile_w_i = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + k0, q0])
                                    q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_w_i)
                                q_proj_tile = pl.assemble(q_proj_tile, q_acc, [0, q0])

                        with pl.spmd(
                            KV_PROJ_SPMD_BLOCKS,
                            name_hint="kv_proj_spmd",
                        ) as kv_proj_tid:
                            kv_core = pl.tile.get_block_idx()
                            for work_id in pl.range(kv_core, KV_PROJ_WORK_ITEMS, KV_PROJ_SPMD_BLOCKS):
                                if work_id < KV_OUT_BLOCKS:
                                    kv0 = work_id * KV_OUT_CHUNK
                                    tile_a = pl.slice(x_gamma_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                    tile_wk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base, kv0])
                                    k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                                    for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                                        k0 = kb * K_CHUNK
                                        tile_a_i = pl.slice(x_gamma_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                        tile_wk_i = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k0, kv0])
                                        k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                                    k_proj_tile = pl.assemble(k_proj_tile, k_acc, [0, kv0])
                                else:
                                    kv_work = work_id - KV_OUT_BLOCKS
                                    kv0 = kv_work * KV_OUT_CHUNK
                                    tile_a = pl.slice(x_gamma_tile, [TOK_TILE, K_CHUNK], [0, 0])
                                    tile_wv = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base, kv0])
                                    v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                                    for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                                        k0 = kb * K_CHUNK
                                        tile_a_i = pl.slice(x_gamma_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                        tile_wv_i = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k0, kv0])
                                        v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                                    v_proj_tile = pl.assemble(v_proj_tile, v_acc, [0, kv0])
                        fused_qkpv_deps[0] = rms_recip_tid
                        fused_qkpv_deps[1] = q_proj_tid
                        fused_qkpv_deps[2] = kv_proj_tid
                        fused_qkpv_deps[3] = kv_tail[kv_tail_idx]
                    attn_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)
                    out_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
                    all_q_padded_tile = pl.create_tensor(
                        [TOK_TILE * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM],
                        dtype=pl.BF16,
                    )
                    rope_ready_deps = pl.array.create(ROPE_READY_DEP_COUNT, pl.TASK_ID)
                    for final_ti0 in pl.range(0, valid_tok, FINALIZE_TOK_GROUP):
                        with pl.scope():
                            finalize_tok = pl.min(FINALIZE_TOK_GROUP, valid_tok - final_ti0)
                            b_i32 = pl.cast(b, pl.INT32)
                            max_blocks_i32 = pl.cast(max_blocks_per_seq, pl.INT32)
                            layer_cache_base_i32 = pl.cast(layer_cache_base, pl.INT32)
                            p0_i32 = group_p0_i32 + pl.cast(p0, pl.INT32)
                            final_ti0_i32 = pl.cast(final_ti0, pl.INT32)
                            finalize_tok_i32 = pl.cast(finalize_tok, pl.INT32)

                            cur_li_phase = pl.create_tensor([ATTN_PHASE_ACC_STAT_ROWS, 1], dtype=pl.FP32)
                            oi_tmp_phase = pl.create_tensor([ATTN_PHASE_ACC_SCORE_ROWS, HEAD_DIM], dtype=pl.FP32)
                            block_ctx_len = chunk_start + group_p0_i32 + p0 + final_ti0 + finalize_tok
                            block_ctx_blocks = (block_ctx_len + SEQ_TILE - 1) // SEQ_TILE
                            pl.array.update_element(fused_qkpv_deps, 4, rope_ready_deps[0])
                            all_q_padded_tile = _manual_rope_kv_cache_phase(
                                all_q_padded_tile,
                                input_inv_rms_tile,
                                q_proj_tile,
                                k_proj_tile,
                                v_proj_tile,
                                q_norm_weight,
                                k_norm_weight,
                                rope_cos,
                                rope_sin,
                                slot_mapping,
                                k_cache,
                                v_cache,
                                layer_cache_base_i32,
                                chunk_start,
                                p0_i32,
                                final_ti0_i32,
                                finalize_tok_i32,
                                slot_token_p0,
                                layer_idx,
                                p0_idx,
                                fused_qkpv_deps,
                                rope_ready_deps,
                            )
                            if block_ctx_blocks == 1:
                                if finalize_tok == FINALIZE_TOK_GROUP:
                                    attn_tile, cur_li_phase, oi_tmp_phase = _attention_phase_window_full_single_block(
                                        attn_tile,
                                        all_q_padded_tile,
                                        block_table,
                                        k_cache,
                                        v_cache,
                                        cur_li_phase,
                                        oi_tmp_phase,
                                        b_i32,
                                        max_blocks_i32,
                                        layer_cache_base_i32,
                                        chunk_start,
                                        p0_i32,
                                        final_ti0_i32,
                                        rope_ready_deps,
                                    )
                                else:
                                    attn_tile, cur_li_phase, oi_tmp_phase = _attention_phase_window(
                                        attn_tile,
                                        all_q_padded_tile,
                                        block_table,
                                        k_cache,
                                        v_cache,
                                        cur_li_phase,
                                        oi_tmp_phase,
                                        b_i32,
                                        max_blocks_i32,
                                        layer_cache_base_i32,
                                        chunk_start,
                                        p0_i32,
                                        final_ti0_i32,
                                        finalize_tok_i32,
                                        rope_ready_deps,
                                    )
                            else:
                                attn_tile, cur_li_phase, oi_tmp_phase = _attention_phase_window(
                                    attn_tile,
                                    all_q_padded_tile,
                                    block_table,
                                    k_cache,
                                    v_cache,
                                    cur_li_phase,
                                    oi_tmp_phase,
                                    b_i32,
                                    max_blocks_i32,
                                    layer_cache_base_i32,
                                    chunk_start,
                                    p0_i32,
                                    final_ti0_i32,
                                    finalize_tok_i32,
                                    rope_ready_deps,
                                )
                    resid1_tile = pl.slice(resid1_mtile, [TOK_TILE, HIDDEN], [p0, 0])
                    out_proj_tile = _out_proj_aic_phase(
                        out_proj_tile,
                        attn_tile,
                        wo,
                        pl.cast(layer_hidden_base, pl.INT32),
                    )
                    with pl.spmd(
                        OUT_RESID_SPMD_BLOCKS,
                        name_hint="out_proj_aiv_spmd",
                    ):
                        out_core = pl.tile.get_block_idx()
                        for ob in pl.range(out_core, OUT_RESID_BLOCKS, OUT_RESID_SPMD_BLOCKS):
                            o0 = ob * OUT_RESID_CHUNK
                            resid_chunk = pl.cast(
                                pl.slice(
                                    hidden_states,
                                    [TOK_TILE, OUT_RESID_CHUNK],
                                    [token_p0, o0],
                                    valid_shape=[valid_tok, OUT_RESID_CHUNK],
                                ),
                                target_type=pl.FP32,
                            )
                            out_proj_chunk = pl.slice(out_proj_tile, [TOK_TILE, OUT_RESID_CHUNK], [0, o0])
                            resid1_tile = pl.assemble(resid1_tile, pl.add(out_proj_chunk, resid_chunk), [0, o0])

                    post_norm_tile = pl.slice(post_norm_mtile, [TOK_TILE, HIDDEN], [p0, 0])
                    # Keep post_norm fully resolved before MLP opens its ring3
                    # band scopes; otherwise the allocator can pre-stage MLP
                    # faster than completed band scopes become reclaimable.
                    with pl.spmd(
                        POST_RMSNORM_SPMD_BLOCKS,
                        name_hint="post_rmsnorm_spmd",
                    ) as post_norm_tid:
                        post_core = pl.tile.get_block_idx()
                        for work_id in pl.range(post_core, RMSNORM_WORK_ITEMS, POST_RMSNORM_SPMD_BLOCKS):
                            ti0 = work_id * RMSNORM_TOK_GROUP
                            if ti0 < valid_tok:
                                rms_tok = pl.min(RMSNORM_TOK_GROUP, valid_tok - ti0)
                                post_sq_sum = pl.full([1, RMSNORM_TOK_GROUP], dtype=pl.FP32, value=0.0)
                                for rb in pl.range(HIDDEN_BLOCKS):
                                    k0 = rb * K_CHUNK
                                    post_x_chunk_sq = pl.slice(
                                        resid1_tile,
                                        [RMSNORM_TOK_GROUP, K_CHUNK],
                                        [ti0, k0],
                                        valid_shape=[rms_tok, K_CHUNK],
                                    )
                                    post_sq_part = pl.reshape(
                                        pl.row_sum(pl.mul(post_x_chunk_sq, post_x_chunk_sq)),
                                        [1, RMSNORM_TOK_GROUP],
                                    )
                                    post_sq_sum = pl.add(post_sq_sum, post_sq_part)
                                post_inv_rms = pl.reshape(
                                    pl.recip(pl.sqrt(pl.add(pl.mul(post_sq_sum, HIDDEN_INV), EPS))),
                                    [RMSNORM_TOK_GROUP, 1],
                                )

                                for kb in pl.range(HIDDEN_BLOCKS):
                                    k0 = kb * K_CHUNK
                                    post_x_chunk_norm = pl.slice(
                                        resid1_tile,
                                        [RMSNORM_TOK_GROUP, K_CHUNK],
                                        [ti0, k0],
                                        valid_shape=[rms_tok, K_CHUNK],
                                    )
                                    gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, k0])
                                    normed = pl.col_expand_mul(
                                        pl.row_expand_mul(post_x_chunk_norm, post_inv_rms),
                                        gamma,
                                    )
                                    post_norm_tile = pl.assemble(
                                        post_norm_tile,
                                        pl.cast(normed, target_type=pl.BF16),
                                        [ti0, k0],
                                    )
                    post_norm_tid_for_ready = post_norm_tid
            post_norm_tids[p0_idx] = post_norm_tid_for_ready

        if tok_blocks > 0:
            # Consume this m-tile immediately after its attention/post-norm tile
            # finishes. Keeping the MLP consumer close to the ring3
            # attention producers lets the ring reclaim older tile scopes before the
            # next batch tile opens more ring3 work.
            m0 = b * MLP_M_TILE
            post_norm_ready = pl.system.task_dummy(deps=[post_norm_tids])
            mlp_out_acc_tile = pl.create_tensor([MLP_M_TILE, HIDDEN], dtype=pl.FP32, manual_dep=True)

            # Seed the accumulator with the first-residual (folds the MLP residual add).
            with pl.spmd(
                DOWN_RESID_SPMD_BLOCKS,
                name_hint="mlp_out_seed_spmd",
            ) as seed_tid:
                seed_core = pl.tile.get_block_idx()
                for hb in pl.range(seed_core, down_n_blocks, DOWN_RESID_SPMD_BLOCKS):
                    h0 = hb * K_CHUNK
                    mlp_out_acc_tile = pl.assemble(
                        mlp_out_acc_tile,
                        pl.slice(resid1_mtile, [MLP_M_TILE, K_CHUNK], [0, h0]),
                        [0, h0],
                    )

            down_chain = pl.array.create(MLP_PROJ_BANDS, pl.TASK_ID)

            for mlp_band in pl.range(MLP_PROJ_BANDS):
                with pl.scope():
                    band_ob0 = mlp_band * MLP_BAND_BLOCKS
                    band_inter0 = mlp_band * MLP_BAND_WIDTH
                    gate_up_acc_b = pl.create_tensor([2 * MLP_M_TILE, MLP_BAND_WIDTH], dtype=pl.FP32)
                    mlp_silu_b = pl.create_tensor([MLP_M_TILE, MLP_BAND_WIDTH], dtype=pl.BF16)

                    band_core_rot = (mlp_band * MLP_BAND_CORE_ROT_STEP) % GATE_UP_SPMD_BLOCKS
                    with pl.manual_scope():
                        with pl.spmd(
                            GATE_UP_SPMD_BLOCKS,
                            name_hint="gate_up_proj_spmd",
                            deps=[post_norm_ready],
                        ) as gate_up_tid:
                            gu_core = (pl.tile.get_block_idx() + band_core_rot) % GATE_UP_SPMD_BLOCKS
                            for rel_ob in pl.range(gu_core, MLP_BAND_BLOCKS, GATE_UP_SPMD_BLOCKS):
                                o0 = (band_ob0 + rel_ob) * MLP_OUT_CHUNK
                                pc0 = pl.slice(post_norm_mtile, [MLP_M_TILE, K_CHUNK], [0, 0])
                                wg0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, o0])
                                gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                                for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                                    k0 = kb * K_CHUNK
                                    pci = pl.slice(post_norm_mtile, [MLP_M_TILE, K_CHUNK], [0, k0])
                                    wgi = pl.slice(
                                        w_gate,
                                        [K_CHUNK, MLP_OUT_CHUNK],
                                        [layer_hidden_base + k0, o0],
                                    )
                                    gate_acc = pl.matmul_acc(gate_acc, pci, wgi)
                                gate_up_acc_b = pl.assemble(
                                    gate_up_acc_b,
                                    gate_acc,
                                    [0, rel_ob * MLP_OUT_CHUNK],
                                )

                            up_core = (gu_core + UP_PROJ_CORE_SHIFT) % GATE_UP_SPMD_BLOCKS
                            for rel_ob in pl.range(up_core, MLP_BAND_BLOCKS, GATE_UP_SPMD_BLOCKS):
                                o0 = (band_ob0 + rel_ob) * MLP_OUT_CHUNK
                                pc0 = pl.slice(post_norm_mtile, [MLP_M_TILE, K_CHUNK], [0, 0])
                                wu0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, o0])
                                up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                                for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                                    k0 = kb * K_CHUNK
                                    pci = pl.slice(post_norm_mtile, [MLP_M_TILE, K_CHUNK], [0, k0])
                                    wui = pl.slice(
                                        w_up,
                                        [K_CHUNK, MLP_OUT_CHUNK],
                                        [layer_hidden_base + k0, o0],
                                    )
                                    up_acc = pl.matmul_acc(up_acc, pci, wui)
                                gate_up_acc_b = pl.assemble(
                                    gate_up_acc_b,
                                    up_acc,
                                    [MLP_M_TILE, rel_ob * MLP_OUT_CHUNK],
                                )

                        with pl.spmd(SILU_SPMD_BLOCKS, name_hint="silu_spmd", deps=[gate_up_tid]) as silu_tid:
                            silu_core = pl.tile.get_block_idx()
                            for rel_sb in pl.range(silu_core, silu_band_blocks, SILU_SPMD_BLOCKS):
                                so0 = rel_sb * SILU_OUT_CHUNK
                                silu_gate = pl.slice(gate_up_acc_b, [MLP_M_TILE, SILU_OUT_CHUNK], [0, so0])
                                silu_up = pl.slice(gate_up_acc_b, [MLP_M_TILE, SILU_OUT_CHUNK], [MLP_M_TILE, so0])
                                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(silu_gate)), 1.0))
                                mlp_chunk = pl.mul(pl.mul(silu_gate, sigmoid), silu_up)
                                mlp_silu_b = pl.assemble(
                                    mlp_silu_b,
                                    pl.cast(mlp_chunk, target_type=pl.BF16),
                                    [0, so0],
                                )

                        with pl.spmd(
                            DOWN_PROJ_SPMD_BLOCKS,
                            name_hint="down_proj_spmd",
                            deps=[seed_tid, silu_tid],
                        ) as down_tid:
                            down_core = pl.tile.get_block_idx()
                            for hb in pl.range(down_core, down_n_blocks, DOWN_PROJ_SPMD_BLOCKS):
                                h0 = hb * K_CHUNK
                                ms0 = pl.slice(mlp_silu_b, [MLP_M_TILE, MLP_OUT_CHUNK], [0, 0])
                                wd0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base + band_inter0, h0])
                                down_acc = pl.matmul(ms0, wd0, out_dtype=pl.FP32)
                                for cb in pl.pipeline(1, band_k_chunks, stage=2):
                                    c0 = cb * MLP_OUT_CHUNK
                                    msi = pl.slice(mlp_silu_b, [MLP_M_TILE, MLP_OUT_CHUNK], [0, c0])
                                    wdi = pl.slice(
                                        w_down,
                                        [MLP_OUT_CHUNK, K_CHUNK],
                                        [layer_inter_base + band_inter0 + c0, h0],
                                    )
                                    down_acc = pl.matmul_acc(down_acc, msi, wdi)
                                mlp_out_acc_tile = pl.assemble(
                                    mlp_out_acc_tile,
                                    down_acc,
                                    [0, h0],
                                    atomic=pl.AtomicType.Add,
                                )
                        down_chain[mlp_band] = down_tid

            valid_tt = pl.min(MLP_M_TILE, prefill_tokens - m0)
            with pl.spmd(
                DOWN_RESID_SPMD_BLOCKS,
                name_hint="mlp_out_cast_spmd",
                deps=[down_chain[i] for i in range(MLP_PROJ_BANDS)],
            ) as cast_tid:
                cast_core = pl.tile.get_block_idx()
                for hb in pl.range(cast_core, down_n_blocks, DOWN_RESID_SPMD_BLOCKS):
                    h0 = hb * K_CHUNK
                    acc_chunk = pl.slice(mlp_out_acc_tile, [MLP_M_TILE, K_CHUNK], [0, h0])
                    if hb == 0:
                        done_value = pl.slice(acc_chunk, [1, 8], [0, 0])
                        window_done_row = pl.assemble(window_done_row, done_value, [0, 0])
                    out_bf = pl.cast(acc_chunk, target_type=pl.BF16)
                    out_valid = pl.slice(out_bf, [MLP_M_TILE, K_CHUNK], [0, 0], valid_shape=[valid_tt, K_CHUNK])
                    out = pl.assemble(out, out_valid, [m0, h0])
            # This carry array lives outside the dynamic layer/window loops.
            # Keep its backing storage stable: assigning through ``arr[idx]``
            # creates an ArrayType SSA phi at the surrounding ``if`` boundary,
            # while the explicit update is emitted as the same in-place C-array
            # slot write without rebinding the array value.
            pl.array.update_element(kv_tail, kv_tail_idx, cast_tid)

    return out


@pl.jit(auto_scope=False)
def prefill_fwd(
    input_ids: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[BATCH_DYN], pl.INT32],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[BATCH_DYN, VOCAB], pl.FP32]:
    input_ids.bind_dynamic(0, PREFILL_TOKENS_DYN)
    seq_lens.bind_dynamic(0, BATCH_DYN)
    chunk_lens.bind_dynamic(0, BATCH_DYN)
    chunk_offsets.bind_dynamic(0, BATCH_DYN)
    out.bind_dynamic(0, BATCH_DYN)
    block_table.bind_dynamic(0, BLOCK_TABLE_FLAT_DYN)
    slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    k_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    v_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)

    batch = pl.tensor.dim(seq_lens, 0)
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)

    final_hidden = pl.create_tensor([BATCH_PAD, HIDDEN], dtype=pl.BF16)
    kv_tail = pl.array.create(NUM_LAYERS * BATCH_PAD, pl.TASK_ID)

    max_chunk_len = pl.tensor.read(chunk_lens, [0])
    for b in pl.range(1, batch):
        max_chunk_len = pl.max(max_chunk_len, pl.tensor.read(chunk_lens, [b]))

    group_blocks = (max_chunk_len + MLP_M_TILE - 1) // MLP_M_TILE
    for p0_idx in pl.range(group_blocks):
        # This runtime scope is the heap lifetime boundary for the window-local
        # hidden buffers below; removing it keeps old windows live until function exit.
        with pl.scope():
            p0 = p0_idx * MLP_M_TILE
            if batch == 1:
                window_hidden_pair = pl.create_tensor([2 * MLP_M_TILE, HIDDEN], dtype=pl.BF16)
                window_done = pl.create_tensor([NUM_LAYERS * BATCH_PAD, 8], dtype=pl.FP32)
                token_base = pl.cast(pl.tensor.read(chunk_offsets, [0]), pl.INDEX)
                chunk_len_b = pl.tensor.read(chunk_lens, [0])
                valid_tok = pl.min(MLP_M_TILE, pl.max(chunk_len_b - p0, 0))

                with pl.spmd(
                    TOKEN_EMBED_SPMD_BLOCKS,
                    name_hint="token_embed_single",
                ):
                    embed_core = pl.tile.get_block_idx()
                    for work_id in pl.range(
                        embed_core,
                        MLP_M_TILE * TOKEN_EMBED_HIDDEN_BLOCKS,
                        TOKEN_EMBED_SPMD_BLOCKS,
                    ):
                        ti = work_id // TOKEN_EMBED_HIDDEN_BLOCKS
                        embed_hidden_block = work_id - ti * TOKEN_EMBED_HIDDEN_BLOCKS
                        if ti < valid_tok:
                            token_idx = token_base + p0 + ti
                            token_id = pl.tensor.read(input_ids, [token_idx])
                            token_row = pl.cast(token_id, target_type=pl.INDEX)
                            k0 = embed_hidden_block * EMBED_HIDDEN_CHUNK
                            hidden_chunk = pl.slice(
                                embed_weight,
                                [1, EMBED_HIDDEN_CHUNK],
                                [token_row, k0],
                            )
                            window_hidden_pair = pl.assemble(
                                window_hidden_pair,
                                hidden_chunk,
                                [ti, k0],
                            )
                for layer_idx in pl.range(num_layers_actual):
                    read_base = (layer_idx % 2) * MLP_M_TILE
                    write_base = ((layer_idx + 1) % 2) * MLP_M_TILE
                    with pl.scope():
                        window_hidden = pl.slice(window_hidden_pair, [MLP_M_TILE, HIDDEN], [read_base, 0])
                        window_next = pl.slice(window_hidden_pair, [MLP_M_TILE, HIDDEN], [write_base, 0])
                        window_done_layer = pl.slice(
                            window_done,
                            [BATCH_PAD, 8],
                            [layer_idx * BATCH_PAD, 0],
                        )
                        window_next = prefill_layer(
                            window_hidden,
                            seq_lens,
                            chunk_lens,
                            chunk_offsets,
                            pl.cast(p0, pl.INDEX),
                            pl.cast(MLP_M_TILE, pl.INDEX),
                            input_rms_weight,
                            wq,
                            wk,
                            wv,
                            q_norm_weight,
                            k_norm_weight,
                            rope_cos,
                            rope_sin,
                            block_table,
                            slot_mapping,
                            k_cache,
                            v_cache,
                            wo,
                            post_rms_weight,
                            w_gate,
                            w_up,
                            w_down,
                            window_next,
                            window_done_layer,
                            layer_idx,
                            kv_tail,
                        )
                if num_layers_actual > 0:
                    window_fence_idx = (num_layers_actual - 1) * BATCH_PAD
                    window_fence = pl.tensor.read(window_done, [window_fence_idx, 0])
                    pl.tensor.write(window_done, [window_fence_idx, 0], window_fence)
                if chunk_len_b > 0:
                    last_group = (chunk_len_b + MLP_M_TILE - 1) // MLP_M_TILE - 1
                    if p0_idx == last_group:
                        local_last = valid_tok - 1
                        with pl.at(
                            level=pl.Level.CORE_GROUP,
                            name_hint="save_prefill_last_token_single",
                        ):
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                final_base = (num_layers_actual % 2) * MLP_M_TILE
                                final_hidden_chunk = pl.slice(
                                    window_hidden_pair,
                                    [1, K_CHUNK],
                                    [final_base + local_last, k0],
                                )
                                final_hidden = pl.assemble(final_hidden, final_hidden_chunk, [0, k0])
            else:
                group_hidden_pair = pl.create_tensor(
                    [2 * BATCH_PAD * MLP_M_TILE, HIDDEN],
                    dtype=pl.BF16,
                )
                window_done = pl.create_tensor([NUM_LAYERS * BATCH_PAD, 8], dtype=pl.FP32)

                with pl.spmd(
                    TOKEN_EMBED_SPMD_BLOCKS,
                    name_hint="token_embed_group",
                ):
                    embed_core = pl.tile.get_block_idx()
                    for work_id in pl.range(
                        embed_core,
                        BATCH_PAD * MLP_M_TILE * TOKEN_EMBED_HIDDEN_BLOCKS,
                        TOKEN_EMBED_SPMD_BLOCKS,
                    ):
                        row_id = work_id // TOKEN_EMBED_HIDDEN_BLOCKS
                        embed_hidden_block = work_id - row_id * TOKEN_EMBED_HIDDEN_BLOCKS
                        embed_b = row_id // MLP_M_TILE
                        local_ti = row_id - embed_b * MLP_M_TILE
                        if embed_b < batch:
                            token_base = pl.cast(pl.tensor.read(chunk_offsets, [embed_b]), pl.INDEX)
                            chunk_len_b = pl.tensor.read(chunk_lens, [embed_b])
                            valid_tok = pl.min(MLP_M_TILE, pl.max(chunk_len_b - p0, 0))
                            if local_ti < valid_tok:
                                token_idx = token_base + p0 + local_ti
                                group_idx = embed_b * MLP_M_TILE + local_ti
                                token_id = pl.tensor.read(input_ids, [token_idx])
                                token_row = pl.cast(token_id, target_type=pl.INDEX)
                                k0 = embed_hidden_block * EMBED_HIDDEN_CHUNK
                                hidden_chunk = pl.slice(
                                    embed_weight,
                                    [1, EMBED_HIDDEN_CHUNK],
                                    [token_row, k0],
                                )
                                group_hidden_pair = pl.assemble(
                                    group_hidden_pair,
                                    hidden_chunk,
                                    [group_idx, k0],
                                )
                group_rows = BATCH_PAD * MLP_M_TILE
                for layer_idx in pl.range(num_layers_actual):
                    read_base = (layer_idx % 2) * group_rows
                    write_base = ((layer_idx + 1) % 2) * group_rows
                    with pl.scope():
                        group_hidden = pl.slice(group_hidden_pair, [BATCH_PAD * MLP_M_TILE, HIDDEN], [read_base, 0])
                        group_next = pl.slice(group_hidden_pair, [BATCH_PAD * MLP_M_TILE, HIDDEN], [write_base, 0])
                        window_done_layer = pl.slice(
                            window_done,
                            [BATCH_PAD, 8],
                            [layer_idx * BATCH_PAD, 0],
                        )
                        group_next = prefill_layer(
                            group_hidden,
                            seq_lens,
                            chunk_lens,
                            chunk_offsets,
                            pl.cast(p0, pl.INDEX),
                            pl.cast(batch * MLP_M_TILE, pl.INDEX),
                            input_rms_weight,
                            wq,
                            wk,
                            wv,
                            q_norm_weight,
                            k_norm_weight,
                            rope_cos,
                            rope_sin,
                            block_table,
                            slot_mapping,
                            k_cache,
                            v_cache,
                            wo,
                            post_rms_weight,
                            w_gate,
                            w_up,
                            w_down,
                            group_next,
                            window_done_layer,
                            layer_idx,
                            kv_tail,
                        )
                if num_layers_actual > 0:
                    window_fence_base = (num_layers_actual - 1) * BATCH_PAD
                    for fence_b in pl.range(batch):
                        fence_chunk_len = pl.tensor.read(chunk_lens, [fence_b])
                        if p0 < fence_chunk_len:
                            window_fence_idx = window_fence_base + fence_b
                            window_fence = pl.tensor.read(window_done, [window_fence_idx, 0])
                            pl.tensor.write(window_done, [window_fence_idx, 0], window_fence)
                for b in pl.parallel(0, batch, 1):
                    chunk_len_b = pl.tensor.read(chunk_lens, [b])
                    if chunk_len_b > 0:
                        last_group = (chunk_len_b + MLP_M_TILE - 1) // MLP_M_TILE - 1
                        if p0_idx == last_group:
                            valid_tok = pl.min(MLP_M_TILE, chunk_len_b - p0)
                            local_last = b * MLP_M_TILE + valid_tok - 1
                            with pl.at(
                                level=pl.Level.CORE_GROUP,
                                name_hint="save_prefill_last_token_group",
                            ):
                                for kb in pl.range(HIDDEN_BLOCKS):
                                    k0 = kb * K_CHUNK
                                    final_base = (num_layers_actual % 2) * BATCH_PAD * MLP_M_TILE
                                    final_hidden_chunk = pl.slice(
                                        group_hidden_pair,
                                        [1, K_CHUNK],
                                        [final_base + local_last, k0],
                                    )
                                    final_hidden = pl.assemble(final_hidden, final_hidden_chunk, [b, k0])

    out = rms_lm_head(final_hidden, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


def build_tensor_specs(
    batch: int = BATCH_PAD,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
    num_layers: int = NUM_LAYERS,
    vocab_size: int = VOCAB,
    use_max_seq: bool = False,
    chunk_start: int = 0,
    chunk_size: int = 0,
):
    import torch
    from golden import TensorSpec

    assert hidden_size == num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    vocab = vocab_size
    if max_seq <= 0:
        raise ValueError(f"max_seq must be positive, got {max_seq}")
    if max_seq > MAX_SEQ:
        raise ValueError(f"max_seq must be <= model MAX_SEQ ({MAX_SEQ}), got {max_seq}")
    if chunk_start < 0:
        raise ValueError(f"chunk_start must be non-negative, got {chunk_start}")
    if chunk_size < 0:
        raise ValueError(f"chunk_size must be non-negative, got {chunk_size}")
    max_blocks_per_seq = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch * max_blocks_per_seq
    layer_cache_rows = num_blocks * num_kv_heads * BLOCK_SIZE
    cache_rows = num_layers * layer_cache_rows

    if use_max_seq:
        prompt_lens_values = torch.full((batch,), max_seq, dtype=torch.int32)
    else:
        n_blocks = max(1, (max_seq + TOK_TILE - 1) // TOK_TILE)
        prompt_lens_values = torch.minimum(
            ((torch.arange(batch, dtype=torch.int32) % n_blocks) + 1) * TOK_TILE,
            torch.full((batch,), max_seq, dtype=torch.int32),
        )

    if chunk_size > 0:
        chunk_end = min(max_seq, chunk_start + chunk_size)
        prompt_lens_values[-1] = max(int(prompt_lens_values[-1].item()), chunk_end)
        seq_lens_values = torch.minimum(
            prompt_lens_values,
            torch.full((batch,), chunk_end, dtype=torch.int32),
        )
        chunk_lens_values = torch.clamp(
            seq_lens_values - chunk_start,
            min=0,
        ).to(torch.int32)
    else:
        seq_lens_values = prompt_lens_values
        chunk_lens_values = seq_lens_values.clone()

    chunk_offsets_values = torch.zeros(batch, dtype=torch.int32)
    if batch > 1:
        chunk_offsets_values[1:] = torch.cumsum(chunk_lens_values[:-1], dim=0)
    total_tokens = int(chunk_lens_values.sum().item())
    if total_tokens <= 0:
        raise ValueError("chunked prefill requires at least one token in the current chunk")

    def init_input_ids():
        return torch.arange(total_tokens, dtype=torch.int32) % vocab

    def init_seq_lens():
        return seq_lens_values.clone()

    def init_chunk_lens():
        return chunk_lens_values.clone()

    def init_chunk_offsets():
        return chunk_offsets_values.clone()

    def init_rms_weight():
        return torch.rand(num_layers, hidden_size) - 0.5

    def init_wq():
        return (torch.rand(num_layers * hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_wk():
        return (torch.rand(num_layers * hidden_size, kv_hidden) - 0.5) / hidden_size ** 0.5

    def init_wv():
        return (torch.rand(num_layers * hidden_size, kv_hidden) - 0.5) / hidden_size ** 0.5

    def init_q_norm_weight():
        return torch.rand(num_layers, head_dim) - 0.5

    def init_k_norm_weight():
        return torch.rand(num_layers, head_dim) - 0.5

    def init_rope_cos():
        return torch.rand(MAX_SEQ, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(MAX_SEQ, head_dim) - 0.5

    def init_block_table():
        return torch.arange(num_blocks, dtype=torch.int32)

    def init_slot_mapping():
        slots = torch.empty(total_tokens, dtype=torch.int32)
        for b in range(batch):
            seq_len = int(seq_lens_values[b].item())
            chunk_len = int(chunk_lens_values[b].item())
            token_idx = int(chunk_offsets_values[b].item())
            chunk_start_b = seq_len - chunk_len
            for local_pos in range(chunk_len):
                pos = chunk_start_b + local_pos
                logical_block = pos // BLOCK_SIZE
                page_offset = pos % BLOCK_SIZE
                phys_block = b * max_blocks_per_seq + logical_block
                slots[token_idx] = phys_block * BLOCK_SIZE + page_offset
                token_idx += 1
        return slots

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_wo():
        return (torch.rand(num_layers * hidden_size, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_post_rms_weight():
        return torch.ones(num_layers, hidden_size)

    def init_w_gate():
        return (torch.rand(num_layers * hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_up():
        return (torch.rand(num_layers * hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

    def init_w_down():
        return (torch.rand(num_layers * intermediate_size, hidden_size) - 0.5) / intermediate_size ** 0.5

    def init_final_norm_weight():
        return torch.ones(1, hidden_size)

    def init_lm_head_weight():
        return (torch.rand(vocab, hidden_size) - 0.5) / hidden_size ** 0.5

    def init_embed_weight():
        return torch.rand(vocab, hidden_size) - 0.5

    return [
        TensorSpec("input_ids", [total_tokens], torch.int32, init_value=init_input_ids),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("chunk_lens", [batch], torch.int32, init_value=init_chunk_lens),
        TensorSpec("chunk_offsets", [batch], torch.int32, init_value=init_chunk_offsets),
        TensorSpec("input_rms_weight", [num_layers, hidden_size], torch.float32,
                   init_value=init_rms_weight),
        TensorSpec("wq", [num_layers * hidden_size, hidden_size], torch.bfloat16,
                   init_value=init_wq),
        TensorSpec("wk", [num_layers * hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wk),
        TensorSpec("wv", [num_layers * hidden_size, kv_hidden], torch.bfloat16,
                   init_value=init_wv),
        TensorSpec("q_norm_weight", [num_layers, head_dim], torch.float32,
                   init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [num_layers, head_dim], torch.float32,
                   init_value=init_k_norm_weight),
        TensorSpec("rope_cos", [MAX_SEQ, head_dim], torch.float32,
                   init_value=init_rope_cos),
        TensorSpec("rope_sin", [MAX_SEQ, head_dim], torch.float32,
                   init_value=init_rope_sin),
        TensorSpec("block_table", [batch * max_blocks_per_seq], torch.int32,
                   init_value=init_block_table),
        TensorSpec("slot_mapping", [total_tokens], torch.int32,
                   init_value=init_slot_mapping),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_k_cache),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16,
                   init_value=init_v_cache),
        TensorSpec("wo", [num_layers * hidden_size, hidden_size], torch.bfloat16,
                   init_value=init_wo),
        TensorSpec("post_rms_weight", [num_layers, hidden_size], torch.float32,
                   init_value=init_post_rms_weight),
        TensorSpec("w_gate", [num_layers * hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_gate),
        TensorSpec("w_up", [num_layers * hidden_size, intermediate_size], torch.bfloat16,
                   init_value=init_w_up),
        TensorSpec("w_down", [num_layers * intermediate_size, hidden_size], torch.bfloat16,
                   init_value=init_w_down),
        TensorSpec("final_norm_weight", [1, hidden_size], torch.float32,
                   init_value=init_final_norm_weight),
        TensorSpec("lm_head_weight", [vocab, hidden_size], torch.bfloat16,
                   init_value=init_lm_head_weight),
        TensorSpec("embed_weight", [vocab, hidden_size], torch.bfloat16,
                   init_value=init_embed_weight),
        TensorSpec("out", [batch, vocab], torch.float32, is_output=True),
    ]


def golden_qwen3_14b_prefill(tensors):
    """Reference implementation for full-layer prefill plus final logits.

    Mirrors the kernel precision path through RMSNorm, Q/K/V projection, RoPE,
    KV cache update, online-softmax attention, output projection, post RMSNorm,
    SwiGLU MLP, final RMSNorm, and the LM head projection.
    """
    import math

    import torch

    input_ids = tensors["input_ids"]
    seq_lens = tensors["seq_lens"]
    chunk_lens = tensors["chunk_lens"]
    chunk_offsets = tensors["chunk_offsets"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]
    final_norm_weight = tensors["final_norm_weight"]
    lm_head_weight = tensors["lm_head_weight"]
    embed_weight = tensors["embed_weight"]

    batch = seq_lens.shape[0]
    total_tokens = input_ids.shape[0]
    max_seq = rope_cos.shape[0]
    hidden_size = embed_weight.shape[1]
    kv_hidden = wk.shape[1]
    head_dim = rope_cos.shape[1]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = EPS
    max_blocks_per_seq = block_table.shape[0] // batch
    num_layers = input_rms_weight.shape[0]
    intermediate_size = w_gate.shape[1]
    layer_cache_rows = batch * max_blocks_per_seq * BLOCK_SIZE
    k_cache_bsnd = k_cache.reshape(-1, kv_hidden)
    v_cache_bsnd = v_cache.reshape(-1, kv_hidden)

    def tiled_lm_head(lhs, rhs_t, k_chunk, vocab_chunk):
        out = torch.zeros(lhs.shape[0], rhs_t.shape[0], dtype=torch.float32)
        for k0 in range(0, lhs.shape[1], k_chunk):
            lhs_chunk = lhs[:, k0:k0 + k_chunk].float()
            out += lhs_chunk @ rhs_t[:, k0:k0 + k_chunk].float().T
        return out

    hidden = embed_weight.index_select(0, input_ids.long()).clone()
    for layer_idx in range(num_layers):
        layer_hidden_base = layer_idx * hidden_size
        layer_inter_base = layer_idx * intermediate_size
        layer_cache_base = layer_idx * layer_cache_rows
        input_rms_weight_f = input_rms_weight[layer_idx:layer_idx + 1, :].float()
        wq_f = wq[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        wk_f = wk[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        wv_f = wv[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        wo_f = wo[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        post_rms_f = post_rms_weight[layer_idx:layer_idx + 1, :].float()
        w_gate_f = w_gate[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        w_up_f = w_up[layer_hidden_base:layer_hidden_base + hidden_size, :].float()
        w_down_f = w_down[layer_inter_base:layer_inter_base + intermediate_size, :].float()

        out_t = torch.zeros(total_tokens, hidden_size, dtype=torch.float32)
        for b in range(batch):
            seq_len_b = int(seq_lens[b].item())
            chunk_len_b = int(chunk_lens[b].item())
            token_base = int(chunk_offsets[b].item())
            if chunk_len_b <= 0:
                continue

            S = chunk_len_b
            chunk_start_b = seq_len_b - chunk_len_b

            # ── Scope 1: RMSNorm + Q/K/V projection ──
            x = hidden[token_base:token_base + S, :].float()
            variance = x.square().mean(dim=-1, keepdim=True) + eps
            inv_rms = 1.0 / torch.sqrt(variance)
            normed_bf16 = (x * inv_rms * input_rms_weight_f).to(torch.bfloat16)

            normed_f32 = normed_bf16.float()
            q_proj_f = normed_f32 @ wq_f
            k_proj_f = normed_f32 @ wk_f
            v_proj_f = normed_f32 @ wv_f

            # Per-head RMSNorm on Q and K (FP32).
            q_norm_f = q_norm_weight[layer_idx:layer_idx + 1, :].float().view(1, 1, head_dim)
            k_norm_f = k_norm_weight[layer_idx:layer_idx + 1, :].float().view(1, 1, head_dim)
            q_proj_view = q_proj_f.view(S, num_heads, head_dim)
            q_var = q_proj_view.pow(2).mean(dim=-1, keepdim=True)
            q_proj_view = q_proj_view * torch.rsqrt(q_var + eps) * q_norm_f
            q_proj_f = q_proj_view.reshape(S, hidden_size)
            k_proj_view = k_proj_f.view(S, num_kv_heads, head_dim)
            k_var = k_proj_view.pow(2).mean(dim=-1, keepdim=True)
            k_proj_view = k_proj_view * torch.rsqrt(k_var + eps) * k_norm_f
            k_proj_f = k_proj_view.reshape(S, kv_hidden)

            # ── Scope 2: RoPE + KV cache + causal attention ──
            cos_row = rope_cos[chunk_start_b:seq_len_b, :]
            sin_row = rope_sin[chunk_start_b:seq_len_b, :]
            cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
            sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

            # K RoPE + cache write.
            k_row = k_proj_f.view(S, num_kv_heads, head_dim)
            k_lo, k_hi = k_row[:, :, :half], k_row[:, :, half:]
            k_rot = torch.cat([
                k_lo * cos_lo.unsqueeze(1) - k_hi * sin_lo.unsqueeze(1),
                k_hi * cos_hi.unsqueeze(1) + k_lo * sin_hi.unsqueeze(1),
            ], dim=-1).to(torch.bfloat16)
            for pos in range(S):
                slot = int(slot_mapping[token_base + pos].item())
                slot_block = slot // BLOCK_SIZE
                slot_offset = slot % BLOCK_SIZE
                for ki in range(num_kv_heads):
                    cache_row = layer_cache_base + slot_block * BLOCK_SIZE + slot_offset
                    cache_col = ki * head_dim
                    k_cache_bsnd[cache_row, cache_col:cache_col + head_dim] = k_rot[pos, ki, :]

            # V cache write.
            v_row_bf16 = v_proj_f.view(S, num_kv_heads, head_dim).to(torch.bfloat16)
            for pos in range(S):
                slot = int(slot_mapping[token_base + pos].item())
                slot_block = slot // BLOCK_SIZE
                slot_offset = slot % BLOCK_SIZE
                for ki in range(num_kv_heads):
                    cache_row = layer_cache_base + slot_block * BLOCK_SIZE + slot_offset
                    cache_col = ki * head_dim
                    v_cache_bsnd[cache_row, cache_col:cache_col + head_dim] = v_row_bf16[pos, ki, :]

            # Q RoPE -> BF16.
            q_row = q_proj_f.view(S, num_heads, head_dim)
            q_lo, q_hi = q_row[:, :, :half], q_row[:, :, half:]
            q_rot_bf16 = torch.cat([
                q_lo * cos_lo.unsqueeze(1) - q_hi * sin_lo.unsqueeze(1),
                q_hi * cos_hi.unsqueeze(1) + q_lo * sin_hi.unsqueeze(1),
            ], dim=-1).to(torch.bfloat16)

            # Causal attention with tiled online softmax.
            max_blocks = (seq_len_b + SEQ_TILE - 1) // SEQ_TILE
            padded_len = max_blocks * SEQ_TILE
            ctx_lens = torch.arange(chunk_start_b + 1, seq_len_b + 1)
            col_idx = torch.arange(SEQ_TILE)
            attn_result = torch.zeros(S, hidden_size, dtype=torch.float32)

            for kvh in range(num_kv_heads):
                for qg in range(q_groups):
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_grp = q_rot_bf16[:S, q_base:q_base + Q_HEAD_BATCH, :]

                    cache_base = b * max_blocks_per_seq
                    k_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                    v_padded = torch.zeros(padded_len, head_dim, dtype=torch.bfloat16)
                    for sb in range(max_blocks):
                        pbid = int(block_table[cache_base + sb].item())
                        cache_row0 = layer_cache_base + pbid * BLOCK_SIZE
                        cache_col = kvh * head_dim
                        s0 = sb * SEQ_TILE
                        k_padded[s0:s0 + BLOCK_SIZE] = k_cache_bsnd[
                            cache_row0:cache_row0 + BLOCK_SIZE,
                            cache_col:cache_col + head_dim,
                        ]
                        v_padded[s0:s0 + BLOCK_SIZE] = v_cache_bsnd[
                            cache_row0:cache_row0 + BLOCK_SIZE,
                            cache_col:cache_col + head_dim,
                        ]

                    oi = li = mi = None

                    for sb in range(max_blocks):
                        s0 = sb * SEQ_TILE
                        k_tile = k_padded[s0:s0 + SEQ_TILE]
                        v_tile = v_padded[s0:s0 + SEQ_TILE]

                        raw_scores = (q_grp @ k_tile.T).float()

                        valid_lens = torch.clamp(ctx_lens - s0, min=0, max=SEQ_TILE)
                        mask = col_idx.unsqueeze(0) < valid_lens.unsqueeze(1)
                        raw_scores[~mask.unsqueeze(1).expand_as(raw_scores)] = torch.finfo(torch.float32).min
                        scores = raw_scores * scale

                        cur_mi = scores.max(dim=-1, keepdim=True).values
                        exp_scores = torch.exp(scores - cur_mi)
                        exp_bf16 = exp_scores.to(torch.bfloat16)
                        cur_li = exp_bf16.float().sum(dim=-1, keepdim=True)

                        oi_tmp = (exp_bf16 @ v_tile).float()

                        if sb == 0:
                            oi, li, mi = oi_tmp, cur_li, cur_mi
                        else:
                            active = valid_lens > 0
                            if active.any():
                                a = active
                                mi_new = torch.maximum(mi[a], cur_mi[a])
                                alpha = torch.exp(mi[a] - mi_new)
                                beta = torch.exp(cur_mi[a] - mi_new)
                                oi[a] = oi[a] * alpha + oi_tmp[a] * beta
                                li[a] = alpha * li[a] + beta * cur_li[a]
                                mi[a] = mi_new

                    ctx = oi / li
                    attn_result[:, q_base * head_dim:(q_base + Q_HEAD_BATCH) * head_dim] = \
                        ctx.reshape(S, Q_HEAD_BATCH * head_dim)

            attn_bf16 = attn_result.to(torch.bfloat16)

            # ── Scope 3: output projection + residual + post RMSNorm + MLP ──
            attn_f = attn_bf16.float()
            hs = hidden[token_base:token_base + S, :].float()

            # Output projection + first residual.
            resid1 = torch.matmul(attn_f, wo_f) + hs

            # Post-attention RMSNorm.
            variance = resid1.pow(2).mean(dim=-1, keepdim=True)
            post_inv_rms = torch.rsqrt(variance + eps)
            normed_bf16 = (resid1 * post_inv_rms * post_rms_f).bfloat16()

            # SwiGLU MLP.
            normed_post_f = normed_bf16.float()
            gate = torch.matmul(normed_post_f, w_gate_f)
            up = torch.matmul(normed_post_f, w_up_f)
            mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
            down = torch.matmul(mlp_bf16.float(), w_down_f)

            # Final residual -> BF16.
            out_t[token_base:token_base + S, :] = (down + resid1).bfloat16().float()

        hidden = out_t.to(torch.bfloat16)

    final_hidden = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)
    for b in range(batch):
        chunk_len_b = int(chunk_lens[b].item())
        token_base = int(chunk_offsets[b].item())
        if chunk_len_b > 0:
            final_hidden[b, :] = hidden[token_base + chunk_len_b - 1, :]

    variance = final_hidden.float().pow(2).mean(dim=-1, keepdim=True)
    final_normed = (
        final_hidden.float()
        * torch.rsqrt(variance + eps)
        * final_norm_weight.float()
    ).bfloat16()
    tensors["out"][:] = tiled_lm_head(final_normed, lm_head_weight, LM_HEAD_K_CHUNK, VOCAB_CHUNK)


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-b", "--batch", type=int, default=BATCH_PAD,
                        help=("User-visible batch size. Host allocates every "
                              "batch-dependent tensor at exactly this size; "
                              "every kernel signature batch-axis dim is a "
                              "pl.dynamic() variable, so a single compiled "
                              "program serves any batch <= host KV-cache "
                              "capacity. Default: %(default)s"))
    parser.add_argument(
        "--enable-chip-swimlane",
        nargs="?",
        const=4,
        default=0,
        type=int,
        metavar="PERF_LEVEL",
        choices=[0, 1, 2, 3, 4],
        help="Enable chip swimlane perf capture at the given granularity level. Bare flag "
             "= level 4 (full). Levels: 1=AICore timing, 2=+dispatch/fanout, 3=+sched "
             "phases, 4=+orch phases; 0 (default) disables.",
    )
    parser.add_argument("--enable-scope-stats", action="store_true", default=False,
                        help="enable PTO2 scope lifetime statistics")
    parser.add_argument("--enable-dep-gen", action="store_true", default=False,
                        help="capture the task dependency graph to dfx_outputs/deps.json "
                             "(render with simpler_setup.tools.deps_viewer). Opt-in AICPU-side "
                             "DFX, off by default. Keep --num-layers small when capturing: the "
                             "per-run SHM record buffer can overflow ('records dropped') on the "
                             "full 40-layer graph.")
    parser.add_argument("--max-seq", type=int, default=DEFAULT_TEST_MAX_SEQ,
                        help="synthetic max sequence length, up to model MAX_SEQ")
    parser.add_argument("--use-max-seq", action="store_true", default=False,
                        help="set all synthetic seq_lens to --max-seq")
    parser.add_argument("--chunk-start", type=int, default=0,
                        help="absolute start position for a synthetic current chunk")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="current chunk size for synthetic chunked-prefill tests; 0 means full prompt")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--save-data", action="store_true", default=False,
                        help="persist inputs + golden for replay (off: large fixtures)")
    parser.add_argument("--no-golden", action="store_true", default=False,
                        help="skip host golden computation and output validation")
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_fwd,
        specs=build_tensor_specs(
            batch=args.batch,
            max_seq=args.max_seq,
            num_layers=args.num_layers,
            use_max_seq=args.use_max_seq,
            chunk_start=args.chunk_start,
            chunk_size=args.chunk_size,
        ),
        golden_fn=None if args.no_golden else golden_qwen3_14b_prefill,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_scope_stats=args.enable_scope_stats,
            enable_dep_gen=args.enable_dep_gen,
        ),
        rtol=5e-3,
        atol=5e-3,
        save_data=args.save_data,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


@pl.jit.host
def qwen3_prefill_host(  # noqa: PLR0913 — HOST entry over prefill_fwd for serving signature-mode compile
    input_ids: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[BATCH_DYN], pl.INT32],
    input_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    wq: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[LAYER_DYN, HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[D.rope_seq, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[D.rope_seq, HEAD_DIM], pl.FP32],
    block_table: pl.Tensor[[BLOCK_TABLE_FLAT_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[PREFILL_TOKENS_DYN], pl.INT32],
    k_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16],
    w_gate: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[LAYER_HIDDEN_ROWS_DYN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[LAYER_DYN, HIDDEN], pl.FP32],
    final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    lm_head_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    embed_weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor:
    return prefill_fwd(
        input_ids,
        seq_lens,
        chunk_lens,
        chunk_offsets,
        input_rms_weight,
        wq,
        wk,
        wv,
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_sin,
        block_table,
        slot_mapping,
        k_cache,
        v_cache,
        wo,
        post_rms_weight,
        w_gate,
        w_up,
        w_down,
        final_norm_weight,
        lm_head_weight,
        embed_weight,
        out,
    )
