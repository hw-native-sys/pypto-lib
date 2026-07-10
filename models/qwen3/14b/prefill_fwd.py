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
(`USER_BATCH_DYN` / `PREFILL_TOKENS_DYN` / `KV_CACHE_ROWS_DYN` /
`BLOCK_TABLE_FLAT_DYN`), so a single compiled program serves any
`user_batch <= host KV-cache capacity`. Host allocates the input/output
hidden states and slot mapping as packed token-major tensors with leading
dim `T = sum(chunk_lens)` (no `[batch, max_seq]` padding). `seq_lens`
stores the absolute sequence length after the current chunk; `chunk_lens`
and `chunk_offsets` identify each batch row's slice inside the packed chunk.

Unlike the decode path, prefill iterates the batch dim with step 1 (one
batch element per outer iteration) and every matmul tile's M dim is
governed by `TOK_TILE` (independent of batch). Therefore prefill does
NOT need decode's `batch_padded` round-up + `valid_shape` zero-pad +
vec-to-vec textract trim machinery on the batch axis. The per-token
`valid_tok = pl.min(TOK_TILE, chunk_len_b - p0)` + `valid_shape` pattern
already handles intra-batch sequence-length variation.
"""
import pypto.language as pl

from config import (
    ATTN_SCALE,
    BATCH,
    BATCH_TILE,
    BLOCK_TABLE_FLAT_DYN,
    EPS,
    HALF_DIM,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    HIDDEN_INV,
    INTERMEDIATE,
    KV_CACHE_ROWS_DYN,
    KV_HIDDEN,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LM_HEAD_K_CHUNK,
    MAX_SEQ as MODEL_MAX_SEQ,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    Q_GROUPS,
    Q_HEAD_BATCH,
    Q_HEAD_PAD,
    Q_PER_KV,
    TOTAL_Q_GROUPS,
    USER_BATCH_DYN,
    VOCAB,
    VOCAB_CHUNK,
)
from rms_lm_head import rms_lm_head

PREFILL_TOKENS_DYN = pl.dynamic("PREFILL_TOKENS_DYN")

# Single-layer prefill constants. Keep these local because config.py is shared
# with the decode kernels and uses decode-tuned tiling constants.
MAX_SEQ = MODEL_MAX_SEQ
DEFAULT_TEST_MAX_SEQ = 128
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
TOK_TILE = 64
ROPE_SPMD_BLOCKS = 32
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
QKPV_TOK_BATCH = 4
QKPV_BATCH_ROWS = QKPV_TOK_BATCH * Q_HEAD_PAD
SEQ_TILE = 128
SB_BATCH = 64
BLOCK_SIZE = SEQ_TILE
MLP_OUT_CHUNK = 128
DOWN_K_PARTS = 3
HIDDEN_BLOCKS = HIDDEN // K_CHUNK
Q_OUT_BLOCKS = HIDDEN // Q_OUT_CHUNK
KV_OUT_BLOCKS = KV_HIDDEN // KV_OUT_CHUNK
MLP_OUT_BLOCKS = INTERMEDIATE // MLP_OUT_CHUNK
MLP_PROJ_BANDS = 2
MLP_BAND_BLOCKS = MLP_OUT_BLOCKS // MLP_PROJ_BANDS
DOWN_PART_BLOCKS = (MLP_OUT_BLOCKS + DOWN_K_PARTS - 1) // DOWN_K_PARTS
DOWN_PART_WORK_ITEMS = HIDDEN_BLOCKS * DOWN_K_PARTS
RMSNORM_TOK_GROUP = 8
RMSNORM_TOK_GROUPS = (TOK_TILE + RMSNORM_TOK_GROUP - 1) // RMSNORM_TOK_GROUP
RMSNORM_WORK_ITEMS = RMSNORM_TOK_GROUPS
RMSNORM_SPMD_BLOCKS = 8
Q_PROJ_SPMD_BLOCKS = 16
KV_PROJ_SPMD_BLOCKS = 8
QK_NORM_SPMD_BLOCKS = NUM_KV_HEADS
POST_RMSNORM_SPMD_BLOCKS = 8
DOWN_RESID_SPMD_BLOCKS = 20
OUT_PROJ_SPMD_BLOCKS = 20
SILU_SPMD_BLOCKS = 24
GATE_PROJ_SPMD_BLOCKS = 24
UP_PROJ_SPMD_BLOCKS = 24
DOWN_PROJ_SPMD_BLOCKS = 24


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
) -> tuple[
    pl.Tensor[[TOK_TILE, HIDDEN], pl.BF16],
    pl.Tensor[[ATTN_PHASE_ACC_STAT_ROWS, 1], pl.FP32],
    pl.Tensor[[ATTN_PHASE_ACC_SCORE_ROWS, HEAD_DIM], pl.FP32],
]:
    cur_mi_phase = pl.create_tensor([ATTN_PHASE_ACC_STAT_ROWS, 1], dtype=pl.FP32)
    if finalize_tok > 0:
        block_ctx_len = chunk_start + p0 + final_ti0 + finalize_tok
        block_ctx_blocks = (block_ctx_len + SEQ_TILE - 1) // SEQ_TILE
        for sb_chunk in pl.range(0, block_ctx_blocks, SB_BATCH):
            for si in pl.range(SB_BATCH):
                sb = sb_chunk + si
                if sb < block_ctx_blocks:
                    for phase_core in pl.spmd(
                        ATTN_PHASE_SPMD_BLOCKS,
                        name_hint="qk_pv_online_phase_spmd",
                        sync_start=True,
                    ):
                        for work_id in pl.range(phase_core, ATTN_PHASE_WORK_ITEMS, ATTN_PHASE_SPMD_BLOCKS):
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
                                        cache_row0 = layer_cache_base + (
                                            pbid * NUM_KV_HEADS + kvh
                                        ) * BLOCK_SIZE
                                        k_tile = pl.slice(k_cache, [SEQ_TILE, HEAD_DIM], [cache_row0, 0])
                                        v_tile = pl.slice(v_cache, [SEQ_TILE, HEAD_DIM], [cache_row0, 0])
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
                                                        micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_PAD
                                                        + gi * ATTN_TOK_GROUP * Q_HEAD_PAD
                                                        + dd * Q_HEAD_PAD
                                                    )
                                                    acc_li_row0 = (
                                                        micro_id * ATTN_TOK_GROUP * TOTAL_Q_GROUPS * Q_HEAD_BATCH_PAD
                                                        + gi * ATTN_TOK_GROUP * Q_HEAD_BATCH_PAD
                                                        + dd * Q_HEAD_BATCH_PAD
                                                    )
                                                    oi_tmp_sb = pl.slice(oi_tmp, [Q_HEAD_BATCH_PAD, HEAD_DIM], [0, 0])
                                                    cur_mi_acc = pl.slice(cur_mi, [Q_HEAD_BATCH_PAD, 1], [0, 0])
                                                    cur_li_acc = pl.slice(cur_li, [Q_HEAD_BATCH_PAD, 1], [0, 0])
                                                    if sb == 0:
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
        for final_core in pl.spmd(FINALIZE_SPMD_BLOCKS, name_hint="attention_finalize_phase_spmd"):
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
) -> tuple[
    pl.Tensor[[TOK_TILE, HIDDEN], pl.BF16],
    pl.Tensor[[ATTN_PHASE_ACC_STAT_ROWS, 1], pl.FP32],
    pl.Tensor[[ATTN_PHASE_ACC_SCORE_ROWS, HEAD_DIM], pl.FP32],
]:
    for phase_core in pl.spmd(
        ATTN_PHASE_SPMD_BLOCKS,
        name_hint="qk_pv_skew_probe_spmd",
        sync_start=True,
    ):
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
            cache_row0 = layer_cache_base + (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
            k_tile = pl.slice(k_cache, [SEQ_TILE, HEAD_DIM], [cache_row0, 0])
            v_tile = pl.slice(v_cache, [SEQ_TILE, HEAD_DIM], [cache_row0, 0])
            for dd0 in pl.pipeline(0, ATTN_TOK_GROUP, QKPV_TOK_BATCH, stage=3):
                ti0 = attn_ti0 + dd0
                ti1 = ti0 + 1
                ti2 = ti0 + 2
                ti3 = ti0 + 3
                q_row0 = ti0 * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                q_row1 = ti1 * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                q_row2 = ti2 * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                q_row3 = ti3 * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                q0 = pl.slice(
                    all_q_padded_tile,
                    [Q_HEAD_PAD, HEAD_DIM],
                    [q_row0, 0],
                )
                q1 = pl.slice(
                    all_q_padded_tile,
                    [Q_HEAD_PAD, HEAD_DIM],
                    [q_row1, 0],
                )
                q2 = pl.slice(
                    all_q_padded_tile,
                    [Q_HEAD_PAD, HEAD_DIM],
                    [q_row2, 0],
                )
                q3 = pl.slice(
                    all_q_padded_tile,
                    [Q_HEAD_PAD, HEAD_DIM],
                    [q_row3, 0],
                )
                q_batch = pl.reshape(
                    pl.concat(
                        pl.concat(
                            pl.reshape(q0, [1, Q_HEAD_PAD * HEAD_DIM]),
                            pl.reshape(q1, [1, Q_HEAD_PAD * HEAD_DIM]),
                        ),
                        pl.concat(
                            pl.reshape(q2, [1, Q_HEAD_PAD * HEAD_DIM]),
                            pl.reshape(q3, [1, Q_HEAD_PAD * HEAD_DIM]),
                        ),
                    ),
                    [QKPV_BATCH_ROWS, HEAD_DIM],
                )
                raw_scores_batch = pl.matmul(
                    q_batch,
                    k_tile,
                    b_trans=True,
                    out_dtype=pl.FP32,
                )
                raw_scores0 = pl.slice(raw_scores_batch, [Q_HEAD_BATCH_PAD, SEQ_TILE], [0, 0])
                raw_scores1 = pl.slice(
                    raw_scores_batch,
                    [Q_HEAD_BATCH_PAD, SEQ_TILE],
                    [Q_HEAD_PAD, 0],
                )
                raw_scores2 = pl.slice(
                    raw_scores_batch,
                    [Q_HEAD_BATCH_PAD, SEQ_TILE],
                    [2 * Q_HEAD_PAD, 0],
                )
                raw_scores3 = pl.slice(
                    raw_scores_batch,
                    [Q_HEAD_BATCH_PAD, SEQ_TILE],
                    [3 * Q_HEAD_PAD, 0],
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
                chunk_pos1 = p0 + ti1
                ctx_len1 = chunk_start + chunk_pos1 + 1
                scores1 = pl.fillpad(
                    pl.set_validshape(
                        pl.mul(raw_scores1, ATTN_SCALE),
                        Q_HEAD_BATCH,
                        ctx_len1,
                    ),
                    pad_value=pl.PadValue.min,
                )
                chunk_pos2 = p0 + ti2
                ctx_len2 = chunk_start + chunk_pos2 + 1
                scores2 = pl.fillpad(
                    pl.set_validshape(
                        pl.mul(raw_scores2, ATTN_SCALE),
                        Q_HEAD_BATCH,
                        ctx_len2,
                    ),
                    pad_value=pl.PadValue.min,
                )
                chunk_pos3 = p0 + ti3
                ctx_len3 = chunk_start + chunk_pos3 + 1
                scores3 = pl.fillpad(
                    pl.set_validshape(
                        pl.mul(raw_scores3, ATTN_SCALE),
                        Q_HEAD_BATCH,
                        ctx_len3,
                    ),
                    pad_value=pl.PadValue.min,
                )
                scores_batch = pl.reshape(
                    pl.concat(
                        pl.concat(
                            pl.reshape(scores0, [1, Q_HEAD_BATCH_PAD * SEQ_TILE]),
                            pl.reshape(scores1, [1, Q_HEAD_BATCH_PAD * SEQ_TILE]),
                        ),
                        pl.concat(
                            pl.reshape(scores2, [1, Q_HEAD_BATCH_PAD * SEQ_TILE]),
                            pl.reshape(scores3, [1, Q_HEAD_BATCH_PAD * SEQ_TILE]),
                        ),
                    ),
                    [QKPV_BATCH_ROWS, SEQ_TILE],
                )
                cur_mi_batch = pl.row_max(scores_batch)
                exp_scores_batch = pl.exp(pl.row_expand_sub(scores_batch, cur_mi_batch))
                exp_scores_bf16_batch = pl.cast(exp_scores_batch, target_type=pl.BF16)
                cur_li_batch = pl.row_sum(
                    pl.cast(exp_scores_bf16_batch, target_type=pl.FP32),
                )
                oi_tmp_batch = pl.matmul(
                    exp_scores_bf16_batch,
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
                    pl.slice(oi_tmp_batch, [Q_HEAD_BATCH_PAD, HEAD_DIM], [0, 0]),
                    [acc_exp_row0, 0],
                )
                cur_li_phase = pl.assemble(
                    cur_li_phase,
                    pl.slice(cur_li_batch, [Q_HEAD_BATCH_PAD, 1], [0, 0]),
                    [acc_li_row0, 0],
                )
                oi_tmp_phase = pl.assemble(
                    oi_tmp_phase,
                    pl.slice(oi_tmp_batch, [Q_HEAD_BATCH_PAD, HEAD_DIM], [Q_HEAD_PAD, 0]),
                    [acc_exp_row0 + Q_HEAD_PAD, 0],
                )
                cur_li_phase = pl.assemble(
                    cur_li_phase,
                    pl.slice(cur_li_batch, [Q_HEAD_BATCH_PAD, 1], [Q_HEAD_PAD, 0]),
                    [acc_li_row0 + Q_HEAD_BATCH_PAD, 0],
                )
                oi_tmp_phase = pl.assemble(
                    oi_tmp_phase,
                    pl.slice(oi_tmp_batch, [Q_HEAD_BATCH_PAD, HEAD_DIM], [2 * Q_HEAD_PAD, 0]),
                    [acc_exp_row0 + 2 * Q_HEAD_PAD, 0],
                )
                cur_li_phase = pl.assemble(
                    cur_li_phase,
                    pl.slice(cur_li_batch, [Q_HEAD_BATCH_PAD, 1], [2 * Q_HEAD_PAD, 0]),
                    [acc_li_row0 + 2 * Q_HEAD_BATCH_PAD, 0],
                )
                oi_tmp_phase = pl.assemble(
                    oi_tmp_phase,
                    pl.slice(oi_tmp_batch, [Q_HEAD_BATCH_PAD, HEAD_DIM], [3 * Q_HEAD_PAD, 0]),
                    [acc_exp_row0 + 3 * Q_HEAD_PAD, 0],
                )
                cur_li_phase = pl.assemble(
                    cur_li_phase,
                    pl.slice(cur_li_batch, [Q_HEAD_BATCH_PAD, 1], [3 * Q_HEAD_PAD, 0]),
                    [acc_li_row0 + 3 * Q_HEAD_BATCH_PAD, 0],
                )

        pl.system.syncall(core_type="mix")

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
    return attn_tile, cur_li_phase, oi_tmp_phase


@pl.jit.inline(auto_scope=False)
def prefill_layer(
    hidden_states: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
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
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16]:
    hidden_states.bind_dynamic(0, PREFILL_TOKENS_DYN)
    out.bind_dynamic(0, PREFILL_TOKENS_DYN)

    # Runtime user_batch (host-visible batch). Outer batch loop
    # iterates with step 1 so every matmul tile's M dim is fully
    # determined by TOK_TILE (no batch-axis pad / trim needed).
    user_batch = pl.tensor.dim(seq_lens, 0)
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    layer_cache_base = layer_idx * layer_cache_rows
    max_blocks_per_seq = pl.tensor.dim(block_table, 0) // user_batch
    for b in pl.parallel(0, user_batch, 1):
        token_base = pl.cast(pl.tensor.read(chunk_offsets, [b]), pl.INDEX)
        seq_len_b = pl.tensor.read(seq_lens, [b])
        chunk_len_b = pl.tensor.read(chunk_lens, [b])
        chunk_start = seq_len_b - chunk_len_b
        tok_blocks = (chunk_len_b + TOK_TILE - 1) // TOK_TILE
        qkv_prev_tids = pl.array.create(2, pl.TASK_ID)
        for p0_idx in pl.range(tok_blocks):
            with pl.scope():
                p0 = p0_idx * TOK_TILE
                token_p0 = token_base + p0
                valid_tok = pl.min(TOK_TILE, chunk_len_b - p0)

                # ── Scope 1: input RMSNorm + Q/K/V projection ──
                normed_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)

                # Stage 1.1: RMSNorm (vector ops).
                for rms_core in pl.spmd(RMSNORM_SPMD_BLOCKS, name_hint="rmsnorm_spmd"):
                    for work_id in pl.range(rms_core, RMSNORM_WORK_ITEMS, RMSNORM_SPMD_BLOCKS):
                        ti0 = work_id * RMSNORM_TOK_GROUP
                        if ti0 < valid_tok:
                            rms_tok = pl.min(RMSNORM_TOK_GROUP, valid_tok - ti0)
                            sq_sum = pl.full([1, RMSNORM_TOK_GROUP], dtype=pl.FP32, value=0.0)
                            for rb in pl.range(HIDDEN_BLOCKS):
                                k0 = rb * K_CHUNK
                                x_chunk = pl.cast(
                                    pl.slice(
                                        hidden_states,
                                        [RMSNORM_TOK_GROUP, K_CHUNK],
                                        [token_p0 + ti0, k0],
                                        valid_shape=[rms_tok, K_CHUNK],
                                    ),
                                    target_type=pl.FP32,
                                )
                                sq_part = pl.reshape(
                                    pl.row_sum(pl.mul(x_chunk, x_chunk)),
                                    [1, RMSNORM_TOK_GROUP],
                                )
                                sq_sum = pl.add(sq_sum, sq_part)
                            inv_rms = pl.reshape(
                                pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))),
                                [RMSNORM_TOK_GROUP, 1],
                            )

                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                x_chunk = pl.cast(
                                    pl.slice(
                                        hidden_states,
                                        [RMSNORM_TOK_GROUP, K_CHUNK],
                                        [token_p0 + ti0, k0],
                                        valid_shape=[rms_tok, K_CHUNK],
                                    ),
                                    target_type=pl.FP32,
                                )
                                gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [layer_idx, k0])
                                normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                                normed_tile = pl.assemble(
                                    normed_tile,
                                    pl.cast(normed, target_type=pl.BF16),
                                    [ti0, k0],
                                )

                # Stage 1.2/1.3: Q/K/V projection.
                q_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
                k_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.FP32)
                v_proj_tile = pl.create_tensor([TOK_TILE, KV_HIDDEN], dtype=pl.FP32)
                with pl.spmd(
                    Q_PROJ_SPMD_BLOCKS,
                    name_hint="q_proj_spmd",
                    deps=[qkv_prev_tids[0], qkv_prev_tids[1]],
                ) as q_proj_tid:
                    q_core = pl.tile.get_block_idx()
                    for ob in pl.range(q_core, Q_OUT_BLOCKS, Q_PROJ_SPMD_BLOCKS):
                        q0 = ob * Q_OUT_CHUNK
                        tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                        tile_w = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base, q0])
                        q_acc = pl.matmul(tile_a, tile_w, out_dtype=pl.FP32)
                        for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            tile_w_i = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [layer_hidden_base + k0, q0])
                            q_acc = pl.matmul_acc(q_acc, tile_a_i, tile_w_i)
                        q_proj_tile = pl.assemble(q_proj_tile, q_acc, [0, q0])

                with pl.spmd(
                    KV_PROJ_SPMD_BLOCKS,
                    name_hint="kv_proj_spmd",
                    deps=[qkv_prev_tids[0], qkv_prev_tids[1]],
                ) as kv_proj_tid:
                    kv_core = pl.tile.get_block_idx()
                    for ob in pl.range(kv_core, KV_OUT_BLOCKS, KV_PROJ_SPMD_BLOCKS):
                        kv0 = ob * KV_OUT_CHUNK

                        tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                        tile_wk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base, kv0])
                        k_acc = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                        for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            tile_wk_i = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k0, kv0])
                            k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                        k_proj_tile = pl.assemble(k_proj_tile, k_acc, [0, kv0])

                        tile_a = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, 0])
                        tile_wv = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base, kv0])
                        v_acc = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                        for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                            k0 = kb * K_CHUNK
                            tile_a_i = pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0])
                            tile_wv_i = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [layer_hidden_base + k0, kv0])
                            v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                        v_proj_tile = pl.assemble(v_proj_tile, v_acc, [0, kv0])

                # ── Scope 2: Q/K norm + RoPE + KV cache update + causal attention ──
                attn_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)
                all_q_padded_tile = pl.create_tensor(
                    [TOK_TILE * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM],
                    dtype=pl.BF16,
                )
                for final_ti0 in pl.range(0, valid_tok, FINALIZE_TOK_GROUP):
                    finalize_tok = pl.min(FINALIZE_TOK_GROUP, valid_tok - final_ti0)
                    for rope_core in pl.spmd(ROPE_SPMD_BLOCKS, name_hint="rope_kv_cache"):
                        for rel_ti in pl.range(rope_core, finalize_tok, ROPE_SPMD_BLOCKS):
                            ti = final_ti0 + rel_ti
                            chunk_pos = p0 + ti
                            pos = chunk_start + chunk_pos
                            cos_row = pl.slice(rope_cos, [1, HEAD_DIM], [pos, 0])
                            sin_row = pl.slice(rope_sin, [1, HEAD_DIM], [pos, 0])
                            cos_lo = pl.slice(cos_row, [1, HALF_DIM], [0, 0])
                            cos_hi = pl.slice(cos_row, [1, HALF_DIM], [0, HALF_DIM])
                            sin_lo = pl.slice(sin_row, [1, HALF_DIM], [0, 0])
                            sin_hi = pl.slice(sin_row, [1, HALF_DIM], [0, HALF_DIM])
                            cache_slot = pl.cast(pl.tensor.read(slot_mapping, [token_base + chunk_pos]), pl.INDEX)
                            cache_slot_block = cache_slot // BLOCK_SIZE
                            cache_slot_offset = cache_slot - cache_slot_block * BLOCK_SIZE
                            q_block_row0 = ti * TOTAL_Q_GROUPS * Q_HEAD_PAD
                            for ki in pl.range(NUM_KV_HEADS):
                                kv_col = ki * HEAD_DIM
                                k_head_raw = pl.slice(k_proj_tile, [1, HEAD_DIM], [ti, kv_col])
                                k_head = pl.full([Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)
                                k_head = pl.assemble(k_head, k_head_raw, [0, 0])
                                k_sq = pl.reshape(pl.row_sum(pl.mul(k_head, k_head)), [Q_HEAD_PAD, 1])
                                k_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(k_sq, HEAD_DIM_INV), EPS)))
                                k_normed = pl.col_expand_mul(
                                    pl.row_expand_mul(k_head, k_inv_rms),
                                    pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
                                )
                                k_lo = pl.reshape(
                                    pl.slice(k_normed, [1, HALF_DIM], [0, 0]),
                                    [1, HALF_DIM],
                                )
                                k_hi = pl.reshape(
                                    pl.slice(k_normed, [1, HALF_DIM], [0, HALF_DIM]),
                                    [1, HALF_DIM],
                                )
                                rot_lo = pl.sub(
                                    pl.col_expand_mul(k_lo, cos_lo),
                                    pl.col_expand_mul(k_hi, sin_lo),
                                )
                                rot_hi = pl.add(
                                    pl.col_expand_mul(k_hi, cos_hi),
                                    pl.col_expand_mul(k_lo, sin_hi),
                                )
                                cache_row = (
                                    layer_cache_base
                                    + (cache_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE
                                    + cache_slot_offset
                                )
                                k_cache = pl.assemble(
                                    k_cache,
                                    pl.cast(rot_lo, target_type=pl.BF16),
                                    [cache_row, 0],
                                )
                                k_cache = pl.assemble(
                                    k_cache,
                                    pl.cast(rot_hi, target_type=pl.BF16),
                                    [cache_row, HALF_DIM],
                                )
                                v_cache = pl.assemble(
                                    v_cache,
                                    pl.cast(
                                        pl.reshape(
                                            pl.slice(v_proj_tile, [1, HEAD_DIM], [ti, ki * HEAD_DIM]),
                                            [1, HEAD_DIM],
                                        ),
                                        target_type=pl.BF16,
                                    ),
                                    [cache_row, 0],
                                )
                                q_base = ki * Q_PER_KV
                                q_block_raw = pl.reshape(
                                    pl.slice(q_proj_tile, [1, Q_HEAD_BATCH * HEAD_DIM], [ti, q_base * HEAD_DIM]),
                                    [Q_HEAD_BATCH, HEAD_DIM],
                                )
                                q_block_pad = pl.full([Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)
                                q_block_pad = pl.assemble(q_block_pad, q_block_raw, [0, 0])
                                q_sq = pl.reshape(
                                    pl.row_sum(pl.mul(q_block_pad, q_block_pad)),
                                    [Q_HEAD_PAD, 1],
                                )
                                q_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(q_sq, HEAD_DIM_INV), EPS)))
                                q_block = pl.col_expand_mul(
                                    pl.row_expand_mul(q_block_pad, q_inv_rms),
                                    pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0]),
                                )
                                q_rot_lo = pl.create_tensor([Q_HEAD_BATCH, HALF_DIM], dtype=pl.FP32)
                                q_rot_hi = pl.create_tensor([Q_HEAD_BATCH, HALF_DIM], dtype=pl.FP32)
                                for qi in pl.range(Q_HEAD_BATCH):
                                    q_lo = pl.slice(q_block, [1, HALF_DIM], [qi, 0])
                                    q_hi = pl.slice(q_block, [1, HALF_DIM], [qi, HALF_DIM])
                                    q_rot_lo = pl.assemble(
                                        q_rot_lo,
                                        pl.sub(
                                            pl.col_expand_mul(q_lo, cos_lo),
                                            pl.col_expand_mul(q_hi, sin_lo),
                                        ),
                                        [qi, 0],
                                    )
                                    q_rot_hi = pl.assemble(
                                        q_rot_hi,
                                        pl.add(
                                            pl.col_expand_mul(q_hi, cos_hi),
                                            pl.col_expand_mul(q_lo, sin_hi),
                                        ),
                                        [qi, 0],
                                    )
                                q_pad_row0 = q_block_row0 + ki * Q_HEAD_PAD
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

                    b_i32 = pl.cast(b, pl.INT32)
                    max_blocks_i32 = pl.cast(max_blocks_per_seq, pl.INT32)
                    layer_cache_base_i32 = pl.cast(layer_cache_base, pl.INT32)
                    p0_i32 = pl.cast(p0, pl.INT32)
                    final_ti0_i32 = pl.cast(final_ti0, pl.INT32)
                    finalize_tok_i32 = pl.cast(finalize_tok, pl.INT32)

                    cur_li_phase = pl.create_tensor([ATTN_PHASE_ACC_STAT_ROWS, 1], dtype=pl.FP32)
                    oi_tmp_phase = pl.create_tensor([ATTN_PHASE_ACC_SCORE_ROWS, HEAD_DIM], dtype=pl.FP32)
                    block_ctx_len = chunk_start + p0 + final_ti0 + finalize_tok
                    block_ctx_blocks = (block_ctx_len + SEQ_TILE - 1) // SEQ_TILE
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
                        )
                # ── Scope 3: output projection + residual + post RMSNorm + MLP ──
                # Stage 3.1: Output projection + first residual.
                out_proj_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
                resid1_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.FP32)
                for out_core in pl.spmd(OUT_PROJ_SPMD_BLOCKS, name_hint="out_proj_aic_spmd"):
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
                for out_core in pl.spmd(OUT_PROJ_SPMD_BLOCKS, name_hint="out_proj_aiv_spmd"):
                    for ob in pl.range(out_core, Q_OUT_BLOCKS, OUT_PROJ_SPMD_BLOCKS):
                        o0 = ob * Q_OUT_CHUNK
                        resid_chunk = pl.cast(
                            pl.slice(
                                hidden_states,
                                [TOK_TILE, Q_OUT_CHUNK],
                                [token_p0, o0],
                                valid_shape=[valid_tok, Q_OUT_CHUNK],
                            ),
                            target_type=pl.FP32,
                        )
                        out_proj_chunk = pl.slice(out_proj_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0])
                        resid1_tile = pl.assemble(resid1_tile, pl.add(out_proj_chunk, resid_chunk), [0, o0])

                # Stage 3.2: Post-attention RMSNorm.
                post_norm_tile = pl.create_tensor([TOK_TILE, HIDDEN], dtype=pl.BF16)
                for post_core in pl.spmd(POST_RMSNORM_SPMD_BLOCKS, name_hint="post_rmsnorm_spmd"):
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

                # Stage 3.3a: MLP gate/up + SiLU.
                gate_acc_tile = pl.create_tensor([TOK_TILE, INTERMEDIATE], dtype=pl.FP32)
                up_acc_tile = pl.create_tensor([TOK_TILE, INTERMEDIATE], dtype=pl.FP32)
                mlp_silu_tile = pl.create_tensor([TOK_TILE, INTERMEDIATE], dtype=pl.BF16)
                for mlp_band in pl.range(MLP_PROJ_BANDS):
                    band_ob0 = mlp_band * MLP_BAND_BLOCKS
                    for gate_core in pl.spmd(GATE_PROJ_SPMD_BLOCKS, name_hint="gate_proj_spmd"):
                        for rel_ob in pl.range(gate_core, MLP_BAND_BLOCKS, GATE_PROJ_SPMD_BLOCKS):
                            ob = band_ob0 + rel_ob
                            o0 = ob * MLP_OUT_CHUNK
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wg0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, o0])
                            gate_acc = pl.matmul(pc0, wg0, out_dtype=pl.FP32)
                            for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wgi = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + k0, o0])
                                gate_acc = pl.matmul_acc(gate_acc, pci, wgi)
                            gate_acc_tile = pl.assemble(gate_acc_tile, gate_acc, [0, o0])

                    for up_core in pl.spmd(UP_PROJ_SPMD_BLOCKS, name_hint="up_proj_spmd"):
                        for rel_ob in pl.range(up_core, MLP_BAND_BLOCKS, UP_PROJ_SPMD_BLOCKS):
                            ob = band_ob0 + rel_ob
                            o0 = ob * MLP_OUT_CHUNK
                            pc0 = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, 0])
                            wu0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base, o0])
                            up_acc = pl.matmul(pc0, wu0, out_dtype=pl.FP32)
                            for kb in pl.pipeline(1, HIDDEN_BLOCKS, stage=2):
                                k0 = kb * K_CHUNK
                                pci = pl.slice(post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0])
                                wui = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [layer_hidden_base + k0, o0])
                                up_acc = pl.matmul_acc(up_acc, pci, wui)
                            up_acc_tile = pl.assemble(up_acc_tile, up_acc, [0, o0])

                for silu_core in pl.spmd(SILU_SPMD_BLOCKS, name_hint="silu_spmd"):
                    for ob in pl.range(silu_core, MLP_OUT_BLOCKS, SILU_SPMD_BLOCKS):
                        o0 = ob * MLP_OUT_CHUNK
                        gate_acc = pl.slice(gate_acc_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, o0])
                        up_acc = pl.slice(up_acc_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, o0])
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                        mlp_silu_tile = pl.assemble(mlp_silu_tile, mlp_chunk_bf16, [0, o0])

                down_partial_tile = pl.create_tensor([DOWN_K_PARTS * TOK_TILE, HIDDEN], dtype=pl.FP32)
                for down_core in pl.spmd(DOWN_PROJ_SPMD_BLOCKS, name_hint="down_proj_spmd"):
                    for down_work in pl.range(down_core, DOWN_PART_WORK_ITEMS, DOWN_PROJ_SPMD_BLOCKS):
                        dob = down_work // DOWN_K_PARTS
                        down_part = down_work - dob * DOWN_K_PARTS
                        d0 = dob * K_CHUNK
                        ob_start = down_part * DOWN_PART_BLOCKS
                        ob_end = pl.min(MLP_OUT_BLOCKS, ob_start + DOWN_PART_BLOCKS)
                        o0_start = ob_start * MLP_OUT_CHUNK
                        mlp_chunk_0 = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, o0_start])
                        w_down_chunk_0 = pl.slice(w_down, [MLP_OUT_CHUNK, K_CHUNK], [layer_inter_base + o0_start, d0])
                        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
                        for ob in pl.pipeline(ob_start + 1, ob_end, stage=2):
                            o0 = ob * MLP_OUT_CHUNK
                            mlp_chunk_i = pl.slice(mlp_silu_tile, [TOK_TILE, MLP_OUT_CHUNK], [0, o0])
                            w_down_chunk_i = pl.slice(
                                w_down,
                                [MLP_OUT_CHUNK, K_CHUNK],
                                [layer_inter_base + o0, d0],
                            )
                            down_acc = pl.matmul_acc(down_acc, mlp_chunk_i, w_down_chunk_i)
                        down_partial_tile = pl.assemble(down_partial_tile, down_acc, [down_part * TOK_TILE, d0])

                for down_resid_core in pl.spmd(DOWN_RESID_SPMD_BLOCKS, name_hint="down_proj_residual_spmd"):
                    for dob in pl.range(down_resid_core, HIDDEN_BLOCKS, DOWN_RESID_SPMD_BLOCKS):
                        d0 = dob * K_CHUNK
                        down_part0 = pl.slice(down_partial_tile, [TOK_TILE, K_CHUNK], [0, d0])
                        down_part1 = pl.slice(down_partial_tile, [TOK_TILE, K_CHUNK], [TOK_TILE, d0])
                        down_part2 = pl.slice(down_partial_tile, [TOK_TILE, K_CHUNK], [2 * TOK_TILE, d0])
                        down_acc_resid = pl.add(pl.add(down_part0, down_part1), down_part2)
                        out_chunk = pl.add(
                            down_acc_resid,
                            pl.slice(resid1_tile, [TOK_TILE, K_CHUNK], [0, d0]),
                        )
                        out_chunk_bf16 = pl.cast(out_chunk, target_type=pl.BF16)
                        out_chunk_valid = pl.slice(
                            out_chunk_bf16,
                            [TOK_TILE, K_CHUNK],
                            [0, 0],
                            valid_shape=[valid_tok, K_CHUNK],
                        )
                        out = pl.assemble(out, out_chunk_valid, [token_p0, d0])

                qkv_prev_tids[0] = q_proj_tid
                qkv_prev_tids[1] = kv_proj_tid

    return out


@pl.jit(auto_scope=False)
def prefill_fwd(
    hidden_states: pl.Tensor[[PREFILL_TOKENS_DYN, HIDDEN], pl.BF16],
    seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
    chunk_offsets: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
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
    out: pl.Out[pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]],
) -> pl.Tensor[[USER_BATCH_DYN, VOCAB], pl.FP32]:
    hidden_states.bind_dynamic(0, PREFILL_TOKENS_DYN)
    seq_lens.bind_dynamic(0, USER_BATCH_DYN)
    chunk_lens.bind_dynamic(0, USER_BATCH_DYN)
    chunk_offsets.bind_dynamic(0, USER_BATCH_DYN)
    out.bind_dynamic(0, USER_BATCH_DYN)
    block_table.bind_dynamic(0, BLOCK_TABLE_FLAT_DYN)
    slot_mapping.bind_dynamic(0, PREFILL_TOKENS_DYN)
    k_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)
    v_cache.bind_dynamic(0, KV_CACHE_ROWS_DYN)

    user_batch = pl.tensor.dim(seq_lens, 0)
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    prefill_tokens = pl.tensor.dim(hidden_states, 0)

    for layer_idx in pl.range(num_layers_actual):
        with pl.scope():
            layer_next_hidden = pl.create_tensor([prefill_tokens, HIDDEN], dtype=pl.BF16)
            hidden_states = prefill_layer(
                hidden_states,
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
                layer_next_hidden,
                layer_idx,
            )

    final_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for b0 in pl.parallel(0, BATCH, BATCH_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gather_prefill_last_token"):
            for bi in pl.range(BATCH_TILE):
                b = b0 + bi
                if b < user_batch:
                    token_base = pl.cast(pl.tensor.read(chunk_offsets, [b]), pl.INDEX)
                    chunk_len_b = pl.tensor.read(chunk_lens, [b])
                    if chunk_len_b > 0:
                        last_token = token_base + pl.cast(chunk_len_b, pl.INDEX) - 1
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            final_hidden_chunk = pl.slice(
                                hidden_states,
                                [1, K_CHUNK],
                                [last_token, k0],
                            )
                            final_hidden = pl.assemble(final_hidden, final_hidden_chunk, [b, k0])

    out = rms_lm_head(final_hidden, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


def build_tensor_specs(
    batch: int = BATCH,
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

    def init_hidden_states():
        return torch.rand(total_tokens, hidden_size) - 0.5

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

    return [
        TensorSpec("hidden_states", [total_tokens, hidden_size], torch.bfloat16,
                   init_value=init_hidden_states),
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

    hidden_states = tensors["hidden_states"]
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

    batch = seq_lens.shape[0]
    total_tokens = hidden_states.shape[0]
    max_seq = rope_cos.shape[0]
    hidden_size = hidden_states.shape[1]
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
    layer_cache_rows = batch * max_blocks_per_seq * num_kv_heads * BLOCK_SIZE

    def tiled_lm_head(lhs, rhs_t, k_chunk, vocab_chunk):
        out = torch.zeros(lhs.shape[0], rhs_t.shape[0], dtype=torch.float32)
        for k0 in range(0, lhs.shape[1], k_chunk):
            lhs_chunk = lhs[:, k0:k0 + k_chunk].float()
            out += lhs_chunk @ rhs_t[:, k0:k0 + k_chunk].float().T
        return out

    hidden = hidden_states.clone()
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
                    cache_row = layer_cache_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                    k_cache[cache_row, :] = k_rot[pos, ki, :]

            # V cache write.
            v_row_bf16 = v_proj_f.view(S, num_kv_heads, head_dim).to(torch.bfloat16)
            for pos in range(S):
                slot = int(slot_mapping[token_base + pos].item())
                slot_block = slot // BLOCK_SIZE
                slot_offset = slot % BLOCK_SIZE
                for ki in range(num_kv_heads):
                    cache_row = layer_cache_base + (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                    v_cache[cache_row, :] = v_row_bf16[pos, ki, :]

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
                        cache_row0 = layer_cache_base + (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                        s0 = sb * SEQ_TILE
                        k_padded[s0:s0 + BLOCK_SIZE] = k_cache[cache_row0:cache_row0 + BLOCK_SIZE, :]
                        v_padded[s0:s0 + BLOCK_SIZE] = v_cache[cache_row0:cache_row0 + BLOCK_SIZE, :]

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
    parser.add_argument("-b", "--batch", type=int, default=BATCH,
                        help=("User-visible batch size. Host allocates every "
                              "batch-dependent tensor at exactly this size; "
                              "every kernel signature batch-axis dim is a "
                              "pl.dynamic() variable, so a single compiled "
                              "program serves any batch <= host KV-cache "
                              "capacity. Default: %(default)s"))
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
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
        golden_fn=golden_qwen3_14b_prefill,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=5e-3,
        atol=5e-3,
        save_data=args.save_data,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
