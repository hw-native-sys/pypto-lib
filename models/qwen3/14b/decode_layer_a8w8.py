# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B A8W8 decode forward."""

import os

import pypto.language as pl

from config import QWEN3_14B as M
from config import QWEN3_14B_TILING as T
from rms_lm_head import rms_lm_head  # LM head for the fused multi-layer decode_fwd

BATCH = M.batch
NUM_KV_HEADS = M.num_kv_heads
HEAD_DIM = M.head_dim
HIDDEN = M.hidden
INTERMEDIATE = M.intermediate
KV_HIDDEN = M.kv_hidden
NUM_LAYERS = M.num_layers
EPS = M.eps
HIDDEN_INV = M.hidden_inv
HEAD_DIM_INV = M.head_dim_inv
ATTN_SCALE = M.attn_scale
HALF_DIM = M.half_dim
Q_PER_KV = M.q_per_kv
Q_HEAD_BATCH = M.q_head_batch
Q_HEAD_PAD = M.q_head_pad
INT8_SCALE_MAX = M.int8_scale_max
INT8_AMAX_EPS = M.int8_amax_eps

MAX_SEQ = int(os.environ.get("PTO2_MANUAL_MAX_SEQ", str(M.max_seq)))
ACTIVE_BATCH = BATCH

RMSNORM_K_CHUNK = 256
XG_BLOCKS = 5
TN = 256
TK = 512
QKV_N_TILE = 512
N_SUB = QKV_N_TILE // TN
Q_ON = HIDDEN // QKV_N_TILE
KV_ON = KV_HIDDEN // QKV_N_TILE
QKV_OK = 1
QKV_K_SLICE = HIDDEN // QKV_OK
QKV_K_CHUNKS = QKV_K_SLICE // TK

BLOCK_SIZE = T.block_size
ATTN_TILE = 16
PAGE_ATTN_PARTS = BLOCK_SIZE // ATTN_TILE
MAX_CTX_BLOCKS = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
MAX_ATTN_PARTS = MAX_CTX_BLOCKS * PAGE_ATTN_PARTS
GP_SIZE = 1
HEAD_GROUPS = NUM_KV_HEADS // GP_SIZE
ROPE_CORES = 32
ROPE_ITEMS_PER_CORE = (NUM_KV_HEADS * BATCH) // ROPE_CORES
NUM_CORES = 24
FA_TABLE_CAP = BATCH * HEAD_GROUPS * MAX_ATTN_PARTS
OS_WORK = BATCH * NUM_KV_HEADS

K_SPLITS_OUT = 5
N_SPLITS_OUT = 40
OUT_INNER_TK = 512
OUT_TN = HIDDEN // N_SPLITS_OUT
OUT_TK = HIDDEN // K_SPLITS_OUT
OUT_N_SUB_K = OUT_TK // OUT_INNER_TK

K_CHUNK = 512
MLP_TN = 256
K_SPLITS_MLP = 5
MLP_K_SLICE = HIDDEN // K_SPLITS_MLP
MLP_GATE_UP_DEP_COUNT = K_SPLITS_MLP
MLP_DUAL_L0_K = 64
MLP_ON = INTERMEDIATE // MLP_TN

MLP_OUT_CHUNK = 256
SILU_INNER_CHUNKS = MLP_TN // MLP_OUT_CHUNK

DOWN_TN = 256
DOWN_K_SLICE = 512
DOWN_TK = 512
DOWN_DUAL_L0_K = 64
DOWN_ON = HIDDEN // DOWN_TN
DOWN_DUAL_PAIR = 2
DOWN_DUAL_ON = DOWN_ON // DOWN_DUAL_PAIR

assert Q_HEAD_PAD % 16 == 0 and Q_HEAD_PAD >= Q_HEAD_BATCH
assert QKV_N_TILE % TN == 0 and HIDDEN % QKV_N_TILE == 0 and KV_HIDDEN % QKV_N_TILE == 0
assert BLOCK_SIZE % ATTN_TILE == 0 and NUM_KV_HEADS % GP_SIZE == 0
assert DOWN_TN == 256 and DOWN_TK == DOWN_K_SLICE == 512
assert N_SPLITS_OUT * OUT_TN == HIDDEN and K_SPLITS_OUT * OUT_TK == HIDDEN

@pl.jit.inline
def _decode_layer(  # noqa: PLR0913 — model signature is intrinsic
    hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    wq_scale: pl.Tensor,
    wk_scale: pl.Tensor,
    wv_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    k_cache_scale: pl.Tensor,
    v_cache_scale: pl.Tensor,
    wo: pl.Tensor,
    wo_scale: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[BATCH, HIDDEN], pl.BF16]:
    layer_hidden_base = layer_idx * HIDDEN
    layer_inter_base = layer_idx * INTERMEDIATE
    num_layers_actual = pl.tensor.dim(input_rms_weight, 0)
    layer_cache_rows = pl.tensor.dim(k_cache, 0) // num_layers_actual
    layer_cache_base = layer_idx * layer_cache_rows
    user_batch = pl.tensor.dim(seq_lens, 0)
    max_blocks_per_seq = pl.tensor.dim(block_table, 0) // user_batch
    q_norm_w = pl.slice(q_norm_weight, [1, HEAD_DIM], [layer_idx, 0])
    k_norm_w = pl.slice(k_norm_weight, [1, HEAD_DIM], [layer_idx, 0])

    normed_states = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    inv_rms_states = pl.create_tensor([BATCH], dtype=pl.FP32)  # deferred 1/rms denominator
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    normed_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
    act_scales = pl.create_tensor([BATCH, 1], dtype=pl.FP32)

    for xg_core in pl.spmd(XG_BLOCKS, name_hint="x_gamma"):
        for kb in pl.pipeline(xg_core, HIDDEN // RMSNORM_K_CHUNK, XG_BLOCKS, stage=2):
            xg_k0 = kb * RMSNORM_K_CHUNK
            xg_chunk = pl.cast(hidden_states[:, xg_k0 : xg_k0 + RMSNORM_K_CHUNK], target_type=pl.FP32)
            xg_gamma = pl.slice(input_rms_weight, [1, RMSNORM_K_CHUNK], [layer_idx, xg_k0])
            xg_scaled = pl.col_expand_mul(xg_chunk, xg_gamma)
            normed_states = pl.assemble(normed_states, pl.cast(xg_scaled, target_type=pl.BF16), [0, xg_k0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms_recip"):
        partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HIDDEN // RMSNORM_K_CHUNK, stage=4):
            rms_k0 = kb * RMSNORM_K_CHUNK
            rms_chunk = pl.cast(hidden_states[:, rms_k0 : rms_k0 + RMSNORM_K_CHUNK], target_type=pl.FP32)
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(rms_chunk, rms_chunk)), [1, BATCH]),
            )
        variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for rms_b in pl.unroll(BATCH):
            pl.tensor.write(inv_rms_states, [rms_b], pl.tensor.read(inv_rms, [rms_b, 0]))

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="act_quant"):
        act_quant_amax_t = pl.full([1, BATCH], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for act_quant_amax_kb_t in pl.range(HIDDEN // RMSNORM_K_CHUNK):
            act_quant_amax_k0_t = act_quant_amax_kb_t * RMSNORM_K_CHUNK
            act_quant_amax_bf16_t = normed_states[:, act_quant_amax_k0_t : act_quant_amax_k0_t + RMSNORM_K_CHUNK]
            act_quant_amax_f_t = pl.cast(act_quant_amax_bf16_t, target_type=pl.FP32)
            act_quant_amax_abs_t = pl.maximum(act_quant_amax_f_t, pl.neg(act_quant_amax_f_t))
            act_quant_amax_row_t = pl.reshape(pl.row_max(act_quant_amax_abs_t), [1, BATCH])
            act_quant_amax_t = pl.maximum(act_quant_amax_t, act_quant_amax_row_t)
        for act_quant_row_t in pl.range(BATCH):
            act_quant_row_amax_t = pl.tensor.read(act_quant_amax_t, [0, act_quant_row_t])
            act_quant_scale_q_t = INT8_SCALE_MAX / act_quant_row_amax_t
            pl.tensor.write(act_scales, [act_quant_row_t, 0], act_quant_row_amax_t / INT8_SCALE_MAX)
            for act_quant_write_kb_t in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                act_quant_write_k0_t = act_quant_write_kb_t * RMSNORM_K_CHUNK
                act_quant_write_bf16_t = pl.slice(
                    normed_states, [1, RMSNORM_K_CHUNK], [act_quant_row_t, act_quant_write_k0_t]
                )
                act_quant_write_f_t = pl.cast(act_quant_write_bf16_t, target_type=pl.FP32)
                act_quant_scaled_t = pl.mul(act_quant_write_f_t, act_quant_scale_q_t)
                act_quant_i32_t = pl.cast(act_quant_scaled_t, target_type=pl.INT32, mode="rint")
                act_quant_i32_t = pl.minimum(
                    pl.maximum(act_quant_i32_t, pl.full([1, RMSNORM_K_CHUNK], dtype=pl.INT32, value=-127)),
                    pl.full([1, RMSNORM_K_CHUNK], dtype=pl.INT32, value=127),
                )
                act_quant_half_t = pl.cast(act_quant_i32_t, target_type=pl.FP16, mode="round")
                act_quant_i8_t = pl.cast(act_quant_half_t, target_type=pl.INT8, mode="trunc")
                normed_i8 = pl.assemble(normed_i8, act_quant_i8_t, [act_quant_row_t, act_quant_write_k0_t])
    for q_grid in pl.spmd(Q_ON * N_SUB, name_hint="q_proj_fused_dequant"):
        q_on = q_grid // N_SUB
        n_sub = q_grid - q_on * N_SUB
        q_n0 = q_on * QKV_N_TILE + n_sub * TN
        q_acc = pl.matmul(
            pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
            wq[layer_hidden_base + 0 : layer_hidden_base + TK, q_n0 : q_n0 + TN],
            out_dtype=pl.INT32,
        )
        for kc in pl.range(1, QKV_K_CHUNKS - 1):
            q_kk = kc * TK
            q_acc = pl.matmul_acc(
                q_acc,
                pl.tensor.set_validshape(normed_i8[:, q_kk : q_kk + TK], ACTIVE_BATCH, TK),
                wq[layer_hidden_base + q_kk : layer_hidden_base + q_kk + TK, q_n0 : q_n0 + TN],
            )
        q_w_scale = pl.reshape(pl.slice(wq_scale, [1, TN], [layer_idx, q_n0]), [1, TN])
        q_last_kk = (QKV_K_CHUNKS - 1) * TK
        q_acc = pl.matmul_acc(
            q_acc,
            pl.tensor.set_validshape(normed_i8[:, q_last_kk : q_last_kk + TK], ACTIVE_BATCH, TK),
            wq[layer_hidden_base + q_last_kk : layer_hidden_base + q_last_kk + TK, q_n0 : q_n0 + TN],
        )
        q_deq_fp32 = pl.cast(q_acc, target_type=pl.FP32, mode="none")
        q_deq_col_scaled = pl.col_expand_mul(q_deq_fp32, q_w_scale)
        q_deq_fused = pl.row_expand_mul(q_deq_col_scaled, act_scales)
        q_proj = pl.assemble(q_proj, q_deq_fused, [0, q_n0])

    for k_grid in pl.spmd(KV_ON * N_SUB, name_hint="k_proj_fused_dequant"):
        k_on = k_grid // N_SUB
        n_sub = k_grid - k_on * N_SUB
        k_n0 = k_on * QKV_N_TILE + n_sub * TN
        k_acc = pl.matmul(
            pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
            wk[layer_hidden_base + 0 : layer_hidden_base + TK, k_n0 : k_n0 + TN],
            out_dtype=pl.INT32,
        )
        for kc in pl.range(1, QKV_K_CHUNKS - 1):
            k_kk = kc * TK
            k_acc = pl.matmul_acc(
                k_acc,
                pl.tensor.set_validshape(normed_i8[:, k_kk : k_kk + TK], ACTIVE_BATCH, TK),
                wk[layer_hidden_base + k_kk : layer_hidden_base + k_kk + TK, k_n0 : k_n0 + TN],
            )
        k_w_scale = pl.reshape(pl.slice(wk_scale, [1, TN], [layer_idx, k_n0]), [1, TN])
        k_last_kk = (QKV_K_CHUNKS - 1) * TK
        k_acc = pl.matmul_acc(
            k_acc,
            pl.tensor.set_validshape(normed_i8[:, k_last_kk : k_last_kk + TK], ACTIVE_BATCH, TK),
            wk[
                layer_hidden_base + k_last_kk : layer_hidden_base + k_last_kk + TK,
                k_n0 : k_n0 + TN,
            ],
        )
        k_deq_fp32 = pl.cast(k_acc, target_type=pl.FP32, mode="none")
        k_deq_col_scaled = pl.col_expand_mul(k_deq_fp32, k_w_scale)
        k_deq_fused = pl.row_expand_mul(k_deq_col_scaled, act_scales)
        k_proj = pl.assemble(k_proj, k_deq_fused, [0, k_n0])

    for v_grid in pl.spmd(KV_ON * N_SUB, name_hint="v_proj_fused_dequant"):
        v_on = v_grid // N_SUB
        n_sub = v_grid - v_on * N_SUB
        v_n0 = v_on * QKV_N_TILE + n_sub * TN
        v_acc = pl.matmul(
            pl.tensor.set_validshape(normed_i8[:, 0:TK], ACTIVE_BATCH, TK),
            wv[layer_hidden_base + 0 : layer_hidden_base + TK, v_n0 : v_n0 + TN],
            out_dtype=pl.INT32,
        )
        for kc in pl.range(1, QKV_K_CHUNKS - 1):
            v_kk = kc * TK
            v_acc = pl.matmul_acc(
                v_acc,
                pl.tensor.set_validshape(normed_i8[:, v_kk : v_kk + TK], ACTIVE_BATCH, TK),
                wv[layer_hidden_base + v_kk : layer_hidden_base + v_kk + TK, v_n0 : v_n0 + TN],
            )
        v_w_scale = pl.reshape(pl.slice(wv_scale, [1, TN], [layer_idx, v_n0]), [1, TN])
        v_last_kk = (QKV_K_CHUNKS - 1) * TK
        v_acc = pl.matmul_acc(
            v_acc,
            pl.tensor.set_validshape(normed_i8[:, v_last_kk : v_last_kk + TK], ACTIVE_BATCH, TK),
            wv[
                layer_hidden_base + v_last_kk : layer_hidden_base + v_last_kk + TK,
                v_n0 : v_n0 + TN,
            ],
        )
        v_deq_fp32 = pl.cast(v_acc, target_type=pl.FP32, mode="none")
        v_deq_col_scaled = pl.col_expand_mul(v_deq_fp32, v_w_scale)
        v_deq_fused = pl.row_expand_mul(v_deq_col_scaled, act_scales)
        v_proj = pl.assemble(v_proj, v_deq_fused, [0, v_n0])
    all_q_padded = pl.create_tensor([BATCH * NUM_KV_HEADS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    all_oi_tmp = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_ATTN_PARTS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.FP32
    )
    all_cur_mi = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_ATTN_PARTS * Q_HEAD_PAD, 1], dtype=pl.FP32
    )
    all_cur_li = pl.create_tensor(
        [BATCH * NUM_KV_HEADS * MAX_ATTN_PARTS * Q_HEAD_PAD, 1], dtype=pl.FP32
    )
    kv_ready = pl.create_tensor([BATCH], dtype=pl.INT32)
    fa_done = pl.create_tensor([FA_TABLE_CAP], dtype=pl.INT32)

    fa_work_table = pl.create_tensor([FA_TABLE_CAP], dtype=pl.INT32)
    fa_total = pl.create_tensor([1], dtype=pl.INT32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_work_build"):
        cursor = pl.read(seq_lens, [0]) * 0  # scalar 0 (INDEX)
        for wb in pl.unroll(BATCH):
            wb_ctx = (pl.read(seq_lens, [wb]) + (ATTN_TILE - 1)) // ATTN_TILE
            for whg in pl.unroll(HEAD_GROUPS):
                for wp in pl.range(wb_ctx):
                    work_id = (wb * HEAD_GROUPS + whg) * MAX_ATTN_PARTS + wp
                    pl.tensor.write(fa_work_table, [cursor + wp], pl.cast(work_id, target_type=pl.INT32))
                cursor = cursor + wb_ctx
        pl.tensor.write(fa_total, [0], pl.cast(cursor, target_type=pl.INT32))

    q_proj_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    inv_rms_col = pl.reshape(inv_rms_states, [BATCH, 1])

    for h in pl.spmd(NUM_KV_HEADS, name_hint="qk_norm"):
        q0 = h * Q_PER_KV * HEAD_DIM
        q_chunk = pl.slice(q_proj, [BATCH, Q_PER_KV * HEAD_DIM], [0, q0])
        q_chunk = pl.row_expand_mul(q_chunk, inv_rms_col)
        q_flat = pl.reshape(q_chunk, [BATCH * Q_PER_KV, HEAD_DIM])
        q_g = pl.col_expand_mul(q_flat, q_norm_w)
        q_ss = pl.row_sum(pl.mul(q_flat, q_flat))
        q_inv = pl.recip(pl.sqrt(pl.add(pl.mul(q_ss, HEAD_DIM_INV), EPS)))
        q_g = pl.row_expand_mul(q_g, q_inv)
        q_proj_norm = pl.assemble(
            q_proj_norm,
            pl.reshape(q_g, [BATCH, Q_PER_KV * HEAD_DIM]),
            [0, q0],
        )

        k0 = h * HEAD_DIM
        k_chunk = pl.slice(k_proj, [BATCH, HEAD_DIM], [0, k0])
        k_chunk = pl.row_expand_mul(k_chunk, inv_rms_col)
        k_g = pl.col_expand_mul(k_chunk, k_norm_w)
        k_ss = pl.row_sum(pl.mul(k_chunk, k_chunk))
        k_inv = pl.recip(pl.sqrt(pl.add(pl.mul(k_ss, HEAD_DIM_INV), EPS)))
        k_g = pl.row_expand_mul(k_g, k_inv)
        k_proj_norm = pl.assemble(k_proj_norm, k_g, [0, k0])
    for h in pl.spmd(NUM_KV_HEADS, name_hint="v_norm"):
        v_norm_k0 = h * HEAD_DIM
        v_norm_chunk = pl.row_expand_mul(
            pl.slice(v_proj, [BATCH, HEAD_DIM], [0, v_norm_k0]), inv_rms_col
        )
        v_proj_norm = pl.assemble(v_proj_norm, v_norm_chunk, [0, v_norm_k0])

    for rope_core in pl.spmd(ROPE_CORES, name_hint="rope_qkv", allow_early_resolve=True):
        for rope_it in pl.pipeline(ROPE_ITEMS_PER_CORE, stage=2):
            g_idx = rope_core * ROPE_ITEMS_PER_CORE + rope_it
            ki = g_idx // BATCH
            b = g_idx % BATCH
            ctx_len = pl.read(seq_lens, [b])
            pos = ctx_len - 1  # absolute position -> RoPE cos/sin row (NOT the cache row)
            wr_slot = pl.cast(pl.tensor.read(slot_mapping, [b]), pl.INDEX)
            wr_slot_block = wr_slot // BLOCK_SIZE
            wr_slot_offset = wr_slot - wr_slot_block * BLOCK_SIZE
            cos_lo = rope_cos[pos : pos + 1, 0:HALF_DIM]
            cos_hi = rope_cos[pos : pos + 1, HALF_DIM:HEAD_DIM]
            sin_lo = rope_sin[pos : pos + 1, 0:HALF_DIM]
            sin_hi = rope_sin[pos : pos + 1, HALF_DIM:HEAD_DIM]
            kv_col = ki * HEAD_DIM
            k_full = k_proj_norm[b : b + 1, kv_col : kv_col + HEAD_DIM]
            k_lo = k_full[:, 0:HALF_DIM]
            k_hi = k_full[:, HALF_DIM:HEAD_DIM]
            rot_lo = pl.sub(pl.mul(k_lo, cos_lo), pl.mul(k_hi, sin_lo))
            rot_hi = pl.add(pl.mul(k_hi, cos_hi), pl.mul(k_lo, sin_hi))
            cache_row = layer_cache_base + (wr_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + wr_slot_offset
            v_row_fp32 = v_proj_norm[b : b + 1, ki * HEAD_DIM : (ki + 1) * HEAD_DIM]
            k_rot_fp32_lo = rot_lo
            k_rot_fp32_hi = rot_hi
            k_rot_abs_lo = pl.maximum(k_rot_fp32_lo, pl.neg(k_rot_fp32_lo))
            k_rot_abs_hi = pl.maximum(k_rot_fp32_hi, pl.neg(k_rot_fp32_hi))
            k_rot_abs_lo_groups = pl.reshape(k_rot_abs_lo, [8, 8])
            k_rot_abs_hi_groups = pl.reshape(k_rot_abs_hi, [8, 8])
            k_cache_amax_lo_parts = pl.row_max(k_rot_abs_lo_groups)
            k_cache_amax_hi_parts = pl.row_max(k_rot_abs_hi_groups)
            k_cache_amax = INT8_AMAX_EPS
            for part in pl.range(8):
                k_cache_amax = pl.max(k_cache_amax, pl.tensor.read(k_cache_amax_lo_parts, [part, 0]))
                k_cache_amax = pl.max(k_cache_amax, pl.tensor.read(k_cache_amax_hi_parts, [part, 0]))
            k_cache_scale_q = INT8_SCALE_MAX / k_cache_amax
            k_cache_scale_fp32 = k_cache_amax / INT8_SCALE_MAX
            kq_lo_scaled = pl.mul(k_rot_fp32_lo, k_cache_scale_q)
            kq_lo_i32 = pl.cast(kq_lo_scaled, target_type=pl.INT32, mode="rint")
            kq_lo_i32 = pl.minimum(
                pl.maximum(kq_lo_i32, pl.full([1, HALF_DIM], dtype=pl.INT32, value=-127)),
                pl.full([1, HALF_DIM], dtype=pl.INT32, value=127),
            )
            kq_lo_i8 = pl.cast(pl.cast(kq_lo_i32, target_type=pl.FP16), target_type=pl.INT8, mode="trunc")
            kq_hi_scaled = pl.mul(k_rot_fp32_hi, k_cache_scale_q)
            kq_hi_i32 = pl.cast(kq_hi_scaled, target_type=pl.INT32, mode="rint")
            kq_hi_i32 = pl.minimum(
                pl.maximum(kq_hi_i32, pl.full([1, HALF_DIM], dtype=pl.INT32, value=-127)),
                pl.full([1, HALF_DIM], dtype=pl.INT32, value=127),
            )
            kq_hi_i8 = pl.cast(pl.cast(kq_hi_i32, target_type=pl.FP16), target_type=pl.INT8, mode="trunc")
            k_cache = pl.assemble(k_cache, kq_lo_i8, [cache_row, 0])
            k_cache = pl.assemble(k_cache, kq_hi_i8, [cache_row, HALF_DIM])
            pl.tensor.write(k_cache_scale, [cache_row, 0], k_cache_scale_fp32)
            v_abs = pl.maximum(v_row_fp32, pl.neg(v_row_fp32))
            v_abs_groups = pl.reshape(v_abs, [16, 8])
            v_cache_amax_parts = pl.row_max(v_abs_groups)
            v_cache_amax = INT8_AMAX_EPS
            for part in pl.range(16):
                v_cache_amax = pl.max(v_cache_amax, pl.tensor.read(v_cache_amax_parts, [part, 0]))
            v_cache_scale_q = INT8_SCALE_MAX / v_cache_amax
            v_cache_scale_fp32 = v_cache_amax / INT8_SCALE_MAX
            vq_scaled = pl.mul(v_row_fp32, v_cache_scale_q)
            vq_i32 = pl.cast(vq_scaled, target_type=pl.INT32, mode="rint")
            vq_i32 = pl.minimum(
                pl.maximum(vq_i32, pl.full([1, HEAD_DIM], dtype=pl.INT32, value=-127)),
                pl.full([1, HEAD_DIM], dtype=pl.INT32, value=127),
            )
            vq_i8 = pl.cast(pl.cast(vq_i32, target_type=pl.FP16), target_type=pl.INT8, mode="trunc")
            v_cache = pl.assemble(v_cache, vq_i8, [cache_row, 0])
            pl.tensor.write(v_cache_scale, [cache_row, 0], v_cache_scale_fp32)

            q_base = ki * Q_PER_KV
            q_pad_row0 = b * NUM_KV_HEADS * Q_HEAD_PAD + ki * Q_HEAD_PAD
            q_heads = pl.reshape(q_proj_norm[b : b + 1, q_base * HEAD_DIM : (q_base + Q_PER_KV) * HEAD_DIM], [Q_PER_KV, HEAD_DIM])
            q_lo = q_heads[:, 0:HALF_DIM]
            q_hi = q_heads[:, HALF_DIM:HEAD_DIM]

            cos_lo_full = pl.full([Q_PER_KV, HALF_DIM], dtype=pl.FP32, value=0.0)
            cos_hi_full = pl.full([Q_PER_KV, HALF_DIM], dtype=pl.FP32, value=0.0)
            sin_lo_full = pl.full([Q_PER_KV, HALF_DIM], dtype=pl.FP32, value=0.0)
            sin_hi_full = pl.full([Q_PER_KV, HALF_DIM], dtype=pl.FP32, value=0.0)
            for qj in pl.unroll(Q_PER_KV):
                cos_lo_full = pl.assemble(cos_lo_full, cos_lo, [qj, 0])
                cos_hi_full = pl.assemble(cos_hi_full, cos_hi, [qj, 0])
                sin_lo_full = pl.assemble(sin_lo_full, sin_lo, [qj, 0])
                sin_hi_full = pl.assemble(sin_hi_full, sin_hi, [qj, 0])

            q_rot_lo = pl.sub(pl.mul(q_lo, cos_lo_full), pl.mul(q_hi, sin_lo_full))
            q_rot_hi = pl.add(pl.mul(q_hi, cos_hi_full), pl.mul(q_lo, sin_hi_full))
            q_rot = pl.concat(q_rot_lo, q_rot_hi)
            all_q_padded = pl.assemble(all_q_padded, pl.cast(q_rot, target_type=pl.BF16), [q_pad_row0, 0])
            q_pad_zero = pl.cast(
                pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM], dtype=pl.FP32, value=0.0),
                target_type=pl.BF16,
            )
            all_q_padded = pl.assemble(all_q_padded, q_pad_zero, [q_pad_row0 + Q_HEAD_BATCH, 0])
            pl.tensor.write(kv_ready, [b], pl.cast(ctx_len * 0 + 1, target_type=pl.INT32))

    down_acc_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    gate_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32, manual_dep=True)
    up_acc_all = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.FP32, manual_dep=True)

    for fa_core in pl.spmd(
        NUM_CORES,
        name_hint="fa_fused",
        sync_start=True,
        allow_early_resolve=True,
    ):
        fa_total_blocks = pl.cast(pl.read(fa_total, [0]), target_type=pl.INDEX)
        for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):
            fa_enc = pl.cast(pl.read(fa_work_table, [fa_w]), target_type=pl.INDEX)
            fa_group = fa_enc // MAX_ATTN_PARTS
            fa_b = fa_group // HEAD_GROUPS
            fa_hg = fa_group - fa_b * HEAD_GROUPS
            fa_ready_raw = pl.read(kv_ready, [fa_b])
            fa_ready = pl.cast(fa_ready_raw, target_type=pl.INDEX)
            fa_part = fa_enc % MAX_ATTN_PARTS
            fa_p = fa_part // PAGE_ATTN_PARTS
            fa_page_part = fa_part - fa_p * PAGE_ATTN_PARTS
            fa_ctx_len = pl.read(seq_lens, [fa_b])
            sb = fa_p  # logical KV block index (no inner loop — old p_blocks was 1)
            s0 = fa_part * ATTN_TILE
            valid_len = pl.min(ATTN_TILE, fa_ctx_len - s0)
            fa_pbid = pl.cast(
                pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX
            ) + (fa_ready - fa_ready)

            for gp in pl.range(GP_SIZE):
                gi = fa_hg * GP_SIZE + gp
                kvh = gi  # Q_GROUPS=1
                q_pad_row_g = fa_b * NUM_KV_HEADS * Q_HEAD_PAD + gi * Q_HEAD_PAD
                q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]
                g_base = (fa_b * NUM_KV_HEADS + gi) * MAX_ATTN_PARTS * Q_HEAD_PAD
                cache_row = layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE + fa_page_part * ATTN_TILE

                k_tile_inline_deq = pl.create_tensor([ATTN_TILE, HEAD_DIM], dtype=pl.BF16)
                for k_scale_ti in pl.range(ATTN_TILE):
                    k_scale = pl.tensor.read(k_cache_scale, [cache_row + k_scale_ti, 0])
                    k_row_i8 = pl.slice(k_cache, [1, HEAD_DIM], [cache_row + k_scale_ti, 0])
                    k_row_fp32 = pl.cast(pl.cast(k_row_i8, target_type=pl.FP16), target_type=pl.FP32)
                    k_tile_inline_deq = pl.assemble(
                        k_tile_inline_deq,
                        pl.cast(pl.mul(k_row_fp32, k_scale), target_type=pl.BF16),
                        [k_scale_ti, 0],
                    )
                raw_scores = pl.matmul(q_padded, k_tile_inline_deq, b_trans=True, out_dtype=pl.FP32)
                scores_valid = pl.tensor.set_validshape(pl.mul(raw_scores, ATTN_SCALE), Q_HEAD_PAD, valid_len)
                scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                cur_mi = pl.row_max(scores)
                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                cur_li = pl.row_sum(exp_scores_fp32)

                v_tile_deq = pl.create_tensor([ATTN_TILE, HEAD_DIM], dtype=pl.BF16)
                for v_scale_ti in pl.range(ATTN_TILE):
                    v_scale = pl.tensor.read(v_cache_scale, [cache_row + v_scale_ti, 0])
                    v_row_i8 = pl.slice(v_cache, [1, HEAD_DIM], [cache_row + v_scale_ti, 0])
                    v_row_fp32 = pl.cast(pl.cast(v_row_i8, target_type=pl.FP16), target_type=pl.FP32)
                    v_tile_deq = pl.assemble(
                        v_tile_deq,
                        pl.cast(pl.mul(v_row_fp32, v_scale), target_type=pl.BF16),
                        [v_scale_ti, 0],
                    )
                oi_raw_deq = pl.matmul(exp_scores_bf16, v_tile_deq, out_dtype=pl.FP32)
                oi_valid_deq = pl.tensor.set_validshape(oi_raw_deq, Q_HEAD_BATCH, HEAD_DIM)
                all_oi_tmp = pl.assemble(all_oi_tmp, oi_valid_deq, [g_base + fa_part * Q_HEAD_PAD, 0])
                all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [g_base + fa_part * Q_HEAD_PAD, 0])
                all_cur_li = pl.assemble(all_cur_li, cur_li, [g_base + fa_part * Q_HEAD_PAD, 0])
            pl.tensor.write(fa_done, [fa_enc], pl.cast(1, target_type=pl.INT32))

    for os_core in pl.spmd(NUM_CORES * 2, name_hint="online_softmax"):
        for os_spmd_idx in pl.range(os_core, OS_WORK, NUM_CORES * 2):
            os_b = os_spmd_idx // NUM_KV_HEADS
            os_gi = os_spmd_idx % NUM_KV_HEADS
            os_hg = os_gi // GP_SIZE
            os_ctx_len = pl.read(seq_lens, [os_b])
            os_ctx_blocks = (os_ctx_len + ATTN_TILE - 1) // ATTN_TILE
            os_kvh = os_gi  # Q_GROUPS=1
            os_q_base = os_kvh * Q_PER_KV
            os_g_base = (os_b * NUM_KV_HEADS + os_gi) * MAX_ATTN_PARTS * Q_HEAD_PAD
            os_w_base = (os_b * HEAD_GROUPS + os_hg) * MAX_ATTN_PARTS

            os_done0 = pl.cast(pl.read(fa_done, [os_w_base]), target_type=pl.FP32)
            oi = all_oi_tmp[os_g_base : os_g_base + Q_HEAD_PAD, :]
            mi = pl.add(
                all_cur_mi[os_g_base : os_g_base + Q_HEAD_PAD, :],
                pl.mul(os_done0, 0.0),
            )
            li = all_cur_li[os_g_base : os_g_base + Q_HEAD_PAD, :]
            for sb in pl.range(1, os_ctx_blocks):
                rec = os_g_base + sb * Q_HEAD_PAD
                os_done_sb = pl.cast(pl.read(fa_done, [os_w_base + sb]), target_type=pl.FP32)
                oi_tmp_valid = all_oi_tmp[rec : rec + Q_HEAD_PAD, :]
                online_cur_mi = pl.add(
                    all_cur_mi[rec : rec + Q_HEAD_PAD, :],
                    pl.mul(os_done_sb, 0.0),
                )
                online_cur_li = all_cur_li[rec : rec + Q_HEAD_PAD, :]
                mi_new = pl.maximum(mi, online_cur_mi)
                alpha = pl.exp(pl.sub(mi, mi_new))
                beta = pl.exp(pl.sub(online_cur_mi, mi_new))
                li = pl.add(pl.mul(alpha, li), pl.mul(beta, online_cur_li))
                oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp_valid, beta))
                mi = mi_new

            ctx = pl.row_expand_div(oi, li)
            ctx_valid = ctx[0:Q_HEAD_BATCH, :]
            ctx_flat_bf16 = pl.cast(pl.reshape(ctx_valid, [1, Q_HEAD_BATCH * HEAD_DIM]), target_type=pl.BF16)
            attn_out = pl.assemble(attn_out, ctx_flat_bf16, [os_b, os_q_base * HEAD_DIM])

    attn_proj_fp32 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    post_norm_partial = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # raw residual h1 (add-back)
    mlp_norm_in = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)  # h1 * post_gamma (gate/up input)
    inv_rms_tile = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
    mlp_down_tile = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)

    attn_out_i8 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.INT8)
    attn_out_scales = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_act_quant"):
        out_quant_amax_t = pl.full([1, BATCH], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for out_quant_amax_kb_t in pl.range(HIDDEN // K_CHUNK):
            out_quant_amax_k0_t = out_quant_amax_kb_t * K_CHUNK
            out_quant_amax_bf16_t = attn_out[:, out_quant_amax_k0_t : out_quant_amax_k0_t + K_CHUNK]
            out_quant_amax_f_t = pl.cast(out_quant_amax_bf16_t, target_type=pl.FP32)
            out_quant_amax_abs_t = pl.maximum(out_quant_amax_f_t, pl.neg(out_quant_amax_f_t))
            out_quant_amax_row_t = pl.reshape(pl.row_max(out_quant_amax_abs_t), [1, BATCH])
            out_quant_amax_t = pl.maximum(out_quant_amax_t, out_quant_amax_row_t)
        for out_quant_row_t in pl.range(BATCH):
            out_quant_row_amax_t = pl.tensor.read(out_quant_amax_t, [0, out_quant_row_t])
            out_quant_scale_q_t = INT8_SCALE_MAX / out_quant_row_amax_t
            pl.tensor.write(attn_out_scales, [out_quant_row_t, 0], out_quant_row_amax_t / INT8_SCALE_MAX)
            for out_quant_write_kb_t in pl.range(HIDDEN // K_CHUNK):
                out_quant_write_k0_t = out_quant_write_kb_t * K_CHUNK
                out_quant_write_bf16_t = pl.slice(
                    attn_out, [1, K_CHUNK], [out_quant_row_t, out_quant_write_k0_t]
                )
                out_quant_write_f_t = pl.cast(out_quant_write_bf16_t, target_type=pl.FP32)
                out_quant_scaled_t = pl.mul(out_quant_write_f_t, out_quant_scale_q_t)
                out_quant_i32_t = pl.cast(out_quant_scaled_t, target_type=pl.INT32, mode="rint")
                out_quant_i32_t = pl.minimum(
                    pl.maximum(out_quant_i32_t, pl.full([1, K_CHUNK], dtype=pl.INT32, value=-127)),
                    pl.full([1, K_CHUNK], dtype=pl.INT32, value=127),
                )
                out_quant_half_t = pl.cast(out_quant_i32_t, target_type=pl.FP16, mode="round")
                out_quant_i8_t = pl.cast(out_quant_half_t, target_type=pl.INT8, mode="trunc")
                attn_out_i8 = pl.assemble(attn_out_i8, out_quant_i8_t, [out_quant_row_t, out_quant_write_k0_t])
    with pl.spmd(N_SPLITS_OUT, name_hint="out_proj") as out_proj_tid:
        n_out_proj = pl.tile.get_block_idx()
        n_op = n_out_proj * OUT_TN
        out_c_acc = pl.full([BATCH, OUT_TN], dtype=pl.INT32, value=0)
        for k_split_out in pl.range(K_SPLITS_OUT):
            k_op = k_split_out * OUT_TK
            out_acc_k = pl.matmul(
                pl.tensor.set_validshape(attn_out_i8[:, k_op : k_op + OUT_INNER_TK], ACTIVE_BATCH, OUT_INNER_TK),
                wo[layer_hidden_base + k_op : layer_hidden_base + OUT_INNER_TK + k_op, n_op : n_op + OUT_TN],
                out_dtype=pl.INT32,
            )
            for out_lk in pl.range(1, OUT_N_SUB_K):
                out_ks_off = out_lk * OUT_INNER_TK
                out_a_k = pl.tensor.set_validshape(
                    attn_out_i8[:, k_op + out_ks_off : k_op + out_ks_off + OUT_INNER_TK],
                    ACTIVE_BATCH,
                    OUT_INNER_TK,
                )
                out_w_k = wo[
                    layer_hidden_base
                    + k_op
                    + out_ks_off : layer_hidden_base
                    + k_op
                    + out_ks_off
                    + OUT_INNER_TK,
                    n_op : n_op + OUT_TN,
                ]
                out_acc_k = pl.matmul_acc(out_acc_k, out_a_k, out_w_k)
            out_c_acc = pl.add(out_c_acc, out_acc_k)
        w_scale_col = pl.reshape(pl.slice(wo_scale, [1, OUT_TN], [layer_idx, n_op]), [1, OUT_TN])
        out_fp32 = pl.mul(pl.col_expand_mul(pl.cast(out_c_acc, target_type=pl.FP32), w_scale_col), attn_out_scales)
        attn_proj_fp32 = pl.assemble(attn_proj_fp32, out_fp32, [0, n_op])

    for k_slice in pl.unroll(K_SPLITS_MLP):
        k_base = k_slice * MLP_K_SLICE
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="residual_rms_cast",
            deps=[out_proj_tid],
        ):
            for kb in pl.pipeline(MLP_K_SLICE // K_CHUNK, stage=2):
                resid_k0 = k_base + kb * K_CHUNK
                resid_attn_chunk = attn_proj_fp32[:, resid_k0 : resid_k0 + K_CHUNK]
                resid_hidden_chunk = hidden_states[:, resid_k0 : resid_k0 + K_CHUNK]
                resid_fp32 = pl.add(resid_attn_chunk, pl.cast(resid_hidden_chunk, target_type=pl.FP32))
                post_norm_partial = pl.assemble(post_norm_partial, pl.cast(resid_fp32, target_type=pl.BF16), [0, resid_k0])
                post_gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [layer_idx, resid_k0])
                mlp_norm_in = pl.assemble(mlp_norm_in, pl.cast(pl.col_expand_mul(resid_fp32, post_gamma), target_type=pl.BF16), [0, resid_k0])

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="post_rms_reduce",
        deps=[out_proj_tid],
    ):
        sq_sum = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HIDDEN // K_CHUNK, stage=2):
            post_rms_k0 = kb * K_CHUNK
            post_rms_attn_chunk = attn_proj_fp32[:, post_rms_k0 : post_rms_k0 + K_CHUNK]
            post_rms_hidden_chunk = hidden_states[:, post_rms_k0 : post_rms_k0 + K_CHUNK]
            resid_chunk = pl.add(post_rms_attn_chunk, pl.cast(post_rms_hidden_chunk, target_type=pl.FP32))
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(resid_chunk, resid_chunk)), [1, BATCH]))
        post_inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))
        post_inv_rms_col = pl.reshape(post_inv_rms, [BATCH, 1])
        inv_rms_tile = pl.assemble(inv_rms_tile, post_inv_rms_col, [0, 0])

    silu_tids = pl.array.create(MLP_ON, pl.TASK_ID)
    gate_up_seed_tids = pl.array.create(MLP_ON, pl.TASK_ID)
    for n_out in pl.parallel(MLP_ON):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_up_seed") as gate_up_seed_tid:
            gate_n0 = n_out * MLP_TN
            zero = pl.full([BATCH, MLP_TN], dtype=pl.FP32, value=0.0)
            gate_acc_all = pl.assemble(gate_acc_all, zero, [0, gate_n0])
            up_acc_all = pl.assemble(up_acc_all, zero, [0, gate_n0])
        gate_up_seed_tids[n_out] = gate_up_seed_tid

    for n_out in pl.parallel(MLP_ON):
        gate_n0 = n_out * MLP_TN
        gate_up_tile_tids = pl.array.create(MLP_GATE_UP_DEP_COUNT, pl.TASK_ID)
        for k_split in pl.range(K_SPLITS_MLP):
            gate_k0 = k_split * MLP_K_SLICE
            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="gate_up_dual_proj",
                deps=[gate_up_seed_tids[n_out]],
                no_dep_args=[gate_acc_all, up_acc_all],
            ) as gate_up_tid:
                pyop_a_mat_0 = pl.tile.load(mlp_norm_in, [0, gate_k0], [BATCH, MLP_DUAL_L0_K], [ACTIVE_BATCH, MLP_DUAL_L0_K], target_memory=pl.MemorySpace.Mat)
                pyop_a_left_0 = pl.move(pyop_a_mat_0, target_memory=pl.MemorySpace.Left)
                pyop_gate_w_mat_0 = pl.tile.load(w_gate, [layer_hidden_base + gate_k0, gate_n0], [MLP_DUAL_L0_K, MLP_TN], [MLP_DUAL_L0_K, MLP_TN], target_memory=pl.MemorySpace.Mat)
                pyop_gate_w_right_0 = pl.move(pyop_gate_w_mat_0, target_memory=pl.MemorySpace.Right)
                pyop_up_w_mat_0 = pl.tile.load(w_up, [layer_hidden_base + gate_k0, gate_n0], [MLP_DUAL_L0_K, MLP_TN], [MLP_DUAL_L0_K, MLP_TN], target_memory=pl.MemorySpace.Mat)
                pyop_up_w_right_0 = pl.move(pyop_up_w_mat_0, target_memory=pl.MemorySpace.Right)
                pyop_gate_c_acc = pl.tile.matmul(pyop_a_left_0, pyop_gate_w_right_0)
                pyop_up_c_acc = pl.tile.matmul(pyop_a_left_0, pyop_up_w_right_0)
                for kk, (pyop_gate_acc_iter, pyop_up_acc_iter) in pl.pipeline(
                    MLP_DUAL_L0_K,
                    MLP_K_SLICE,
                    MLP_DUAL_L0_K,
                    stage=2,
                    init_values=(pyop_gate_c_acc, pyop_up_c_acc),
                ):
                    pyop_a_mat_k = pl.tile.load(mlp_norm_in, [0, gate_k0 + kk], [BATCH, MLP_DUAL_L0_K], [ACTIVE_BATCH, MLP_DUAL_L0_K], target_memory=pl.MemorySpace.Mat)
                    pyop_gate_w_mat_k = pl.tile.load(w_gate, [layer_hidden_base + gate_k0 + kk, gate_n0], [MLP_DUAL_L0_K, MLP_TN], [MLP_DUAL_L0_K, MLP_TN], target_memory=pl.MemorySpace.Mat)
                    pyop_up_w_mat_k = pl.tile.load(w_up, [layer_hidden_base + gate_k0 + kk, gate_n0], [MLP_DUAL_L0_K, MLP_TN], [MLP_DUAL_L0_K, MLP_TN], target_memory=pl.MemorySpace.Mat)
                    pyop_a_left_k = pl.move(pyop_a_mat_k, target_memory=pl.MemorySpace.Left)
                    pyop_gate_w_right_k = pl.move(pyop_gate_w_mat_k, target_memory=pl.MemorySpace.Right)
                    pyop_gate_next = pl.tile.matmul_acc(pyop_gate_acc_iter, pyop_a_left_k, pyop_gate_w_right_k)
                    pyop_up_w_right_k = pl.move(pyop_up_w_mat_k, target_memory=pl.MemorySpace.Right)
                    pyop_up_next = pl.tile.matmul_acc(pyop_up_acc_iter, pyop_a_left_k, pyop_up_w_right_k)
                    pyop_gate_c_acc, pyop_up_c_acc = pl.yield_(pyop_gate_next, pyop_up_next)
                gate_acc_all = pl.tile.store(pyop_gate_c_acc, [0, gate_n0], gate_acc_all, atomic=pl.AtomicType.Add)
                up_acc_all = pl.tile.store(pyop_up_c_acc, [0, gate_n0], up_acc_all, atomic=pl.AtomicType.Add)
            gate_up_tile_tids[k_split] = gate_up_tid
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="silu",
            deps=[gate_up_tile_tids],
            no_dep_args=[mlp_down_tile],
        ) as silu_tid:
            silu_inv_rms_chunk = inv_rms_tile[:, 0:1]
            for sub in pl.range(SILU_INNER_CHUNKS):
                silu_off = gate_n0 + sub * MLP_OUT_CHUNK
                gate_chunk = gate_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]
                up_chunk = up_acc_all[:, silu_off : silu_off + MLP_OUT_CHUNK]
                scaled_gate = pl.row_expand_mul(gate_chunk, silu_inv_rms_chunk)
                scaled_up = pl.row_expand_mul(up_chunk, silu_inv_rms_chunk)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(scaled_gate)), 1.0))
                mlp_chunk = pl.mul(pl.mul(scaled_gate, sigmoid), scaled_up)
                mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
                mlp_down_tile = pl.assemble(mlp_down_tile, mlp_chunk_bf16, [0, silu_off])
        silu_tids[n_out] = silu_tid
    silu_barrier = pl.system.task_dummy(deps=[silu_tids])

    down_dual_tids = pl.array.create(DOWN_DUAL_ON, pl.TASK_ID)
    for n_pair in pl.parallel(DOWN_DUAL_ON):
        down_n0 = n_pair * DOWN_DUAL_PAIR * DOWN_TN
        down_n1 = down_n0 + DOWN_TN
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.NONE, slot_num=2)],
            no_dep_args=[down_acc_all],
            name_hint="down_dual_proj",
            deps=[silu_barrier],
        ) as down_dual_tid:
            down_a_mat_0 = pl.tile.load(
                mlp_down_tile,
                [0, 0],
                [BATCH, DOWN_DUAL_L0_K],
                [ACTIVE_BATCH, DOWN_DUAL_L0_K],
                target_memory=pl.MemorySpace.Mat,
            )
            down_a_left_0 = pl.move(down_a_mat_0, target_memory=pl.MemorySpace.Left)
            down_w0_mat_0 = pl.tile.load(
                w_down,
                [layer_inter_base, down_n0],
                [DOWN_DUAL_L0_K, DOWN_TN],
                [DOWN_DUAL_L0_K, DOWN_TN],
                target_memory=pl.MemorySpace.Mat,
            )
            down_w0_right_0 = pl.move(down_w0_mat_0, target_memory=pl.MemorySpace.Right)
            down_acc0 = pl.tile.matmul(down_a_left_0, down_w0_right_0)
            down_w1_mat_0 = pl.tile.load(
                w_down,
                [layer_inter_base, down_n1],
                [DOWN_DUAL_L0_K, DOWN_TN],
                [DOWN_DUAL_L0_K, DOWN_TN],
                target_memory=pl.MemorySpace.Mat,
            )
            down_w1_right_0 = pl.move(down_w1_mat_0, target_memory=pl.MemorySpace.Right)
            down_acc1 = pl.tile.matmul(down_a_left_0, down_w1_right_0)
            for down_k0, (down_acc0_iter, down_acc1_iter) in pl.pipeline(
                DOWN_DUAL_L0_K,
                INTERMEDIATE,
                DOWN_DUAL_L0_K,
                stage=2,
                init_values=(down_acc0, down_acc1),
            ):
                down_a_mat = pl.tile.load(
                    mlp_down_tile,
                    [0, down_k0],
                    [BATCH, DOWN_DUAL_L0_K],
                    [ACTIVE_BATCH, DOWN_DUAL_L0_K],
                    target_memory=pl.MemorySpace.Mat,
                )
                down_a_left = pl.move(down_a_mat, target_memory=pl.MemorySpace.Left)
                down_w0_mat = pl.tile.load(
                    w_down,
                    [layer_inter_base + down_k0, down_n0],
                    [DOWN_DUAL_L0_K, DOWN_TN],
                    [DOWN_DUAL_L0_K, DOWN_TN],
                    target_memory=pl.MemorySpace.Mat,
                )
                down_w0_right = pl.move(down_w0_mat, target_memory=pl.MemorySpace.Right)
                down_next0 = pl.tile.matmul_acc(down_acc0_iter, down_a_left, down_w0_right)
                down_w1_mat = pl.tile.load(
                    w_down,
                    [layer_inter_base + down_k0, down_n1],
                    [DOWN_DUAL_L0_K, DOWN_TN],
                    [DOWN_DUAL_L0_K, DOWN_TN],
                    target_memory=pl.MemorySpace.Mat,
                )
                down_w1_right = pl.move(down_w1_mat, target_memory=pl.MemorySpace.Right)
                down_next1 = pl.tile.matmul_acc(down_acc1_iter, down_a_left, down_w1_right)
                down_acc0, down_acc1 = pl.yield_(down_next0, down_next1)
            down_acc_all = pl.tile.store(down_acc0, [0, down_n0], down_acc_all)
            down_acc_all = pl.tile.store(down_acc1, [0, down_n1], down_acc_all)
        down_dual_tids[n_pair] = down_dual_tid

    with pl.spmd(
        DOWN_ON,
        name_hint="down_cast_residual",
        deps=[down_dual_tids[i] for i in range(DOWN_DUAL_ON)],
    ) as _down_cast_tid:
        n_out = pl.tile.get_block_idx()
        down_cast_n0 = n_out * DOWN_TN
        resid_block_bf16 = post_norm_partial[:, down_cast_n0 : down_cast_n0 + DOWN_TN]
        resid_block = pl.cast(resid_block_bf16, target_type=pl.FP32)
        acc_chunk_bf16 = pl.cast(
            down_acc_all[:, down_cast_n0 : down_cast_n0 + DOWN_TN],
            target_type=pl.BF16,
        )
        acc_chunk = pl.cast(acc_chunk_bf16, target_type=pl.FP32)
        out_chunk = pl.add(acc_chunk, resid_block)
        out_bf16 = pl.cast(out_chunk, target_type=pl.BF16)
        out = pl.assemble(out, out_bf16, [0, down_cast_n0])
    return out


@pl.jit
def _decode_layer_test_entry(  # noqa: PLR0913 - mirrors the model layer signature
    hidden_states: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    wq_scale: pl.Tensor,
    wk_scale: pl.Tensor,
    wv_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    k_cache_scale: pl.Tensor,
    v_cache_scale: pl.Tensor,
    wo: pl.Tensor,
    wo_scale: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    return _decode_layer(
        hidden_states,
        input_rms_weight,
        wq,
        wk,
        wv,
        wq_scale,
        wk_scale,
        wv_scale,
        q_norm_weight,
        k_norm_weight,
        seq_lens,
        block_table,
        slot_mapping,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        k_cache_scale,
        v_cache_scale,
        wo,
        wo_scale,
        w_gate,
        w_up,
        w_down,
        post_rms_weight,
        out,
        0,
    )


@pl.jit
def decode_fwd(  # noqa: PLR0913 — device-side fused NUM_LAYERS decode + LM head
    hidden_states: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    wq_scale: pl.Tensor,
    wk_scale: pl.Tensor,
    wv_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    k_norm_weight: pl.Tensor,
    seq_lens: pl.Tensor,
    block_table: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    k_cache_scale: pl.Tensor,
    v_cache_scale: pl.Tensor,
    wo: pl.Tensor,
    wo_scale: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    post_rms_weight: pl.Tensor,
    final_norm_weight: pl.Tensor,
    lm_head_weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    cur = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    for cb0 in pl.parallel(0, BATCH, BATCH):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_hidden"):
            for ckb in pl.range(HIDDEN // RMSNORM_K_CHUNK):
                ck0 = ckb * RMSNORM_K_CHUNK
                cur = pl.assemble(
                    cur, pl.slice(hidden_states, [BATCH, RMSNORM_K_CHUNK], [cb0, ck0]), [cb0, ck0]
                )
    for layer_idx in pl.range(NUM_LAYERS):
        next_hidden = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
        cur = _decode_layer(
            cur, input_rms_weight,
            wq, wk, wv, wq_scale, wk_scale, wv_scale,
            q_norm_weight, k_norm_weight,
            seq_lens, block_table, slot_mapping, rope_cos, rope_sin,
            k_cache, v_cache, k_cache_scale, v_cache_scale,
            wo, wo_scale,
            w_gate, w_up, w_down, post_rms_weight, next_hidden, layer_idx,
        )
    out = rms_lm_head(cur, final_norm_weight, lm_head_weight, seq_lens, out)
    return out


def _decode_layer_test_inputs(initialize: bool):
    """Build one-layer inputs for CI compile and device smoke tests."""
    import torch

    torch.manual_seed(1234)

    def tensor(shape, dtype, *, scale=1.0):
        value = torch.empty(shape, dtype=dtype)
        if not initialize:
            return value
        if dtype == torch.int8:
            return torch.randint(-2, 3, shape, dtype=dtype)
        return value.normal_(mean=0.0, std=scale)

    seq_lens = torch.arange(1, BATCH + 1, dtype=torch.int32)
    block_table = torch.arange(BATCH, dtype=torch.int32)
    slot_mapping = torch.arange(BATCH, dtype=torch.int32) * BLOCK_SIZE + seq_lens - 1
    cache_rows = BATCH * NUM_KV_HEADS * BLOCK_SIZE
    weight_scale = 1.0 / INT8_SCALE_MAX

    return [
        tensor([BATCH, HIDDEN], torch.bfloat16, scale=0.1),
        torch.ones([1, HIDDEN], dtype=torch.float32),
        tensor([HIDDEN, HIDDEN], torch.int8),
        tensor([HIDDEN, KV_HIDDEN], torch.int8),
        tensor([HIDDEN, KV_HIDDEN], torch.int8),
        torch.full([1, HIDDEN], weight_scale, dtype=torch.float32),
        torch.full([1, KV_HIDDEN], weight_scale, dtype=torch.float32),
        torch.full([1, KV_HIDDEN], weight_scale, dtype=torch.float32),
        torch.ones([1, HEAD_DIM], dtype=torch.float32),
        torch.ones([1, HEAD_DIM], dtype=torch.float32),
        seq_lens,
        block_table,
        slot_mapping,
        torch.ones([BLOCK_SIZE, HEAD_DIM], dtype=torch.float32),
        torch.zeros([BLOCK_SIZE, HEAD_DIM], dtype=torch.float32),
        tensor([cache_rows, HEAD_DIM], torch.int8),
        tensor([cache_rows, HEAD_DIM], torch.int8),
        torch.full([cache_rows, 1], weight_scale, dtype=torch.float32),
        torch.full([cache_rows, 1], weight_scale, dtype=torch.float32),
        tensor([HIDDEN, HIDDEN], torch.int8),
        torch.full([1, HIDDEN], weight_scale, dtype=torch.float32),
        tensor([HIDDEN, INTERMEDIATE], torch.bfloat16, scale=0.002),
        tensor([HIDDEN, INTERMEDIATE], torch.bfloat16, scale=0.002),
        tensor([INTERMEDIATE, HIDDEN], torch.bfloat16, scale=0.002),
        torch.ones([1, HIDDEN], dtype=torch.float32),
    ]


def _patch_test_aicore_bitcast_helpers(work_dir) -> int:
    """Make generated bitcast helpers callable from AICore test kernels."""
    from pathlib import Path

    needle = "static inline To ptoas_bitcast(From from) {"
    replacement = "static __aicore__ inline To ptoas_bitcast(From from) {"
    patched = 0
    for cpp in Path(work_dir).rglob("*.cpp"):
        try:
            source = cpp.read_text()
        except UnicodeDecodeError:
            continue
        if needle not in source:
            continue
        cpp.write_text(source.replace(needle, replacement))
        patched += 1
    return patched


def _main() -> None:
    import argparse

    import torch
    from pypto.backend import BackendType, set_backend_type
    from pypto.runtime import RunConfig

    parser = argparse.ArgumentParser(description="Compile or run one Qwen3-14B A8W8 decode layer.")
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()

    backend_type = BackendType.Ascend950 if args.platform.startswith("a5") else BackendType.Ascend910B
    set_backend_type(backend_type)
    compile_only = args.platform.endswith("sim")
    inputs = _decode_layer_test_inputs(initialize=not compile_only)
    out = torch.empty([BATCH, HIDDEN], dtype=torch.bfloat16)

    if compile_only:
        program = _decode_layer_test_entry.compile_for_test(*inputs, out)
        print(f"Compiled A8W8 decode layer with {len(program.functions)} function(s).")
        return

    out.zero_()
    run_config = RunConfig(
        platform=args.platform,
        device_id=args.device,
        backend_type=backend_type,
        enable_dep_gen=False,
        dump_passes=False,
    )
    program = _decode_layer_test_entry.compile(*inputs, out, config=run_config)
    patched = _patch_test_aicore_bitcast_helpers(program.output_dir)
    print(f"Patched {patched} AICore bitcast helper(s).")
    program(*inputs, out, config=run_config)
    output = out.float()
    if not torch.isfinite(output).all():
        raise RuntimeError("A8W8 decode layer produced non-finite output")
    print(
        "A8W8 decode layer smoke passed: "
        f"shape={tuple(out.shape)}, max_abs={output.abs().max().item():.6f}"
    )


if __name__ == "__main__":
    _main()
