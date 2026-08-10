# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=4
"""DeepSeek-V4 HCA (Hierarchical Compressed Attention) decode orchestration — `compress_ratio == 128` path.
Active in layers 3/5 of the model (2 of the 8 layers in demo). Has the main compressor (ratio=128,
overlap=False) but NO indexer; the compressed-portion topk for sparse_attn comes from a deterministic
index computation, not from a learned indexer score.
Companion files: attention_swa.py (ratio=0)
                 attention_csa_draft.py (ratio=4)."""


import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from attention_tp import (
    GROUP_T_MAX,
    TP_ATTN_SINK_SIZE,
    TP_CHOICES,
    TP_O_A_SIZE,
    TP_O_B_SIZE,
    TP_Q_B_SIZE,
    TP_SIZE,
    gather_sp_bf16,
)

from config import (
    FLASH as M,
    DECODE_BATCH,
    TP,
    DECODE_SEQ,
    BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    DECODE_CMP_BLOCK_NUM,
    DECODE_ORI_BLOCK_NUM,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    KV_ORI_TABLE_MAX_BLOCKS,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope, qkv_proj_rope_local
from rmsnorm import rms_norm
from rope_interleave import rope_interleave
from decode_compressor_ratio128 import compressor_ratio128
from decode_sparse_attn_hca import (
    CMP_TOPK as HCA_SPARSE_CMP_TOPK,
    sparse_attn_hca,
    sparse_attn_hca_tp,
)

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
LEGACY_B = DECODE_BATCH // TP
B = DECODE_BATCH if TP_SIZE > 1 else LEGACY_B
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
GLOBAL_H = M.num_attention_heads
H = GLOBAL_H
LOCAL_H = GLOBAL_H // TP_SIZE
LOCAL_Q = LOCAL_H * M.head_dim
SP_T = GROUP_T_MAX // TP_SIZE
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
GLOBAL_O_GROUPS = M.o_groups
O_GROUPS = GLOBAL_O_GROUPS
LOCAL_O_GROUPS = GLOBAL_O_GROUPS // TP_SIZE
O_GROUP_IN = GLOBAL_H * HEAD_DIM // GLOBAL_O_GROUPS

# kernel-local (HCA: ratio-128 main compressor, no indexer)
COMPRESS_RATIO = 128  # HCA
OVERLAP = COMPRESS_RATIO == 4   # always False for HCA
COFF = 1 + int(OVERLAP)         # always 1 for HCA
MAIN_OUT_DIM = COFF * HEAD_DIM
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
ORI_TABLE_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
ORI_BLOCK_NUM = DECODE_ORI_BLOCK_NUM
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
# Main compressor state pool (kv + score channels merged into one paged FP32 buffer).
COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE
COMPRESS_STATE_PHYSICAL_BLOCKS = 64
COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE
COMPRESS_STATE_BLOCK_NUM = COMPRESS_STATE_PHYSICAL_BLOCKS
COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("HCA_STATE_BLOCK_NUM_DYN")
COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
COMPRESS_TOPK = MAX_SEQ_LEN // COMPRESS_RATIO   # demo 32; flash 128 (= 16384/128); max compressed positions
# HCA has no indexer: the compressed tail is every slot the cache holds, so the
# only bound is the cache capacity (`index_topk` belongs to the ratio-4 indexer).
# Longest context served = COMPRESS_TOPK * COMPRESS_RATIO = MAX_SEQ_LEN.
HCA_TOPK_LIMIT = COMPRESS_TOPK

HCA_CMP_TOPK = HCA_SPARSE_CMP_TOPK

# tiling
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE
HCA_TOPK_TOKEN_TILE = 8   # tokens per cache-window topk SPMD block
HCA_WB_TOKEN_TILE = 8  # tokens per cache-writeback SPMD block

assert GLOBAL_H % TP_SIZE == 0
assert GLOBAL_O_GROUPS % TP_SIZE == 0
if TP_SIZE > 1:
    assert T == GROUP_T_MAX


@pl.jit.inline
def attention_hca(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    # KV cache split into ori (sliding window) and cmp (compressed) pools to match sparse_attn's contract.
    # cmp_kv is shared with the compressor: it writes the compressed row directly into this pool.
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj (fused into sparse_attn)
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    """HCA decode orchestration for compress_ratio=128."""
    t_dim = pl.tensor.dim(x_hc, 0)
    b_dim = t_dim // S
    topk_blocks = t_dim // HCA_TOPK_TOKEN_TILE
    wb_blocks = t_dim // HCA_WB_TOKEN_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    rope_cos_t = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.BF16)
    cmp_cos = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    # Interleave-duplicated / sign-folded compressed-position rope rows. The ratio-128
    # compressor's rmsnorm_rope_cache_write rebuilt this j>>1 dup-gather itself; pl.gather
    # lowers to a per-row TGATHER loop, so it is hoisted here (once, B rows) and read as a
    # plain load downstream.
    cmp_cos_il = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([B, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_rope"):
        for b in pl.range(b_dim):
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

    rope_interleave(cmp_cos, cmp_sin, cmp_cos_il, cmp_sin_signed)

    x_normed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    # Defers kv_proj_matmul one hop behind rms_norm so qr_proj_matmul dispatches first.
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)        # unused on HCA path
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, late_dep,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    for wb_blk in pl.spmd(wb_blocks, name_hint="hca_cache_writeback"):
        wb_t0 = wb_blk * HCA_WB_TOKEN_TILE
        for write_dt in pl.range(HCA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            write_row_i64 = pl.read(ori_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]

    cmp_kv_proj = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
    compressor_ratio128(
        x_normed, cmp_kv_proj,
        compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_cos_il, cmp_sin_signed, cmp_kv,
        position_ids, cmp_slot_mapping, state_slot_mapping,
        late_dep,
    )

    # Sparse-index build fanned out over an SPMD (8 tokens/block) instead of one
    # serial CORE_GROUP loop. The two window-slot abs_pos branches collapse into
    # one: column k -> ring slot k, live iff k <= abs_pos. sparse_attn pairs each
    # K/V by its stored raw value (order-agnostic), so the full-ring rotation is
    # dead. The compressed-slot ramp is fused into the same block.
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    topk_all = pl.create_tensor([t_dim, HCA_CMP_TOPK], dtype=pl.INT32)
    for topk_block in pl.spmd(topk_blocks, name_hint="hca_cache_topk"):
        topk_t0 = topk_block * HCA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(HCA_TOPK_TOKEN_TILE):
            topk_t = topk_t0 + topk_dt
            if topk_t < t_dim:
                topk_b = topk_t // S
                topk_abs_pos = pl.read(position_ids, [topk_t])

                topk_cmp_valid = pl.min(
                    HCA_TOPK_LIMIT,
                    pl.min((topk_abs_pos + 1) // COMPRESS_RATIO, pl.read(kv_seq_lens, [topk_b]) // COMPRESS_RATIO),
                )
                for topk_ck in pl.range(HCA_CMP_TOPK):
                    if topk_ck < topk_cmp_valid:
                        pl.write(topk_all, [topk_t, topk_ck], pl.cast(topk_ck, pl.INT32))
                    else:
                        pl.write(topk_all, [topk_t, topk_ck], pl.cast(-1, pl.INT32))

    sparse_attn_hca(
        q, kv_cache, window_swa_indices,
        cmp_kv, cmp_block_table, topk_all,
        attn_sink, rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit.inline(auto_scope=False)
def attention_hca_tp(
    x_hc: pl.Tensor[[SP_T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[
        [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[DECODE_BATCH, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[DECODE_BATCH, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[GROUP_T_MAX], pl.INT64],
    window_swa_indices: pl.Tensor[[GROUP_T_MAX, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[GROUP_T_MAX], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[GROUP_T_MAX], pl.INT64],
    state_slot_mapping: pl.Tensor[[GROUP_T_MAX], pl.INT64],
    position_ids: pl.Tensor[[GROUP_T_MAX], pl.INT32],
    kv_seq_lens: pl.Tensor[[DECODE_BATCH], pl.INT32],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[SP_T, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    scatter_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.FP32],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run HCA on local HC rows with TP-sharded Q/O heads."""
    x_mixed = pl.create_tensor([SP_T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([SP_T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([SP_T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    x_normed_local = pl.create_tensor([SP_T, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_local)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    x_normed_group = pl.create_tensor([GROUP_T_MAX, D], dtype=pl.BF16)
    x_normed_group = gather_sp_bf16(
        x_normed_local,
        x_normed_group,
        gather_window,
        gather_signal,
        my_rank,
        late_dep,
    )

    rope_cos_t = pl.create_tensor([GROUP_T_MAX, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([GROUP_T_MAX, ROPE_HEAD_DIM], dtype=pl.BF16)
    cmp_cos = pl.create_tensor([DECODE_BATCH, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([DECODE_BATCH, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_rope"):
        for b in pl.range(DECODE_BATCH):
            first_t = b * S
            first_pos_b = pl.read(position_ids, [first_t])
            cmp_offset_b = COMPRESS_RATIO - (first_pos_b % COMPRESS_RATIO)
            cmp_pos_b = pl.cast(first_pos_b + cmp_offset_b - COMPRESS_RATIO, pl.INDEX)
            cmp_cos_row = freqs_cos[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_sin_row = freqs_sin[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_cos[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_cos_row, target_type=pl.FP32)
            cmp_sin[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_sin_row, target_type=pl.FP32)
            for s in pl.range(S):
                token = b * S + s
                pos = pl.cast(pl.read(position_ids, [token]), pl.INDEX)
                cos_row = pl.cast(
                    freqs_cos[pos : pos + 1, 0 : ROPE_HEAD_DIM],
                    target_type=pl.FP32,
                )
                sin_row = pl.cast(
                    freqs_sin[pos : pos + 1, 0 : ROPE_HEAD_DIM],
                    target_type=pl.FP32,
                )
                rope_cos_t[token : token + 1, 0 : ROPE_HEAD_DIM] = pl.cast(
                    cos_row,
                    target_type=pl.BF16,
                    mode="rint",
                )
                rope_sin_t[token : token + 1, 0 : ROPE_HEAD_DIM] = pl.cast(
                    sin_row,
                    target_type=pl.BF16,
                    mode="rint",
                )
    q = pl.create_tensor([GROUP_T_MAX, LOCAL_H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([GROUP_T_MAX, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([GROUP_T_MAX, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([GROUP_T_MAX, 1], dtype=pl.FP32)
    qkv_proj_rope_local(
        x_normed_group,
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
        late_dep,
    )

    kv_cache_flat = pl.reshape(kv_cache, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for wb_blk in pl.spmd(GROUP_T_MAX // HCA_WB_TOKEN_TILE, name_hint="hca_cache_writeback"):
        wb_t0 = wb_blk * HCA_WB_TOKEN_TILE
        for write_dt in pl.range(HCA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            write_row_i64 = pl.read(ori_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[
                    write_t : write_t + 1,
                    0 : HEAD_DIM,
                ]

    cmp_cos_il_group = pl.create_tensor([DECODE_BATCH, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed_group = pl.create_tensor([DECODE_BATCH, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_interleave(cmp_cos, cmp_sin, cmp_cos_il_group, cmp_sin_signed_group)

    cmp_kv_proj = pl.create_tensor([GROUP_T_MAX, HEAD_DIM], dtype=pl.FP32)
    compressor_ratio128(
        x_normed_group,
        cmp_kv_proj,
        compress_state,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_cos_il_group,
        cmp_sin_signed_group,
        cmp_kv,
        position_ids,
        cmp_slot_mapping,
        state_slot_mapping,
        late_dep,
    )

    topk_all = pl.create_tensor([GROUP_T_MAX, HCA_CMP_TOPK], dtype=pl.INT32)
    for topk_block in pl.spmd(GROUP_T_MAX // HCA_TOPK_TOKEN_TILE, name_hint="hca_cache_topk"):
        topk_t0 = topk_block * HCA_TOPK_TOKEN_TILE
        for topk_dt in pl.range(HCA_TOPK_TOKEN_TILE):
            topk_t = topk_t0 + topk_dt
            topk_b = topk_t // S
            topk_abs_pos = pl.read(position_ids, [topk_t])
            topk_cmp_valid = pl.min(
                HCA_TOPK_LIMIT,
                pl.min(
                    (topk_abs_pos + 1) // COMPRESS_RATIO,
                    pl.read(kv_seq_lens, [topk_b]) // COMPRESS_RATIO,
                ),
            )
            for topk_ck in pl.range(HCA_CMP_TOPK):
                if topk_ck < topk_cmp_valid:
                    pl.write(topk_all, [topk_t, topk_ck], pl.cast(topk_ck, pl.INT32))
                else:
                    pl.write(topk_all, [topk_t, topk_ck], pl.cast(-1, pl.INT32))

    attn_out = pl.create_tensor([SP_T, D], dtype=pl.BF16)
    attn_out = sparse_attn_hca_tp(
        q,
        kv_cache,
        window_swa_indices,
        cmp_kv,
        cmp_block_table,
        topk_all,
        attn_sink,
        rope_cos_t,
        rope_sin_t,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
        scatter_window,
        scatter_signal,
        my_rank,
    )
    return hc_post(attn_out, x_hc, post_t, comb_t, x_out)


@pl.jit
def attention_hca_test(
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
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B_DYN, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
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
    state_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    x_out.bind_dynamic(0, T_DYN)

    attention_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, state_slot_mapping,
        position_ids, kv_seq_lens,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
    )
    return x_out


@pl.jit
def attention_hca_tp_test(
    x_hc: pl.Tensor[[SP_T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[
        [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32
    ]],
    compress_state_block_table: pl.Tensor[[DECODE_BATCH, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[DECODE_BATCH, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[GROUP_T_MAX], pl.INT64],
    window_swa_indices: pl.Tensor[[GROUP_T_MAX, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[GROUP_T_MAX], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[GROUP_T_MAX], pl.INT64],
    state_slot_mapping: pl.Tensor[[GROUP_T_MAX], pl.INT64],
    position_ids: pl.Tensor[[GROUP_T_MAX], pl.INT32],
    kv_seq_lens: pl.Tensor[[DECODE_BATCH], pl.INT32],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[SP_T, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    scatter_window: pld.DistributedTensor[[GROUP_T_MAX, D], pl.FP32],
    scatter_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    return attention_hca_tp(
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
        compress_state,
        compress_state_block_table,
        kv_cache,
        cmp_kv,
        cmp_block_table,
        ori_slot_mapping,
        window_swa_indices,
        window_swa_lens,
        cmp_slot_mapping,
        state_slot_mapping,
        position_ids,
        kv_seq_lens,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        x_out,
        gather_window,
        gather_signal,
        scatter_window,
        scatter_signal,
        my_rank,
    )


@pl.jit.host
def l3_attention_hca_tp(
    x_hc: pl.Tensor[[TP_SIZE, SP_T, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_Q], pl.INT8],
    wq_b_scale: pl.Tensor[[TP_SIZE, LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[TP_SIZE, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[TP_SIZE, COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[
        [TP_SIZE, COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32
    ]],
    compress_state_block_table: pl.Tensor[
        [TP_SIZE, DECODE_BATCH, COMPRESS_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[pl.Tensor[
        [TP_SIZE, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    cmp_kv: pl.InOut[pl.Tensor[
        [TP_SIZE, CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ]],
    cmp_block_table: pl.Tensor[[TP_SIZE, DECODE_BATCH, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[TP_SIZE, GROUP_T_MAX], pl.INT64],
    window_swa_indices: pl.Tensor[[TP_SIZE, GROUP_T_MAX, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[TP_SIZE, GROUP_T_MAX], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[TP_SIZE, GROUP_T_MAX], pl.INT64],
    state_slot_mapping: pl.Tensor[[TP_SIZE, GROUP_T_MAX], pl.INT64],
    position_ids: pl.Tensor[[TP_SIZE, GROUP_T_MAX], pl.INT32],
    kv_seq_lens: pl.Tensor[[TP_SIZE, DECODE_BATCH], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, LOCAL_O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[TP_SIZE, SP_T, HC_MULT, D], pl.FP32]],
):
    gather_window_buffer = pld.alloc_window_buffer([GROUP_T_MAX, D], dtype=pl.BF16)
    gather_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    scatter_window_buffer = pld.alloc_window_buffer([GROUP_T_MAX, D], dtype=pl.FP32)
    scatter_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buffer, [GROUP_T_MAX, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        scatter_window = pld.window(scatter_window_buffer, [GROUP_T_MAX, D], dtype=pl.FP32)
        scatter_signal = pld.window(scatter_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        attention_hca_tp_test(
            x_hc[rank],
            hc_attn_fn[rank],
            hc_attn_scale[rank],
            hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank],
            wq_b[rank],
            wq_b_scale[rank],
            wkv[rank],
            gamma_cq[rank],
            gamma_ckv[rank],
            freqs_cos[rank],
            freqs_sin[rank],
            cmp_wkv[rank],
            cmp_wgate[rank],
            cmp_ape[rank],
            cmp_norm_w[rank],
            compress_state[rank],
            compress_state_block_table[rank],
            kv_cache[rank],
            cmp_kv[rank],
            cmp_block_table[rank],
            ori_slot_mapping[rank],
            window_swa_indices[rank],
            window_swa_lens[rank],
            cmp_slot_mapping[rank],
            state_slot_mapping[rank],
            position_ids[rank],
            kv_seq_lens[rank],
            attn_sink[rank],
            wo_a[rank],
            wo_b[rank],
            wo_b_scale[rank],
            x_out[rank],
            gather_window,
            gather_signal,
            scatter_window,
            scatter_signal,
            rank,
            device=rank,
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

    tokens = tensors["x_hc"].shape[0]
    batch = tokens // S
    from hc_post import golden_hc_post

    # ---- Block.hc_pre ----
    x_mixed = torch.zeros(tokens, D, dtype=torch.bfloat16)
    post_t = torch.zeros(tokens, HC_MULT)
    comb_t = torch.zeros(tokens, HC_MULT * HC_MULT)
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
    ratio = COMPRESS_RATIO
    rd = ROPE_HEAD_DIM

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_T = torch.empty(tokens, rd, dtype=freqs_cos.dtype)
    rope_sin_T = torch.empty(tokens, rd, dtype=freqs_sin.dtype)
    for t in range(tokens):
        pos = int(position_ids[t].item())
        rope_cos_T[t] = freqs_cos[pos]
        rope_sin_T[t] = freqs_sin[pos]

    # q + win kv (W8A8 q_proj)
    q = torch.zeros(tokens, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(tokens, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(tokens, 1, dtype=torch.float32)
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
    window_swa_indices = tensors["window_swa_indices"]
    cmp_kv = tensors["cmp_kv"]
    cmp_block_table = tensors["cmp_block_table"]
    attn_out = torch.zeros(tokens, D, dtype=torch.bfloat16)

    half_rd = rd // 2
    cmp_cos = torch.empty(batch, half_rd, dtype=torch.float32)
    cmp_sin = torch.empty(batch, half_rd, dtype=torch.float32)
    for b in range(batch):
        first_pos_b = int(position_ids[b * S].item())
        cmp_offset_b = ratio - (first_pos_b % ratio)
        cmp_pos_b = first_pos_b + cmp_offset_b - ratio
        cmp_cos[b] = freqs_cos[cmp_pos_b, :half_rd].float()
        cmp_sin[b] = freqs_sin[cmp_pos_b, :half_rd].float()

    # The compressor ABI is token-major flat.
    cmp_kv_proj = torch.zeros(tokens, HEAD_DIM, dtype=torch.float32)
    position_ids_flat = position_ids.reshape(-1).to(torch.int32).contiguous()
    cmp_slot_mapping_flat = tensors["cmp_slot_mapping"].reshape(-1).to(torch.int64).contiguous()
    state_slot_mapping_flat = tensors["state_slot_mapping"].reshape(-1).to(torch.int64).contiguous()
    golden_compressor({
        "x": x_normed,
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
        "position_ids": position_ids_flat,
        "cmp_slot_mapping": cmp_slot_mapping_flat,
        "state_slot_mapping": state_slot_mapping_flat,
    })

    ori_slot_mapping = tensors["ori_slot_mapping"].to(torch.int64)
    for t in range(tokens):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]

    topk_all = torch.full((tokens, HCA_CMP_TOPK), -1, dtype=torch.int32)
    for t in range(tokens):
        b = t // S
        abs_pos = int(position_ids[t].item())
        cmp_valid = min(HCA_TOPK_LIMIT, (abs_pos + 1) // ratio, int(kv_seq_lens[b].item()) // ratio)
        if cmp_valid:
            topk_all[t, :cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32)

    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "window_swa_indices": window_swa_indices,
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

    # ===== Block.hc_post =====
    y = torch.zeros(tokens, HC_MULT, D, dtype=torch.float32)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })

    tensors["x_out"][:] = y


def golden_attention_hca_tp(tensors):
    """Torch reference for SP-local HC rows and TP-sharded attention heads."""
    import torch

    from decode_compressor_ratio128 import golden_compressor
    from decode_sparse_attn_hca import golden_sparse_attn_tp
    from hc_post import golden_hc_post
    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_rms_norm

    x_normed_local = []
    post_local = []
    comb_local = []
    for rank in range(TP_SIZE):
        x_mixed = torch.zeros(SP_T, D, dtype=torch.bfloat16)
        post = torch.zeros(SP_T, HC_MULT, dtype=torch.float32)
        comb = torch.zeros(SP_T, HC_MULT * HC_MULT, dtype=torch.float32)
        golden_hc_pre({
            "x": tensors["x_hc"][rank],
            "hc_fn": tensors["hc_attn_fn"][rank],
            "hc_scale": tensors["hc_attn_scale"][rank],
            "hc_base": tensors["hc_attn_base"][rank],
            "x_mixed": x_mixed,
            "post": post,
            "comb": comb,
        })
        x_normed_local.append(golden_rms_norm(x_mixed, tensors["attn_norm_w"][rank]))
        post_local.append(post)
        comb_local.append(comb)

    x_normed_group = torch.cat(x_normed_local, dim=0)
    position_ids = tensors["position_ids"][0].to(torch.int64)
    rope_cos_group = tensors["freqs_cos"][0].index_select(0, position_ids).contiguous()
    rope_sin_group = tensors["freqs_sin"][0].index_select(0, position_ids).contiguous()
    rope_cos = rope_cos_group.unsqueeze(0).repeat(TP_SIZE, 1, 1).contiguous()
    rope_sin = rope_sin_group.unsqueeze(0).repeat(TP_SIZE, 1, 1).contiguous()

    q = torch.zeros(TP_SIZE, GROUP_T_MAX, LOCAL_H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(TP_SIZE, GROUP_T_MAX, HEAD_DIM, dtype=torch.bfloat16)
    for rank in range(TP_SIZE):
        qr = torch.zeros(GROUP_T_MAX, Q_LORA, dtype=torch.int8)
        qr_scale = torch.zeros(GROUP_T_MAX, 1, dtype=torch.float32)
        golden_qkv_proj_rope({
            "x": x_normed_group,
            "wq_a": tensors["wq_a"][rank],
            "wq_b": tensors["wq_b"][rank],
            "wq_b_scale": tensors["wq_b_scale"][rank],
            "wkv": tensors["wkv"][rank],
            "rope_cos": rope_cos_group,
            "rope_sin": rope_sin_group,
            "gamma_cq": tensors["gamma_cq"][rank],
            "gamma_ckv": tensors["gamma_ckv"][rank],
            "q": q[rank],
            "kv": kv[rank],
            "qr": qr,
            "qr_scale": qr_scale,
        })

    half_rope = ROPE_HEAD_DIM // 2
    for rank in range(TP_SIZE):
        rank_positions = tensors["position_ids"][rank].to(torch.int64)
        cmp_cos = torch.empty(DECODE_BATCH, half_rope, dtype=torch.float32)
        cmp_sin = torch.empty(DECODE_BATCH, half_rope, dtype=torch.float32)
        for batch in range(DECODE_BATCH):
            first_pos = int(rank_positions[batch * S].item())
            cmp_offset = COMPRESS_RATIO - (first_pos % COMPRESS_RATIO)
            cmp_pos = first_pos + cmp_offset - COMPRESS_RATIO
            cmp_cos[batch] = tensors["freqs_cos"][rank, cmp_pos, :half_rope].float()
            cmp_sin[batch] = tensors["freqs_sin"][rank, cmp_pos, :half_rope].float()

        golden_compressor({
            "x": x_normed_group,
            "kv": torch.zeros(GROUP_T_MAX, HEAD_DIM, dtype=torch.float32),
            "compress_state": tensors["compress_state"][rank],
            "compress_state_block_table": tensors["compress_state_block_table"][rank],
            "wkv": tensors["cmp_wkv"][rank],
            "wgate": tensors["cmp_wgate"][rank],
            "ape": tensors["cmp_ape"][rank],
            "norm_w": tensors["cmp_norm_w"][rank],
            "cos": cmp_cos,
            "sin": cmp_sin,
            "cmp_kv_cache": tensors["cmp_kv"][rank],
            "position_ids": tensors["position_ids"][rank],
            "cmp_slot_mapping": tensors["cmp_slot_mapping"][rank],
            "state_slot_mapping": tensors["state_slot_mapping"][rank],
        })

        cache_flat = tensors["kv_cache"][rank].view(-1, HEAD_DIM)
        for token in range(GROUP_T_MAX):
            write_row = int(tensors["ori_slot_mapping"][rank, token].item())
            if write_row >= 0:
                cache_flat[write_row] = kv[rank, token]

    topk_one = torch.full((GROUP_T_MAX, HCA_CMP_TOPK), -1, dtype=torch.int32)
    for token in range(GROUP_T_MAX):
        batch = token // S
        abs_pos = int(tensors["position_ids"][0, token].item())
        cmp_valid = min(
            HCA_TOPK_LIMIT,
            (abs_pos + 1) // COMPRESS_RATIO,
            int(tensors["kv_seq_lens"][0, batch].item()) // COMPRESS_RATIO,
        )
        if cmp_valid:
            topk_one[token, :cmp_valid] = torch.arange(cmp_valid, dtype=torch.int32)
    topk_all = topk_one.unsqueeze(0).repeat(TP_SIZE, 1, 1).contiguous()

    attn_out = torch.zeros(TP_SIZE, SP_T, D, dtype=torch.bfloat16)
    golden_sparse_attn_tp({
        "q": q,
        "ori_kv": tensors["kv_cache"],
        "window_swa_indices": tensors["window_swa_indices"],
        "cmp_kv": tensors["cmp_kv"],
        "cmp_block_table": tensors["cmp_block_table"],
        "cmp_sparse_indices": topk_all,
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos,
        "freqs_sin": rope_sin,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    for rank in range(TP_SIZE):
        golden_hc_post({
            "x": attn_out[rank],
            "residual": tensors["x_hc"][rank],
            "post": post_local[rank],
            "comb": comb_local[rank],
            "y": tensors["x_out"][rank],
        })


def build_tensor_specs(start_pos=None, batch=B):
    tokens = batch * S
    import torch  # type: ignore[import]
    from utils import (
        block_table,
        compressed_slot_mapping,
        hca_decode_start_set,
        kv_seq_lens_from_starts,
        ori_slot_mapping,
        position_ids_from_starts,
        resolve_start_positions,
        state_slot_mapping,
        swa_indices_and_lens,
    )
    from golden import TensorSpec
    from utils import build_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

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
        return torch.empty(tokens, HC_MULT, D).uniform_(-1, 1)
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
        return torch.randn(MAIN_OUT_DIM, D) * 0.0246
    def init_cmp_wgate():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0316
    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.0340
    def init_cmp_norm_w():
        return 0.1001 + 0.0549 * torch.randn(HEAD_DIM)
    def init_compress_state():
        return torch.zeros(COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM)
    def init_compress_state_block_table():
        return block_table(
            batch=batch,
            table_blocks=COMPRESS_STATE_MAX_BLOCKS,
            physical_blocks=COMPRESS_STATE_PHYSICAL_BLOCKS,
        )
    def init_kv_cache():
        return init_normalized_cache((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))
    def init_cmp_kv():
        return init_normalized_cache((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_window_block_table():
        return block_table(batch=batch, table_blocks=ORI_TABLE_MAX_BLOCKS, physical_blocks=ORI_MAX_BLOCKS)

    def init_cmp_block_table():
        return block_table(
            batch=batch,
            table_blocks=CMP_MAX_BLOCKS,
            physical_blocks=CMP_BLOCK_NUM if TP_SIZE > 1 else CMP_MAX_BLOCKS,
        )

    def init_attn_sink():
        return torch.zeros(H)
    def init_default_start_pos():
        # Canonical HCA start-position set (ratio-128 compressor branches + 8k long-context).
        return hca_decode_start_set(
            batch=batch, compress_ratio=COMPRESS_RATIO, state_block_size=COMPRESS_STATE_BLOCK_SIZE)
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
    def init_state_slot_mapping():
        return state_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S),
            init_compress_state_block_table(),
            state_block_size=COMPRESS_STATE_BLOCK_SIZE,
        ).reshape(-1).contiguous()
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
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
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec("compress_state", [COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], torch.float32, init_value=init_compress_state),
        TensorSpec("compress_state_block_table", [batch, COMPRESS_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [batch, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("ori_slot_mapping", [tokens], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("window_swa_indices", [tokens, WIN], torch.int32, init_value=init_window_swa_indices),
        TensorSpec("window_swa_lens", [tokens], torch.int32, init_value=init_window_swa_lens),
        TensorSpec("cmp_slot_mapping", [tokens], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [tokens], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("position_ids", [tokens], torch.int32, init_value=init_position_ids),
        TensorSpec("kv_seq_lens", [batch], torch.int32, init_value=init_kv_seq_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [tokens, HC_MULT, D], torch.float32, is_output=True),
    ]


def build_tp_tensor_specs(start_pos=None):
    """Build full-group decode state with SP rows and contiguous TP weight shards."""
    import torch
    from golden import TensorSpec

    base_specs = build_tensor_specs(start_pos, batch=DECODE_BATCH)
    base = {
        spec.name: spec.create_tensor()
        for spec in base_specs
        if spec.name != "x_out"
    }

    def replicated(name):
        value = base[name]
        repeats = (TP_SIZE,) + (1,) * value.ndim
        return value.unsqueeze(0).repeat(repeats).contiguous()

    def quant_w_per_output_channel(weight):
        amax = weight.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = weight.float() * scale_quant.view(1, GLOBAL_H * HEAD_DIM)
        weight_i32 = torch.round(scaled).to(torch.int32)
        weight_i32 = torch.clamp(weight_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        return weight_i32.to(torch.float16).to(torch.int8), (1.0 / scale_quant).float()

    def quant_w_per_row(weight):
        amax = weight.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = weight.float() * scale_quant.unsqueeze(-1)
        weight_i32 = torch.round(scaled).to(torch.int32)
        weight_i32 = torch.clamp(weight_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        return weight_i32.to(torch.float16).to(torch.int8), (1.0 / scale_quant).float()

    x_hc = base["x_hc"].reshape(TP_SIZE, SP_T, HC_MULT, D).contiguous()

    wq_b_full = (torch.randn(Q_LORA, GLOBAL_H * HEAD_DIM) / Q_LORA ** 0.5).to(torch.bfloat16)
    wq_b_full_i8, wq_b_full_scale = quant_w_per_output_channel(wq_b_full)
    wq_b = torch.stack([
        shard.contiguous()
        for shard in torch.chunk(wq_b_full_i8, TP_SIZE, dim=1)
    ])
    wq_b_scale = wq_b_full_scale.reshape(TP_SIZE, LOCAL_Q).contiguous()

    attn_sink_full = torch.zeros(GLOBAL_H, dtype=torch.float32)
    attn_sink = attn_sink_full.reshape(TP_SIZE, LOCAL_H).contiguous()
    wo_a_full = torch.randn(GLOBAL_O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    wo_a = torch.stack([
        shard.contiguous()
        for shard in torch.chunk(wo_a_full.to(torch.bfloat16), TP_SIZE, dim=0)
    ])
    wo_b_full = (
        torch.randn(D, GLOBAL_O_GROUPS * O_LORA)
        / (GLOBAL_O_GROUPS * O_LORA) ** 0.5
    ).to(torch.bfloat16)
    wo_b_full_i8, wo_b_scale_one = quant_w_per_row(wo_b_full)
    wo_b = torch.stack([
        shard.contiguous()
        for shard in torch.chunk(wo_b_full_i8, TP_SIZE, dim=1)
    ])
    wo_b_scale = wo_b_scale_one.unsqueeze(0).repeat(TP_SIZE, 1).contiguous()

    values = {
        name: replicated(name)
        for name in (
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_attn_base",
            "attn_norm_w",
            "wq_a",
            "wkv",
            "gamma_cq",
            "gamma_ckv",
            "freqs_cos",
            "freqs_sin",
            "cmp_wkv",
            "cmp_wgate",
            "cmp_ape",
            "cmp_norm_w",
            "compress_state",
            "compress_state_block_table",
            "kv_cache",
            "cmp_kv",
            "cmp_block_table",
            "ori_slot_mapping",
            "window_swa_indices",
            "window_swa_lens",
            "cmp_slot_mapping",
            "state_slot_mapping",
            "position_ids",
            "kv_seq_lens",
        )
    }

    return [
        TensorSpec("x_hc", [TP_SIZE, SP_T, HC_MULT, D], torch.float32, init_value=lambda: x_hc),
        TensorSpec("hc_attn_fn", [TP_SIZE, MIX_HC, HC_DIM], torch.float32,
                   init_value=lambda: values["hc_attn_fn"], resident="stacked"),
        TensorSpec("hc_attn_scale", [TP_SIZE, 3], torch.float32,
                   init_value=lambda: values["hc_attn_scale"], resident="stacked"),
        TensorSpec("hc_attn_base", [TP_SIZE, MIX_HC], torch.float32,
                   init_value=lambda: values["hc_attn_base"], resident="stacked"),
        TensorSpec("attn_norm_w", [TP_SIZE, D], torch.bfloat16,
                   init_value=lambda: values["attn_norm_w"], resident="stacked"),
        TensorSpec("wq_a", [TP_SIZE, D, Q_LORA], torch.bfloat16,
                   init_value=lambda: values["wq_a"], resident="stacked"),
        TensorSpec("wq_b", [TP_SIZE, Q_LORA, LOCAL_Q], torch.int8,
                   init_value=lambda: wq_b, resident="stacked"),
        TensorSpec("wq_b_scale", [TP_SIZE, LOCAL_Q], torch.float32,
                   init_value=lambda: wq_b_scale, resident="stacked"),
        TensorSpec("wkv", [TP_SIZE, D, HEAD_DIM], torch.bfloat16,
                   init_value=lambda: values["wkv"], resident="stacked"),
        TensorSpec("gamma_cq", [TP_SIZE, Q_LORA], torch.bfloat16,
                   init_value=lambda: values["gamma_cq"], resident="stacked"),
        TensorSpec("gamma_ckv", [TP_SIZE, HEAD_DIM], torch.bfloat16,
                   init_value=lambda: values["gamma_ckv"], resident="stacked"),
        TensorSpec("freqs_cos", [TP_SIZE, MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16,
                   init_value=lambda: values["freqs_cos"]),
        TensorSpec("freqs_sin", [TP_SIZE, MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16,
                   init_value=lambda: values["freqs_sin"]),
        TensorSpec("cmp_wkv", [TP_SIZE, MAIN_OUT_DIM, D], torch.bfloat16,
                   init_value=lambda: values["cmp_wkv"], resident="stacked"),
        TensorSpec("cmp_wgate", [TP_SIZE, MAIN_OUT_DIM, D], torch.bfloat16,
                   init_value=lambda: values["cmp_wgate"], resident="stacked"),
        TensorSpec("cmp_ape", [TP_SIZE, COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32,
                   init_value=lambda: values["cmp_ape"], resident="stacked"),
        TensorSpec("cmp_norm_w", [TP_SIZE, HEAD_DIM], torch.bfloat16,
                   init_value=lambda: values["cmp_norm_w"], resident="stacked"),
        TensorSpec(
            "compress_state",
            [TP_SIZE, COMPRESS_STATE_BLOCK_NUM, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM],
            torch.float32,
            init_value=lambda: values["compress_state"],
            is_output=True,
        ),
        TensorSpec(
            "compress_state_block_table",
            [TP_SIZE, DECODE_BATCH, COMPRESS_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=lambda: values["compress_state_block_table"],
        ),
        TensorSpec(
            "kv_cache",
            [TP_SIZE, ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: values["kv_cache"],
            is_output=True,
        ),
        TensorSpec(
            "cmp_kv",
            [TP_SIZE, CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: values["cmp_kv"],
            is_output=True,
        ),
        TensorSpec("cmp_block_table", [TP_SIZE, DECODE_BATCH, CMP_MAX_BLOCKS], torch.int32,
                   init_value=lambda: values["cmp_block_table"]),
        TensorSpec("ori_slot_mapping", [TP_SIZE, GROUP_T_MAX], torch.int64,
                   init_value=lambda: values["ori_slot_mapping"]),
        TensorSpec("window_swa_indices", [TP_SIZE, GROUP_T_MAX, WIN], torch.int32,
                   init_value=lambda: values["window_swa_indices"]),
        TensorSpec("window_swa_lens", [TP_SIZE, GROUP_T_MAX], torch.int32,
                   init_value=lambda: values["window_swa_lens"]),
        TensorSpec("cmp_slot_mapping", [TP_SIZE, GROUP_T_MAX], torch.int64,
                   init_value=lambda: values["cmp_slot_mapping"]),
        TensorSpec("state_slot_mapping", [TP_SIZE, GROUP_T_MAX], torch.int64,
                   init_value=lambda: values["state_slot_mapping"]),
        TensorSpec("position_ids", [TP_SIZE, GROUP_T_MAX], torch.int32,
                   init_value=lambda: values["position_ids"]),
        TensorSpec("kv_seq_lens", [TP_SIZE, DECODE_BATCH], torch.int32,
                   init_value=lambda: values["kv_seq_lens"]),
        TensorSpec("attn_sink", [TP_SIZE, LOCAL_H], torch.float32, init_value=lambda: attn_sink),
        TensorSpec("wo_a", [TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16,
                   init_value=lambda: wo_a, resident="stacked"),
        TensorSpec("wo_b", [TP_SIZE, D, LOCAL_O_GROUPS * O_LORA], torch.int8,
                   init_value=lambda: wo_b, resident="stacked"),
        TensorSpec("wo_b_scale", [TP_SIZE, D], torch.float32,
                   init_value=lambda: wo_b_scale, resident="stacked"),
        TensorSpec("x_out", [TP_SIZE, SP_T, HC_MULT, D], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=None, choices=list(TP_CHOICES))
    parser.add_argument("--tp-q-b", type=int, default=TP_Q_B_SIZE, choices=list(TP_CHOICES))
    parser.add_argument("--tp-attn-sink", type=int, default=TP_ATTN_SINK_SIZE, choices=list(TP_CHOICES))
    parser.add_argument("--tp-o-a", type=int, default=TP_O_A_SIZE, choices=list(TP_CHOICES))
    parser.add_argument("--tp-o-b", type=int, default=TP_O_B_SIZE, choices=list(TP_CHOICES))
    parser.add_argument("-d", "--device", type=str, default=",".join(str(rank) for rank in range(TP_SIZE)))
    parser.add_argument("-b", "--batch", type=int, default=B,
                        help=f"runtime request count; a multiple of 4 up to {B} (the compile-time "
                             "upper bound). The token axis is pl.dynamic, so one compiled program "
                             "serves every value.")
    parser.add_argument("--start-pos", type=int, default=None,
                        help="Uniform fixture-only start_pos override for all batches; "
                             "default (unset) uses the canonical per-batch HCA set that includes the 8k point.")
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    parsed_tp = (args.tp_q_b, args.tp_attn_sink, args.tp_o_a, args.tp_o_b)
    resolved_tp = (TP_Q_B_SIZE, TP_ATTN_SINK_SIZE, TP_O_A_SIZE, TP_O_B_SIZE)
    if parsed_tp != resolved_tp:
        parser.error(f"import-time attention TP {resolved_tp} does not match parsed TP {parsed_tp}")
    if TP_SIZE == 1:
        if args.batch < 4 or args.batch > LEGACY_B or args.batch % 4 != 0:
            parser.error(f"--batch must be a multiple of 4 in [4, {LEGACY_B}], got {args.batch}")
    elif args.batch != DECODE_BATCH:
        parser.error(f"distributed HCA requires the full-group batch {DECODE_BATCH}, got {args.batch}")

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < TP_SIZE:
        parser.error(f"need at least {TP_SIZE} devices, got {device_ids}")

    if TP_SIZE == 1:
        fn = attention_hca_test
        specs = build_tensor_specs(args.start_pos, batch=args.batch)
        golden_fn = golden_attention_hca
        compile_cfg = dict(dump_passes=args.dump_passes)
        runtime_cfg = dict(
            platform=args.platform,
            device_id=device_ids[0],
            enable_l2_swimlane=args.enable_l2_swimlane,
        )
    else:
        fn = l3_attention_hca_tp
        specs = build_tp_tensor_specs(args.start_pos)
        golden_fn = golden_attention_hca_tp
        compile_cfg = dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:TP_SIZE],
                num_sub_workers=0,
            ),
        )
        runtime_cfg = dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        )

    result = run_jit(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        compile_cfg=compile_cfg,
        runtime_cfg=runtime_cfg,
        atol=1e-2,
        compare_fn={
            # Tightened from CANN's 1e-2 bar: the realistic layer-9 hc_attn gates keep
            # x_out well-conditioned, so it holds 0% over 3e-3 (worst rdiff well under 1).
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
        rtol=1e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
