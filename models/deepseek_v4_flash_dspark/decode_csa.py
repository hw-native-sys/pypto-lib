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
    KV_ORI_BLOCK_NUM,
    INT8_SCALE_MAX,
    INT8_AMAX_EPS,
)
from decode_compressor_ratio4 import compressor_ratio4
from hc_post import hc_post
from hc_pre import hc_pre
from decode_indexer import indexer
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import rms_norm
from decode_o_proj import (
    ATTENTION_WINDOW_ROWS,
    GROUP_T_PAD,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    LOCAL_T,
    LOCAL_T_PAD,
    O_WINDOW_ROWS,
    decode_o_proj,
    decode_o_proj_tp1,
    o_group_a2a,
    o_proj_reduce_scatter,
)
from decode_sparse_attn_csa import (
    ROPE_CS_T_TILE,
    T_PAD,
    sparse_attn_csa,
)

# Dynamic shape variables.
B_DYN = pl.dynamic("B_DYN")  # per-request axis
T_DYN = pl.dynamic("T_DYN")  # T = B * S
ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")
IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic("IDX_CACHE_BLOCK_NUM_DYN")
MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic("CSA_STATE_BLOCK_NUM_DYN")
INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("INNER_STATE_BLOCK_NUM_DYN")

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
MAX_SEQ_LEN = 1_048_576
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
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
MAIN_STATE_LEN = COFF * COMPRESS_RATIO
MAIN_STATE_MAX_BLOCKS = (MAIN_STATE_LEN + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE
INNER_OUT_DIM = COFF * IDX_HEAD_DIM
INNER_STATE_DIM = 2 * INNER_OUT_DIM
INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE
INNER_STATE_LEN = COFF * COMPRESS_RATIO
INNER_STATE_MAX_BLOCKS = (INNER_STATE_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE
ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
CMP_MAX_ROWS = MAX_SEQ_LEN // COMPRESS_RATIO
CMP_MAX_BLOCKS = (CMP_MAX_ROWS + BLOCK_SIZE - 1) // BLOCK_SIZE
IDX_MAX_BLOCKS = CMP_MAX_BLOCKS

# tiling
CSA_WB_TOKEN_TILE = 8

if T != LOCAL_T:
    raise ValueError(f"CSA token capacity {T} must equal TP local token capacity {LOCAL_T}")
if T_PAD != LOCAL_T_PAD:
    raise ValueError(f"CSA token capacity {T_PAD} must equal TP local token capacity {LOCAL_T_PAD}")


@pl.jit.inline(auto_scope=False)
def decode_csa(
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
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
    idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32],
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Run one rank of the complete tensor-parallel CSA layer."""
    t_dim = pl.tensor.dim(x_hc, 0)
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    idx_topk_scores = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.FP32)
    idx_topk = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.INT32)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    wb_blocks = t_dim // CSA_WB_TOKEN_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.scope():
        hc_pre(
            x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base,
            x_mixed, post_t, comb_t,
        )

    idx_cos_il = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    idx_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_cos_il = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_interleave") as rope_tid:
        il_ones = pl.full([4, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_col = pl.col_expand_mul(
            il_ones,
            pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32),
        )
        il_dup_f = pl.cast(
            pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
        for rope_t0 in pl.range(0, t_dim, 4):
            idx_cos_il[rope_t0 : rope_t0 + 4, :] = pl.gather(
                pl.cast(freqs_cos[rope_t0 : rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32),
                dim=-1,
                index=il_dup_idx,
            )
            idx_sin_signed[rope_t0 : rope_t0 + 4, :] = pl.mul(
                pl.gather(
                    pl.cast(freqs_sin[rope_t0 : rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32),
                    dim=-1,
                    index=il_dup_idx,
                ),
                il_sign,
            )
            cmp_cos_il[rope_t0 : rope_t0 + 4, :] = pl.gather(
                pl.cast(cmp_freqs_cos[rope_t0 : rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32),
                dim=-1,
                index=il_dup_idx,
            )
            cmp_sin_signed[rope_t0 : rope_t0 + 4, :] = pl.mul(
                pl.gather(
                    pl.cast(cmp_freqs_sin[rope_t0 : rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32),
                    dim=-1,
                    index=il_dup_idx,
                ),
                il_sign,
            )

    x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.scope():
        rms_norm(x_mixed, attn_norm_w, x_normed_t)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    with pl.scope():
        late_dep = pl.system.task_dummy(deps=[rope_tid])
        qkv_proj_rope(
            x_normed_t, wq_a, wq_b, wq_b_scale, wkv,
            freqs_cos, freqs_sin, gamma_cq, gamma_ckv,
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
                    kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[
                        write_t : write_t + 1, 0 : HEAD_DIM
                    ]

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
        indexer(
            x_normed_t, qr, qr_scale, idx_wq_b, idx_wq_b_scale,
            weights_proj, idx_cos_il, idx_sin_signed, cmp_cos_il, cmp_sin_signed,
            hadamard_idx,
            idx_kv_unused, inner_compress_state, inner_compress_state_block_table,
            inner_wkv, inner_wgate, inner_ape, inner_norm_w,
            idx_kv_cache, idx_kv_scale, idx_block_table,
            idx_topk_scores, idx_topk,
            position_ids, idx_slot_mapping, inner_state_slot_mapping,
            kv_seq_lens, late_dep,
        )

    position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])
    attention_grouped = pl.create_tensor([O_GROUPS * LOCAL_T_PAD, O_GROUP_IN], dtype=pl.BF16)
    attention_grouped, _heads_tid = sparse_attn_csa(
        q, kv_cache, window_swa_indices,
        cmp_kv, cmp_block_table, idx_topk,
        position_ids_t1, attn_sink, freqs_cos, freqs_sin,
        attention_grouped,
    )

    attention_local_flat = pl.create_tensor([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_local_flat, attention_signal = o_group_a2a(
        attention_grouped, attention_local_flat,
        attention_window, attention_signal,
        group_base, tp_rank, local_t,
    )

    attention_local_groups = pl.reshape(attention_local_flat, [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN])
    o_partial = pl.create_tensor([GROUP_T_PAD, D], dtype=pl.FP32)
    o_partial, projection_tid = decode_o_proj(
        attention_local_groups,
        wo_a, wo_b, wo_b_scale,
        local_t, o_partial,
    )
    o_local = pl.create_tensor([LOCAL_T_PAD, D], dtype=pl.BF16)
    o_local, o_signal = o_proj_reduce_scatter(
        o_partial, o_local,
        o_window, o_signal,
        group_base, tp_rank, local_t, projection_tid,
    )
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    for block in pl.spmd(t_dim // CSA_WB_TOKEN_TILE, name_hint="csa_o_local_bridge"):
        token_start = block * CSA_WB_TOKEN_TILE
        attn_out[token_start : token_start + CSA_WB_TOKEN_TILE, 0:D] = o_local[
            token_start : token_start + CSA_WB_TOKEN_TILE, 0:D
        ]
    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def decode_csa_test(
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[B_DYN, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32]
    ],
    inner_compress_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[
        pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]
    ],
    idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32],
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Compile one rank of the complete tensor-parallel CSA layer."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, B_DYN)
    inner_compress_state.bind_dynamic(0, INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(0, B_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    idx_kv_cache.bind_dynamic(0, IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(0, IDX_CACHE_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(0, B_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    idx_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    x_out.bind_dynamic(0, T_DYN)

    return decode_csa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin,
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
        attention_window, attention_signal, o_window, o_signal,
        group_base, tp_rank, local_t,
    )


@pl.jit.host
def l3_decode_csa(
    x_hc: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[TP_SIZE, COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[
        [TP_SIZE, MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32
    ]],
    compress_state_block_table: pl.Tensor[[TP_SIZE, B_DYN, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[TP_SIZE, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[TP_SIZE, D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[TP_SIZE, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[TP_SIZE, INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[TP_SIZE, INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[TP_SIZE, COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[TP_SIZE, IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[pl.Tensor[
        [TP_SIZE, INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32
    ]],
    inner_compress_state_block_table: pl.Tensor[[TP_SIZE, B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[[TP_SIZE, CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_block_table: pl.Tensor[[TP_SIZE, B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]
    ],
    idx_block_table: pl.Tensor[[TP_SIZE, B_DYN, IDX_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[TP_SIZE, T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    position_ids: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[TP_SIZE, B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch the complete CSA layer on one physical TP group."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    cmp_freqs_cos.bind_dynamic(1, T_DYN)
    cmp_freqs_sin.bind_dynamic(1, T_DYN)
    compress_state.bind_dynamic(1, MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(1, B_DYN)
    inner_compress_state.bind_dynamic(1, INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(1, B_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(1, CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(1, B_DYN)
    idx_kv_cache.bind_dynamic(1, IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(1, IDX_CACHE_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(1, B_DYN)
    ori_slot_mapping.bind_dynamic(1, T_DYN)
    window_swa_indices.bind_dynamic(1, T_DYN)
    window_swa_lens.bind_dynamic(1, T_DYN)
    cmp_slot_mapping.bind_dynamic(1, T_DYN)
    idx_slot_mapping.bind_dynamic(1, T_DYN)
    state_slot_mapping.bind_dynamic(1, T_DYN)
    inner_state_slot_mapping.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    kv_seq_lens.bind_dynamic(1, B_DYN)
    x_out.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        decode_csa_test(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            cmp_freqs_cos[rank], cmp_freqs_sin[rank],
            cmp_wkv[rank], cmp_wgate[rank], cmp_ape[rank], cmp_norm_w[rank],
            compress_state[rank], compress_state_block_table[rank],
            idx_wq_b[rank], idx_wq_b_scale[rank], weights_proj[rank], hadamard_idx[rank],
            inner_wkv[rank], inner_wgate[rank], inner_ape[rank], inner_norm_w[rank],
            inner_compress_state[rank], inner_compress_state_block_table[rank],
            kv_cache[rank], cmp_kv[rank], cmp_block_table[rank],
            idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank],
            ori_slot_mapping[rank], window_swa_indices[rank], window_swa_lens[rank],
            cmp_slot_mapping[rank], idx_slot_mapping[rank],
            state_slot_mapping[rank], inner_state_slot_mapping[rank],
            position_ids[rank], kv_seq_lens[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            x_out[rank],
            attention_window, attention_signal, o_window, o_signal,
            0, rank, local_t,
            device=rank,
        )
    return x_out


@pl.jit.inline(auto_scope=False)
def decode_csa_tp1(
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[
        [MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[B_DYN, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32],
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
    wb_blocks = t_dim // CSA_WB_TOKEN_TILE
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post_t = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    with pl.scope():
        hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    idx_cos_il = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    idx_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_cos_il = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_interleave") as rope_tid:
        il_ones = pl.full([4, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_col = pl.col_expand_mul(
            il_ones,
            pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32),
        )
        il_dup_f = pl.cast(
            pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        il_dup_idx = pl.cast(il_dup_f, target_type=pl.INT32)
        il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
        il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
        for rope_t0 in pl.range(0, t_dim, 4):
            idx_cos_il[rope_t0 : rope_t0 + 4, :] = pl.gather(
                pl.cast(freqs_cos[rope_t0 : rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32),
                dim=-1,
                index=il_dup_idx,
            )
            idx_sin_signed[rope_t0 : rope_t0 + 4, :] = pl.mul(
                pl.gather(
                    pl.cast(freqs_sin[rope_t0 : rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32),
                    dim=-1,
                    index=il_dup_idx,
                ),
                il_sign,
            )
            cmp_cos_il[rope_t0 : rope_t0 + 4, :] = pl.gather(
                pl.cast(cmp_freqs_cos[rope_t0 : rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32),
                dim=-1,
                index=il_dup_idx,
            )
            cmp_sin_signed[rope_t0 : rope_t0 + 4, :] = pl.mul(
                pl.gather(
                    pl.cast(cmp_freqs_sin[rope_t0 : rope_t0 + 4, 0:HALF_ROPE], target_type=pl.FP32),
                    dim=-1,
                    index=il_dup_idx,
                ),
                il_sign,
            )

    x_normed_t = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.scope():
        rms_norm(x_mixed, attn_norm_w, x_normed_t)
    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    idx_topk_scores = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.FP32)
    idx_topk = pl.create_tensor([t_dim, IDX_TOPK], dtype=pl.INT32)
    with pl.scope():
        # Dispatch barrier: kv_proj_matmul, kv_score_proj and weights_proj resolve one hop
        # after rms_norm, leaving qr_proj_matmul first.
        late_dep = pl.system.task_dummy(deps=[rope_tid])
        qkv_proj_rope(
            x_normed_t, wq_a, wq_b, wq_b_scale, wkv,
            freqs_cos, freqs_sin, gamma_cq, gamma_ckv,
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
                    kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[
                        write_t : write_t + 1, 0 : HEAD_DIM
                    ]

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
        indexer(
            x_normed_t, qr, qr_scale, idx_wq_b, idx_wq_b_scale,
            weights_proj, idx_cos_il, idx_sin_signed, cmp_cos_il, cmp_sin_signed,
            hadamard_idx,
            idx_kv_unused, inner_compress_state, inner_compress_state_block_table,
            inner_wkv, inner_wgate, inner_ape, inner_norm_w,
            idx_kv_cache, idx_kv_scale, idx_block_table,
            idx_topk_scores, idx_topk,
            position_ids, idx_slot_mapping, inner_state_slot_mapping,
            kv_seq_lens, late_dep,
        )

    # sparse_attn_csa folds the compressed-slot masking + valid-block flags in from the
    # raw indexer topk + position.
    position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)
    o_packed_heads, heads_dep = sparse_attn_csa(
        q, kv_cache, window_swa_indices,
        cmp_kv, cmp_block_table, idx_topk, position_ids_t1,
        attn_sink, freqs_cos, freqs_sin,
        o_packed_heads,
    )
    attn_out = decode_o_proj_tp1(
        o_packed_heads,
        wo_a, wo_b, wo_b_scale,
        attn_out, heads_dep,
    )

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit
def decode_csa_tp1_test(
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[B_DYN, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[
        pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32]
    ],
    inner_compress_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[B_DYN, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[
        pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]
    ],
    idx_block_table: pl.Tensor[[B_DYN, IDX_MAX_BLOCKS], pl.INT32],
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
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, MAIN_STATE_BLOCK_NUM_DYN)
    inner_compress_state.bind_dynamic(0, INNER_STATE_BLOCK_NUM_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    idx_kv_cache.bind_dynamic(0, IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(0, IDX_CACHE_BLOCK_NUM_DYN)
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

    decode_csa_tp1(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin,
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


# fixture
if LOCAL_T < 2 * ROPE_CS_T_TILE:
    raise ValueError("CSA fixture token capacity must leave one RoPE row tile for subcapacity")
if LOCAL_T % ROPE_CS_T_TILE != 0:
    raise ValueError("CSA fixture token capacity must align to the RoPE row tile")
if (LOCAL_T - ROPE_CS_T_TILE) % S != 0:
    raise ValueError("CSA fixture subcapacity must preserve whole decode requests")


def golden_decode_csa_tp1(tensors):
    """Torch reference for the ratio-4 compression-step CSA orchestration."""
    import torch

    from decode_compressor_ratio4 import golden_compressor
    from decode_o_proj import golden_decode_o_proj_tp1
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

    rope_cos_t = tensors["freqs_cos"]
    rope_sin_t = tensors["freqs_sin"]

    def interleave_rope(cos, sin):
        cos_il = cos[:, :HALF_ROPE].float().repeat_interleave(2, dim=-1)
        sin_il = sin[:, :HALF_ROPE].float().repeat_interleave(2, dim=-1)
        sign = torch.ones(ROPE_HEAD_DIM, dtype=torch.float32)
        sign[0::2] = -1.0
        return cos_il.contiguous(), (sin_il * sign).contiguous()

    idx_cos, idx_sin = interleave_rope(rope_cos_t, rope_sin_t)
    cmp_cos, cmp_sin = interleave_rope(
        tensors["cmp_freqs_cos"], tensors["cmp_freqs_sin"]
    )

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
    idx_topk_scores = torch.full((tokens, IDX_TOPK), float("-inf"), dtype=torch.float32)
    idx_topk = torch.full((tokens, IDX_TOPK), -1, dtype=torch.int32)
    golden_indexer({
        "x": x_normed,
        "qr": qr_i8,
        "qr_scale": qr_scale,
        "wq_b": tensors["idx_wq_b"],
        "wq_b_scale": tensors["idx_wq_b_scale"],
        "weights_proj": tensors["weights_proj"],
        "cos": idx_cos,
        "sin": idx_sin,
        "cmp_cos": cmp_cos,
        "cmp_sin": cmp_sin,
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
        "topk_scores": idx_topk_scores,
        "topk_idxs": idx_topk,
        "position_ids": position_ids_bsd.reshape(-1),
        "idx_slot_mapping": idx_slot_mapping_bsd.reshape(-1),
        "inner_state_slot_mapping": inner_state_slot_mapping_bsd.reshape(-1),
        "kv_seq_lens": tensors["kv_seq_lens"],
    })

    ori_slot_mapping = tensors["ori_slot_mapping"].to(torch.int64)
    for t in range(tokens):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            blk_id = write_row // BLOCK_SIZE
            intra = write_row % BLOCK_SIZE
            kv_cache[blk_id, intra, 0] = kv[t]

    o_packed_heads = torch.zeros(O_GROUPS, T_PAD, O_GROUP_IN, dtype=torch.bfloat16)
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "window_swa_indices": window_swa_indices,
        "cmp_kv": cmp_kv,
        "cmp_block_table": cmp_block_table,
        "idx_topk": idx_topk,
        "position_ids": position_ids.view(tokens, 1),
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "o_packed_heads": o_packed_heads,
    })
    attn_out = golden_decode_o_proj_tp1(
        o_packed_heads,
        tensors["wo_a"],
        tensors["wo_b"],
        tensors["wo_b_scale"],
        tokens,
    )

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
        swa_indices_and_lens,
        token_local_rope,
    )
    from golden import TensorSpec
    from hc_pre import golden_hc_pre

    starts = resolve_start_positions(
        start_pos,
        batch=batch,
        seq=S,
        max_seq_len=MAX_SEQ_LEN,
        default_fn=lambda: csa_decode_start_set(
            batch=batch,
            seq=S,
            compress_ratio=COMPRESS_RATIO,
            state_block_size=INNER_STATE_BLOCK_SIZE,
            window=WIN,
        ),
    )
    positions = position_ids_from_starts(starts, seq=S)
    kv_seq_lens = kv_seq_lens_from_starts(starts, seq=S)

    shared_freqs_cos, shared_freqs_sin = token_local_rope(
        M,
        COMPRESS_RATIO,
        positions.reshape(-1),
        max_seq_len=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
    )
    cmp_rope_positions = torch.where(
        (positions.to(torch.int64) + 1) % COMPRESS_RATIO == 0,
        positions.to(torch.int64) - (COMPRESS_RATIO - 1),
        torch.zeros_like(positions, dtype=torch.int64),
    )
    shared_cmp_freqs_cos, shared_cmp_freqs_sin = token_local_rope(
        M,
        COMPRESS_RATIO,
        cmp_rope_positions.reshape(-1),
        max_seq_len=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
    )

    main_state_block_num = batch * MAIN_STATE_MAX_BLOCKS
    inner_state_block_num = batch * INNER_STATE_MAX_BLOCKS
    main_state_block_table = block_table(
        batch=batch,
        table_blocks=MAIN_STATE_MAX_BLOCKS,
        physical_blocks=main_state_block_num,
    )
    inner_state_block_table = block_table(
        batch=batch,
        table_blocks=INNER_STATE_MAX_BLOCKS,
        physical_blocks=inner_state_block_num,
    )

    def ring_slots(table, state_len, state_block_size):
        ring_rows = positions.to(torch.int64) % state_len
        pages = torch.gather(
            table.to(torch.int64), 1, ring_rows // state_block_size
        )
        return pages * state_block_size + ring_rows % state_block_size

    main_state_slots = ring_slots(
        main_state_block_table, MAIN_STATE_LEN, MAIN_STATE_BLOCK_SIZE
    )
    inner_state_slots = ring_slots(
        inner_state_block_table, INNER_STATE_LEN, INNER_STATE_BLOCK_SIZE
    )

    max_visible_rows = int((kv_seq_lens.to(torch.int64) // COMPRESS_RATIO).max())
    max_active_pages = max(1, (max_visible_rows + BLOCK_SIZE - 1) // BLOCK_SIZE)
    cmp_block_num = batch * max_active_pages
    idx_cache_block_num = batch * max_active_pages
    window_block_table = block_table(
        batch=batch,
        table_blocks=ORI_MAX_BLOCKS,
        physical_blocks=ORI_BLOCK_NUM,
    )
    cmp_block_table = block_table(
        batch=batch,
        table_blocks=CMP_MAX_BLOCKS,
        physical_blocks=cmp_block_num,
    )
    idx_block_table = block_table(
        batch=batch,
        table_blocks=IDX_MAX_BLOCKS,
        physical_blocks=idx_cache_block_num,
    )
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

    # Real layer-8 (CSA, ratio-4) hc_attn scale/base; fn is synthetic at the real magnitude.
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
        return torch.randn(
            main_state_block_num,
            MAIN_STATE_BLOCK_SIZE,
            MAIN_STATE_DIM,
        ) * 0.05

    def init_compress_state_block_table():
        return main_state_block_table.clone()

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
        return torch.randn(
            inner_state_block_num,
            INNER_STATE_BLOCK_SIZE,
            INNER_STATE_DIM,
        ) * 0.05

    def init_inner_compress_state_block_table():
        return inner_state_block_table.clone()

    def init_kv_cache():
        return init_normalized_cache((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_window_block_table():
        return window_block_table.clone()

    def init_cmp_kv():
        return init_normalized_cache((cmp_block_num, BLOCK_SIZE, 1, HEAD_DIM))

    def init_cmp_block_table():
        return cmp_block_table.clone()

    def init_idx_kv_cache():
        return init_normalized_cache(
            (idx_cache_block_num, BLOCK_SIZE, 1, IDX_HEAD_DIM)
        )

    def init_idx_block_table():
        return idx_block_table.clone()

    def init_attn_sink():
        return torch.ones(H) * 4.0

    def init_position_ids():
        return positions.reshape(-1).contiguous()

    def init_kv_seq_lens():
        return kv_seq_lens.clone()

    def init_ori_slot_mapping():
        return ori_slot_mapping(
            positions,
            init_window_block_table(),
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()

    def init_window_swa_metadata():
        return swa_indices_and_lens(
            positions,
            init_window_block_table(),
            block_size=BLOCK_SIZE,
            window=WIN,
        )

    def init_window_swa_indices():
        return init_window_swa_metadata()[0].contiguous()

    def init_window_swa_lens():
        return init_window_swa_metadata()[1].contiguous()

    def init_cmp_slot_mapping():
        return compressed_slot_mapping(
            positions,
            init_cmp_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()

    def init_idx_slot_mapping():
        return compressed_slot_mapping(
            positions,
            init_idx_block_table(),
            compress_ratio=COMPRESS_RATIO,
            block_size=BLOCK_SIZE,
        ).reshape(-1).contiguous()

    def init_state_slot_mapping():
        return main_state_slots.reshape(-1).contiguous()

    def init_inner_state_slot_mapping():
        return inner_state_slots.reshape(-1).contiguous()

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
        shared_idx_kv_cache.float().reshape(
            idx_cache_block_num * BLOCK_SIZE, IDX_HEAD_DIM
        )
    )
    shared_idx_kv_cache_i8 = _idx_kv_i8.view(
        idx_cache_block_num, BLOCK_SIZE, 1, IDX_HEAD_DIM
    )
    shared_idx_kv_scale = _idx_kv_sc.view(
        idx_cache_block_num, BLOCK_SIZE, 1, 1
    )

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
        TensorSpec("freqs_cos", [tokens, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_freqs_cos.clone()),
        TensorSpec("freqs_sin", [tokens, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_freqs_sin.clone()),
        TensorSpec("cmp_freqs_cos", [tokens, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_cmp_freqs_cos.clone()),
        TensorSpec("cmp_freqs_sin", [tokens, ROPE_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_cmp_freqs_sin.clone()),
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec(
            "compress_state", [main_state_block_num, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
            torch.float32, init_value=init_compress_state, is_output=True,
        ),
        TensorSpec("compress_state_block_table", [batch, MAIN_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec("idx_wq_b", [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], torch.int8, init_value=lambda: idx_wq_b_i8),
        TensorSpec("idx_wq_b_scale", [IDX_N_HEADS * IDX_HEAD_DIM], torch.float32, init_value=lambda: idx_wq_b_scale),
        TensorSpec("weights_proj", [D, IDX_N_HEADS], torch.bfloat16, init_value=lambda: shared_weights_proj.clone()),
        TensorSpec("hadamard_idx", [IDX_HEAD_DIM, IDX_HEAD_DIM], torch.bfloat16, init_value=lambda: shared_hadamard_idx.clone()),
        TensorSpec("inner_wkv", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wkv),
        TensorSpec("inner_wgate", [INNER_OUT_DIM, D], torch.bfloat16, init_value=init_inner_wgate),
        TensorSpec("inner_ape", [COMPRESS_RATIO, INNER_OUT_DIM], torch.float32, init_value=init_inner_ape),
        TensorSpec("inner_norm_w", [IDX_HEAD_DIM], torch.bfloat16, init_value=init_inner_norm_w),
        TensorSpec(
            "inner_compress_state", [inner_state_block_num, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            torch.float32, init_value=init_inner_compress_state, is_output=True,
        ),
        TensorSpec("inner_compress_state_block_table", [batch, INNER_STATE_MAX_BLOCKS], torch.int32, init_value=init_inner_compress_state_block_table),
        TensorSpec("kv_cache", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec(
            "cmp_kv", [cmp_block_num, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16, init_value=init_cmp_kv, is_output=True,
        ),
        TensorSpec("cmp_block_table", [batch, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec(
            "idx_kv_cache", [idx_cache_block_num, BLOCK_SIZE, 1, IDX_HEAD_DIM],
            torch.int8, init_value=lambda: shared_idx_kv_cache_i8.clone(), is_output=True,
        ),
        TensorSpec(
            "idx_kv_scale", [idx_cache_block_num, BLOCK_SIZE, 1, 1],
            torch.float32, init_value=lambda: shared_idx_kv_scale.clone(), is_output=True,
        ),
        TensorSpec("idx_block_table", [batch, IDX_MAX_BLOCKS], torch.int32, init_value=init_idx_block_table),
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


def build_distributed_tensor_specs(local_t, start_pos=None):
    """Build one logical CSA layer fixture split over the TP token ranks."""
    import torch

    from golden import ScalarSpec, TensorSpec

    if (local_t < ROPE_CS_T_TILE or local_t > LOCAL_T
            or local_t % ROPE_CS_T_TILE != 0 or local_t % S != 0):
        raise ValueError(
            f"local_t must be a multiple of {ROPE_CS_T_TILE} "
            f"in [{ROPE_CS_T_TILE}, {LOCAL_T}], got {local_t}"
        )

    local_batch = local_t // S
    base_specs = build_tensor_specs(start_pos=start_pos, batch=local_batch)
    base_values = {
        spec.name: spec.create_tensor()
        for spec in base_specs
        if spec.name != "x_out"
    }

    def replicate(value):
        repeats = (TP_SIZE,) + (1,) * value.ndim
        return value.unsqueeze(0).repeat(repeats).contiguous()

    values = {name: replicate(value) for name, value in base_values.items()}

    values["wo_a"] = base_values["wo_a"].reshape(
        TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN,
    ).contiguous()
    values["wo_b"] = base_values["wo_b"].reshape(
        D, TP_SIZE, LOCAL_O_WIDTH,
    ).permute(1, 0, 2).contiguous()
    values["wo_b_scale"] = replicate(base_values["wo_b_scale"])

    resident_names = frozenset({
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv",
        "gamma_cq", "gamma_ckv", "freqs_cos", "freqs_sin",
        "cmp_freqs_cos", "cmp_freqs_sin",
        "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
        "idx_wq_b", "idx_wq_b_scale", "weights_proj", "hadamard_idx",
        "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
        "compress_state", "compress_state_block_table",
        "inner_compress_state", "inner_compress_state_block_table",
        "kv_cache", "cmp_kv", "cmp_block_table",
        "idx_kv_cache", "idx_kv_scale", "idx_block_table",
        "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    })
    specs = []
    for spec in base_specs:
        if spec.name == "x_out":
            specs.append(TensorSpec(
                "x_out", [TP_SIZE, local_t, HC_MULT, D],
                torch.float32, is_output=True,
            ))
            continue
        value = values[spec.name]
        distributed_spec = TensorSpec(
            spec.name,
            list(value.shape),
            spec.dtype,
            init_value=lambda value=value: value.clone(),
            is_output=spec.is_output,
        )
        if spec.name in resident_names:
            distributed_spec.resident = "stacked"
        specs.append(distributed_spec)
    specs.append(ScalarSpec("local_t", torch.int32, local_t))
    return specs


def golden_decode_csa(tensors):
    """Run the complete TP1 reference independently for each token rank."""
    full_wo_a = tensors["wo_a"].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    full_wo_b = tensors["wo_b"].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)
    full_wo_b_scale = tensors["wo_b_scale"][0]

    for rank in range(TP_SIZE):
        rank_tensors = {
            name: value[rank]
            for name, value in tensors.items()
            if name != "local_t"
        }
        rank_tensors["wo_a"] = full_wo_a
        rank_tensors["wo_b"] = full_wo_b
        rank_tensors["wo_b_scale"] = full_wo_b_scale
        golden_decode_csa_tp1(rank_tensors)


def build_full_compare(mapping_shape, *, leading_rank_axis, diagnostic_x_out=False):
    """Compare the six mutable pools only at allocator-mapped rows."""
    from golden import error_distribution, mapped_pool_ratio_allclose, ratio_reldiff

    common = {
        "mapping_shape": mapping_shape,
        "leading_rank_axis": leading_rank_axis,
    }
    # Simulator x_out is diagnostic until CSA sparse-attention numerics match hardware.
    x_out_compare = (
        error_distribution(always_pass=False)
        if diagnostic_x_out
        else ratio_reldiff(diff_thd=4e-3, pct_thd=0.008, max_diff_hd=1)
    )
    return {
        "compress_state": mapped_pool_ratio_allclose(
            "state_slot_mapping",
            block_size=MAIN_STATE_BLOCK_SIZE,
            pool_name="main compressor state",
            atol=1e-3,
            rtol=1e-3,
            **common,
        ),
        "inner_compress_state": mapped_pool_ratio_allclose(
            "inner_state_slot_mapping",
            block_size=INNER_STATE_BLOCK_SIZE,
            pool_name="inner compressor state",
            atol=1e-3,
            rtol=1e-3,
            **common,
        ),
        "kv_cache": mapped_pool_ratio_allclose(
            "ori_slot_mapping",
            block_size=BLOCK_SIZE,
            pool_name="original KV cache",
            atol=1e-4,
            rtol=1.0 / 128,
            **common,
        ),
        "cmp_kv": mapped_pool_ratio_allclose(
            "cmp_slot_mapping",
            block_size=BLOCK_SIZE,
            pool_name="compressed KV cache",
            atol=1e-4,
            rtol=1.0 / 128,
            **common,
        ),
        "idx_kv_cache": mapped_pool_ratio_allclose(
            "idx_slot_mapping",
            block_size=BLOCK_SIZE,
            pool_name="indexer KV cache",
            atol=1,
            rtol=0,
            max_error_ratio=0.01,
            **common,
        ),
        "idx_kv_scale": mapped_pool_ratio_allclose(
            "idx_slot_mapping",
            block_size=BLOCK_SIZE,
            pool_name="indexer KV scale",
            atol=1e-4,
            rtol=1.0 / 128,
            max_error_ratio=0.01,
            **common,
        ),
        "x_out": x_out_compare,
    }


if __name__ == "__main__":
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES), help="tensor-parallel world size")
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help=f"comma-separated device ids; --tp {TP_SIZE} needs {TP_SIZE}",
    )
    parser.add_argument(
        "--start-pos", type=str, default=None,
        help="absolute decode start position; a scalar sets batch=1, "
             "and a comma-separated list sets batch to its length",
    )
    parser.add_argument("--enable-chip-swimlane", type=int, choices=(0, 1, 2, 3, 4), default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.tp != TP_SIZE:
        parser.error(f"--tp must remain {TP_SIZE} after import-time specialization")
    if args.device is None:
        args.device = ",".join(str(rank) for rank in range(TP_SIZE))
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(f"--device must be a comma-separated integer list, got {args.device!r}")
    if any(device < 0 for device in device_ids):
        parser.error(f"--device IDs must be non-negative, got {device_ids}")
    if len(set(device_ids)) != len(device_ids):
        parser.error(f"--device IDs must be distinct, got {device_ids}")

    if len(device_ids) != TP_SIZE:
        parser.error(f"--tp {TP_SIZE} needs exactly {TP_SIZE} device(s), got {device_ids}")
    start_pos = None
    batch = B
    if args.start_pos is not None:
        try:
            start_values = [
                int(value.strip())
                for value in args.start_pos.split(",")
                if value.strip() != ""
            ]
        except ValueError:
            parser.error(f"--start-pos must contain integers, got {args.start_pos!r}")
        if not start_values:
            parser.error("--start-pos must contain at least one integer")
        if len(start_values) > B:
            parser.error(f"--start-pos accepts at most {B} values, got {len(start_values)}")
        batch = len(start_values)
        start_pos = start_values[0] if len(start_values) == 1 else start_values

    local_t = batch * S

    if TP_SIZE == 1:
        result = run_jit(
            fn=decode_csa_tp1_test,
            specs=build_tensor_specs(start_pos=start_pos, batch=batch),
            golden_fn=golden_decode_csa_tp1,
            compile_only=args.compile_only,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=device_ids[0],
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            rtol=1e-2,
            atol=1e-2,
            compare_fn=build_full_compare(
                (local_t,),
                leading_rank_axis=False,
                diagnostic_x_out=args.platform.endswith("sim"),
            ),
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
        raise SystemExit(0)

    compile_cfg = dict(
        dump_passes=args.dump_passes,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
    )
    result = run_jit(
        fn=l3_decode_csa,
        specs=build_distributed_tensor_specs(local_t, start_pos=start_pos),
        golden_fn=golden_decode_csa,
        compile_only=args.compile_only,
        compile_cfg=compile_cfg,
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn=build_full_compare(
            (TP_SIZE, local_t),
            leading_rank_axis=True,
            diagnostic_x_out=args.platform.endswith("sim"),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
