# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI marker: run on >=2 NPUs via $DEVICE_RANGE instead of single $DEVICE_ID
"""DeepSeek-V4 decode layer smoke: SWA attention DP2 followed by MoE EP2.

Each rank owns a local decode micro-batch for the attention stage. The resulting
per-rank hidden states feed the existing two-rank EP MoE path.
"""

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from decode_attention_swa import (
    B,
    BLOCK_SIZE,
    D,
    HC_DIM,
    HC_MULT,
    H,
    HEAD_DIM,
    MAX_SEQ_LEN,
    MIX_HC,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    ORI_MAX_BLOCKS,
    Q_LORA,
    ROPE_HEAD_DIM,
    S,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_IDX_TOPK,
    SPARSE_TOPK,
    T,
    WIN,
    build_tensor_specs as build_attention_tensor_specs,
    golden_attention_swa,
)
from decode_qkv_proj_rope import qkv_proj_rope
from decode_rmsnorm import attn_norm
from decode_sparse_attn import sparse_attn
from hc_pre import (
    COMB_T_TILE,
    D_TILE,
    HC_DIM_INV,
    HC_EPS,
    HC_PAD,
    HC_SINKHORN_ITER,
    LINEAR_K_TILE,
    LINEAR_T_TILE,
    MIX_PAD,
    NORM_EPS,
    T_TILE,
)
from moe_ep import (
    IDX_PAD,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    TOPK,
    VOCAB,
    W_PAD,
    build_tensor_specs as build_moe_tensor_specs,
    golden_moe_ep,
    moe_ep,
)

assert N_RANKS == 2, "decode layer smoke is wired for attention DP2 + MoE EP2"

CACHE_COPY_TILE = 16


@pl.jit.inline
def attn_hc_pre(
    x: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    post: pl.Tensor[[T, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32],
):
    x_flat = pl.reshape(x, [T, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    mixes = pl.create_tensor([T, MIX_PAD], dtype=pl.FP32)
    for ob in pl.spmd(T // LINEAR_T_TILE, name_hint="attn_hc_linear"):
        t0 = ob * LINEAR_T_TILE
        sq_sum = pl.full([1, LINEAR_T_TILE], dtype=pl.FP32, value=0.0)

        mix_acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
            kl0 = kb * LINEAR_K_TILE
            x_lin = pl.cast(
                x_flat[t0:t0 + LINEAR_T_TILE, kl0:kl0 + LINEAR_K_TILE],
                target_type=pl.FP32,
            )
            x_sq = pl.mul(x_lin, x_lin)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(x_sq), [1, LINEAR_T_TILE]))
            w_lin = pl.slice(
                hc_fn,
                [MIX_PAD, LINEAR_K_TILE],
                [0, kl0],
                valid_shape=[MIX_HC, LINEAR_K_TILE],
            )
            if kb == 0:
                mix_acc = pl.matmul(x_lin, w_lin, b_trans=True, out_dtype=pl.FP32)
            else:
                mix_acc = pl.matmul_acc(mix_acc, x_lin, w_lin, b_trans=True)

        mean_sq = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_rms_val = pl.rsqrt(mean_sq, high_precision=True)
        inv_rms_col = pl.reshape(inv_rms_val, [LINEAR_T_TILE, 1])
        mixes[t0:t0 + LINEAR_T_TILE, 0:MIX_PAD] = pl.row_expand_mul(mix_acc, inv_rms_col)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_hc_split_pre_post"):
        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_scaled = pl.mul(mixes[0:T, 0:HC_PAD], scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val_store = pl.add(pre_sig, HC_EPS)

        post_base = pl.reshape(hc_base[HC_MULT:HC_MULT + HC_PAD], [1, HC_PAD])
        post_scaled = pl.mul(mixes[0:T, HC_MULT:HC_MULT + HC_PAD], scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_pad = pl.mul(post_sig, 2.0)

        comb_base = pl.reshape(
            hc_base[HC_MULT * 2:HC_MULT * 2 + HC_MULT * HC_MULT],
            [1, HC_MULT * HC_MULT],
        )
        comb_scaled = pl.mul(
            mixes[0:T, HC_MULT * 2:HC_MULT * 2 + HC_MULT * HC_MULT],
            scale2,
        )
        comb_logits = pl.add(comb_scaled, pl.col_expand(comb_scaled, comb_base))

    for ob in pl.spmd(T // COMB_T_TILE, name_hint="attn_hc_write_post"):
        t0 = ob * COMB_T_TILE
        post_tile = pl.load(
            post_pad,
            [t0, 0],
            [COMB_T_TILE, HC_PAD],
            valid_shapes=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        pl.store(post_tile, [t0, 0], post)

    for ob in pl.spmd(T // COMB_T_TILE, name_hint="attn_hc_comb_sinkhorn"):
        t0 = ob * COMB_T_TILE
        row0 = pl.load(
            comb_logits,
            [t0, 0 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shapes=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        row1 = pl.load(
            comb_logits,
            [t0, 1 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shapes=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        row2 = pl.load(
            comb_logits,
            [t0, 2 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shapes=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        row3 = pl.load(
            comb_logits,
            [t0, 3 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shapes=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

        row_max_tmp = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row0_max = pl.row_max(row0_p, row_max_tmp)
        row1_max = pl.row_max(row1_p, row_max_tmp)
        row2_max = pl.row_max(row2_p, row_max_tmp)
        row3_max = pl.row_max(row3_p, row_max_tmp)
        row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
        row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
        row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
        row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
        row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
        row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
        row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
        row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
        row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
        row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
        row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
        row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)

        row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
        row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
        row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
        row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
        row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
        row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
        row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
        row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

        row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
        col_sum = pl.add(col_sum, HC_EPS)
        row0_cur = pl.div(row0_eff, col_sum)
        row1_cur = pl.div(row1_eff, col_sum)
        row2_cur = pl.div(row2_eff, col_sum)
        row3_cur = pl.div(row3_eff, col_sum)

        for sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
            row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
            row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
            row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
            row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
            row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
            row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
            row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
            col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_norm, col_sum)
            row1_cur = pl.div(row1_norm, col_sum)
            row2_cur = pl.div(row2_norm, col_sum)
            row3_cur = pl.div(row3_norm, col_sum)

        row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
        pl.store(row0_out, [t0, 0 * HC_MULT], comb)
        pl.store(row1_out, [t0, 1 * HC_MULT], comb)
        pl.store(row2_out, [t0, 2 * HC_MULT], comb)
        pl.store(row3_out, [t0, 3 * HC_MULT], comb)

    for ob in pl.spmd(T // T_TILE, name_hint="attn_hc_mix_x"):
        t0 = ob * T_TILE
        pre_tile = pre_val_store[t0:t0 + T_TILE, 0:HC_PAD]
        pre_tile_t = pl.transpose(pre_tile, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.range(D // D_TILE):
            d0 = db * D_TILE
            x0 = pl.cast(x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_TILE], target_type=pl.FP32)
            x1 = pl.cast(x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_TILE], target_type=pl.FP32)
            x2 = pl.cast(x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_TILE], target_type=pl.FP32)
            x3 = pl.cast(x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_TILE], target_type=pl.FP32)
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            x_mixed[t0:t0 + T_TILE, d0:d0 + D_TILE] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
    return x_mixed


@pl.jit.inline
def attn_hc_post(
    x: pl.Tensor[[T, D], pl.BF16],
    residual: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
):
    residual_flat = pl.reshape(residual, [T, HC_DIM])
    y_flat = pl.reshape(y, [T, HC_DIM])

    for tb in pl.spmd(T // 2, name_hint="attn_hc_post"):
        for tt in pl.range(2):
            t = tb * 2 + tt
            x_row = pl.cast(x[t:t + 1, 0:D], target_type=pl.FP32)
            for out_h in pl.range(HC_MULT):
                post_w = pl.read(post, [t, out_h])
                y_row = pl.mul(x_row, post_w)
                for in_h in pl.range(HC_MULT):
                    comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                    res_d = in_h * D
                    residual_row = pl.cast(
                        residual_flat[t:t + 1, res_d:res_d + D],
                        target_type=pl.FP32,
                    )
                    y_row = pl.add(y_row, pl.mul(residual_row, comb_w))
                y_d = out_h * D
                y_bf16 = pl.cast(y_row, target_type=pl.BF16, mode="rint")
                y_flat[t:t + 1, y_d:y_d + D] = y_bf16
    y = pl.reshape(y_flat, [T, HC_MULT, D])
    return y


@pl.jit
def attention_swa_dp(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
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
    kv_cache: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    kv_cache_out: pl.Out[pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_comm_dummy: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    start_pos: pl.Tensor[[B], pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    x_mixed = attn_hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post_t,
        comb_t,
    )

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_step"):
        for b in pl.parallel(B):
            start_pos_b = pl.read(start_pos, [b])
            for s_idx in pl.range(S):
                pos_b = pl.cast(start_pos_b + s_idx, pl.INDEX)
                cos_row = pl.cast(
                    pl.slice(freqs_cos, [1, ROPE_HEAD_DIM], [pos_b, 0]),
                    target_type=pl.FP32,
                )
                sin_row = pl.cast(
                    pl.slice(freqs_sin, [1, ROPE_HEAD_DIM], [pos_b, 0]),
                    target_type=pl.FP32,
                )
                t = b * S + s_idx
                rope_cos_t = pl.assemble(rope_cos_t, pl.cast(cos_row, target_type=pl.BF16, mode="rint"), [t, 0])
                rope_sin_t = pl.assemble(rope_sin_t, pl.cast(sin_row, target_type=pl.BF16, mode="rint"), [t, 0])

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    x_normed_t = pl.create_tensor([T, D], dtype=pl.BF16)
    x_normed_t = attn_norm(x_mixed, attn_norm_w, x_normed_t)
    q = qkv_proj_rope(
        x_normed_t,
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

    kv_cache_in_flat = pl.reshape(kv_cache, [B * ORI_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM])
    kv_cache_out_flat = pl.reshape(kv_cache_out, [B * ORI_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM])
    for copy_block in pl.spmd((B * ORI_MAX_BLOCKS * BLOCK_SIZE) // CACHE_COPY_TILE, name_hint="swa_cache_copy"):
        copy_row0 = copy_block * CACHE_COPY_TILE
        kv_cache_out_flat[
            copy_row0:copy_row0 + CACHE_COPY_TILE,
            0:HEAD_DIM,
        ] = kv_cache_in_flat[copy_row0:copy_row0 + CACHE_COPY_TILE, 0:HEAD_DIM]
    for write_t in pl.spmd(T, name_hint="swa_cache_writeback"):
        write_b = write_t // S
        write_s = write_t - write_b * S
        write_start_b = pl.read(start_pos, [write_b])
        write_slot = (write_start_b + write_s) % WIN
        write_blk = pl.cast(pl.read(block_table, [write_b, write_slot // BLOCK_SIZE]), pl.INDEX)
        write_row = write_blk * BLOCK_SIZE + write_slot % BLOCK_SIZE
        kv_cache_out_flat[write_row:write_row + 1, 0:HEAD_DIM] = kv[write_t:write_t + 1, 0:HEAD_DIM]

    sparse_topk = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    cmp_kv_dummy = pl.create_tensor([B * SPARSE_CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_block_table_dummy = pl.create_tensor([B, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_overlay_topk"):
        for topk_b in pl.range(B):
            pl.write(cmp_block_table_dummy, [topk_b, 0], pl.cast(topk_b * SPARSE_CMP_MAX_BLOCKS, pl.INT32))
            topk_start_b = pl.read(start_pos, [topk_b])
            for topk_s in pl.range(S):
                topk_t = topk_b * S + topk_s
                topk_abs_pos = topk_start_b + topk_s
                if topk_abs_pos >= WIN - 1:
                    topk_win_start = (topk_abs_pos % WIN) + 1
                    for topk_k in pl.range(WIN):
                        topk_val = (topk_win_start + topk_k) % WIN
                        topk_out = topk_val
                        for topk_os in pl.range(S):
                            if topk_os <= topk_s:
                                if topk_val == (topk_start_b + topk_os) % WIN:
                                    topk_out = WIN + topk_os
                        pl.write(sparse_topk, [topk_t, topk_k], pl.cast(topk_out, pl.INT32))
                else:
                    for topk_k in pl.range(WIN):
                        if topk_k <= topk_abs_pos:
                            topk_out = topk_k
                            for topk_os in pl.range(S):
                                if topk_os <= topk_s:
                                    if topk_k == (topk_start_b + topk_os) % WIN:
                                        topk_out = WIN + topk_os
                            pl.write(sparse_topk, [topk_t, topk_k], pl.cast(topk_out, pl.INT32))
                        else:
                            pl.write(sparse_topk, [topk_t, topk_k], pl.cast(-1, pl.INT32))
                for topk_pad in pl.range(SPARSE_IDX_TOPK):
                    pl.write(sparse_topk, [topk_t, WIN + topk_pad], pl.cast(-1, pl.INT32))

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn(
        q,
        kv_cache,
        block_table,
        kv,
        cmp_kv_dummy,
        cmp_block_table_dummy,
        sparse_topk,
        attn_sink,
        seqused_kv,
        rope_cos_t,
        rope_sin_t,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )

    x_out = attn_hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out


@pl.jit.host
def host_orch(
    # attention input / weights, rank-major for DP2
    x_hc: pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[N_RANKS, B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[N_RANKS, B, ORI_MAX_BLOCKS], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    seqused_kv: pl.Tensor[[N_RANKS, B], pl.INT32],
    wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    start_pos: pl.Tensor[[N_RANKS, B], pl.INT32],
    # MoE weights, rank-major for EP2
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, T], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    # composed outputs
    kv_cache_out: pl.Out[
        pl.Tensor[[N_RANKS, B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.BF16]],
    # scalars last: runtime TaskArgs forbids tensor args after scalar args.
    layer_id: pl.Scalar[pl.INT32],
):
    pub_counts_buf = pld.alloc_window_buffer(N_RANKS * N_RANKS * N_LOCAL * 4)
    count_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D * 2)
    recv_scale_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_w_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * W_PAD * 4)
    recv_r_route_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * IDX_PAD * 4)
    routed_y_buf_buf = pld.alloc_window_buffer(N_ROUTES * D * 2)
    combine_done_buf = pld.alloc_window_buffer(N_RANKS * 4)
    x_attn = pl.create_tensor([N_RANKS, T, HC_MULT, D], dtype=pl.BF16)

    for r in pl.range(pld.world_size()):
        attn_comm_dummy = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        attention_swa_dp(
            x_hc[r],
            hc_attn_fn[r],
            hc_attn_scale[r],
            hc_attn_base[r],
            attn_norm_w[r],
            wq_a[r],
            wq_b[r],
            wq_b_scale[r],
            wkv[r],
            gamma_cq[r],
            gamma_ckv[r],
            freqs_cos[r],
            freqs_sin[r],
            kv_cache[r],
            block_table[r],
            attn_sink[r],
            seqused_kv[r],
            wo_a[r],
            wo_b[r],
            wo_b_scale[r],
            x_attn[r],
            kv_cache_out[r],
            attn_comm_dummy,
            start_pos[r],
            device=r,
        )

    for r in pl.range(pld.world_size()):
        pub_counts = pld.window(pub_counts_buf, [N_RANKS * N_RANKS, N_LOCAL], dtype=pl.INT32)
        count_done = pld.window(count_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.FP16)
        recv_scale = pld.window(recv_scale_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_w = pld.window(recv_w_buf, [N_LOCAL * RECV_MAX, W_PAD], dtype=pl.FP32)
        recv_r_route = pld.window(recv_r_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_done = pld.window(combine_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        moe_ep(
            x_attn[r],
            hc_ffn_fn[r],
            hc_ffn_scale[r],
            hc_ffn_base[r],
            norm_w[r],
            gate_w[r],
            gate_bias[r],
            tid2eid[r],
            input_ids[r],
            routed_w1[r],
            routed_w1_scale[r],
            routed_w3[r],
            routed_w3_scale[r],
            routed_w2[r],
            routed_w2_scale[r],
            shared_w1[r],
            shared_w1_scale[r],
            shared_w3[r],
            shared_w3_scale[r],
            shared_w2[r],
            shared_w2_scale[r],
            x_next[r],
            pub_counts,
            count_done,
            recv_x,
            recv_scale,
            recv_w,
            recv_r_route,
            routed_y_buf,
            combine_done,
            layer_id,
            r,
            device=r,
        )


def golden_decode_layer(tensors):
    import torch

    x_attn = torch.empty_like(tensors["x_hc"])
    tensors["kv_cache_out"].copy_(tensors["kv_cache"])
    for r in range(N_RANKS):
        golden_attention_swa({
            "x_hc": tensors["x_hc"][r],
            "hc_attn_fn": tensors["hc_attn_fn"][r],
            "hc_attn_scale": tensors["hc_attn_scale"][r],
            "hc_attn_base": tensors["hc_attn_base"][r],
            "attn_norm_w": tensors["attn_norm_w"][r],
            "wq_a": tensors["wq_a"][r],
            "wq_b": tensors["wq_b"][r],
            "wq_b_scale": tensors["wq_b_scale"][r],
            "wkv": tensors["wkv"][r],
            "gamma_cq": tensors["gamma_cq"][r],
            "gamma_ckv": tensors["gamma_ckv"][r],
            "freqs_cos": tensors["freqs_cos"][r],
            "freqs_sin": tensors["freqs_sin"][r],
            "kv_cache": tensors["kv_cache_out"][r],
            "block_table": tensors["block_table"][r],
            "attn_sink": tensors["attn_sink"][r],
            "seqused_kv": tensors["seqused_kv"][r],
            "wo_a": tensors["wo_a"][r],
            "wo_b": tensors["wo_b"][r],
            "wo_b_scale": tensors["wo_b_scale"][r],
            "x_out": x_attn[r],
            "start_pos": tensors["start_pos"][r],
        })

    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn
    golden_moe_ep(moe_tensors)


def _ranked_init(single_spec, *, replicated=False):
    import torch

    def init():
        if replicated:
            value = single_spec.create_tensor()
            return value.unsqueeze(0).expand(N_RANKS, *value.shape).contiguous()
        return torch.stack([single_spec.create_tensor() for _ in range(N_RANKS)], dim=0)

    return init


def build_tensor_specs(start_pos=None, layer_id=10):
    import torch
    from golden import ScalarSpec, TensorSpec

    attention_specs = {
        spec.name: spec for spec in build_attention_tensor_specs(start_pos) if isinstance(spec, TensorSpec)
    }
    moe_specs = build_moe_tensor_specs(layer_id)
    moe_tensor_specs = {spec.name: spec for spec in moe_specs if isinstance(spec, TensorSpec)}

    replicated_attention = {
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm_w",
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
        "freqs_cos",
        "freqs_sin",
        "attn_sink",
        "wo_a",
        "wo_b",
        "wo_b_scale",
    }
    attention_order = [
        "x_hc",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm_w",
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
        "freqs_cos",
        "freqs_sin",
        "kv_cache",
        "block_table",
        "attn_sink",
        "seqused_kv",
        "wo_a",
        "wo_b",
        "wo_b_scale",
        "start_pos",
    ]

    specs = []
    for name in attention_order:
        spec = attention_specs[name]
        specs.append(
            TensorSpec(
                name,
                [N_RANKS, *spec.shape],
                spec.dtype,
                init_value=_ranked_init(spec, replicated=name in replicated_attention),
            )
        )

    for spec in moe_specs:
        if not isinstance(spec, TensorSpec):
            continue
        if spec.name in {"x_hc", "x_next"}:
            continue
        if spec.name == "tid2eid":
            def init_tid2eid():
                base = torch.arange(VOCAB, dtype=torch.int32).reshape(VOCAB, 1) * TOPK
                offs = torch.arange(TOPK, dtype=torch.int32).reshape(1, TOPK)
                table = (base + offs) % N_EXPERTS_GLOBAL
                return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

            specs.append(TensorSpec("tid2eid", spec.shape, spec.dtype, init_value=init_tid2eid))
        elif spec.name == "input_ids":
            def init_input_ids():
                ids = torch.arange(T, dtype=torch.int64)
                return ids.unsqueeze(0).expand(N_RANKS, -1).contiguous()

            specs.append(TensorSpec("input_ids", spec.shape, spec.dtype, init_value=init_input_ids))
        else:
            specs.append(moe_tensor_specs[spec.name])

    specs.extend([
        TensorSpec(
            "kv_cache_out",
            [N_RANKS, B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            is_output=True,
        ),
        TensorSpec("x_next", [N_RANKS, T, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("layer_id", torch.int32, layer_id),
    ])
    return specs


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0,1",
                        help="comma-separated device ids; need at least 2")
    parser.add_argument("--start-pos", type=int, default=None,
                        help="If set, use this single start_pos for all batches.")
    parser.add_argument("--layer-id", type=int, default=10)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=host_orch,
        specs=build_tensor_specs(start_pos=args.start_pos, layer_id=args.layer_id),
        golden_fn=golden_decode_layer,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "kv_cache_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            # The composed attention->MoE smoke feeds quantized MoE with the
            # real attention output distribution; keep KV strict and use a
            # wider FFN envelope for this end-to-end bring-up check.
            "x_next": ratio_reldiff(diff_thd=0.1, pct_thd=0.05),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
