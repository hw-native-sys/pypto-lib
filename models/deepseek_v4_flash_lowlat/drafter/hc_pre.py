# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Drafter HC pre, layer tail, and head mix."""

from math import gcd
import pypto.language as pl
from .config import FLASH as M
from .config import DECODE_BATCH, DECODE_SEQ, TP, PREFILL_BATCH, PREFILL_SEQ


T_DYN = pl.dynamic("NATIVE_D_T_DYN")  # T = B * S

D = M.hidden_size

HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

HC_DIM = M.hc_dim

HC_DIM_INV = 1.0 / HC_DIM

HC_SINKHORN_ITER = M.hc_sinkhorn_iters

HC_EPS = M.hc_eps

NORM_EPS = M.rms_norm_eps

MIX_PAD = 32  # mix_hc (24) padded to a 32-wide cube N / vector row

PRE_HC_PAD = 8  # hc (4) padded for 32B-aligned vector ops

PRE_T_TILE = 8  # other values miscompare

LINEAR_T_TILE = 16  # cube matmul rows must be a 16-row boxed tile

COMB_T_TILE = 8

PRE_POST_WORKERS = 16

PRE_RMS_K_TILE = 512

LINEAR_K_TILE = 256

D_TILE = 256

PRE_D_SPMD = gcd(4096, D)

LINEAR_OK = 4

LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK

LINEAR_WORKERS = 24  # hc_pre_linear workers, one per AIC

assert HC_MULT == 4, f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}"


@pl.jit.inline
def hc_pre_gates(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[T_DYN, PRE_HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
):
    """Compute pre/post gates and Sinkhorn combinations with padded linear intermediates."""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + PRE_T_TILE - 1) // PRE_T_TILE
    t_linear = (
        (t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE
    ) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # for per-group comb base loads in comb_sinkhorn

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)

    # rms: full-K sum-of-squares per token-tile -> inv_rms.
    for t in pl.spmd(token_tiles, name_hint="hc_pre_rms", allow_early_resolve=True):
        t0 = t * PRE_T_TILE
        valid_rows = pl.min(PRE_T_TILE, t_dim - t0)
        sq_sum = pl.full([1, PRE_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // PRE_RMS_K_TILE, stage=4):
            k0 = kb * PRE_RMS_K_TILE
            if valid_rows == PRE_T_TILE:
                x_chunk_full = x_flat[t0 : t0 + PRE_T_TILE, k0 : k0 + PRE_RMS_K_TILE]
                x_sq_full = pl.mul(x_chunk_full, x_chunk_full)
                x_sq_row_full = pl.reshape(pl.row_sum(x_sq_full), [1, PRE_T_TILE])
                sq_sum = pl.add(sq_sum, x_sq_row_full)
            else:
                x_chunk_tail = pl.slice(
                    x_flat, [PRE_T_TILE, PRE_RMS_K_TILE], [t0, k0], valid_shape=[valid_rows, PRE_RMS_K_TILE]
                )
                x_sq_tail = pl.mul(x_chunk_tail, x_chunk_tail)
                x_sq_row_tail = pl.reshape(pl.row_sum(x_sq_tail), [1, PRE_T_TILE])
                sq_sum = pl.add(sq_sum, x_sq_row_tail)
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv = pl.reshape(pl.rsqrt(sq_mean, high_precision=True), [PRE_T_TILE, 1])
        inv_rms[t0 : t0 + PRE_T_TILE, 0:1] = inv

    # linear: split-K matmul -> per-split partials. The t_dim..t_linear pad rows are
    # zero-filled by valid_shape, never materialized.
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    linear_workers = pl.min(linear_units, LINEAR_WORKERS)
    for linear_worker in pl.spmd(linear_workers, name_hint="hc_pre_linear", allow_early_resolve=True):
        # Weight reads bypass L2.
        pl.set_cache_policy(hc_fn, pl.CachePolicy.BYPASS)
        for task in pl.range(linear_worker, linear_units, linear_workers):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            linear_split = task % LINEAR_OK
            k_base = linear_split * LINEAR_K_PER_SPLIT
            t_rows = pl.min(
                LINEAR_T_TILE, t_dim - t0
            )  # last row-block spills past t_dim; valid_shape zero-fills the tail
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
                k0 = k_base + kb * LINEAR_K_TILE
                x_linear_chunk = pl.slice(
                    x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[t_rows, LINEAR_K_TILE]
                )
                w_chunk = pl.slice(
                    hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE]
                )
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True, init_cond=(kb == 0))
            partial_row0 = linear_split * t_linear + t0
            mixes_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:MIX_PAD] = acc

    # Partials are reduced in ascending K order.
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for linear_block in pl.spmd(
        t_linear // LINEAR_T_TILE, name_hint="hc_pre_linear_reduce", allow_early_resolve=True
    ):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)
        mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total

    # split_pre_post: inv_rms-scaled pre gate -> pre_val_store (for mix_x), post gate -> post.
    # Both compute at HC_PAD width; post narrows to HC_MULT via a valid-shape slice (an 8-wide
    # 32B tile, 4 cols valid -- a bare 4-wide slice allocs a 16B tile ptoas rejects). comb gate
    # lives in comb_sinkhorn.
    # Only the final partial token tile uses these fixed-size staging buffers.
    post_tail_store = pl.create_tensor([PRE_T_TILE, PRE_HC_PAD], dtype=pl.FP32)
    for ob_worker in pl.spmd(
        pl.min(token_tiles, PRE_POST_WORKERS), name_hint="split_pre_post", allow_early_resolve=True
    ):
        for ob in pl.range(ob_worker, token_tiles, pl.min(token_tiles, PRE_POST_WORKERS)):
            t0 = ob * PRE_T_TILE
            valid_rows = pl.min(PRE_T_TILE, t_dim - t0)
            inv_col = inv_rms[t0 : t0 + PRE_T_TILE, 0:1]

            pre_base = pl.reshape(hc_base[0:PRE_HC_PAD], [1, PRE_HC_PAD])
            pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0 : t0 + PRE_T_TILE, 0:PRE_HC_PAD], inv_col), scale0)
            pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
            pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
            pre_val = pl.add(pre_sig, HC_EPS)
            pre_val_store[t0 : t0 + PRE_T_TILE, 0:PRE_HC_PAD] = pre_val

            post_base = pl.reshape(hc_base[HC_MULT : HC_MULT + PRE_HC_PAD], [1, PRE_HC_PAD])
            post_scaled = pl.mul(
                pl.row_expand_mul(mixes_raw[t0 : t0 + PRE_T_TILE, HC_MULT : HC_MULT + PRE_HC_PAD], inv_col), scale1
            )
            post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
            post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
            post_pad = pl.mul(post_sig, 2.0)
            if valid_rows == PRE_T_TILE:
                post[t0 : t0 + PRE_T_TILE, 0:HC_MULT] = pl.slice(
                    post_pad, [PRE_T_TILE, PRE_HC_PAD], [0, 0], valid_shape=[PRE_T_TILE, HC_MULT]
                )
            else:
                post_tail_store[0:PRE_T_TILE, 0:PRE_HC_PAD] = post_pad
                post_tile = pl.load(
                    post_tail_store,
                    [0, 0],
                    [PRE_T_TILE, PRE_HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(post_tile, [t0, 0], post)

    # comb_sinkhorn: comb gate from mixes_raw cols 8/12/16/20, softmax, then a
    # column-first 20-iteration Sinkhorn -> comb.
    comb_tail_store = pl.create_tensor([COMB_T_TILE, PRE_HC_PAD * HC_MULT], dtype=pl.FP32)
    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn", allow_early_resolve=True):
        t0 = ob * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.load(
            inv_rms, [t0, 0], [COMB_T_TILE, 1], valid_shape=[valid_rows, 1], target_memory=pl.MemorySpace.Vec
        )
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(
            mixes_raw,
            [t0, comb_off + 0 * HC_MULT],
            [COMB_T_TILE, PRE_HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g1 = pl.load(
            mixes_raw,
            [t0, comb_off + 1 * HC_MULT],
            [COMB_T_TILE, PRE_HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g2 = pl.load(
            mixes_raw,
            [t0, comb_off + 2 * HC_MULT],
            [COMB_T_TILE, PRE_HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g3 = pl.load(
            mixes_raw,
            [t0, comb_off + 3 * HC_MULT],
            [COMB_T_TILE, PRE_HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb0 = pl.load(
            hc_base_2d,
            [0, comb_off + 0 * HC_MULT],
            [1, PRE_HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb1 = pl.load(
            hc_base_2d,
            [0, comb_off + 1 * HC_MULT],
            [1, PRE_HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb2 = pl.load(
            hc_base_2d,
            [0, comb_off + 2 * HC_MULT],
            [1, PRE_HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb3 = pl.load(
            hc_base_2d,
            [0, comb_off + 3 * HC_MULT],
            [1, PRE_HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
        row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
        row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
        row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

        row_max_tmp = pl.create_tile([COMB_T_TILE, PRE_HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, PRE_HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
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
        if row_recip:
            row0_inv = pl.recip(row0_sum)
            row0_prob = pl.row_expand_mul(row0_exp, row0_inv)
            row0_soft = pl.add(row0_prob, HC_EPS)
            row1_inv = pl.recip(row1_sum)
            row1_prob = pl.row_expand_mul(row1_exp, row1_inv)
            row1_soft = pl.add(row1_prob, HC_EPS)
            row2_inv = pl.recip(row2_sum)
            row2_prob = pl.row_expand_mul(row2_exp, row2_inv)
            row2_soft = pl.add(row2_prob, HC_EPS)
            row3_inv = pl.recip(row3_sum)
            row3_prob = pl.row_expand_mul(row3_exp, row3_inv)
            row3_soft = pl.add(row3_prob, HC_EPS)
        else:
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

        row_sum_tmp_iter = pl.create_tile(
            [COMB_T_TILE, PRE_HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
        col_sum = pl.add(col_sum, HC_EPS)
        row0_cur = pl.div(row0_eff, col_sum)
        row1_cur = pl.div(row1_eff, col_sum)
        row2_cur = pl.div(row2_eff, col_sum)
        row3_cur = pl.div(row3_eff, col_sum)

        for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
            row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
            row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
            row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
            if row_recip:
                row0_inv = pl.recip(row0_rowsum)
                row0_norm = pl.row_expand_mul(row0_cur, row0_inv)
                row1_inv = pl.recip(row1_rowsum)
                row1_norm = pl.row_expand_mul(row1_cur, row1_inv)
                row2_inv = pl.recip(row2_rowsum)
                row2_norm = pl.row_expand_mul(row2_cur, row2_inv)
                row3_inv = pl.recip(row3_rowsum)
                row3_norm = pl.row_expand_mul(row3_cur, row3_inv)
            else:
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

        if valid_rows == COMB_T_TILE:
            row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
            row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
            row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
            row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
            pl.store(row0_out, [t0, 0 * HC_MULT], comb)
            pl.store(row1_out, [t0, 1 * HC_MULT], comb)
            pl.store(row2_out, [t0, 2 * HC_MULT], comb)
            pl.store(row3_out, [t0, 3 * HC_MULT], comb)
        else:
            pl.store(row0_cur, [0, 0 * PRE_HC_PAD], comb_tail_store)
            pl.store(row1_cur, [0, 1 * PRE_HC_PAD], comb_tail_store)
            pl.store(row2_cur, [0, 2 * PRE_HC_PAD], comb_tail_store)
            pl.store(row3_cur, [0, 3 * PRE_HC_PAD], comb_tail_store)
            row0_tail = pl.load(
                comb_tail_store,
                [0, 0 * PRE_HC_PAD],
                [COMB_T_TILE, PRE_HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row1_tail = pl.load(
                comb_tail_store,
                [0, 1 * PRE_HC_PAD],
                [COMB_T_TILE, PRE_HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row2_tail = pl.load(
                comb_tail_store,
                [0, 2 * PRE_HC_PAD],
                [COMB_T_TILE, PRE_HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row3_tail = pl.load(
                comb_tail_store,
                [0, 3 * PRE_HC_PAD],
                [COMB_T_TILE, PRE_HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
            pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
            pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
            pl.store(row3_tail, [t0, 3 * HC_MULT], comb)

    return pre_val_store


def _hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """Compute HC gates and BF16 pre-mixed activations."""
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + PRE_T_TILE - 1) // PRE_T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    pre_val_store = pl.create_tensor([t_linear, PRE_HC_PAD], dtype=pl.FP32)
    hc_pre_gates(x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, False)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D/D_SPMD blocks per token tile.
    x_mixed_tail_store = pl.create_tensor([PRE_T_TILE, D], dtype=pl.BF16)
    for blk in pl.spmd(token_tiles * (D // PRE_D_SPMD), name_hint="mix_x", allow_early_resolve=True):
        t0 = (blk // (D // PRE_D_SPMD)) * PRE_T_TILE
        d_base = (blk % (D // PRE_D_SPMD)) * PRE_D_SPMD
        valid_rows = pl.min(PRE_T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(pre_val_store[t0 : t0 + PRE_T_TILE, 0:PRE_HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:PRE_T_TILE], [PRE_T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:PRE_T_TILE], [PRE_T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:PRE_T_TILE], [PRE_T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:PRE_T_TILE], [PRE_T_TILE, 1])
        for db in pl.pipeline(PRE_D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [PRE_T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[valid_rows, D_TILE])
            x1 = pl.slice(x_flat, [PRE_T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[valid_rows, D_TILE])
            x2 = pl.slice(x_flat, [PRE_T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
            x3 = pl.slice(x_flat, [PRE_T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            if valid_rows == PRE_T_TILE:
                x_mixed[t0 : t0 + PRE_T_TILE, d0 : d0 + D_TILE] = y_bf16
            else:
                x_mixed_tail_store[0:PRE_T_TILE, d0 : d0 + D_TILE] = y_bf16
                y_out = pl.load(
                    x_mixed_tail_store,
                    [0, d0],
                    [PRE_T_TILE, D_TILE],
                    valid_shape=[valid_rows, D_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(y_out, [t0, d0], x_mixed)
    return x_mixed


hc_pre = pl.jit.inline(_hc_pre)


EPS = M.rms_norm_eps



HEAD_HC_PAD = 16

HC_VPAD = 8

HEAD_T_TILE = 16


RMS_T_TILE = 64

HEAD_RMS_K_TILE = 256



HEAD_D_SPMD = 2048

RMS_OK = 8

ROW_ALIGN = max(HEAD_T_TILE, LINEAR_T_TILE, RMS_T_TILE)


def _hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    x_hc.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x_hc, 0)
    token_tiles = (t_dim + HEAD_T_TILE - 1) // HEAD_T_TILE
    rms_tiles = (t_dim + RMS_T_TILE - 1) // RMS_T_TILE
    linear_full_rows = (t_dim // LINEAR_T_TILE) * LINEAR_T_TILE
    t_linear = ((t_dim + ROW_ALIGN - 1) // ROW_ALIGN) * ROW_ALIGN
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, D])
    # rms: split-K sum-of-squares, fanned over (rms-tile x K-slice). RMS_T_TILE is
    # taller than the gate's T_TILE so the fan-out stays inside the vector array
    # instead of queueing several waves of short tasks.
    sq_part = pl.create_tensor([RMS_OK, t_linear], dtype=pl.FP32)
    for task in pl.spmd(rms_tiles * RMS_OK, name_hint="hc_head_rms"):
        t0 = (task // RMS_OK) * RMS_T_TILE
        ok = task % RMS_OK
        k_base = ok * (HC_DIM // RMS_OK)
        valid_rows = pl.min(RMS_T_TILE, t_dim - t0)
        sq_sum = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_OK // HEAD_RMS_K_TILE, stage=4):
            k0 = k_base + kb * HEAD_RMS_K_TILE
            if valid_rows == RMS_T_TILE:
                x_rms_full = x_flat[t0 : t0 + RMS_T_TILE, k0 : k0 + HEAD_RMS_K_TILE]
                sq_col_full = pl.row_sum(pl.mul(x_rms_full, x_rms_full))
                sq_sum = pl.add(sq_sum, pl.reshape(sq_col_full, [1, RMS_T_TILE]))
            else:
                x_rms_tail = pl.slice(
                    x_flat,
                    [RMS_T_TILE, HEAD_RMS_K_TILE],
                    [t0, k0],
                    valid_shape=[valid_rows, HEAD_RMS_K_TILE],
                )
                sq_col_tail = pl.row_sum(pl.mul(x_rms_tail, x_rms_tail))
                sq_sum = pl.add(sq_sum, pl.reshape(sq_col_tail, [1, RMS_T_TILE]))
        sq_part = pl.assemble(sq_part, sq_sum, [ok, t0])

    # One writer per row block computes the complete head projection.
    mixes_raw = pl.create_tensor([t_linear, HEAD_HC_PAD], dtype=pl.FP32)
    if linear_full_rows > 0:
        with pl.spmd(
            linear_full_rows // LINEAR_T_TILE,
            name_hint="hc_head_linear",
        ) as _linear_tid:
            # Weight reads bypass L2.
            pl.set_cache_policy(hc_head_fn, pl.CachePolicy.BYPASS)
            task = pl.tile.get_block_idx()
            t0 = task * LINEAR_T_TILE
            acc_full = pl.create_tensor([LINEAR_T_TILE, HEAD_HC_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
                k0 = kb * LINEAR_K_TILE
                x_lin_full = x_flat[t0 : t0 + LINEAR_T_TILE, k0 : k0 + LINEAR_K_TILE]
                w_full = pl.slice(
                    hc_head_fn,
                    [HEAD_HC_PAD, LINEAR_K_TILE],
                    [0, k0],
                    valid_shape=[HC_MULT, LINEAR_K_TILE],
                )
                acc_full = pl.matmul_acc(acc_full, x_lin_full, w_full, b_trans=True, init_cond=(kb == 0))
            mixes_raw = pl.assemble(mixes_raw, acc_full, [t0, 0])

    # At most one incomplete M=16 block exists. Keep it in a separate conditional
    # task so every aligned block above retains the original static-M Cube path.
    if linear_full_rows < t_dim:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="hc_head_linear_tail",
        ):
            pl.set_cache_policy(hc_head_fn, pl.CachePolicy.BYPASS)
            tail_rows = t_dim - linear_full_rows
            acc_tail = pl.create_tensor([LINEAR_T_TILE, HEAD_HC_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
                k0 = kb * LINEAR_K_TILE
                x_lin_tail = pl.slice(
                    x_flat,
                    [LINEAR_T_TILE, LINEAR_K_TILE],
                    [linear_full_rows, k0],
                    valid_shape=[tail_rows, LINEAR_K_TILE],
                )
                w_tail = pl.slice(
                    hc_head_fn,
                    [HEAD_HC_PAD, LINEAR_K_TILE],
                    [0, k0],
                    valid_shape=[HC_MULT, LINEAR_K_TILE],
                )
                acc_tail = pl.matmul_acc(acc_tail, x_lin_tail, w_tail, b_trans=True, init_cond=(kb == 0))
            mixes_raw = pl.assemble(
                mixes_raw,
                acc_tail,
                [linear_full_rows, 0],
            )

    # reduce: gate + hc mix, fanned over (token-tile x D-slice). The rsqrt/sigmoid gate is
    # recomputed per task instead of being published by its own scope.
    for blk in pl.spmd(token_tiles * (D // HEAD_D_SPMD), name_hint="hc_head_reduce"):
        t0 = (blk // (D // HEAD_D_SPMD)) * HEAD_T_TILE
        d_base = (blk % (D // HEAD_D_SPMD)) * HEAD_D_SPMD
        valid_rows = pl.min(HEAD_T_TILE, t_dim - t0)
        scale = pl.read(hc_head_scale, [0])
        base = pl.reshape(pl.slice(hc_head_base, [HC_VPAD], [0], valid_shape=[HC_MULT]), [1, HC_VPAD])
        # sq_part is [RMS_OK, T], so the per-token total is a column sum; transposing the
        # slab makes it one row_sum landing in the [T_TILE, 1] shape row_expand_mul wants
        ssq_slab = pl.set_validshape(sq_part[0:RMS_OK, t0 : t0 + HEAD_T_TILE], RMS_OK, HEAD_T_TILE)
        ssq = pl.row_sum(pl.transpose(ssq_slab, axis1=0, axis2=1))
        inv_row = pl.rsqrt(pl.add(pl.mul(ssq, HC_DIM_INV), EPS), high_precision=True)
        inv_col = pl.reshape(inv_row, [HEAD_T_TILE, 1])
        mix = mixes_raw[t0 : t0 + HEAD_T_TILE, 0:HC_VPAD]
        scaled = pl.mul(pl.row_expand_mul(mix, inv_col), scale)
        logits = pl.add(scaled, pl.col_expand(scaled, base))
        pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), HC_EPS)
        # pin the valid shape: the gate carries valid=?x? from the dynamic-offset slice
        pre_val = pl.set_validshape(pre_val, HEAD_T_TILE, HC_VPAD)
        pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:HEAD_T_TILE], [HEAD_T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:HEAD_T_TILE], [HEAD_T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:HEAD_T_TILE], [HEAD_T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:HEAD_T_TILE], [HEAD_T_TILE, 1])
        for db in pl.pipeline(HEAD_D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            if valid_rows == HEAD_T_TILE:
                x_h0_full = x_flat[t0 : t0 + HEAD_T_TILE, 0 * D + d0 : 0 * D + d0 + D_TILE]
                x_h1_full = x_flat[t0 : t0 + HEAD_T_TILE, 1 * D + d0 : 1 * D + d0 + D_TILE]
                x_h2_full = x_flat[t0 : t0 + HEAD_T_TILE, 2 * D + d0 : 2 * D + d0 + D_TILE]
                x_h3_full = x_flat[t0 : t0 + HEAD_T_TILE, 3 * D + d0 : 3 * D + d0 + D_TILE]
                y01_full = pl.add(
                    pl.row_expand_mul(x_h0_full, pre0),
                    pl.row_expand_mul(x_h1_full, pre1),
                )
                y23_full = pl.add(
                    pl.row_expand_mul(x_h2_full, pre2),
                    pl.row_expand_mul(x_h3_full, pre3),
                )
                y_tile_full = pl.add(y01_full, y23_full)
                y_flat[t0 : t0 + HEAD_T_TILE, d0 : d0 + D_TILE] = pl.cast(
                    y_tile_full,
                    target_type=pl.BF16,
                    mode="rint",
                )
            else:
                x_h0_tail = pl.slice(
                    x_flat,
                    [HEAD_T_TILE, D_TILE],
                    [t0, 0 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                x_h1_tail = pl.slice(
                    x_flat,
                    [HEAD_T_TILE, D_TILE],
                    [t0, 1 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                x_h2_tail = pl.slice(
                    x_flat,
                    [HEAD_T_TILE, D_TILE],
                    [t0, 2 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                x_h3_tail = pl.slice(
                    x_flat,
                    [HEAD_T_TILE, D_TILE],
                    [t0, 3 * D + d0],
                    valid_shape=[valid_rows, D_TILE],
                )
                y01_tail = pl.add(
                    pl.row_expand_mul(x_h0_tail, pre0),
                    pl.row_expand_mul(x_h1_tail, pre1),
                )
                y23_tail = pl.add(
                    pl.row_expand_mul(x_h2_tail, pre2),
                    pl.row_expand_mul(x_h3_tail, pre3),
                )
                y_tile_tail = pl.add(y01_tail, y23_tail)
                y_bf16_tail = pl.cast(y_tile_tail, target_type=pl.BF16, mode="rint")
                y_valid_tail = pl.set_validshape(y_bf16_tail, valid_rows, D_TILE)
                y_flat[t0 : t0 + HEAD_T_TILE, d0 : d0 + D_TILE] = y_valid_tail

    y = pl.reshape(y_flat, [t_dim, D])
    return y


hc_head = pl.jit.inline(_hc_head)


POST_T_TILE = 4

INACTIVE_FILL_T_TILE = 16

INACTIVE_FILL_D_TILE = 256

assert (DECODE_BATCH // TP * DECODE_SEQ) % POST_T_TILE == 0

assert (PREFILL_BATCH * PREFILL_SEQ) % POST_T_TILE == 0


@pl.jit.inline
def hc_post_prefill(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Compute prefill active rows and deterministically pad the static tail."""
    t_dim = pl.tensor.dim(x, 0)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > t_dim:
        active_tokens = t_dim

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])
    active_tiles = (active_tokens + POST_T_TILE - 1) // POST_T_TILE

    if active_tokens > 0:
        for block in pl.spmd(active_tiles * HC_MULT, name_hint="hc_post_prefill"):
            token_block = block // HC_MULT
            out_h = block % HC_MULT
            t0 = token_block * POST_T_TILE
            for t in pl.pipeline(t0, t0 + POST_T_TILE, stage=2):
                if t < active_tokens:
                    post_w = pl.read(post, [t, out_h])
                    x_row = pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)
                    y_row = pl.mul(x_row, post_w)
                    for in_h in pl.pipeline(HC_MULT, stage=4):
                        comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                        res_d = in_h * D
                        res_row = residual_flat[t : t + 1, res_d : res_d + D]
                        y_row = pl.add(y_row, pl.mul(res_row, comb_w))
                    y_flat[t : t + 1, out_h * D : out_h * D + D] = y_row

    inactive_tokens = t_dim - active_tokens
    if inactive_tokens > 0:
        inactive_tiles = (inactive_tokens + INACTIVE_FILL_T_TILE - 1) // INACTIVE_FILL_T_TILE
        for fill_block in pl.spmd(inactive_tiles * HC_MULT, name_hint="hc_post_inactive_pad"):
            fill_tile = fill_block // HC_MULT
            out_h = fill_block % HC_MULT
            fill_t0 = active_tokens + fill_tile * INACTIVE_FILL_T_TILE
            zero = pl.full([1, INACTIVE_FILL_D_TILE], dtype=pl.FP32, value=0.0)
            for fill_dt in pl.range(INACTIVE_FILL_T_TILE):
                fill_t = fill_t0 + fill_dt
                if fill_t < t_dim:
                    for fill_d0 in pl.range(0, D, INACTIVE_FILL_D_TILE):
                        out_d0 = out_h * D + fill_d0
                        y_flat[fill_t : fill_t + 1, out_d0 : out_d0 + INACTIVE_FILL_D_TILE] = zero
    return y
