# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ordered HC pre-mix and the prefill wave dependency."""

import pypto.language as pl


if __package__:
    from .config import FLASH as M
else:
    from config import FLASH as M
if __package__:
    pass
else:
    pass
if __package__:
    from .decode_shapes import DECODE_BATCH, DECODE_SEQ
else:
    from decode_shapes import DECODE_BATCH, DECODE_SEQ

if __package__:
    from .rmsnorm import rms_norm_row
else:
    from rmsnorm import rms_norm_row

T_DYN = pl.dynamic('LOWLAT_TARGET_T_DYN')
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
COMB_T_TILE = 8
RMS_K_TILE = 512
LINEAR_K_TILE = 256
D_TILE = 256
D_SPMD = 1024
LINEAR_OK = 8
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
RMS_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // RMS_K_TILE
RMS_CHUNKS = HC_DIM // RMS_K_TILE
assert HC_MULT == 4, f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}"


@pl.jit.inline
def _hc_pre_gates(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    """One pl.spmd task per work-type, ordered by their GM read/write dependencies.

    A mixed split-K task produces linear and RMS chunk partials.
    The ordered reduction feeds the pre/post gates.
    Sinkhorn and mix_x independently consume the reduction outputs.
    The projection buffer is populated through the last live eight-row tile.
    Its extra Cube alignment rows are unused; the ready variant also fills them.
    """
    t_dim = pl.tensor.dim(x, 0)
    t_linear = (
        (t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE
    ) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # for per-group comb base loads in comb_sinkhorn

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)

    # linear: split-K matmul -> per-split partials. The t_dim..t_linear pad rows are
    # zero-filled by valid_shape, never materialized.
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    rms_partials = pl.create_tensor([(t_linear // T_TILE) * RMS_CHUNKS, T_TILE], dtype=pl.FP32)
    for task in pl.spmd(
        (t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="hc_pre_linear_rms", allow_early_resolve=True
    ):
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
            w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
            acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True, init_cond=(kb == 0))
        partial_row0 = linear_split * t_linear + t0
        mixes_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:MIX_PAD] = acc

        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            rms_t0 = t0 + pl.cast(lane, pl.INDEX) * T_TILE
            rms_rows = pl.min(T_TILE, t_dim - rms_t0)
            if rms_rows > 0:
                rms_base = (rms_t0 // T_TILE) * RMS_CHUNKS + linear_split * RMS_CHUNKS_PER_SPLIT
                for chunk in pl.range(RMS_CHUNKS_PER_SPLIT):
                    rms_k = k_base + chunk * RMS_K_TILE
                    rms_x = pl.slice(
                        x_flat, [T_TILE, RMS_K_TILE], [rms_t0, rms_k], valid_shape=[rms_rows, RMS_K_TILE]
                    )
                    sq_row = pl.reshape(pl.row_sum(pl.mul(rms_x, rms_x)), [1, T_TILE])
                    rms_partials[rms_base + chunk : rms_base + chunk + 1, 0:T_TILE] = sq_row

    # Reduce in ascending K order and keep the pre/post gate inputs on chip.
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="hc_pre_reduce_gates", allow_early_resolve=True):
        t0 = block * T_TILE
        mixes_total = mixes_partials[t0 : t0 + T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)
        mixes_raw[t0 : t0 + T_TILE, 0:MIX_PAD] = mixes_total

        # Both gates use an aligned HC_PAD tile. Only the live post columns are stored.
        post_rows = pl.min(T_TILE, t_dim - t0)
        # Fold the original 512-element row sums in ascending K order.
        rms_chunks = pl.slice(rms_partials, [RMS_CHUNKS, T_TILE], [block * RMS_CHUNKS, 0])
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for chunk in pl.range(RMS_CHUNKS):
            sq_sum = pl.add(sq_sum, pl.slice(rms_chunks, [1, T_TILE], [chunk, 0]))
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_col = pl.reshape(pl.rsqrt(sq_mean, high_precision=True), [T_TILE, 1])
        inv_rms[t0 : t0 + T_TILE, 0:1] = inv_col

        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_input = mixes_total[0:T_TILE, 0:HC_PAD]
        pre_norm = pl.row_expand_mul(pre_input, inv_col)
        pre_scaled = pl.mul(pre_norm, scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD] = pre_val

        post_base = pl.reshape(hc_base[HC_MULT : HC_MULT + HC_PAD], [1, HC_PAD])
        # The post gate starts four FP32 columns into the reduction tile.
        # A2/A3 UB extracts require the source column offset to be 32-byte aligned.
        post_cols = pl.cast(pl.arange(HC_MULT, [1, HC_PAD], dtype=pl.INT32), target_type=pl.FP32)
        post_grid = pl.col_expand(pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=0.0), post_cols)
        post_indices = pl.cast(post_grid, target_type=pl.INT32)
        post_input = pl.gather(mixes_total, dim=-1, index=post_indices)
        post_norm = pl.row_expand_mul(post_input, inv_col)
        post_scaled = pl.mul(post_norm, scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_pad = pl.mul(post_sig, 2.0)
        post[t0 : t0 + T_TILE, 0:HC_MULT] = pl.slice(
            post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[post_rows, HC_MULT]
        )

    # comb_sinkhorn: comb gate from mixes_raw cols 8/12/16/20, softmax, then a
    # column-first 20-iteration Sinkhorn -> comb.
    for ob in pl.spmd(
        (t_dim + COMB_T_TILE - 1) // COMB_T_TILE, name_hint="comb_sinkhorn", allow_early_resolve=True
    ):
        t0 = ob * COMB_T_TILE
        comb_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(
            mixes_raw,
            [t0, comb_off + 0 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g1 = pl.load(
            mixes_raw,
            [t0, comb_off + 1 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g2 = pl.load(
            mixes_raw,
            [t0, comb_off + 2 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g3 = pl.load(
            mixes_raw,
            [t0, comb_off + 3 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb0 = pl.load(
            hc_base_2d,
            [0, comb_off + 0 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb1 = pl.load(
            hc_base_2d,
            [0, comb_off + 1 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb2 = pl.load(
            hc_base_2d,
            [0, comb_off + 2 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb3 = pl.load(
            hc_base_2d,
            [0, comb_off + 3 * HC_MULT],
            [1, HC_PAD],
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

        row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
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

        row_sum_tmp_iter = pl.create_tile(
            [COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
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

        row0_out = pl.set_validshape(row0_cur, comb_rows, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, comb_rows, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, comb_rows, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, comb_rows, HC_MULT)
        pl.store(row0_out, [t0, 0 * HC_MULT], comb)
        pl.store(row1_out, [t0, 1 * HC_MULT], comb)
        pl.store(row2_out, [t0, 2 * HC_MULT], comb)
        pl.store(row3_out, [t0, 3 * HC_MULT], comb)

    return pre_val_store


@pl.jit.inline
def hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    pre_val_store = _hc_pre_gates(x, hc_fn, hc_scale, hc_base, post, comb)
    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D/D_SPMD blocks per token tile.
    for blk in pl.spmd(
        ((t_dim + T_TILE - 1) // T_TILE) * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True
    ):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        mix_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[mix_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[mix_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[mix_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[mix_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(y_bf16, mix_rows, D_TILE)
    return x_mixed


@pl.jit.inline
def hc_pre_norm(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    """Fork normalization and Sinkhorn from the gate reduction."""
    pre_val_store = _hc_pre_gates(x, hc_fn, hc_scale, hc_base, post, comb)
    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    norm_w_view = pl.reshape(norm_w, [1, D])
    with pl.spmd(t_dim, name_hint="hc_mix_attn_norm", allow_early_resolve=True) as norm_tid:
        tok = pl.tile.get_block_idx()
        pre0 = pl.read(pre_val_store, [tok, 0])
        pre1 = pl.read(pre_val_store, [tok, 1])
        pre2 = pl.read(pre_val_store, [tok, 2])
        pre3 = pl.read(pre_val_store, [tok, 3])
        y0 = pl.mul(pl.tile.load(x_flat, [tok, 0], [1, D]), pre0)
        y1 = pl.mul(pl.tile.load(x_flat, [tok, D], [1, D]), pre1)
        pair01 = pl.add(y0, y1)
        y2 = pl.mul(pl.tile.load(x_flat, [tok, 2 * D], [1, D]), pre2)
        y3 = pl.mul(pl.tile.load(x_flat, [tok, 3 * D], [1, D]), pre3)
        y = pl.add(pair01, pl.add(y2, y3))
        mixed_bf16 = pl.cast(y, target_type=pl.BF16, mode="rint")
        w_bf16 = pl.tile.load(norm_w_view, [0, 0], [1, D])
        normed = rms_norm_row(mixed_bf16, w_bf16)
        pl.tile.store(normed, [tok, 0], x_normed, shapes=[1, D])
    return norm_tid


@pl.jit.inline
def _hc_post_and_mix_inputs(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    """Compute post/comb on one row owner and leave the pre-mix path independent."""
    t_dim = pl.tensor.dim(x, 0)
    t_linear = (
        (t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE
    ) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # for per-group comb base loads in comb_sinkhorn

    # linear: split-K matmul -> per-split partials. The t_dim..t_linear pad rows are
    # zero-filled by valid_shape, never materialized.
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    rms_partials = pl.create_tensor([(t_linear // T_TILE) * RMS_CHUNKS, T_TILE], dtype=pl.FP32)
    for task in pl.spmd(
        (t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="hc_pre_linear_rms", allow_early_resolve=True
    ):
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
            w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
            acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True, init_cond=(kb == 0))
        partial_row0 = linear_split * t_linear + t0
        mixes_partials[partial_row0 : partial_row0 + LINEAR_T_TILE, 0:MIX_PAD] = acc

        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            rms_t0 = t0 + pl.cast(lane, pl.INDEX) * T_TILE
            rms_rows = pl.min(T_TILE, t_dim - rms_t0)
            if rms_rows > 0:
                rms_base = (rms_t0 // T_TILE) * RMS_CHUNKS + linear_split * RMS_CHUNKS_PER_SPLIT
                for chunk in pl.range(RMS_CHUNKS_PER_SPLIT):
                    rms_k = k_base + chunk * RMS_K_TILE
                    rms_x = pl.slice(
                        x_flat, [T_TILE, RMS_K_TILE], [rms_t0, rms_k], valid_shape=[rms_rows, RMS_K_TILE]
                    )
                    sq_row = pl.reshape(pl.row_sum(pl.mul(rms_x, rms_x)), [1, T_TILE])
                    rms_partials[rms_base + chunk : rms_base + chunk + 1, 0:T_TILE] = sq_row

    # Reduce in ascending K order and keep the pre/post gate inputs on chip.
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="hc_reduce_post_and_sinkhorn", allow_early_resolve=True):
        t0 = block * T_TILE
        mixes_total = mixes_partials[t0 : t0 + T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)

        # Both gates use an aligned HC_PAD tile. Only the live post columns are stored.
        post_rows = pl.min(T_TILE, t_dim - t0)
        # Fold the original 512-element row sums in ascending K order.
        rms_chunks = pl.slice(rms_partials, [RMS_CHUNKS, T_TILE], [block * RMS_CHUNKS, 0])
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for chunk in pl.range(RMS_CHUNKS):
            sq_sum = pl.add(sq_sum, pl.slice(rms_chunks, [1, T_TILE], [chunk, 0]))
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_col = pl.reshape(pl.rsqrt(sq_mean, high_precision=True), [T_TILE, 1])

        post_base = pl.reshape(hc_base[HC_MULT : HC_MULT + HC_PAD], [1, HC_PAD])
        # The post gate starts four FP32 columns into the reduction tile.
        # A2/A3 UB extracts require the source column offset to be 32-byte aligned.
        post_cols = pl.cast(pl.arange(HC_MULT, [1, HC_PAD], dtype=pl.INT32), target_type=pl.FP32)
        post_grid = pl.col_expand(pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=0.0), post_cols)
        post_indices = pl.cast(post_grid, target_type=pl.INT32)
        post_input = pl.gather(mixes_total, dim=-1, index=post_indices)
        post_norm = pl.row_expand_mul(post_input, inv_col)
        post_scaled = pl.mul(post_norm, scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_pad = pl.mul(post_sig, 2.0)
        post[t0 : t0 + T_TILE, 0:HC_MULT] = pl.slice(
            post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[post_rows, HC_MULT]
        )


        # The post and comb use the same eight-row owner and ordered reductions.
        comb_rows = post_rows
        inv_col_t = inv_col
        comb_off = HC_MULT * 2
        cols0 = pl.arange(pl.cast(comb_off, pl.INT32), [1, HC_PAD], dtype=pl.INT32)
        cols_f32 = pl.cast(cols0, target_type=pl.FP32)
        grid0 = pl.col_expand(pl.full([COMB_T_TILE, HC_PAD], dtype=pl.FP32, value=0.0), cols_f32)
        indices0 = pl.cast(pl.add(grid0, 0.0), target_type=pl.INT32)
        mix_g0 = pl.set_validshape(pl.gather(mixes_total, dim=-1, index=indices0), COMB_T_TILE, HC_MULT)
        indices1 = pl.cast(pl.add(grid0, 4.0), target_type=pl.INT32)
        mix_g1 = pl.set_validshape(pl.gather(mixes_total, dim=-1, index=indices1), COMB_T_TILE, HC_MULT)
        indices2 = pl.cast(pl.add(grid0, 8.0), target_type=pl.INT32)
        mix_g2 = pl.set_validshape(pl.gather(mixes_total, dim=-1, index=indices2), COMB_T_TILE, HC_MULT)
        indices3 = pl.cast(pl.add(grid0, 12.0), target_type=pl.INT32)
        mix_g3 = pl.set_validshape(pl.gather(mixes_total, dim=-1, index=indices3), COMB_T_TILE, HC_MULT)
        cb0 = pl.slice(hc_base_2d, [1, HC_PAD], [0, comb_off + 0 * HC_MULT], valid_shape=[1, HC_MULT])
        cb1 = pl.slice(hc_base_2d, [1, HC_PAD], [0, comb_off + 1 * HC_MULT], valid_shape=[1, HC_MULT])
        cb2 = pl.slice(hc_base_2d, [1, HC_PAD], [0, comb_off + 2 * HC_MULT], valid_shape=[1, HC_MULT])
        cb3 = pl.slice(hc_base_2d, [1, HC_PAD], [0, comb_off + 3 * HC_MULT], valid_shape=[1, HC_MULT])
        row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
        row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
        row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
        row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

        row0_max = pl.row_max(row0_p)
        row1_max = pl.row_max(row1_p)
        row2_max = pl.row_max(row2_p)
        row3_max = pl.row_max(row3_p)
        row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
        row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
        row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
        row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
        row0_sum = pl.row_sum(row0_exp)
        row1_sum = pl.row_sum(row1_exp)
        row2_sum = pl.row_sum(row2_exp)
        row3_sum = pl.row_sum(row3_exp)
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

        col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
        col_sum = pl.add(col_sum, HC_EPS)
        row0_cur = pl.div(row0_eff, col_sum)
        row1_cur = pl.div(row1_eff, col_sum)
        row2_cur = pl.div(row2_eff, col_sum)
        row3_cur = pl.div(row3_eff, col_sum)

        for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_rowsum = pl.add(pl.row_sum(row0_cur), HC_EPS)
            row1_rowsum = pl.add(pl.row_sum(row1_cur), HC_EPS)
            row2_rowsum = pl.add(pl.row_sum(row2_cur), HC_EPS)
            row3_rowsum = pl.add(pl.row_sum(row3_cur), HC_EPS)
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

        row0_out = pl.set_validshape(row0_cur, comb_rows, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, comb_rows, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, comb_rows, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, comb_rows, HC_MULT)
        comb[t0:t0 + COMB_T_TILE, 0 * HC_MULT:(0 + 1) * HC_MULT] = row0_out
        comb[t0:t0 + COMB_T_TILE, 1 * HC_MULT:(1 + 1) * HC_MULT] = row1_out
        comb[t0:t0 + COMB_T_TILE, 2 * HC_MULT:(2 + 1) * HC_MULT] = row2_out
        comb[t0:t0 + COMB_T_TILE, 3 * HC_MULT:(3 + 1) * HC_MULT] = row3_out

    return mixes_partials, rms_partials, t_linear


@pl.jit.inline
def hc_pre_ready_mix(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    mixes_partials, rms_partials, t_linear = _hc_post_and_mix_inputs(x, hc_fn, hc_scale, hc_base, post, comb)
    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D/D_SPMD blocks per token tile.
    for blk in pl.spmd(
        ((t_dim + T_TILE - 1) // T_TILE) * (D // D_SPMD), name_hint="hc_mix_from_partials", allow_early_resolve=True
    ):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        mix_rows = pl.min(T_TILE, t_dim - t0)
        gate_block = t0 // T_TILE
        mixes_total = mixes_partials[t0:t0 + T_TILE, 0:HC_PAD]
        for split in pl.range(1, LINEAR_OK):
            offset = split * t_linear + t0
            mixes_total = pl.add(mixes_total, mixes_partials[offset:offset + T_TILE, 0:HC_PAD])
        rms_chunks = pl.slice(rms_partials, [RMS_CHUNKS, T_TILE], [gate_block * RMS_CHUNKS, 0])
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for chunk in pl.range(RMS_CHUNKS):
            sq_sum = pl.add(sq_sum, pl.slice(rms_chunks, [1, T_TILE], [chunk, 0]))
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_col = pl.reshape(pl.rsqrt(sq_mean, high_precision=True), [T_TILE, 1])
        scale0 = pl.read(hc_scale, [0])
        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_input = mixes_total[0:T_TILE, 0:HC_PAD]
        pre_norm = pl.row_expand_mul(pre_input, inv_col)
        pre_scaled = pl.mul(pre_norm, scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[mix_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[mix_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[mix_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[mix_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(y_bf16, mix_rows, D_TILE)
    return x_mixed


@pl.jit.inline
def hc_pre_norm_ready_mix(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    """Fork normalization and Sinkhorn from the gate reduction."""
    mixes_partials, rms_partials, t_linear = _hc_post_and_mix_inputs(x, hc_fn, hc_scale, hc_base, post, comb)
    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    norm_w_view = pl.reshape(norm_w, [1, D])
    with pl.spmd(t_dim, name_hint="hc_mix_attn_norm_from_partials", allow_early_resolve=True) as norm_tid:
        tok = pl.tile.get_block_idx()
        t0 = tok // T_TILE * T_TILE
        gate_block = t0 // T_TILE
        mixes_total = mixes_partials[t0:t0 + T_TILE, 0:HC_PAD]
        for split in pl.range(1, LINEAR_OK):
            offset = split * t_linear + t0
            mixes_total = pl.add(mixes_total, mixes_partials[offset:offset + T_TILE, 0:HC_PAD])
        rms_chunks = pl.slice(rms_partials, [RMS_CHUNKS, T_TILE], [gate_block * RMS_CHUNKS, 0])
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for chunk in pl.range(RMS_CHUNKS):
            sq_sum = pl.add(sq_sum, pl.slice(rms_chunks, [1, T_TILE], [chunk, 0]))
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_col = pl.reshape(pl.rsqrt(sq_mean, high_precision=True), [T_TILE, 1])
        scale0 = pl.read(hc_scale, [0])
        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_input = mixes_total[0:T_TILE, 0:HC_PAD]
        pre_norm = pl.row_expand_mul(pre_input, inv_col)
        pre_scaled = pl.mul(pre_norm, scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)
        pre0 = pl.read(pre_val, [tok-t0, 0])
        pre1 = pl.read(pre_val, [tok-t0, 1])
        pre2 = pl.read(pre_val, [tok-t0, 2])
        pre3 = pl.read(pre_val, [tok-t0, 3])
        y0 = pl.mul(pl.tile.load(x_flat, [tok, 0], [1, D]), pre0)
        y1 = pl.mul(pl.tile.load(x_flat, [tok, D], [1, D]), pre1)
        pair01 = pl.add(y0, y1)
        y2 = pl.mul(pl.tile.load(x_flat, [tok, 2 * D], [1, D]), pre2)
        y3 = pl.mul(pl.tile.load(x_flat, [tok, 3 * D], [1, D]), pre3)
        y = pl.add(pair01, pl.add(y2, y3))
        mixed_bf16 = pl.cast(y, target_type=pl.BF16, mode="rint")
        w_bf16 = pl.tile.load(norm_w_view, [0, 0], [1, D])
        normed = rms_norm_row(mixed_bf16, w_bf16)
        pl.tile.store(normed, [tok, 0], x_normed, shapes=[1, D])
    return norm_tid


if __package__:
    from .decode_shapes import DECODE_TOKENS as _NATIVE_ROWS
else:
    from decode_shapes import DECODE_TOKENS as _NATIVE_ROWS

if _NATIVE_ROWS == 6:
    hc_pre = hc_pre_ready_mix
    hc_pre_norm = hc_pre_norm_ready_mix


@pl.jit.inline
def hc_pre_with_input_ready(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    input_ready: pl.Scalar[pl.TASK_ID],
):
    """One pl.spmd task per work-type, ordered by their GM read/write dependencies.

    rms -> linear -> linear_reduce -> split_pre_post / comb_sinkhorn / mix_x. Cross-scope
    buffers are sized to t_linear, the token count padded up to whole 16-row cube tiles.
    Reduction populates the full alignment extent; consumers read only eight-row tiles
    covering live tokens, as in hc_pre.
    """
    t_dim = pl.tensor.dim(x, 0)
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
    for t in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="hc_pre_rms", deps=[input_ready], allow_early_resolve=True):
        t0 = t * T_TILE
        rms_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            k0 = kb * RMS_K_TILE
            x_chunk = pl.slice(x_flat, [T_TILE, RMS_K_TILE], [t0, k0], valid_shape=[rms_rows, RMS_K_TILE])
            x_sq = pl.mul(x_chunk, x_chunk)
            x_sq_row = pl.reshape(pl.row_sum(x_sq), [1, T_TILE])
            sq_sum = pl.add(sq_sum, x_sq_row)
        sq_mean = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv = pl.reshape(pl.rsqrt(sq_mean, high_precision=True), [T_TILE, 1])
        inv_rms[t0 : t0 + T_TILE, 0:1] = inv

    # linear: split-K matmul -> per-split partials. The t_dim..t_linear pad rows are
    # zero-filled by valid_shape, never materialized.
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    for task in pl.spmd(
        (t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="hc_pre_linear", deps=[input_ready], allow_early_resolve=True
    ):
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
            w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[MIX_HC, LINEAR_K_TILE])
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
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    for ob in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="split_pre_post", allow_early_resolve=True):
        t0 = ob * T_TILE
        post_rows = pl.min(T_TILE, t_dim - t0)
        inv_col = inv_rms[t0 : t0 + T_TILE, 0:1]

        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_norm = pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, 0:HC_PAD], inv_col)
        pre_scaled = pl.mul(pre_norm, scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD] = pre_val

        post_base = pl.reshape(hc_base[HC_MULT : HC_MULT + HC_PAD], [1, HC_PAD])
        post_norm = pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, HC_MULT : HC_MULT + HC_PAD], inv_col)
        post_scaled = pl.mul(post_norm, scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_pad = pl.mul(post_sig, 2.0)
        post[t0 : t0 + T_TILE, 0:HC_MULT] = pl.slice(
            post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[post_rows, HC_MULT]
        )

    # comb_sinkhorn: comb gate from mixes_raw cols 8/12/16/20, softmax, then a
    # column-first 20-iteration Sinkhorn -> comb.
    for ob in pl.spmd(
        (t_dim + COMB_T_TILE - 1) // COMB_T_TILE, name_hint="comb_sinkhorn", allow_early_resolve=True
    ):
        t0 = ob * COMB_T_TILE
        comb_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(
            mixes_raw,
            [t0, comb_off + 0 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g1 = pl.load(
            mixes_raw,
            [t0, comb_off + 1 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g2 = pl.load(
            mixes_raw,
            [t0, comb_off + 2 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g3 = pl.load(
            mixes_raw,
            [t0, comb_off + 3 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[COMB_T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb0 = pl.load(
            hc_base_2d,
            [0, comb_off + 0 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb1 = pl.load(
            hc_base_2d,
            [0, comb_off + 1 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb2 = pl.load(
            hc_base_2d,
            [0, comb_off + 2 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb3 = pl.load(
            hc_base_2d,
            [0, comb_off + 3 * HC_MULT],
            [1, HC_PAD],
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

        row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
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

        row_sum_tmp_iter = pl.create_tile(
            [COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
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

        row0_out = pl.set_validshape(row0_cur, comb_rows, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, comb_rows, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, comb_rows, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, comb_rows, HC_MULT)
        pl.store(row0_out, [t0, 0 * HC_MULT], comb)
        pl.store(row1_out, [t0, 1 * HC_MULT], comb)
        pl.store(row2_out, [t0, 2 * HC_MULT], comb)
        pl.store(row3_out, [t0, 3 * HC_MULT], comb)

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D/D_SPMD blocks per token tile.
    for blk in pl.spmd(
        ((t_dim + T_TILE - 1) // T_TILE) * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True
    ):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        mix_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[mix_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[mix_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[mix_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[mix_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(y_bf16, mix_rows, D_TILE)
    return x_mixed


@pl.jit
def hc_pre_test(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    hc_pre(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
    return x_mixed


def _golden_a2a3_cube_linear(x_flat_2d, hc_fn):
    """Emulate the A2/A3 f32 Cube MAD reduction used by the HC projection."""
    import torch

    cube_acc_k = 4
    group_block = 64
    assert LINEAR_K_PER_SPLIT % cube_acc_k == 0
    groups_per_split = LINEAR_K_PER_SPLIT // cube_acc_k
    t_dim = x_flat_2d.shape[0]
    x_groups = x_flat_2d.reshape(t_dim, 1, LINEAR_OK, groups_per_split, cube_acc_k)
    w_groups = hc_fn.reshape(1, MIX_HC, LINEAR_OK, groups_per_split, cube_acc_k)
    split_mixes = torch.zeros(t_dim, MIX_HC, LINEAR_OK, dtype=torch.float32)
    for group0 in range(0, groups_per_split, group_block):
        x_block = x_groups[..., group0 : group0 + group_block, :].double()
        w_block = w_groups[..., group0 : group0 + group_block, :].double()
        group_dots = (x_block * w_block).sum(dim=-1)
        for group in range(group_dots.shape[-1]):
            split_mixes = (split_mixes.double() + group_dots[..., group]).float()
    mixes = torch.zeros(t_dim, MIX_HC, dtype=torch.float32)
    for split in range(LINEAR_OK):
        mixes += split_mixes[..., split]
    return mixes


def golden_hc_pre(tensors):
    """Torch reference matching the target HC-pre reduction and nonlinearities."""
    import torch

    if __package__:
        from .utils import fp32_matmul_fp32
    else:
        from utils import fp32_matmul_fp32

    x = tensors["x"].float()
    hc_fn = tensors["hc_fn"].float()
    hc_scale = tensors["hc_scale"].float()
    hc_base = tensors["hc_base"].float()
    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)
    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0 : k0 + RMS_K_TILE]
        square_row = x_chunk * x_chunk
        while square_row.shape[-1] > 1:
            square_row = square_row[:, 0::2] + square_row[:, 1::2]
        sq_sum += square_row
    variance = sq_sum * HC_DIM_INV + NORM_EPS
    rsqrt = torch.rsqrt(variance.double()).float()
    mixes = fp32_matmul_fp32(x_flat_2d, hc_fn.t(), split_k=LINEAR_OK)
    mixes *= rsqrt
    pre_logits = mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]
    pre_exp = torch.exp(-pre_logits.double()).float()
    pre = torch.reciprocal((pre_exp + 1.0).double()).float() + HC_EPS
    post_logits = mixes[..., HC_MULT : HC_MULT * 2] * hc_scale[1] + hc_base[HC_MULT : HC_MULT * 2]
    post_exp = torch.exp(-post_logits.double()).float()
    post_t = 2 * torch.reciprocal((post_exp + 1.0).double()).float()
    comb_logits = mixes[..., HC_MULT * 2 :] * hc_scale[2] + hc_base[HC_MULT * 2 :]
    comb_t = comb_logits.view(t_dim, HC_MULT, HC_MULT)
    comb_max = comb_t.max(dim=-1, keepdim=True).values
    comb_exp = torch.exp((comb_t - comb_max).double()).float()

    def sum_four(value):
        return value[..., 0:1] + value[..., 1:2] + (value[..., 2:3] + value[..., 3:4])

    def column_sum(value):
        return value[:, 0:1, :] + value[:, 1:2, :] + (value[:, 2:3, :] + value[:, 3:4, :])

    comb_t = (comb_exp.double() / sum_four(comb_exp).double()).float() + HC_EPS
    comb_t = (comb_t.double() / (column_sum(comb_t) + HC_EPS).double()).float()
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = (comb_t.double() / (sum_four(comb_t) + HC_EPS).double()).float()
        comb_t = (comb_t.double() / (column_sum(comb_t) + HC_EPS).double()).float()
    y0 = x[:, 0, :] * pre[:, 0:1]
    y1 = x[:, 1, :] * pre[:, 1:2]
    y2 = x[:, 2, :] * pre[:, 2:3]
    y3 = x[:, 3, :] * pre[:, 3:4]
    y = y0 + y1 + (y2 + y3)
    tensors["x_mixed"][:] = y.to(torch.bfloat16).reshape(t_dim, D)
    tensors["post"][:] = post_t.reshape(t_dim, HC_MULT)
    tensors["comb"][:] = comb_t.reshape(t_dim, HC_MULT * HC_MULT)


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def init_x():
        return torch.randn(T, HC_MULT, D) * 0.05

    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0519

    def init_hc_scale():
        return torch.tensor([0.076099, 0.032597, 0.226994])

    def init_hc_base():
        return torch.tensor(
            [
                5.9166,
                -3.6223,
                -2.9324,
                -3.3124,
                -3.91,
                -0.9384,
                -3.3256,
                -2.524,
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
                -0.501,
                -3.5992,
                -2.752,
                -3.3493,
                3.1587,
            ]
        )

    return [
        TensorSpec("x", [T, HC_MULT, D], torch.float32, init_value=init_x),
        TensorSpec("hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
        TensorSpec("hc_scale", [3], torch.float32, init_value=init_hc_scale),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
        TensorSpec("x_mixed", [T, D], torch.bfloat16),
        TensorSpec("post", [T, HC_MULT], torch.float32),
        TensorSpec("comb", [T, HC_MULT * HC_MULT], torch.float32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    MODES = {"decode": (DECODE_BATCH, DECODE_SEQ)}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode"], default="decode")
    parser.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="Validate one token count; default covers configured, single, short, and multi-tile tails.",
    )
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    if args.tokens is not None and args.tokens <= 0:
        parser.error("--tokens must be positive")
    modes_to_run = [args.mode]
    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        if args.tokens is not None:
            token_counts = [args.tokens]
        elif args.golden_data is not None:
            token_counts = [B * S]
        else:
            token_counts = list(dict.fromkeys([B * S, 1, 6, 9, 17]))
        for tokens in token_counts:
            print(f"--- hc_pre {mode_name}: T={tokens} ---")
            result = run(
                fn=hc_pre_test,
                specs=build_tensor_specs(1, tokens),
                golden_fn=golden_hc_pre,
                runtime_dir=args.runtime_dir,
                golden_data=args.golden_data,
                config=dict(
                    dump_passes=args.dump_passes,
                    backend_type=args.platform.removesuffix("sim"),
                    execution_mode=args.sim or args.platform.endswith("sim"),
                    device_id=args.device,
                    enable_chip_swimlane=args.enable_chip_swimlane,
                ),
                rtol=0.001,
                atol=0.001,
                compare_fn={
                    "x_mixed": ratio_allclose(atol=0.0001, rtol=1.0 / 128),
                    "post": ratio_allclose(atol=2.5e-05, rtol=0.005),
                    "comb": ratio_allclose(atol=2.5e-05, rtol=0.005),
                },
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(1)
