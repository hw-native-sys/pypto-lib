# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 hc_pre (dynamic shape) — T-adaptive: split-K (decode) + fused (prefill).

hc_pre wants opposite tilings in the two regimes, so it dispatches on T = B*S at
runtime (hc_pre inlines into each decode/prefill attention kernel, so each context
keeps only its branch):

  T <= LINEAR_T_TILE  -> _hc_pre_decode   (small T: one token-tile)
  else                -> _hc_pre_prefill  (large T: many token-tiles)

Decode (T=8): the fused single-spmd shape runs the whole op on one core (1 of 24
AIC / 1 of 48 AIV) — one token-tile is one spmd block. _hc_pre_decode instead
fans out across scopes (mirroring hc_head.py's pure-AIC split-K):

  cast        x (BF16) -> x_fp32 (FP32), so the projection is a pure-AIC matmul
  x_pad       zero-fill x_fp32 rows [t_dim:t_linear] (decode pads 8->16 rows)
  rms         sum-of-squares over HC_DIM -> inv_rms (overlaps seed+linear)
  seed        zero-seed mixes_raw for the atomic-add accumulation
  linear      SPLIT-K matmul: (t_linear/LINEAR_T_TILE)*LINEAR_OK tasks, each one
              token-tile x one 1/OK K-slice, atomic-adding its FP32 partial into
              mixes_raw. Decode goes 1 cube task -> LINEAR_OK (~40us -> ~7us).
              assemble(atomic=Add) is device-only -- the a2a3sim / a5sim sims do
              not model it, so those two sim CI checks are skipped for hc_pre.
  split_pre_post  scale mixes_raw by inv_rms, then sigmoid -> pre / post-pad and
                  comb logits.
  write_post  narrow post-pad [.,HC_PAD] -> post [.,HC_MULT].
  comb_sinkhorn   softmax + Sinkhorn normalization -> comb (20 serial iters; a
                  latency floor that more cores cannot shorten).
  mix_x       x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D (D/D_SPMD cores).

Prefill (T=128): _hc_pre_prefill is the fused single-spmd variant (#533, with the
#653 pad-free `valid_shape` + `fillpad` matmul) — its token-tiles already saturate
the chip, so the decode fan-out would only add AICPU dispatch overhead (the
per-tile scopes multiply into hundreds of tasks). The dispatch leaves this path
unchanged.

Device a2a3, best-of-N vs the pad-free fused baseline (post-#653): decode ~75us ->
~68us (split-K parallelizes the otherwise-1-cube matmul); prefill ~unchanged
(~87us, same fused path). The decode matmul reads pre-cast FP32 x_fp32, so it is a
clean cube-only kernel; the mixed prefill kernel still needs the pypto#1761 fix.
"""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# kernel-local
MIX_PAD = 32  # MIX_HC (24) padded to a 32-wide cube N / vector row
HC_PAD = 8  # HC_MULT (4) padded for 32B-aligned vector ops
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)

# tiling
T_TILE = 8  # vector row-tile (RMS / cast / split / mix_x)
LINEAR_T_TILE = 16  # cube matmul rows must be a 16-row boxed tile
COMB_T_TILE = 8  # sinkhorn row-tile
RMS_K_CHUNK = 512  # cast / rms K-fragment
LINEAR_K_CHUNK = 256  # cube K-fragment per matmul_acc (32x256x4 FP32 weight fits L0B)
D_CHUNK = 512  # mix_x inner D-fragment (BF16 load = 1KB, 512B-aligned)
D_SPMD = 512  # mix_x D per spmd block: decode fans 4096 reduce over D/D_SPMD cores
CAST_K_SPMD = 2048  # cast K per spmd block: decode fans the BF16->FP32 cast over HC_DIM/CAST_K_SPMD cores
# Split the K=HC_DIM reduction into LINEAR_OK slices that atomic-add their FP32
# partials, filling idle cubes at small T (decode: 1 token-tile -> LINEAR_OK
# cube tasks) and shortening each task's matmul_acc chain. Higher OK fills more
# decode cubes; prefill (8 token-tiles) packs OK*8 tasks into waves of ~24.
LINEAR_OK = 16
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // LINEAR_K_CHUNK

assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0
assert (DECODE_BATCH * DECODE_SEQ) % COMB_T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % COMB_T_TILE == 0
assert DECODE_BATCH * DECODE_SEQ <= LINEAR_T_TILE
assert T_MAX % LINEAR_T_TILE == 0
assert HC_DIM % LINEAR_OK == 0 and LINEAR_K_PER_SPLIT % LINEAR_K_CHUNK == 0
assert D % D_SPMD == 0 and D_SPMD % D_CHUNK == 0
assert HC_DIM % CAST_K_SPMD == 0 and CAST_K_SPMD % RMS_K_CHUNK == 0
# The 4-way pre / comb reduce, the comb_off = HC_MULT*2 column offset, and the
# pre0..pre3 / row0..row3 unrolling all assume HC_MULT == 4.
assert HC_MULT == 4, (
    f"hc_pre is hand-specialized to HC_MULT == 4, got {HC_MULT}; "
    "regenerate the pre0..pre3 / row0..row3 unrolling for the new hc_mult before using it."
)



# --- fused-path tiling (large T / prefill): one mixed cube+vec task per token-tile ---
MIX_RAW_ROWS = (T_MAX // T_TILE) * LINEAR_T_TILE
LINEAR_K_TILE = 256  # fused matmul K-fragment (32x256x4 FP32 weight fits L0B; 512 overflows)
D_TILE = 512  # fused mix_x D-fragment; halves the D-loop trip count versus 256


@pl.jit.inline
def _hc_pre_decode(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    t_dim = pl.tensor.dim(x, 0)
    t_linear = pl.max(t_dim, LINEAR_T_TILE)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])

    inv_rms = pl.create_tensor([T_MAX, 1], dtype=pl.FP32)
    x_fp32 = pl.create_tensor([T_MAX, HC_DIM], dtype=pl.FP32)  # FP32 activations for the pure-AIC matmul + rms
    mixes_raw = pl.create_tensor([T_MAX, MIX_PAD], dtype=pl.FP32)

    for blk in pl.spmd((t_dim // T_TILE) * (HC_DIM // CAST_K_SPMD), name_hint="hc_pre_cast"):
        t0 = (blk // (HC_DIM // CAST_K_SPMD)) * T_TILE
        k_base = (blk % (HC_DIM // CAST_K_SPMD)) * CAST_K_SPMD
        for kb in pl.pipeline(CAST_K_SPMD // RMS_K_CHUNK, stage=4):
            k0 = k_base + kb * RMS_K_CHUNK
            x_chunk = pl.cast(x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK], target_type=pl.FP32)
            x_fp32[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK] = x_chunk
    if t_linear > t_dim:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_pre_x_pad"):
            for k0 in pl.pipeline(0, HC_DIM, RMS_K_CHUNK, stage=4):
                x_fp32[t_dim:t_linear, k0:k0 + RMS_K_CHUNK] = pl.full(
                    [LINEAR_T_TILE - T_TILE, RMS_K_CHUNK], dtype=pl.FP32, value=0.0
                )

    for t in pl.spmd(t_dim // T_TILE, name_hint="hc_pre_rms"):
        t0 = t * T_TILE
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_CHUNK, stage=4):
            k0 = kb * RMS_K_CHUNK
            x_chunk = x_fp32[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK]
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]))
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS), high_precision=True), [T_TILE, 1])
        inv_rms = pl.assemble(inv_rms, inv, [t0, 0])

    # Split-K via atomic-add: zero-seed mixes_raw, then each (token-tile, K-slice)
    # task atomic-adds its [LINEAR_T_TILE, MIX_PAD] partial. The seed write -> RMW
    # WAW dependency orders the seed first; token tiles touch disjoint rows, so
    # only the LINEAR_OK tasks per row block contend. (assemble(atomic=Add) is a
    # device-only accumulate -- the a2a3sim / a5sim simulators do not model it, so
    # those two sim CI checks are skipped for hc_pre.)
    for tc in pl.spmd(t_linear // T_TILE, name_hint="hc_pre_seed"):
        ts0 = tc * T_TILE
        mixes_raw[ts0:ts0 + T_TILE, 0:MIX_PAD] = pl.full([T_TILE, MIX_PAD], dtype=pl.FP32, value=0.0)

    for task in pl.spmd((t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="hc_pre_linear"):
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        k_base = (task % LINEAR_OK) * LINEAR_K_PER_SPLIT
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
            k0 = k_base + kb * LINEAR_K_CHUNK
            x_linear_chunk = x_fp32[t0:t0 + LINEAR_T_TILE, k0:k0 + LINEAR_K_CHUNK]
            w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_CHUNK], [0, k0], valid_shape=[MIX_HC, LINEAR_K_CHUNK])
            if kb == 0:
                acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
            else:
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
        mixes_raw = pl.assemble(mixes_raw, acc, [t0, 0], atomic=pl.AtomicType.Add)

    # Scale the raw projection by inv_rms, then sigmoid -> pre (kept for mix_x via
    # pre_val_store) and post (written straight out); comb logits bridge to the
    # sinkhorn scope. inv_rms is folded in per column-group (cheap on [T, <=16]).
    pre_val_store = pl.create_tensor([T_MAX, HC_PAD], dtype=pl.FP32)
    post_pad_store = pl.create_tensor([T_MAX, HC_PAD], dtype=pl.FP32)
    # MIX_PAD (not HC_MULT*HC_MULT=16): comb_sinkhorn loads each group HC_PAD-wide
    # at offset k*HC_MULT, so group 3 reads cols [12:20] -- the 16-wide alloc made
    # that load descriptor exceed the tensor even though valid_shapes bounds the
    # real transfer to [12:16]. The 32-wide alloc keeps every descriptor in-bounds.
    comb_logits = pl.create_tensor([T_MAX, MIX_PAD], dtype=pl.FP32)
    for ob in pl.spmd(t_dim // T_TILE, name_hint="split_pre_post"):
        t0 = ob * T_TILE
        inv_col = inv_rms[t0:t0 + T_TILE, 0:1]

        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_val_store = pl.assemble(pre_val_store, pre_val, [t0, 0])

        post_base = pl.reshape(hc_base[HC_MULT:HC_MULT + HC_PAD], [1, HC_PAD])
        post_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, HC_MULT:HC_MULT + HC_PAD], inv_col), scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_pad = pl.mul(post_sig, 2.0)
        post_pad_store = pl.assemble(post_pad_store, post_pad, [t0, 0])

        comb_base = pl.reshape(hc_base[HC_MULT * 2:HC_MULT * 2 + HC_MULT * HC_MULT], [1, HC_MULT * HC_MULT])
        comb_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, HC_MULT * 2:HC_MULT * 2 + HC_MULT * HC_MULT], inv_col), scale2)
        comb_logits_tile = pl.add(comb_scaled, pl.col_expand(comb_scaled, comb_base))
        comb_logits = pl.assemble(comb_logits, comb_logits_tile, [t0, 0])

    for ob in pl.spmd(t_dim // COMB_T_TILE, name_hint="write_post"):
        t0 = ob * COMB_T_TILE
        post_tile = pl.load(post_pad_store, [t0, 0], [COMB_T_TILE, HC_PAD],
                            valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        pl.store(post_tile, [t0, 0], post)

    for ob in pl.spmd(t_dim // COMB_T_TILE, name_hint="comb_sinkhorn"):
        t0 = ob * COMB_T_TILE
        row0 = pl.load(comb_logits, [t0, 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row1 = pl.load(comb_logits, [t0, 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row2 = pl.load(comb_logits, [t0, 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row3 = pl.load(comb_logits, [t0, 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
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

        row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
        pl.store(row0_out, [t0, 0 * HC_MULT], comb)
        pl.store(row1_out, [t0, 1 * HC_MULT], comb)
        pl.store(row2_out, [t0, 2 * HC_MULT], comb)
        pl.store(row3_out, [t0, 3 * HC_MULT], comb)

    for blk in pl.spmd((t_dim // T_TILE) * (D // D_SPMD), name_hint="mix_x"):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        pre_tile_t = pl.transpose(pre_val_store[t0:t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_CHUNK, stage=2):
            d0 = d_base + db * D_CHUNK
            x0 = pl.cast(x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_CHUNK], target_type=pl.FP32)
            x1 = pl.cast(x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_CHUNK], target_type=pl.FP32)
            x2 = pl.cast(x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_CHUNK], target_type=pl.FP32)
            x3 = pl.cast(x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_CHUNK], target_type=pl.FP32)
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            x_mixed[t0:t0 + T_TILE, d0:d0 + D_CHUNK] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
    return x_mixed


@pl.jit.inline
def _hc_pre_prefill(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])

    # Raw RMS+linear result, spilled per tile and read straight back within the
    # same task. mixes_gm is sized to the static upper bound T_MAX; the loop
    # below only touches the t_dim real rows. The cube (matmul/AIC) writes it and
    # the vector epilogue (AIV) reads it back in the SAME task — the AIV-side
    # MTE3->MTE2 fence orders the self-RAW correctly; the cube<->vec pipe sync is
    # the part that needs pypto#1761.
    mixes_gm = pl.create_tensor([T_MAX, MIX_PAD], dtype=pl.FP32)
    mix_raw_gm = pl.create_tensor([MIX_RAW_ROWS, MIX_PAD], dtype=pl.FP32)

    for ob in pl.spmd(t_dim // T_TILE, name_hint="hc_pre"):
        t0 = ob * T_TILE
        linear_t0 = (t0 // LINEAR_T_TILE) * LINEAR_T_TILE
        linear_row_off = t0 - linear_t0
        mix_raw_t0 = ob * LINEAR_T_TILE

        # --- linear: RMS norm + hc_fn projection -> mixes_gm[t0] ---
        lin_rows = pl.min(LINEAR_T_TILE, t_dim - linear_t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        mix_acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
            kl0 = kb * LINEAR_K_TILE
            x_lin = pl.cast(x_flat[t0:t0 + T_TILE, kl0:kl0 + LINEAR_K_TILE], target_type=pl.FP32)
            x_lin_src = pl.slice(
                x_flat,
                [LINEAR_T_TILE, LINEAR_K_TILE],
                [linear_t0, kl0],
                valid_shape=[lin_rows, LINEAR_K_TILE],
            )
            x_lin_matmul = pl.cast(
                pl.fillpad(x_lin_src, pad_value=pl.PadValue.zero),
                target_type=pl.FP32,
            )
            x_sq = pl.mul(x_lin, x_lin)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(x_sq), [1, T_TILE]))
            w_lin = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, kl0], valid_shape=[MIX_HC, LINEAR_K_TILE])
            if kb == 0:
                mix_acc = pl.matmul(x_lin_matmul, w_lin, b_trans=True, out_dtype=pl.FP32)
            else:
                mix_acc = pl.matmul_acc(mix_acc, x_lin_matmul, w_lin, b_trans=True)
        mean_sq = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_rms_val = pl.rsqrt(mean_sq, high_precision=True)
        inv_rms_col = pl.reshape(inv_rms_val, [T_TILE, 1])
        mix_raw_gm[mix_raw_t0:mix_raw_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mix_acc
        mix_acc_real = mix_raw_gm[mix_raw_t0 + linear_row_off:mix_raw_t0 + linear_row_off + T_TILE, 0:MIX_PAD]
        mixes_gm[t0:t0 + T_TILE, 0:MIX_PAD] = pl.row_expand_mul(mix_acc_real, inv_rms_col)

        # --- post = 2*sigmoid(mixes[:, hc:2hc]*s1 + base) -> store ---
        post_base = pl.load(hc_base_2d, [0, HC_MULT], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)
        post_in = pl.load(mixes_gm, [t0, HC_MULT], [T_TILE, HC_PAD], target_memory=pl.MemorySpace.Vec)
        post_scaled = pl.mul(post_in, scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_tile = pl.set_validshape(pl.mul(post_sig, 2.0), T_TILE, HC_MULT)
        pl.store(post_tile, [t0, 0], post)

        # --- pre = sigmoid(mixes[:, :hc]*s0 + base) + eps. Kept in Vec, consumed
        # by mix_x below in the SAME scope (no GM round-trip). ---
        pre_base = pl.load(hc_base_2d, [0, 0], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)
        pre_in = pl.load(mixes_gm, [t0, 0], [T_TILE, HC_PAD], target_memory=pl.MemorySpace.Vec)
        pre_scaled = pl.mul(pre_in, scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_eps = pl.add(pre_sig, HC_EPS)

        # --- mix_x = sum_h pre[:, h] * x[:, h, :]. Transpose so each head is a
        # 32B-aligned row, then materialize each [T_TILE,1] scale into its own
        # buffer (tmuls by 1.0). ---
        pre_eps_t = pl.transpose(pre_eps, axis1=0, axis2=1)  # [HC_PAD, T_TILE]
        pre0 = pl.mul(pl.reshape(pre_eps_t[0:1, 0:T_TILE], [T_TILE, 1]), 1.0)
        pre1 = pl.mul(pl.reshape(pre_eps_t[1:2, 0:T_TILE], [T_TILE, 1]), 1.0)
        pre2 = pl.mul(pl.reshape(pre_eps_t[2:3, 0:T_TILE], [T_TILE, 1]), 1.0)
        pre3 = pl.mul(pl.reshape(pre_eps_t[3:4, 0:T_TILE], [T_TILE, 1]), 1.0)
        for db in pl.range(D // D_TILE):
            d0 = db * D_TILE
            x0 = pl.cast(pl.load(x_flat, [t0, 0 * D + d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec), target_type=pl.FP32)
            x1 = pl.cast(pl.load(x_flat, [t0, 1 * D + d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec), target_type=pl.FP32)
            x2 = pl.cast(pl.load(x_flat, [t0, 2 * D + d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec), target_type=pl.FP32)
            x3 = pl.cast(pl.load(x_flat, [t0, 3 * D + d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec), target_type=pl.FP32)
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            pl.store(pl.cast(y_tile, target_type=pl.BF16, mode="rint"), [t0, d0], x_mixed)

        # --- comb = sinkhorn(reshape(mixes[:, 2hc:]*s2 + base, hc, hc)). Each
        # group read 8-wide DIRECTLY from mixes_gm (offsets 8/12/16/20 fit in the
        # MIX_PAD=32 row); scale2 + base applied per group in-scope. ---
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(mixes_gm, [t0, comb_off + 0 * HC_MULT], [T_TILE, HC_PAD], valid_shapes=[T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g1 = pl.load(mixes_gm, [t0, comb_off + 1 * HC_MULT], [T_TILE, HC_PAD], valid_shapes=[T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g2 = pl.load(mixes_gm, [t0, comb_off + 2 * HC_MULT], [T_TILE, HC_PAD], valid_shapes=[T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g3 = pl.load(mixes_gm, [t0, comb_off + 3 * HC_MULT], [T_TILE, HC_PAD], valid_shapes=[T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb0 = pl.load(hc_base_2d, [0, comb_off + 0 * HC_MULT], [1, HC_PAD], valid_shapes=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb1 = pl.load(hc_base_2d, [0, comb_off + 1 * HC_MULT], [1, HC_PAD], valid_shapes=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb2 = pl.load(hc_base_2d, [0, comb_off + 2 * HC_MULT], [1, HC_PAD], valid_shapes=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb3 = pl.load(hc_base_2d, [0, comb_off + 3 * HC_MULT], [1, HC_PAD], valid_shapes=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row0 = pl.add(pl.mul(mix_g0, scale2), pl.col_expand(mix_g0, cb0))
        row1 = pl.add(pl.mul(mix_g1, scale2), pl.col_expand(mix_g1, cb1))
        row2 = pl.add(pl.mul(mix_g2, scale2), pl.col_expand(mix_g2, cb2))
        row3 = pl.add(pl.mul(mix_g3, scale2), pl.col_expand(mix_g3, cb3))
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

        row_max_tmp = pl.create_tile([T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
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

        row0_valid = pl.set_validshape(row0_soft, T_TILE, HC_MULT)
        row1_valid = pl.set_validshape(row1_soft, T_TILE, HC_MULT)
        row2_valid = pl.set_validshape(row2_soft, T_TILE, HC_MULT)
        row3_valid = pl.set_validshape(row3_soft, T_TILE, HC_MULT)
        row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
        row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
        row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
        row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

        row_sum_tmp_iter = pl.create_tile([T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
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

        row0_out = pl.set_validshape(row0_cur, T_TILE, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, T_TILE, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, T_TILE, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, T_TILE, HC_MULT)
        pl.store(row0_out, [t0, 0 * HC_MULT], comb)
        pl.store(row1_out, [t0, 1 * HC_MULT], comb)
        pl.store(row2_out, [t0, 2 * HC_MULT], comb)
        pl.store(row3_out, [t0, 3 * HC_MULT], comb)
    return x_mixed


@pl.jit.inline
def hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    # Decode (T = B*S small, one token-tile) starves the chip in the fused single
    # task; split-K + per-axis fan-out fills 24 AIC / 48 AIV. Prefill already fills
    # the chip per token-tile, so the extra scopes are pure dispatch overhead there.
    # hc_pre inlines into each decode/prefill attention kernel, so each context keeps
    # only its branch.
    t_dim_sel = pl.tensor.dim(x, 0)
    if t_dim_sel <= LINEAR_T_TILE:
        _hc_pre_decode(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
    else:
        _hc_pre_prefill(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
    return x_mixed


@pl.jit
def hc_pre_test(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
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


def golden_hc_pre(tensors):
    """Torch reference, direct port of model.py Block.hc_pre + hc_split_sinkhorn."""
    import torch

    x = tensors["x"].float()  # [T, hc, D]
    hc_fn = tensors["hc_fn"].float()  # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()  # [3]
    hc_base = tensors["hc_base"].float()  # [mix_hc]

    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)

    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + NORM_EPS)

    mix_cols = []
    for m in range(MIX_HC):
        mix_col = torch.zeros(t_dim, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_CHUNK):
            x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_CHUNK]
            w_chunk = hc_fn[m:m + 1, k0:k0 + LINEAR_K_CHUNK]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1)  # [T, mix_hc]

    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1]
                               + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(t_dim, HC_MULT, HC_MULT)

    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y = torch.zeros(t_dim, D, dtype=torch.float32)
    for h in range(HC_MULT):
        y += x[:, h, :] * pre[:, h:h + 1]

    def _to_device_bf16(value):
        rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
        return rounded.view(torch.float32).to(torch.bfloat16)

    tensors["x_mixed"][:] = _to_device_bf16(y).reshape(t_dim, D)
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
        return torch.tensor([
            5.9166, -3.6223, -2.9324, -3.3124,
            -3.9100, -0.9384, -3.3256, -2.5240,
            2.0706, -2.5728, 0.1424, -3.9453,
            -3.8859, 3.4634, -3.3799, -2.6077,
            -2.7191, -2.4846, 2.0395, -0.5010,
            -3.5992, -2.7520, -3.3493, 3.1587,
        ])

    return [
        TensorSpec("x", [T, HC_MULT, D], torch.bfloat16, init_value=init_x),
        TensorSpec("hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
        TensorSpec("hc_scale", [3], torch.float32, init_value=init_hc_scale),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
        TensorSpec("x_mixed", [T, D], torch.bfloat16, is_output=True),
        TensorSpec("post", [T, HC_MULT], torch.float32, is_output=True),
        TensorSpec("comb", [T, HC_MULT * HC_MULT], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- hc_pre {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=hc_pre_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_hc_pre,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "post":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "comb":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
