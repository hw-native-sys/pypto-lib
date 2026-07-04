# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-dep-gen  # CI marker: full-occupancy pl.system.syncall -> dep_gen (DFX) trips 507018 (pypto#1931)
"""DeepSeek-V4 hc_pre -- hyper-connection pre-mix, fused into ONE syncall-barriered task.

x[T, hc, D] holds hc streams of the hidden state (x_flat = x.reshape(T, hc*D)). With the
projection hc_fn[mix_hc, hc*D], the per-group scales hc_scale[3] and biases hc_base[mix_hc]:

    inv_rms[T, 1]   = rsqrt(mean_k(x_flat^2) + eps)
    mixes[T, mix_hc]= inv_rms * (x_flat @ hc_fn^T)              # 1/rms deferred past the matmul
    pre [T, hc]     = sigmoid(mixes[:, :hc]   * s0 + base) + eps
    post[T, hc]     = 2 * sigmoid(mixes[:, hc:2hc] * s1 + base)
    comb[T, hc, hc] = sinkhorn(softmax(reshape(mixes[:, 2hc:] * s2 + base, hc, hc)))
    x_mixed[T, D]   = sum_h pre[:, h] * x[:, h, :]

FUSION (unified decode + prefill): the whole op is ONE ``pl.spmd(NUM_CORES=24)`` launch --
24 persistent blocks == 24 AIC + 48 AIV == every 910B core (the full-occupancy contract a
hard ``pl.system.syncall(core_type="mix")`` requires). The body is 3 phases separated by 2
barriers; each barrier publishes the prior phase's cross-core HBM writes:

    Phase A  cast (x BF16 -> x_fp32 FP32)  +  seed (zero mixes_raw)          AIV
    ── syncall ──  (x_fp32 + zeroed mixes_raw visible)
    Phase B  linear (split-K matmul -> mixes_raw, AIC)  +  rms (split-K SoS -> sq_sum_acc, AIV)
    ── syncall ──  (sq_sum_acc + all LINEAR_OK atomic-add partials visible)
    Phase D  per task: rsqrt(sq_sum) + scale/sigmoid GATE, then                    AIV
             comb_sinkhorn (-> comb) | mix_x (-> x_mixed) | write_post (-> post)

The old Phase C (a separate rsqrt+scale+sigmoid gate pass, published by a 3rd barrier) is
FOLDED into Phase D: since the C->D handoff was cross-core (different grid-stride striping),
it needed a barrier -- but each D task can just recompute the ONE gate it consumes from the
barrier-2-published mixes_raw + sq_sum_acc on its OWN core (rsqrt+sigmoid is nearly free on
this latency-bound kernel). That deletes a whole hard FFTS barrier (-8% decode / -11% prefill
on the device L2 swimlane, latency-bound so barrier removal > byte removal). comb_logits /
post_pad_store survive only as SAME-CORE scratch (assemble then load-back, no barrier) because
their downstream pl.load->pl.store needs an HC_PAD-wide 32B-aligned tile; the pre gate is
consumed in tensor-world so it needs no buffer.

Scheduling within a phase: each independent task-type is its OWN unconditional grid-stride
loop (cube work strides over ``core`` 0..23, vector work over the global AIV-lane id
``core*2 + aiv_id`` 0..47). Grid-stride IS the static round-robin schedule -- a pool larger
than the core count wraps into further waves; a smaller one leaves extra cores running a
0-trip loop that falls straight through to the next type (so a lane with no sinkhorn work
starts mix_x immediately, overlapping the sinkhorn on a busy lane). LOOP ORDER is the
scheduling policy: Phase D runs comb_sinkhorn FIRST -- its 20-iteration serial recurrence is
a latency floor no core count shortens, so starting it first overlaps it with the
throughput-bound mix_x on the other lanes.

Split-K linear atomic-adds LINEAR_OK FP32 partials into mixes_raw; Barrier 1's zero-seed +
Barrier 2's publish replace the old seed->RMW WAW edge. assemble(atomic=Add) is device-only
and the hard barrier is not modeled by the a2a3sim / a5sim simulators, so those two sim CI
checks are skipped for hc_pre -- validate on real device. mix_hc(24) pads to a 32-wide cube
N (MIX_PAD) and hc(4) to an 8-wide vector row (HC_PAD).
"""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


T_DYN = pl.dynamic("T_DYN")  # T = B * S


D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

MIX_PAD = 32  # mix_hc (24) padded to a 32-wide cube N / vector row
HC_PAD = 8  # hc (4) padded for 32B-aligned vector ops

# Full-occupancy launch: NUM_CORES persistent blocks == physical AIC count (910B = 24), so a
# hard mix-syncall reaches every AIC + AIV. Must equal the physical count exactly -- pypto does
# NOT cap the launch, and >24 makes the runtime multi-round the blocks so some physical core is
# in a later round when the barrier fires -> AICore timeout 507018.
NUM_CORES = 24

# tiling (unchanged)
T_TILE = 8  # vector row-tile (RMS / cast / split / mix_x)
LINEAR_T_TILE = 16  # cube matmul rows must be a 16-row boxed tile
COMB_T_TILE = 8  # sinkhorn row-tile
RMS_K_CHUNK = 512  # cast / rms K-fragment
LINEAR_K_CHUNK = 256  # cube K-fragment per matmul_acc (32x256x4 FP32 weight fits L0B)
D_CHUNK = 256  # mix_x inner D-fragment (BF16 load = 1KB, 512B-aligned)
D_SPMD = 1024  # mix_x D per spmd block: decode fans 4096 reduce over D/D_SPMD cores
CAST_K_SPMD = 2048  # cast K per spmd block: decode fans the BF16->FP32 cast over HC_DIM/CAST_K_SPMD cores
# Split the K=HC_DIM reduction into LINEAR_OK slices that atomic-add their FP32
# partials, filling idle cubes at small T (decode: 1 token-tile -> LINEAR_OK
# cube tasks) and shortening each task's matmul_acc chain. Higher OK fills more
# decode cubes; prefill (8 token-tiles) packs OK*8 tasks into waves of ~24.
LINEAR_OK = 16
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // LINEAR_K_CHUNK

# Split the RMS sum-of-squares K reduction over RMS_OK cores, mirroring LINEAR_OK: at decode
# (1 token-tile) the full 16384-wide reduce is otherwise a single-lane straggler. Each
# (token-tile, K-slice) task atomic-adds its FP32 partial sum-of-squares into sq_sum_acc
# (zero-seeded in Phase A); Phase C reads the barrier-published total and applies rsqrt inline
# per group (no separate inv_rms buffer / no within-phase RAW).
RMS_OK = 16
RMS_K_PER_SPLIT = HC_DIM // RMS_OK
RMS_CHUNKS_PER_SPLIT = RMS_K_PER_SPLIT // RMS_K_CHUNK

# per-phase fan-out factors (compile-time constants; the token-tile factor is dynamic in T)
CAST_KS = HC_DIM // CAST_K_SPMD  # cast tasks per token-tile (8)
MIXX_DS = D // D_SPMD  # mix_x tasks per token-tile (4)

assert HC_MULT == 4, (
    f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}; "
    "regenerate the pre0..pre3 / row0..row3 unrolling before using it."
)
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % LINEAR_T_TILE == 0
assert DECODE_BATCH * DECODE_SEQ <= LINEAR_T_TILE
assert HC_DIM % LINEAR_K_CHUNK == 0
assert HC_DIM % CAST_K_SPMD == 0 and CAST_K_SPMD % RMS_K_CHUNK == 0
assert HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0
assert D % D_SPMD == 0 and D_SPMD % D_CHUNK == 0


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
    t_dim = pl.tensor.dim(x, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_reshaped = pl.reshape(hc_base, [1, MIX_HC])

    # Cross-barrier intermediates: allocated ONCE in GM/HBM, live across the phases.
    x_fp32 = pl.create_tensor([t_linear, HC_DIM], dtype=pl.FP32)  # FP32 activations for the pure-AIC matmul + rms
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    # comb_logits / post_pad_store are now SAME-CORE scratch: the sinkhorn / write_post task
    # writes its token-tile's gate then reads it back on the SAME core (no barrier). They are
    # kept (not deleted) because both feed a downstream pl.load->pl.store that needs an
    # HC_PAD-wide (32B-aligned) tile -- a narrow [T_TILE, HC_MULT] FP32 tile is 16B rows and
    # pto.alloc_tile rejects it. pre (mix_x) is consumed in tensor-world, so it needs no buffer.
    post_pad_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)  # HC_PAD-wide same-core scratch; write_post narrows to post
    # MIX_PAD (not HC_MULT*HC_MULT=16): comb_sinkhorn loads each group HC_PAD-wide at
    # offset k*HC_MULT, so group 3 reads cols [12:20] -- a 16-wide alloc puts that load
    # descriptor out of bounds. The 32-wide alloc keeps every descriptor in-bounds.
    comb_logits = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    sq_sum_acc = pl.create_tensor([1, t_linear], dtype=pl.FP32)  # RMS split-K sum-of-squares (row layout: [1,T_TILE] tiles stay 32B-aligned)

    # Per-phase grid-stride bounds (dynamic in t_dim; grid-stride round-robins any T over cores).
    tt_n = t_dim // T_TILE            # token-tiles (pre / post / comb / rsqrt / sinkhorn / write_post base)
    cast_n = tt_n * CAST_KS           # cast fans over token-tile x K-slice
    seed_n = t_linear // T_TILE       # seed zeros t_linear rows (includes the 8->16 pad rows)
    lin_n = (t_linear // LINEAR_T_TILE) * LINEAR_OK  # linear fans over row-block x OK K-slice
    rms_n = tt_n * RMS_OK             # rms fans over token-tile x K-slice (split-K sum-of-squares)
    mixx_n = tt_n * MIXX_DS           # mix_x fans over token-tile x D-slice
    pool_d = 2 * tt_n + mixx_n        # phase-D flattened pool: sinkhorn(tt_n)|mix_x(mixx_n)|write_post(tt_n)

    with pl.spmd(NUM_CORES, name_hint="hc_pre_fused", sync_start=True) as _hc_tid:  # inline form requires the TaskId capture
        core = pl.tile.get_block_idx()  # 0 .. NUM_CORES-1

        # ===================== PHASE A: cast (AIV) + seed (AIV) =====================
        # Both AIV, independent. Barrier 1 publishes x_fp32 and the zeroed mixes_raw.
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id  # 0..47
            for blk in pl.range(lane, cast_n, NUM_CORES * 2):
                t0 = (blk // CAST_KS) * T_TILE
                k_base = (blk % CAST_KS) * CAST_K_SPMD
                for kb in pl.pipeline(CAST_K_SPMD // RMS_K_CHUNK, stage=4):
                    k0 = k_base + kb * RMS_K_CHUNK
                    x_chunk = pl.cast(x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK], target_type=pl.FP32)
                    x_fp32 = pl.assemble(x_fp32, x_chunk, [t0, k0])
            for tc in pl.range(lane, seed_n, NUM_CORES * 2):
                ts0 = tc * T_TILE
                mixes_raw = pl.assemble(mixes_raw, pl.full([T_TILE, MIX_PAD], dtype=pl.FP32, value=0.0), [ts0, 0])
                sq_sum_acc = pl.assemble(sq_sum_acc, pl.full([1, T_TILE], dtype=pl.FP32, value=0.0), [0, ts0])
        pl.system.syncall(core_type="mix")

        # ===================== PHASE B: linear (AIC) + rms (AIV) ====================
        # Cube split-K matmul strides over the 24 AIC; vector RMS over the 48 AIV lanes,
        # concurrently. Barrier 2 publishes inv_rms + every LINEAR_OK atomic-add partial.
        for task in pl.range(core, lin_n, NUM_CORES):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            k_base = (task % LINEAR_OK) * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)  # last row-block spills past t_dim; valid_shape zero-fills the tail
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
                k0 = k_base + kb * LINEAR_K_CHUNK
                x_linear_chunk = pl.slice(x_fp32, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, k0], valid_shape=[t_rows, LINEAR_K_CHUNK])
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_CHUNK], [0, k0], valid_shape=[MIX_HC, LINEAR_K_CHUNK])
                if kb == 0:
                    acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
            mixes_raw = pl.assemble(mixes_raw, acc, [t0, 0], atomic=pl.AtomicType.Add)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for task in pl.range(lane, rms_n, NUM_CORES * 2):
                t0 = (task // RMS_OK) * T_TILE
                k_base = (task % RMS_OK) * RMS_K_PER_SPLIT
                sq_part = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(RMS_CHUNKS_PER_SPLIT, stage=4):
                    k0 = k_base + kb * RMS_K_CHUNK
                    x_chunk = x_fp32[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK]
                    sq_part = pl.add(sq_part, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]))
                sq_sum_acc = pl.assemble(sq_sum_acc, sq_part, [0, t0], atomic=pl.AtomicType.Add)
        pl.system.syncall(core_type="mix")

        # ===================== PHASE C+D FUSED: gate + sinkhorn/mix_x/write_post (AIV) =====
        # Phase C (rsqrt + scale + sigmoid gates) is FOLDED into each Phase-D task: every
        # task recomputes the gate it needs from the barrier-2-published mixes_raw + sq_sum_acc
        # on its OWN core, just before using it. Recompute is nearly free here (latency-bound
        # kernel), and it deletes an entire cross-core barrier plus the pre/post/comb gate
        # round-trips -- the C->D handoff was the reason the old Barrier 3 existed. comb_logits
        # stays only as SAME-CORE scratch (assemble then load back, no barrier).
        #
        # The three kinds share ONE flattened work pool so they run on DISJOINT cores in
        # parallel -- otherwise per-type loops all start at the same lane base, stacking
        # sinkhorn (a 20-iter serial-recurrence latency floor) + that lane's mix_x/write_post
        # onto lane 0 while the other lanes idle. sinkhorn is placed first in the pool so its
        # long chain starts immediately alongside the mix_x lanes. No trailing barrier
        # (kernel end publishes the outputs).
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            # ONE flattened pool [sinkhorn(tt_n) | mix_x(mixx_n) | write_post(tt_n)] so grid-
            # stride lands the three kinds on DISJOINT lanes -- sinkhorn, all mix_x D-slices,
            # and write_post run on different cores concurrently (not serialized per lane).
            for gw in pl.range(lane, pool_d, NUM_CORES * 2):
                if gw < tt_n:
                    t0 = gw * COMB_T_TILE
                    # Fold Phase C's comb-logit gate here: rsqrt(sq_sum) inline, scale the raw
                    # mix by inv * scale2, add the group bias -> comb_logits[t0]. Written and
                    # read back on THIS core (no barrier); the tile-world sinkhorn below is
                    # unchanged (still loads per-group padded tiles from comb_logits).
                    ssq_row = sq_sum_acc[0:1, t0:t0 + COMB_T_TILE]  # [1,T_TILE] row keeps the rsqrt tile 32B-aligned
                    inv_col = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [COMB_T_TILE, 1])
                    comb_base = hc_reshaped[0:1, HC_MULT * 2:HC_MULT * 2 + HC_MULT * HC_MULT]
                    comb_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + COMB_T_TILE, HC_MULT * 2:HC_MULT * 2 + HC_MULT * HC_MULT], inv_col), scale2)
                    comb_logits_tile = pl.add(comb_scaled, pl.col_expand(comb_scaled, comb_base))
                    comb_logits = pl.assemble(comb_logits, comb_logits_tile, [t0, 0])
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

                elif gw < tt_n + mixx_n:
                    blk = gw - tt_n
                    t0 = (blk // MIXX_DS) * T_TILE
                    d_base = (blk % MIXX_DS) * D_SPMD
                    # Fold Phase C's pre gate: pre = sigmoid(mix[:, :hc] * inv * scale0 + base) + eps.
                    # Recomputed here from mixes_raw (every D-slice of a token-tile redoes this
                    # tiny [T_TILE, HC_PAD] sigmoid -- free, and it drops the pre_val HBM buffer).
                    ssq_row = sq_sum_acc[0:1, t0:t0 + T_TILE]  # [1,T_TILE] row keeps the rsqrt tile 32B-aligned
                    inv_col = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [T_TILE, 1])
                    pre_base = hc_reshaped[0:1, 0:HC_PAD]
                    pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
                    pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
                    pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
                    # pin the valid shape: the inline gate carries valid=?x? from the dynamic-
                    # offset mixes_raw slice, but the transpose below wants a static 8x8 tile.
                    pre_val = pl.set_validshape(pre_val, T_TILE, HC_PAD)
                    pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
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
                        x_mixed = pl.assemble(x_mixed, pl.cast(y_tile, target_type=pl.BF16, mode="rint"), [t0, d0])

                else:
                    ob = gw - tt_n - mixx_n
                    t0 = ob * COMB_T_TILE
                    # Fold Phase C's post gate: post = 2 * sigmoid(mix[:, hc:2hc] * inv * scale1 + base).
                    # The gate is tensor-world (mixes_raw slice -> sigmoid), and pl.store needs a Vec
                    # TILE -- so post_pad is stashed HC_PAD-wide in same-core scratch and loaded back as
                    # an aligned [COMB_T_TILE, HC_PAD] tile (a direct [.,HC_MULT] store would need a
                    # 16B-row FP32 tile, which pto.alloc_tile rejects). pl.store then narrows valid
                    # [COMB_T_TILE, HC_MULT] into the 4-wide post output.
                    ssq_row = sq_sum_acc[0:1, t0:t0 + COMB_T_TILE]  # [1,COMB_T_TILE] row keeps the rsqrt tile 32B-aligned
                    inv_col = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [COMB_T_TILE, 1])
                    post_base = hc_reshaped[0:1, HC_MULT:HC_MULT + HC_PAD]
                    post_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0:t0 + COMB_T_TILE, HC_MULT:HC_MULT + HC_PAD], inv_col), scale1)
                    post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
                    post_pad = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
                    post_pad_store = pl.assemble(post_pad_store, post_pad, [t0, 0])
                    post_tile = pl.load(post_pad_store, [t0, 0], [COMB_T_TILE, HC_PAD],
                                        valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    pl.store(post_tile, [t0, 0], post)
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
    parser.add_argument("--no-dep-gen", action="store_true", default=False,
                        help="deprecated no-op: dep_gen is already forced off for hc_pre "
                             "(full-occupancy syncall is incompatible with dep_gen, pypto#1931); "
                             "kept for CLI / CI back-compat.")
    args = parser.parse_args()

    # hc_pre's fused body is specialized to Ascend 910B: NUM_CORES=24 IS the physical AIC
    # count, and the hard full-occupancy mix-syncall hangs (AICore timeout 507018) unless the
    # launch fills every physical core of the SoC. A5 (Ascend950) has a different core count,
    # so 24 blocks would leave physical participants unreached -> reject it here rather than
    # hang. Supporting A5 needs a backend-aware participant count + re-tuning/re-validation.
    if args.platform in ("a5", "a5sim"):
        raise SystemExit(
            f"hc_pre is specialized to Ascend 910B (NUM_CORES={NUM_CORES} == physical AIC count); "
            f"its full-occupancy mix-syncall would hang on {args.platform!r}. Run with -p a2a3."
        )

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
                # dep_gen forced OFF: this kernel's full-occupancy pl.system.syncall is
                # incompatible with dep_gen -- the DFX instrumentation perturbs core
                # occupancy and trips AICore timeout 507018 (pypto#1931).
                enable_dep_gen=False,
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
