# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-dep-gen  # CI marker: full-occupancy pl.system.syncall -> dep_gen (DFX) trips 507018 (pypto#1931)
"""DeepSeek-V4 hc_pre -- hyper-connection pre-mix with fused and separate-task implementations.

x[T, hc, D] holds hc streams of the hidden state (x_flat = x.reshape(T, hc*D)). With the
projection hc_fn[mix_hc, hc*D], the per-group scales hc_scale[3] and biases hc_base[mix_hc]:

    inv_rms[T, 1]   = rsqrt(mean_k(x_flat^2) + eps)
    mixes[T, mix_hc]= inv_rms * (x_flat @ hc_fn^T)              # 1/rms deferred past the matmul
    pre [T, hc]     = sigmoid(mixes[:, :hc]   * s0 + base) + eps
    post[T, hc]     = 2 * sigmoid(mixes[:, hc:2hc] * s1 + base)
    comb[T, hc, hc] = sinkhorn(softmax(reshape(mixes[:, 2hc:] * s2 + base, hc, hc)))
    x_mixed[T, D]   = sum_h pre[:, h] * x[:, h, :]

TWO IMPLEMENTATIONS, selected by the ``HC_PRE_IMPL`` module flag (env ``DSV4_HC_PRE_IMPL``
or ``--impl``); both are UNIFIED over decode + prefill and compute the identical math above:
  * ``_hc_pre_syncall`` -- the #684 fusion documented below: ONE full-occupancy
    ``pl.spmd(24)`` with 2 hard ``pl.system.syncall`` barriers, run with dep_gen OFF.
  * ``_hc_pre_separate`` (default) -- each work-type is its OWN ``pl.spmd`` task
    (rms / linear / deterministic linear reduce / vector pre / vector post /
    split_pre_post / write_post / comb_sinkhorn / mix_x), ordered by the runtime
    task graph, with tile sizes aligned to the syncall path. The standalone harness
    enables dep-gen to capture that graph.
The rest of this docstring describes the ``_hc_pre_syncall`` fusion.

FUSION (unified decode + prefill): the whole op is ONE ``pl.spmd(NUM_CORES=24)`` launch --
24 persistent blocks == 24 AIC + 48 AIV == every 910B core (the full-occupancy contract a
hard ``pl.system.syncall(core_type="mix")`` requires). The body is 3 phases separated by 2
barriers; each barrier publishes the prior phase's cross-core HBM writes:

    Phase B  linear (split-K matmul -> disjoint partials, AIC) +
             rms (split-K SoS -> disjoint partials, AIV)
    ── syncall ──  (all split-K partials visible)
    Phase C  reduce linear and RMS partials in ascending K-split order      AIV
    ── syncall ──  (mixes_raw + sq_sum_acc visible)
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

Split-K linear and RMS tasks write disjoint FP32 partial buffers; a fixed ascending-split
reduction makes results independent of task completion order. The hard barrier is not modeled
by the a2a3sim / a5sim simulators, so those two sim CI checks are skipped for hc_pre -- validate
on real device. mix_hc(24) pads to a 32-wide cube N (MIX_PAD) and hc(4) to an 8-wide vector row
(HC_PAD).
"""


import os

import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


T_DYN = pl.dynamic("T_DYN")  # T = B * S

# Implementation selector (both versions are UNIFIED -- one code path for decode AND
# prefill; no separate decode/prefill dispatch):
#   "syncall" -- the #684 fused kernel: ONE full-occupancy pl.spmd(24) with
#               2 hard pl.system.syncall barriers; runs with dep_gen OFF (pypto#1931).
#   "separate" (default) -- each work-type is its own pl.spmd task, ordered by the
#               runtime task graph; dep-gen may capture those dependencies for DFX.
# Switch via env DSV4_HC_PRE_IMPL={syncall,separate} or by reassigning this module global
# before the kernel is traced (the __main__ below wires it to --impl).
HC_PRE_IMPL = os.environ.get("DSV4_HC_PRE_IMPL", "separate").lower()


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
T_TILE = 8  # vector row-tile (RMS / split / mix_x)
LINEAR_T_TILE = 16  # cube matmul rows must be a 16-row boxed tile
COMB_T_TILE = 8  # sinkhorn row-tile
# RMS K-fragment. 256 (not 512) because PRO's HC_DIM = 4 * 7168 = 28672
# factors as 4096 * 7: with RMS_OK = 16 the per-split K is 1792, which 512 does
# not divide. Keeping RMS_OK at 16 preserves the decode fan-out this file is
# tuned around, so the fragment is the knob that moves. 256 also matches
# LINEAR_K_CHUNK / D_CHUNK.
RMS_K_CHUNK = 256
LINEAR_K_CHUNK = 256  # cube K-fragment per matmul_acc (32x256x4 FP32 weight fits L0B)
D_CHUNK = 256  # mix_x inner D-fragment (BF16 load = 1KB, 512B-aligned)
D_SPMD = 1024  # mix_x D per spmd block: decode fans 4096 reduce over D/D_SPMD cores
# Split the K=HC_DIM reduction into LINEAR_OK disjoint partials.
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK
LINEAR_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // LINEAR_K_CHUNK

# Split the RMS sum-of-squares K reduction into RMS_OK disjoint partials.
RMS_OK = 16
RMS_K_PER_SPLIT = HC_DIM // RMS_OK
RMS_CHUNKS_PER_SPLIT = RMS_K_PER_SPLIT // RMS_K_CHUNK

# Per-phase fan-out factor (the token-tile factor is dynamic in T).
MIXX_DS = D // D_SPMD  # mix_x tasks per token-tile (PRO: 7168/1024 = 7)

assert HC_MULT == 4, (
    f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}; "
    "regenerate the pre0..pre3 / row0..row3 unrolling before using it."
)
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % LINEAR_T_TILE == 0
assert DECODE_BATCH * DECODE_SEQ <= LINEAR_T_TILE
assert HC_DIM % LINEAR_OK == 0
assert LINEAR_K_PER_SPLIT % LINEAR_K_CHUNK == 0
assert HC_DIM % RMS_OK == 0 and RMS_K_PER_SPLIT % RMS_K_CHUNK == 0
assert D % D_SPMD == 0 and D_SPMD % D_CHUNK == 0


@pl.jit.inline
def _hc_pre_syncall(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    # x arrives as FP32 (hc residual stream is FP32 end-to-end), so there is no x_fp32
    # staging buffer: linear / rms read x_flat directly.
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    linear_partial_rows = LINEAR_OK * t_linear
    mixes_partials = pl.create_tensor([linear_partial_rows, MIX_PAD], dtype=pl.FP32)
    # post_pad_store is SAME-CORE scratch: write_post writes its token-tile's post gate then
    # reads it back on the SAME core (no barrier). It is kept (not deleted) because its
    # downstream pl.load->pl.store needs an HC_PAD-wide (32B-aligned) tile -- a narrow
    # [T_TILE, HC_MULT] FP32 tile is 16B rows and pto.alloc_tile rejects it. pre (mix_x) and
    # comb (loaded straight from mixes_raw) are consumed in-tile, so they need no scratch.
    post_pad_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)  # HC_PAD-wide same-core scratch; write_post narrows to post
    # inv_rms same-core scratch for the comb gate: the per-token inv is a [T_TILE,1] COLUMN,
    # which the comb groups (Tiles) need as a Tile too. pto rejects a transposed [T_TILE,1]
    # tile (row-major, 4B row < 32B) and forbids Tile x Tensor ops, so the tensor-world inv
    # is spilled to this [t_linear,1] buffer and loaded back as a [T_TILE,1] Tile (8 floats,
    # vs the 128-float [.,16] comb_logits round-trip this replaces).
    inv_gm = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    sq_sum_acc = pl.create_tensor([1, t_linear], dtype=pl.FP32)
    sq_partials = pl.create_tensor([RMS_OK, t_linear], dtype=pl.FP32)

    # Per-phase grid-stride bounds (dynamic in t_dim; grid-stride round-robins any T over cores).
    tt_n = t_dim // T_TILE            # token-tiles (pre / post / comb / rsqrt / sinkhorn / write_post base)
    lin_n = (t_linear // LINEAR_T_TILE) * LINEAR_OK  # linear fans over row-block x OK K-slice
    rms_n = tt_n * RMS_OK             # rms fans over token-tile x K-slice (split-K sum-of-squares)
    linear_reduce_n = t_linear // LINEAR_T_TILE
    reduce_n = linear_reduce_n + tt_n
    mixx_n = tt_n * MIXX_DS           # mix_x fans over token-tile x D-slice
    pool_d = 2 * tt_n + mixx_n        # phase-D flattened pool: sinkhorn(tt_n)|mix_x(mixx_n)|write_post(tt_n)

    with pl.spmd(NUM_CORES, name_hint="hc_pre_fused", sync_start=True, allow_early_resolve=True) as _hc_tid:  # inline form requires the TaskId capture
        core = pl.tile.get_block_idx()  # 0 .. NUM_CORES-1

        # Cube linear partials and vector RMS partials run concurrently.
        for task in pl.range(core, lin_n, NUM_CORES):
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            linear_split = task % LINEAR_OK
            k_base = linear_split * LINEAR_K_PER_SPLIT
            t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)  # last row-block spills past t_dim; valid_shape zero-fills the tail
            acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
                k0 = k_base + kb * LINEAR_K_CHUNK
                x_linear_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, k0], valid_shape=[t_rows, LINEAR_K_CHUNK])
                w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_CHUNK], [0, k0], valid_shape=[MIX_HC, LINEAR_K_CHUNK])
                if kb == 0:
                    acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
            mixes_partials = pl.assemble(
                mixes_partials,
                acc,
                [linear_split * t_linear + t0, 0],
            )
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for task in pl.range(lane, rms_n, NUM_CORES * 2):
                t0 = (task // RMS_OK) * T_TILE
                rms_split = task % RMS_OK
                k_base = rms_split * RMS_K_PER_SPLIT
                sq_part = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
                for kb in pl.pipeline(RMS_CHUNKS_PER_SPLIT, stage=4):
                    k0 = k_base + kb * RMS_K_CHUNK
                    x_chunk = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK]
                    sq_part = pl.add(sq_part, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]))
                sq_partials = pl.assemble(sq_partials, sq_part, [rms_split, t0])
        pl.system.syncall(core_type="mix")

        # Split-K partials are reduced in ascending K order.
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane = core * 2 + aiv_id
            for reduce_task in pl.range(lane, reduce_n, NUM_CORES * 2):
                if reduce_task < linear_reduce_n:
                    linear_t0 = reduce_task * LINEAR_T_TILE
                    mixes_total = mixes_partials[
                        linear_t0 : linear_t0 + LINEAR_T_TILE,
                        0:MIX_PAD,
                    ]
                    for linear_split in pl.range(1, LINEAR_OK):
                        partial_t0 = linear_split * t_linear + linear_t0
                        mixes_total = pl.add(
                            mixes_total,
                            mixes_partials[
                                partial_t0 : partial_t0 + LINEAR_T_TILE,
                                0:MIX_PAD,
                            ],
                        )
                    mixes_raw = pl.assemble(mixes_raw, mixes_total, [linear_t0, 0])
                else:
                    rms_t0 = (reduce_task - linear_reduce_n) * T_TILE
                    sq_total = sq_partials[0:1, rms_t0 : rms_t0 + T_TILE]
                    for rms_split in pl.range(1, RMS_OK):
                        sq_total = pl.add(
                            sq_total,
                            sq_partials[rms_split : rms_split + 1, rms_t0 : rms_t0 + T_TILE],
                        )
                    sq_sum_acc = pl.assemble(sq_sum_acc, sq_total, [0, rms_t0])
        pl.system.syncall(core_type="mix")

        # ===================== PHASE C+D FUSED: gate + sinkhorn/mix_x/write_post (AIV) =====
        # Phase C (rsqrt + scale + sigmoid gates) is FOLDED into each Phase-D task: every
        # task recomputes the gate it needs from the published mixes_raw + sq_sum_acc
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
                    # Fold Phase C's comb gate here AND skip the comb_logits GM round-trip:
                    # load the 4 comb groups DIRECTLY from mixes_raw (pad-capable loads at cols
                    # 8/12/16/20 within the 32-wide MIX_PAD row), then apply inv_rms * scale2 +
                    # group bias PER GROUP in-tile. The previous version assembled a [.,16]
                    # comb_logits tile to GM and loaded the 4 misaligned 4-wide groups back
                    # (pto cannot slice a 16B sub-tile in-register); loading straight from the
                    # matmul output drops that store+load, matching the prefill path.
                    # inv_rms for the comb gate must be a TILE (the comb groups below are
                    # Tiles). Compute it in tensor-world, spill to inv_gm, and load it back as a
                    # [T_TILE,1] Tile (a transposed [T_TILE,1] tile is rejected -- 4B row).
                    ssq_row = sq_sum_acc[0:1, t0:t0 + COMB_T_TILE]  # [1,T_TILE] tensor
                    inv_col_tensor = pl.reshape(pl.rsqrt(pl.add(pl.mul(ssq_row, HC_DIM_INV), NORM_EPS), high_precision=True), [COMB_T_TILE, 1])
                    inv_gm = pl.assemble(inv_gm, inv_col_tensor, [t0, 0])
                    inv_col_t = pl.load(inv_gm, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)  # [T_TILE,1] tile
                    comb_off = HC_MULT * 2
                    mix_g0 = pl.load(mixes_raw, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    mix_g1 = pl.load(mixes_raw, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    mix_g2 = pl.load(mixes_raw, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    mix_g3 = pl.load(mixes_raw, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    cb0 = pl.load(hc_reshaped, [0, comb_off + 0 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    cb1 = pl.load(hc_reshaped, [0, comb_off + 1 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    cb2 = pl.load(hc_reshaped, [0, comb_off + 2 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    cb3 = pl.load(hc_reshaped, [0, comb_off + 3 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
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

                    row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
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
                        x0 = x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_CHUNK]
                        x1 = x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_CHUNK]
                        x2 = x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_CHUNK]
                        x3 = x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_CHUNK]
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
                                        valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
                    pl.store(post_tile, [t0, 0], post)
    return x_mixed


@pl.jit.inline
def _hc_pre_separate(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    """Multi-scope (separate-task) hc_pre -- the pre-#684 structure, applied to ALL T.

    Identical math to _hc_pre_syncall, but each work-type is its OWN pl.spmd task instead
    of one full-occupancy pl.spmd + hard pl.system.syncall barriers. The runtime task
    graph orders the scopes by their GM read/write dependencies (linear -> linear reduce ->
    split_pre_post -> write_post / comb_sinkhorn / mix_x). The scheduler derives that ordering
    from GM dependencies; the standalone harness enables dep-gen only to capture the graph.
    Tile sizes are aligned to the
    tuned syncall version (D_CHUNK / D_SPMD / LINEAR_* / t_linear round-up); cross-barrier
    buffers are sized to the dynamic t_linear (the 8->16 padded row count), not a static
    T_MAX. Kept as a switchable alternative to the fused kernel (perf is not the goal here).
    """
    t_dim = pl.tensor.dim(x, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # for per-group comb base loads in comb_sinkhorn

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    # x arrives as FP32 from the prior hc_post (the hc residual stream is FP32 end-to-end),
    # so the old BF16->FP32 cast scope + x_fp32 staging buffer are gone: linear / rms read
    # x_flat directly. The 8->16 pad rows of the cube tile are never materialized -- the
    # linear matmul masks them with valid_shape (zero-fill past t_dim), and rms only reads
    # the t_dim real rows.
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    linear_partial_rows = LINEAR_OK * t_linear
    mixes_partials = pl.create_tensor([linear_partial_rows, MIX_PAD], dtype=pl.FP32)

    # rms: full-K sum-of-squares per token-tile -> inv_rms (one scope, no split-K).
    for t in pl.spmd(t_dim // T_TILE, name_hint="hc_pre_rms", allow_early_resolve=True):
        t0 = t * T_TILE
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_CHUNK, stage=4):
            k0 = kb * RMS_K_CHUNK
            x_chunk = x_flat[t0:t0 + T_TILE, k0:k0 + RMS_K_CHUNK]
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]))
        inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS), high_precision=True), [T_TILE, 1])
        inv_rms = pl.assemble(inv_rms, inv, [t0, 0])

    # Linear split-K partials are reduced in ascending K order.
    for task in pl.spmd((t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="hc_pre_linear", allow_early_resolve=True):
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        linear_split = task % LINEAR_OK
        k_base = linear_split * LINEAR_K_PER_SPLIT
        t_rows = pl.min(LINEAR_T_TILE, t_dim - t0)  # last row-block spills past t_dim; valid_shape zero-fills the tail
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, LINEAR_CHUNKS_PER_SPLIT, stage=2):
            k0 = k_base + kb * LINEAR_K_CHUNK
            x_linear_chunk = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_CHUNK], [t0, k0], valid_shape=[t_rows, LINEAR_K_CHUNK])
            w_chunk = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_CHUNK], [0, k0], valid_shape=[MIX_HC, LINEAR_K_CHUNK])
            if kb == 0:
                acc = pl.matmul(x_linear_chunk, w_chunk, b_trans=True, out_dtype=pl.FP32)
            else:
                acc = pl.matmul_acc(acc, x_linear_chunk, w_chunk, b_trans=True)
        mixes_partials = pl.assemble(mixes_partials, acc, [linear_split * t_linear + t0, 0])

    for linear_block in pl.spmd(
        t_linear // LINEAR_T_TILE,
        name_hint="hc_pre_linear_reduce",
        allow_early_resolve=True,
    ):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            mixes_total = pl.add(
                mixes_total,
                mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD],
            )
        mixes_raw = pl.assemble(mixes_raw, mixes_total, [linear_t0, 0])

    # The pre gate uses an explicit A5 FP32 vector reduction, one token per task.
    # Each K256 fragment is reduced by TROWSUM and the fragment results accumulate
    # in ascending K order. Keep this scope independent from the matching post
    # projection: the proven schedule has one scope for rows 0..3 and one for 4..7.
    pre_raw_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    for task in pl.spmd(t_dim, name_hint="hc_pre_pre_linear_vector", allow_early_resolve=True):
        trow = task
        dot_out = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // LINEAR_K_CHUNK, stage=1):
            k0 = kb * LINEAR_K_CHUNK
            pre_x_chunk = pl.load(
                x_flat,
                [trow, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            pre_w0_chunk = pl.load(
                hc_fn,
                [0, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            pre_w1_chunk = pl.load(
                hc_fn,
                [1, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            pre_w2_chunk = pl.load(
                hc_fn,
                [2, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            pre_w3_chunk = pl.load(
                hc_fn,
                [3, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            pre_row_sum_tmp = pl.tile.create(
                [8, LINEAR_K_CHUNK],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            chunk0 = pl.row_sum(pl.mul(pre_x_chunk, pre_w0_chunk), pre_row_sum_tmp)
            chunk1 = pl.row_sum(pl.mul(pre_x_chunk, pre_w1_chunk), pre_row_sum_tmp)
            chunk2 = pl.row_sum(pl.mul(pre_x_chunk, pre_w2_chunk), pre_row_sum_tmp)
            chunk3 = pl.row_sum(pl.mul(pre_x_chunk, pre_w3_chunk), pre_row_sum_tmp)
            chunk0_row = pl.set_validshape(pl.reshape(chunk0, [1, 8]), 1, 1)
            chunk1_row = pl.set_validshape(pl.reshape(chunk1, [1, 8]), 1, 1)
            chunk2_row = pl.set_validshape(pl.reshape(chunk2, [1, 8]), 1, 1)
            chunk3_row = pl.set_validshape(pl.reshape(chunk3, [1, 8]), 1, 1)
            chunk_vec_init = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
            chunk_vec0 = pl.tile.assemble(chunk_vec_init, chunk0_row, [0, 0])
            chunk_vec1 = pl.tile.assemble(chunk_vec0, chunk1_row, [1, 0])
            chunk_vec2 = pl.tile.assemble(chunk_vec1, chunk2_row, [2, 0])
            chunk_vec3 = pl.tile.assemble(chunk_vec2, chunk3_row, [3, 0])
            dot_out = pl.add(dot_out, chunk_vec3)

        pre_raw_row = pl.transpose(dot_out, axis1=0, axis2=1)
        pre_raw_valid = pl.set_validshape(pre_raw_row, 1, HC_PAD)
        pl.store(pre_raw_valid, [trow, 0], pre_raw_store)

    post_raw_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    for task in pl.spmd(t_dim, name_hint="hc_pre_post_linear_vector", allow_early_resolve=True):
        trow = task
        dot_out = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // LINEAR_K_CHUNK, stage=1):
            k0 = kb * LINEAR_K_CHUNK
            post_x_chunk = pl.load(
                x_flat,
                [trow, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            post_w0_chunk = pl.load(
                hc_fn,
                [HC_MULT + 0, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            post_w1_chunk = pl.load(
                hc_fn,
                [HC_MULT + 1, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            post_w2_chunk = pl.load(
                hc_fn,
                [HC_MULT + 2, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            post_w3_chunk = pl.load(
                hc_fn,
                [HC_MULT + 3, k0],
                [8, LINEAR_K_CHUNK],
                valid_shape=[1, LINEAR_K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            post_row_sum_tmp = pl.tile.create(
                [8, LINEAR_K_CHUNK],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            chunk0 = pl.row_sum(pl.mul(post_x_chunk, post_w0_chunk), post_row_sum_tmp)
            chunk1 = pl.row_sum(pl.mul(post_x_chunk, post_w1_chunk), post_row_sum_tmp)
            chunk2 = pl.row_sum(pl.mul(post_x_chunk, post_w2_chunk), post_row_sum_tmp)
            chunk3 = pl.row_sum(pl.mul(post_x_chunk, post_w3_chunk), post_row_sum_tmp)
            chunk0_row = pl.set_validshape(pl.reshape(chunk0, [1, 8]), 1, 1)
            chunk1_row = pl.set_validshape(pl.reshape(chunk1, [1, 8]), 1, 1)
            chunk2_row = pl.set_validshape(pl.reshape(chunk2, [1, 8]), 1, 1)
            chunk3_row = pl.set_validshape(pl.reshape(chunk3, [1, 8]), 1, 1)
            chunk_vec_init = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
            chunk_vec0 = pl.tile.assemble(chunk_vec_init, chunk0_row, [0, 0])
            chunk_vec1 = pl.tile.assemble(chunk_vec0, chunk1_row, [1, 0])
            chunk_vec2 = pl.tile.assemble(chunk_vec1, chunk2_row, [2, 0])
            chunk_vec3 = pl.tile.assemble(chunk_vec2, chunk3_row, [3, 0])
            dot_out = pl.add(dot_out, chunk_vec3)

        post_raw_row = pl.transpose(dot_out, axis1=0, axis2=1)
        post_raw_valid = pl.set_validshape(post_raw_row, 1, HC_PAD)
        pl.store(post_raw_valid, [trow, 0], post_raw_store)

    # split_pre_post: inv_rms-scaled pre gate -> pre_val_store (for mix_x), post gate -> post.
    # Both compute at HC_PAD width; post narrows to HC_MULT via a valid-shape slice (an 8-wide
    # 32B tile, 4 cols valid -- a bare 4-wide slice allocs a 16B tile ptoas rejects). comb gate
    # lives in comb_sinkhorn.
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    for ob in pl.spmd(t_dim // T_TILE, name_hint="split_pre_post", allow_early_resolve=True):
        t0 = ob * T_TILE

        pre_inv = pl.load(
            inv_rms,
            [t0, 0],
            [T_TILE, 1],
            target_memory=pl.MemorySpace.Vec,
        )
        pre_base = pl.load(
            hc_base_2d,
            [0, 0],
            [1, HC_PAD],
            target_memory=pl.MemorySpace.Vec,
        )
        pre_mixes = pl.load(
            pre_raw_store,
            [t0, 0],
            [T_TILE, HC_PAD],
            valid_shape=[T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        pre_scaled = pl.mul(pl.row_expand_mul(pre_mixes, pre_inv), scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_valid = pl.set_validshape(pre_val, T_TILE, HC_MULT)
        pl.store(pre_valid, [t0, 0], pre_val_store)

        post_mixes = pl.load(
            post_raw_store,
            [t0, 0],
            [T_TILE, HC_PAD],
            valid_shape=[T_TILE, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        post_inv = pl.load(inv_rms, [t0, 0], [T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        post_base = pl.load(hc_base_2d, [0, HC_MULT], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)
        post_scaled = pl.mul(pl.row_expand_mul(post_mixes, post_inv), scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_pad = pl.mul(post_sig, 2.0)
        post_valid = pl.set_validshape(post_pad, T_TILE, HC_MULT)
        pl.store(post_valid, [t0, 0], post)

    # comb_sinkhorn: comb gate (direct from mixes_raw, no comb_logits round-trip) + softmax +
    # 20-iter Sinkhorn (column-first) -> comb. inv_rms is already a [t_linear,1] column buffer,
    # so the [T_TILE,1] inv tile loads directly (no transpose / no spill); the 4 comb groups
    # load pad-capable from mixes_raw at cols 8/12/16/20, then inv_rms * scale2 + group bias.
    for ob in pl.spmd(t_dim // COMB_T_TILE, name_hint="comb_sinkhorn", allow_early_resolve=True):
        t0 = ob * COMB_T_TILE
        inv_col_t = pl.load(inv_rms, [t0, 0], [COMB_T_TILE, 1], target_memory=pl.MemorySpace.Vec)
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(mixes_raw, [t0, comb_off + 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g1 = pl.load(mixes_raw, [t0, comb_off + 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g2 = pl.load(mixes_raw, [t0, comb_off + 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g3 = pl.load(mixes_raw, [t0, comb_off + 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shape=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb0 = pl.load(hc_base_2d, [0, comb_off + 0 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb1 = pl.load(hc_base_2d, [0, comb_off + 1 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb2 = pl.load(hc_base_2d, [0, comb_off + 2 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb3 = pl.load(hc_base_2d, [0, comb_off + 3 * HC_MULT], [1, HC_PAD], valid_shape=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
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

        row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
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

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D (D/D_SPMD tasks per tile).
    for blk in pl.spmd((t_dim // T_TILE) * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        pre_tile_t = pl.transpose(pre_val_store[t0:t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_CHUNK, stage=2):
            d0 = d_base + db * D_CHUNK
            x0 = x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_CHUNK]
            x1 = x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_CHUNK]
            x2 = x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_CHUNK]
            x3 = x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_CHUNK]
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            x_mixed[t0:t0 + T_TILE, d0:d0 + D_CHUNK] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
    return x_mixed


def _bind_hc_pre():
    """Define and return the public `hc_pre` inline kernel for the selected HC_PRE_IMPL.

    pypto constrains how the choice can be expressed:
      * a kernel-body branch on the module-global string (``if HC_PRE_IMPL == ...``) fails
        -- pypto's frontend treats it as device control flow and cannot resolve the name;
      * an alias (``hc_pre = _hc_pre_syncall``) fails at the call site -- pypto matches the
        call-site NAME against the callee's registered ``__name__`` ("hc_pre" != the impl).
    So `hc_pre` must BE a decorated inline named ``hc_pre`` that calls the chosen impl by
    its literal name; the selection is a plain-Python ``if`` at import (never seen by a
    kernel). Re-invoke to rebind after changing HC_PRE_IMPL (the __main__ --impl path).
    """
    if HC_PRE_IMPL == "separate":
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
            _hc_pre_separate(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
            return x_mixed
    else:
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
            _hc_pre_syncall(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
            return x_mixed
    return hc_pre


# Public entry point. Callers do `from hc_pre import hc_pre`; env DSV4_HC_PRE_IMPL (or the
# __main__ --impl flag, which rebinds) picks the implementation at import time.
hc_pre = _bind_hc_pre()


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


_A5_FP32_VECTOR_LANES = 64


def _golden_a5_trowsum_fp32(values):
    """Mirror A5 TROWSUM's FP32 reduction order along the last dimension."""
    import torch

    if values.dtype != torch.float32:
        raise ValueError(f"A5 FP32 TROWSUM golden requires float32, got {values.dtype}")
    if values.shape[-1] % _A5_FP32_VECTOR_LANES != 0:
        raise ValueError(
            f"A5 FP32 TROWSUM width must be divisible by {_A5_FP32_VECTOR_LANES}, "
            f"got {values.shape[-1]}"
        )

    # A5 processes 64 FP32 lanes per vector repeat. vcadd recursively adds
    # adjacent pairs inside each repeat; TRowReduce then accumulates the repeat
    # scalars from zero in ascending address order.
    groups = values.reshape(*values.shape[:-1], -1, _A5_FP32_VECTOR_LANES)
    while groups.shape[-1] > 1:
        pairs = groups.reshape(*groups.shape[:-1], -1, 2)
        groups = pairs[..., 0] + pairs[..., 1]

    group_sums = groups[..., 0]
    total = torch.zeros_like(group_sums[..., :1])
    for group in range(group_sums.shape[-1]):
        total += group_sums[..., group:group + 1]
    return total


def _golden_a5_high_precision_rsqrt(value):
    """Match A5 high-precision FP32 rsqrt without host FP32 double rounding."""
    import torch

    return torch.rsqrt(value.to(torch.float64)).to(torch.float32)


def _golden_a5_rms_inv(x_flat_2d):
    """Host mirror of hc_pre_rms: K256 chunks, A5 TROWSUM, then HP rsqrt."""
    import torch

    if x_flat_2d.dtype != torch.float32:
        raise ValueError(f"hc_pre RMS golden requires float32, got {x_flat_2d.dtype}")
    if x_flat_2d.shape[-1] != HC_DIM:
        raise ValueError(f"hc_pre RMS width must be HC_DIM={HC_DIM}, got {x_flat_2d.shape[-1]}")

    sq_sum = torch.zeros(
        x_flat_2d.shape[0], 1, dtype=torch.float32, device=x_flat_2d.device
    )
    for k0 in range(0, HC_DIM, RMS_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_CHUNK]
        sq_sum += _golden_a5_trowsum_fp32(x_chunk * x_chunk)

    rms_arg = sq_sum * HC_DIM_INV + NORM_EPS
    return _golden_a5_high_precision_rsqrt(rms_arg)


def _golden_a5_gate_raw(x_flat_2d, hc_fn, row_start):
    """Mirror one four-row A5 gate projection's K256 FP32 reduction order."""
    import torch

    if x_flat_2d.dtype != torch.float32 or hc_fn.dtype != torch.float32:
        raise ValueError(
            f"A5 gate golden requires float32 inputs, got {x_flat_2d.dtype} and {hc_fn.dtype}"
        )
    if x_flat_2d.shape[-1] != HC_DIM or hc_fn.shape[-1] != HC_DIM:
        raise ValueError(
            f"A5 gate golden widths must be HC_DIM={HC_DIM}, "
            f"got {x_flat_2d.shape[-1]} and {hc_fn.shape[-1]}"
        )
    if row_start < 0 or row_start + HC_MULT > hc_fn.shape[0]:
        raise ValueError(
            f"A5 gate rows [{row_start}, {row_start + HC_MULT}) exceed "
            f"hc_fn rows={hc_fn.shape[0]}"
        )

    gate_raw = torch.zeros(
        x_flat_2d.shape[0], HC_MULT, dtype=torch.float32, device=x_flat_2d.device
    )
    gate_fn = hc_fn[row_start:row_start + HC_MULT]
    for k0 in range(0, HC_DIM, LINEAR_K_CHUNK):
        x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_CHUNK]
        w_chunk = gate_fn[:, k0:k0 + LINEAR_K_CHUNK]
        products = x_chunk[:, None, :] * w_chunk[None, :, :]
        gate_raw += _golden_a5_trowsum_fp32(products)[..., 0]
    return gate_raw


def _golden_a5_pre_raw(x_flat_2d, hc_fn):
    """Mirror the separate pre projection over hc_fn rows 0..3."""
    return _golden_a5_gate_raw(x_flat_2d, hc_fn, row_start=0)


def _golden_a5_post_raw(x_flat_2d, hc_fn):
    """Mirror the separate post projection over hc_fn rows 4..7."""
    return _golden_a5_gate_raw(x_flat_2d, hc_fn, row_start=HC_MULT)


def _golden_hc_pre_pre_raw(x_flat_2d, hc_fn, cube_mixes_raw):
    """Select the pre projection reduction used by the active implementation."""
    if HC_PRE_IMPL == "separate":
        return _golden_a5_pre_raw(x_flat_2d, hc_fn)
    return cube_mixes_raw[..., :HC_MULT].clone()


def _golden_hc_pre_post_raw(x_flat_2d, hc_fn, cube_mixes_raw):
    """Select the post projection reduction used by the active HC-pre implementation."""
    if HC_PRE_IMPL == "separate":
        return _golden_a5_post_raw(x_flat_2d, hc_fn)
    return cube_mixes_raw[..., HC_MULT:HC_MULT * 2].clone()


def golden_hc_pre(tensors):
    """Torch reference, direct port of model.py Block.hc_pre + hc_split_sinkhorn."""
    import torch

    x = tensors["x"].float()  # [T, hc, D]
    hc_fn = tensors["hc_fn"].float()  # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()  # [3]
    hc_base = tensors["hc_base"].float()  # [mix_hc]

    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)

    rsqrt = _golden_a5_rms_inv(x_flat_2d)

    # Per-K-chunk partial dots in one reduction, then accumulated in chunk order
    # to match the cube K-fragment accumulation. Summing over n_chunk with a
    # single torch.sum instead reorders the adds and is not bit-exact.
    n_chunk = HC_DIM // LINEAR_K_CHUNK
    x_k = x_flat_2d.reshape(t_dim, 1, n_chunk, LINEAR_K_CHUNK)
    w_k = hc_fn.reshape(1, MIX_HC, n_chunk, LINEAR_K_CHUNK)
    per_chunk = (x_k * w_k).sum(dim=3)                       # [T, mix_hc, n_chunk]
    mixes = torch.zeros(t_dim, MIX_HC, dtype=torch.float32)  # [T, mix_hc]
    for c in range(n_chunk):
        mixes += per_chunk[:, :, c]
    pre_raw = _golden_hc_pre_pre_raw(x_flat_2d, hc_fn, mixes)
    post_raw = _golden_hc_pre_post_raw(x_flat_2d, hc_fn, mixes)
    mixes *= rsqrt

    pre_mixes = pre_raw * rsqrt
    pre = torch.sigmoid(pre_mixes * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_mixes = post_raw * rsqrt
    post_t = 2 * torch.sigmoid(post_mixes * hc_scale[1] + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(t_dim, HC_MULT, HC_MULT)

    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y0 = x[:, 0, :] * pre[:, 0:1]
    y1 = x[:, 1, :] * pre[:, 1:2]
    y2 = x[:, 2, :] * pre[:, 2:3]
    y3 = x[:, 3, :] * pre[:, 3:4]
    y = (y0 + y1) + (y2 + y3)

    # Match the kernel's mode="rint" cast (round to nearest, ties to even).
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
        return torch.tensor([
            5.9166, -3.6223, -2.9324, -3.3124,
            -3.9100, -0.9384, -3.3256, -2.5240,
            2.0706, -2.5728, 0.1424, -3.9453,
            -3.8859, 3.4634, -3.3799, -2.6077,
            -2.7191, -2.4846, 2.0395, -0.5010,
            -3.5992, -2.7520, -3.3493, 3.1587,
        ])

    return [
        TensorSpec("x", [T, HC_MULT, D], torch.float32, init_value=init_x),
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
                        help="deprecated no-op: dep_gen is auto-selected per --impl (OFF for "
                             "'syncall' per pypto#1931, ON for 'separate'); kept for CLI / CI back-compat.")
    parser.add_argument("--impl", choices=["syncall", "separate"], default=HC_PRE_IMPL,
                        help="hc_pre implementation: 'syncall' (fused single task, #684) or "
                             "'separate' (multi-scope task graph, pre-#684). Both are unified "
                             "over decode+prefill. Default from env DSV4_HC_PRE_IMPL.")
    args = parser.parse_args()

    # Select the implementation for this run: set the flag, then rebind the module-global
    # `hc_pre` to the chosen inline kernel BEFORE run_jit traces hc_pre_test (which resolves
    # `hc_pre` from the module namespace at trace time).
    HC_PRE_IMPL = args.impl
    hc_pre = _bind_hc_pre()
    print(f"hc_pre implementation: {HC_PRE_IMPL}")

    # hc_pre "syncall" is specialized to Ascend 910B: it sets NUM_CORES=24 == the physical
    # AIC count and its hard full-occupancy mix-syncall hangs (AICore timeout 507018) unless
    # the launch fills every physical core; A5 (Ascend950) has a different AIC count (36), so
    # the syncall impl is rejected on A5. The "separate" impl has no such barrier -- it uses
    # data-derived pl.spmd(work_count) scopes ordered by their GM dependencies, so it is
    # core-count-agnostic and runs on A5. Its tile sizes are 910B-tuned (perf, not correctness);
    # a5 perf may be suboptimal
    # until re-swept, but precision is unaffected.
    if args.platform in ("a5", "a5sim") and HC_PRE_IMPL == "syncall":
        raise SystemExit(
            f"hc_pre 'syncall' impl is specialized to Ascend 910B (NUM_CORES={NUM_CORES} == physical "
            f"AIC count); its full-occupancy mix-syncall would hang (AICore timeout 507018) on "
            f"{args.platform!r}. Re-run with --impl separate on {args.platform!r}, or -p a2a3."
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
                # dep_gen: the "syncall" version's full-occupancy pl.system.syncall is
                # incompatible with dep_gen -- the DFX instrumentation perturbs core
                # occupancy and trips AICore timeout 507018 (pypto#1931) -- so it runs with
                # dep_gen OFF. The "separate" version has no hard syncall, so the harness
                # enables dep-gen to capture its scheduler-derived dependency graph.
                enable_dep_gen=(HC_PRE_IMPL == "separate"),
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
