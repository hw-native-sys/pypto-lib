# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim  # A2/A3 device validation; simulator CI is deferred for this model.
"""Persistent routed and shared MoE expert execution.

The planned entry consumes compact slot and row-count tables produced by route
grouping. The balanced wrapper constructs those tables for standalone fixtures
with arbitrary active slots. A 24-block grid processes complete expert chains,
with gate/up matmuls, SwiGLU, activation quantization, down matmul and dequantization
kept in one mixed task. Each cube core uses two vector lanes through UP_DOWN
regions; intermediate activations stay on chip.
"""


import pypto.language as pl

from config import (INT8_SCALE_MAX, INT8_AMAX_EPS)
from expert_routed import (
    D, IDX_PAD, MOE_INTER, N_BANK, N_SLOTS_B, RECV_MAX, RECV_TILE, SWIGLU_LIMIT,
    build_tensor_specs, golden_expert_routed, compare_recv_y,
)


# Persistent blocks == the a2a3 AIC count. No barrier lives in this kernel, so a
# mismatch on another SoC is a scheduling fact, not a hang.
NUM_CORES = 24
WORK_SLOTS_PAD = (N_SLOTS_B + IDX_PAD - 1) // IDX_PAD * IDX_PAD

# One cluster = 1 cube core + 2 buddy vector cores.
AIV_LANES = 2
ROW_HALF = RECV_TILE // AIV_LANES

# tiling
FUSED_N_TILE = MOE_INTER
# The whole intermediate shard is one cube N tile, which is what lets the amax
# stay a within-row reduction -- and it is also what bounds this form by buffer
# size. Acc holds gate_acc and up_acc double-buffered at 2 * 2 * ROW_TILE *
# MOE_INTER * 4 B, so MOE_INTER > 512 overflows L0C no matter how K is tiled;
# below that, K is what keeps Mat (double-buffered operands plus the 64 KiB c2v
# ring) under its limit. TP=8 gives MOE_INTER=256 and K stays at 512.
assert MOE_INTER <= 512, (
    f"the fused routed expert needs MOE_INTER <= 512, got {MOE_INTER}: gate_acc and "
    "up_acc do not fit L0C. Use expert_routed.py's four-task form at this TP degree."
)
# Both cube operands are bounded the same way: the gate/up weight tile is
# MOE_INTER x FUSED_K_TILE and the down weight tile is FUSED_Y_TILE x MOE_INTER,
# each double-buffered, and Mat has to hold them beside the 64 KiB c2v ring.
FUSED_K_TILE = max(64, min(512, 131072 // MOE_INTER))
FUSED_Y_TILE = max(64, min(512, 131072 // MOE_INTER))


@pl.jit.inline(auto_scope=False)
def expert_routed_persistent_planned(
    recv_x: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    work_slot: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    work_rows: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    routed_w1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_BANK, D], pl.FP32],
    recv_y: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.BF16],
):
    recv_x_flat = pl.reshape(recv_x, [N_SLOTS_B * RECV_MAX, D])
    recv_y_flat = pl.reshape(recv_y, [N_SLOTS_B * RECV_MAX, D])

    # Fold the expert axis into the row axis so every matmul operand is 2-D and
    # the accumulator comes back 2-D (no [1, RECV_TILE, TILE] reshape unwrap).
    w1_2d = pl.reshape(routed_w1, [N_BANK * MOE_INTER, D])
    w3_2d = pl.reshape(routed_w3, [N_BANK * MOE_INTER, D])
    w2_2d = pl.reshape(routed_w2, [N_BANK * D, MOE_INTER])

    with pl.scope():
        # sync_start: the combine grid downstream is submitted while this task is
        # still running and parks blocks on AIV lanes waiting for recv_y. Without
        # a synchronized start this grid takes whatever lanes are free and leaves
        # some of its own blocks queued behind those occupants -- normally a
        # 10-20 us start spread, but occasionally a block starts only after
        # another has already finished, which stretches the task ~113 -> ~146 us
        # on that card, and the layer waits for it. Starting all NUM_CORES blocks
        # together holds the spread at ~1.5 us; the grid is the whole AIC array,
        # so refusing a partial start gives nothing away.
        with pl.spmd(
            NUM_CORES, name_hint="exp_routed_balanced", sync_start=True,
            allow_early_resolve=True,
        ):
            core = pl.tile.get_block_idx()  # 0 .. NUM_CORES-1

            for w in pl.range(core, N_SLOTS_B, NUM_CORES):
                s = pl.cast(pl.read(work_slot, [w]), pl.INDEX)
                eid = pl.cast(pl.read(slot_expert, [s, 0]), pl.INDEX)
                n_rows = pl.read(work_rows, [w])
                e_w1 = eid * MOE_INTER
                e_w2 = eid * D
                slot_base = s * RECV_MAX

                # 0-or-1 trip: the zero case IS the empty-slot skip.
                for t in pl.range((n_rows + RECV_TILE - 1) // RECV_TILE):
                    t0 = t * RECV_TILE
                    flat_t0 = slot_base + t0
                    valid_rows = pl.min(RECV_TILE, n_rows - t0)

                    gate_acc = pl.create_tensor([RECV_TILE, FUSED_N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                        k0 = kb * FUSED_K_TILE
                        x_k = pl.slice(recv_x_flat, [RECV_TILE, FUSED_K_TILE], [flat_t0, k0],
                                       valid_shape=[valid_rows, FUSED_K_TILE])
                        w1_k = w1_2d[e_w1 : e_w1 + FUSED_N_TILE, k0 : k0 + FUSED_K_TILE]
                        gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True, init_cond=(kb == 0))

                    up_acc = pl.create_tensor([RECV_TILE, FUSED_N_TILE], dtype=pl.INT32)
                    for ukb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                        uk0 = ukb * FUSED_K_TILE
                        x_u = pl.slice(recv_x_flat, [RECV_TILE, FUSED_K_TILE], [flat_t0, uk0],
                                       valid_shape=[valid_rows, FUSED_K_TILE])
                        w3_k = w3_2d[e_w1 : e_w1 + FUSED_N_TILE, uk0 : uk0 + FUSED_K_TILE]
                        up_acc = pl.matmul_acc(up_acc, x_u, w3_k, b_trans=True, init_cond=(ukb == 0))

                    w1_sc = routed_w1_scale[eid : eid + 1, 0:FUSED_N_TILE]
                    w3_sc = routed_w3_scale[eid : eid + 1, 0:FUSED_N_TILE]

                    # ---- vector phase 1: dequant + SwiGLU + per-row A8 requant ----
                    # UP_DOWN halves the shard on rows, so each lane holds ROW_HALF
                    # COMPLETE rows and pl.row_max is a within-lane reduction.
                    for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                        # A lane-derived read address is what marks the load
                        # per-lane for the region's half-width scan. Column-sliced
                        # in place rather than through a 1-D view of the whole
                        # table: a 2-D -> 1-D reshape of an inline parameter loses
                        # its inferred metadata once moe.py nests this call.
                        lane_r0 = pl.cast(aiv_id * ROW_HALF, pl.INDEX)

                        x_sc = pl.reshape(
                            recv_scale_dq[s : s + 1, lane_r0 : lane_r0 + ROW_HALF],
                            [ROW_HALF, 1])
                        w_col = pl.reshape(
                            recv_weights[s : s + 1, lane_r0 : lane_r0 + ROW_HALF],
                            [ROW_HALF, 1])
                        gate_sh = pl.aiv_shard(gate_acc)
                        up_sh = pl.aiv_shard(up_acc)
                        gate_f = pl.cast(gate_sh, target_type=pl.FP32, mode="none")
                        up_f = pl.cast(up_sh, target_type=pl.FP32, mode="none")
                        gate_f = pl.col_expand_mul(pl.row_expand_mul(gate_f, x_sc), w1_sc)
                        up_f = pl.col_expand_mul(pl.row_expand_mul(up_f, x_sc), w3_sc)
                        if SWIGLU_LIMIT > 0.0:
                            gate_f = pl.minimum(gate_f, SWIGLU_LIMIT)
                            up_f = pl.maximum(pl.minimum(up_f, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                        expert_exp_argument = pl.neg(gate_f)
                        erexp1_result = expert_exp_argument
                        erexp1_zero = expert_exp_argument
                        erexp1_zero = pl.sub(expert_exp_argument, expert_exp_argument)
                        erexp2_safe = expert_exp_argument
                        erexp2_safe = pl.minimum(pl.maximum(expert_exp_argument, -80.0), 80.0)
                        erexp2_scaled = pl.mul(erexp2_safe, 1.4426950216293335)
                        erexp2_n_i32 = pl.cast(erexp2_scaled, target_type=pl.INT32, mode='rint')
                        erexp2_n = expert_exp_argument
                        erexp2_n = pl.cast(erexp2_n_i32, target_type=pl.FP32)
                        erexp2_ln_hi = erexp1_zero
                        erexp2_ln_lo = erexp1_zero
                        erexp2_ln_hi = pl.add(erexp1_zero, 0.6931471824645996)
                        erexp2_ln_lo = pl.add(erexp1_zero, -1.9046542121259336e-09)
                        erexp3_product = erexp2_n
                        erexp3_error = erexp2_n
                        erexp3_a_split = pl.mul(erexp2_n, 4097.0)
                        erexp3_a_delta = pl.sub(erexp3_a_split, erexp2_n)
                        erexp3_a_hi = pl.sub(erexp3_a_split, erexp3_a_delta)
                        erexp3_a_lo = pl.sub(erexp2_n, erexp3_a_hi)
                        erexp3_b_split = pl.mul(erexp2_ln_hi, 4097.0)
                        erexp3_b_delta = pl.sub(erexp3_b_split, erexp2_ln_hi)
                        erexp3_b_hi = pl.sub(erexp3_b_split, erexp3_b_delta)
                        erexp3_b_lo = pl.sub(erexp2_ln_hi, erexp3_b_hi)
                        erexp3_product = pl.mul(erexp2_n, erexp2_ln_hi)
                        erexp3_high_product = pl.mul(erexp3_a_hi, erexp3_b_hi)
                        erexp3_error = pl.sub(erexp3_high_product, erexp3_product)
                        erexp3_high_low = pl.mul(erexp3_a_hi, erexp3_b_lo)
                        erexp3_error = pl.add(erexp3_error, erexp3_high_low)
                        erexp3_low_high = pl.mul(erexp3_a_lo, erexp3_b_hi)
                        erexp3_error = pl.add(erexp3_error, erexp3_low_high)
                        erexp3_low_product = pl.mul(erexp3_a_lo, erexp3_b_lo)
                        erexp3_error = pl.add(erexp3_error, erexp3_low_product)
                        erexp2_ssa1_ph = erexp3_product
                        erexp2_ssa1_pe = erexp3_error
                        erexp4_product = erexp2_n
                        erexp4_error = erexp2_n
                        erexp4_a_split = pl.mul(erexp2_n, 4097.0)
                        erexp4_a_delta = pl.sub(erexp4_a_split, erexp2_n)
                        erexp4_a_hi = pl.sub(erexp4_a_split, erexp4_a_delta)
                        erexp4_a_lo = pl.sub(erexp2_n, erexp4_a_hi)
                        erexp4_b_split = pl.mul(erexp2_ln_lo, 4097.0)
                        erexp4_b_delta = pl.sub(erexp4_b_split, erexp2_ln_lo)
                        erexp4_b_hi = pl.sub(erexp4_b_split, erexp4_b_delta)
                        erexp4_b_lo = pl.sub(erexp2_ln_lo, erexp4_b_hi)
                        erexp4_product = pl.mul(erexp2_n, erexp2_ln_lo)
                        erexp4_high_product = pl.mul(erexp4_a_hi, erexp4_b_hi)
                        erexp4_error = pl.sub(erexp4_high_product, erexp4_product)
                        erexp4_high_low = pl.mul(erexp4_a_hi, erexp4_b_lo)
                        erexp4_error = pl.add(erexp4_error, erexp4_high_low)
                        erexp4_low_high = pl.mul(erexp4_a_lo, erexp4_b_hi)
                        erexp4_error = pl.add(erexp4_error, erexp4_low_high)
                        erexp4_low_product = pl.mul(erexp4_a_lo, erexp4_b_lo)
                        erexp4_error = pl.add(erexp4_error, erexp4_low_product)
                        erexp2_ssa2_qh = erexp4_product
                        erexp2_ssa2_qe = erexp4_error
                        erexp2_negative_ph = expert_exp_argument
                        erexp2_negative_ph = pl.neg(erexp2_ssa1_ph)
                        erexp5_total = erexp2_safe
                        erexp5_error = erexp2_safe
                        erexp5_total = pl.add(erexp2_safe, erexp2_negative_ph)
                        erexp5_z = pl.sub(erexp5_total, erexp2_safe)
                        erexp5_total_minus_z = pl.sub(erexp5_total, erexp5_z)
                        erexp5_a_error = pl.sub(erexp2_safe, erexp5_total_minus_z)
                        erexp5_b_error = pl.sub(erexp2_negative_ph, erexp5_z)
                        erexp5_error = pl.add(erexp5_a_error, erexp5_b_error)
                        erexp2_ssa3_rh = erexp5_total
                        erexp2_ssa3_rl = erexp5_error
                        erexp2_negative_pe = expert_exp_argument
                        erexp2_negative_pe = pl.neg(erexp2_ssa1_pe)
                        erexp7_total = erexp2_ssa3_rh
                        erexp7_error = erexp2_ssa3_rh
                        erexp7_total = pl.add(erexp2_ssa3_rh, erexp2_negative_pe)
                        erexp7_z = pl.sub(erexp7_total, erexp2_ssa3_rh)
                        erexp7_total_minus_z = pl.sub(erexp7_total, erexp7_z)
                        erexp7_a_error = pl.sub(erexp2_ssa3_rh, erexp7_total_minus_z)
                        erexp7_b_error = pl.sub(erexp2_negative_pe, erexp7_z)
                        erexp7_error = pl.add(erexp7_a_error, erexp7_b_error)
                        erexp6_ssa1_total = erexp7_total
                        erexp6_ssa1_error = erexp7_error
                        erexp6_low = erexp2_ssa3_rl
                        erexp6_low = pl.add(erexp2_ssa3_rl, erexp1_zero)
                        erexp6_tail = erexp6_ssa1_error
                        erexp6_tail = pl.add(erexp6_ssa1_error, erexp6_low)
                        erexp8_total = erexp6_ssa1_total
                        erexp8_error = erexp6_ssa1_total
                        erexp8_total = pl.add(erexp6_ssa1_total, erexp6_tail)
                        erexp8_z = pl.sub(erexp8_total, erexp6_ssa1_total)
                        erexp8_total_minus_z = pl.sub(erexp8_total, erexp8_z)
                        erexp8_a_error = pl.sub(erexp6_ssa1_total, erexp8_total_minus_z)
                        erexp8_b_error = pl.sub(erexp6_tail, erexp8_z)
                        erexp8_error = pl.add(erexp8_a_error, erexp8_b_error)
                        erexp6_ssa2_high = erexp8_total
                        erexp6_ssa2_low = erexp8_error
                        erexp2_ssa4_rh = erexp6_ssa2_high
                        erexp2_ssa4_rl = erexp6_ssa2_low
                        erexp2_negative_qh = expert_exp_argument
                        erexp2_negative_qe = expert_exp_argument
                        erexp2_negative_qh = pl.neg(erexp2_ssa2_qh)
                        erexp2_negative_qe = pl.neg(erexp2_ssa2_qe)
                        erexp10_total = erexp2_ssa4_rh
                        erexp10_error = erexp2_ssa4_rh
                        erexp10_total = pl.add(erexp2_ssa4_rh, erexp2_negative_qh)
                        erexp10_z = pl.sub(erexp10_total, erexp2_ssa4_rh)
                        erexp10_total_minus_z = pl.sub(erexp10_total, erexp10_z)
                        erexp10_a_error = pl.sub(erexp2_ssa4_rh, erexp10_total_minus_z)
                        erexp10_b_error = pl.sub(erexp2_negative_qh, erexp10_z)
                        erexp10_error = pl.add(erexp10_a_error, erexp10_b_error)
                        erexp9_ssa1_total = erexp10_total
                        erexp9_ssa1_error = erexp10_error
                        erexp9_low = erexp2_ssa4_rl
                        erexp9_low = pl.add(erexp2_ssa4_rl, erexp2_negative_qe)
                        erexp9_tail = erexp9_ssa1_error
                        erexp9_tail = pl.add(erexp9_ssa1_error, erexp9_low)
                        erexp11_total = erexp9_ssa1_total
                        erexp11_error = erexp9_ssa1_total
                        erexp11_total = pl.add(erexp9_ssa1_total, erexp9_tail)
                        erexp11_z = pl.sub(erexp11_total, erexp9_ssa1_total)
                        erexp11_total_minus_z = pl.sub(erexp11_total, erexp11_z)
                        erexp11_a_error = pl.sub(erexp9_ssa1_total, erexp11_total_minus_z)
                        erexp11_b_error = pl.sub(erexp9_tail, erexp11_z)
                        erexp11_error = pl.add(erexp11_a_error, erexp11_b_error)
                        erexp9_ssa2_high = erexp11_total
                        erexp9_ssa2_low = erexp11_error
                        erexp2_ssa5_rh = erexp9_ssa2_high
                        erexp2_ssa5_rl = erexp9_ssa2_low
                        erexp2_ch = erexp1_zero
                        erexp2_cl = erexp1_zero
                        erexp2_ch = pl.add(erexp1_zero, 4.7794772561329454e-14)
                        erexp2_cl = pl.add(erexp1_zero, 7.62544404448643e-22)
                        erexp13_product = erexp2_ch
                        erexp13_error = erexp2_ch
                        erexp13_a_split = pl.mul(erexp2_ch, 4097.0)
                        erexp13_a_delta = pl.sub(erexp13_a_split, erexp2_ch)
                        erexp13_a_hi = pl.sub(erexp13_a_split, erexp13_a_delta)
                        erexp13_a_lo = pl.sub(erexp2_ch, erexp13_a_hi)
                        erexp13_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp13_b_delta = pl.sub(erexp13_b_split, erexp2_ssa5_rh)
                        erexp13_b_hi = pl.sub(erexp13_b_split, erexp13_b_delta)
                        erexp13_b_lo = pl.sub(erexp2_ssa5_rh, erexp13_b_hi)
                        erexp13_product = pl.mul(erexp2_ch, erexp2_ssa5_rh)
                        erexp13_high_product = pl.mul(erexp13_a_hi, erexp13_b_hi)
                        erexp13_error = pl.sub(erexp13_high_product, erexp13_product)
                        erexp13_high_low = pl.mul(erexp13_a_hi, erexp13_b_lo)
                        erexp13_error = pl.add(erexp13_error, erexp13_high_low)
                        erexp13_low_high = pl.mul(erexp13_a_lo, erexp13_b_hi)
                        erexp13_error = pl.add(erexp13_error, erexp13_low_high)
                        erexp13_low_product = pl.mul(erexp13_a_lo, erexp13_b_lo)
                        erexp13_error = pl.add(erexp13_error, erexp13_low_product)
                        erexp12_ssa1_high = erexp13_product
                        erexp12_ssa1_error = erexp13_error
                        erexp12_cross1 = pl.mul(erexp2_ch, erexp2_ssa5_rl)
                        erexp12_cross2 = pl.mul(erexp2_cl, erexp2_ssa5_rh)
                        erexp12_cross3 = pl.mul(erexp2_cl, erexp2_ssa5_rl)
                        erexp12_low = erexp12_ssa1_error
                        erexp12_low = pl.add(erexp12_ssa1_error, erexp12_cross1)
                        erexp12_low = pl.add(erexp12_low, erexp12_cross2)
                        erexp12_low = pl.add(erexp12_low, erexp12_cross3)
                        erexp14_total = erexp12_ssa1_high
                        erexp14_error = erexp12_ssa1_high
                        erexp14_total = pl.add(erexp12_ssa1_high, erexp12_low)
                        erexp14_z = pl.sub(erexp14_total, erexp12_ssa1_high)
                        erexp14_total_minus_z = pl.sub(erexp14_total, erexp14_z)
                        erexp14_a_error = pl.sub(erexp12_ssa1_high, erexp14_total_minus_z)
                        erexp14_b_error = pl.sub(erexp12_low, erexp14_z)
                        erexp14_error = pl.add(erexp14_a_error, erexp14_b_error)
                        erexp12_ssa2_high = erexp14_total
                        erexp12_ssa2_low = erexp14_error
                        erexp2_ssa6_ch = erexp12_ssa2_high
                        erexp2_ssa6_cl = erexp12_ssa2_low
                        erexp2_coefficient_hi15 = erexp1_zero
                        erexp2_coefficient_lo15 = erexp1_zero
                        erexp2_coefficient_hi15 = pl.add(erexp1_zero, 7.647163609812713e-13)
                        erexp2_coefficient_lo15 = pl.add(erexp1_zero, 1.2200710471178288e-20)
                        erexp16_total = erexp2_ssa6_ch
                        erexp16_error = erexp2_ssa6_ch
                        erexp16_total = pl.add(erexp2_ssa6_ch, erexp2_coefficient_hi15)
                        erexp16_z = pl.sub(erexp16_total, erexp2_ssa6_ch)
                        erexp16_total_minus_z = pl.sub(erexp16_total, erexp16_z)
                        erexp16_a_error = pl.sub(erexp2_ssa6_ch, erexp16_total_minus_z)
                        erexp16_b_error = pl.sub(erexp2_coefficient_hi15, erexp16_z)
                        erexp16_error = pl.add(erexp16_a_error, erexp16_b_error)
                        erexp15_ssa1_total = erexp16_total
                        erexp15_ssa1_error = erexp16_error
                        erexp15_low = erexp2_ssa6_cl
                        erexp15_low = pl.add(erexp2_ssa6_cl, erexp2_coefficient_lo15)
                        erexp15_tail = erexp15_ssa1_error
                        erexp15_tail = pl.add(erexp15_ssa1_error, erexp15_low)
                        erexp17_total = erexp15_ssa1_total
                        erexp17_error = erexp15_ssa1_total
                        erexp17_total = pl.add(erexp15_ssa1_total, erexp15_tail)
                        erexp17_z = pl.sub(erexp17_total, erexp15_ssa1_total)
                        erexp17_total_minus_z = pl.sub(erexp17_total, erexp17_z)
                        erexp17_a_error = pl.sub(erexp15_ssa1_total, erexp17_total_minus_z)
                        erexp17_b_error = pl.sub(erexp15_tail, erexp17_z)
                        erexp17_error = pl.add(erexp17_a_error, erexp17_b_error)
                        erexp15_ssa2_high = erexp17_total
                        erexp15_ssa2_low = erexp17_error
                        erexp2_ssa7_ch = erexp15_ssa2_high
                        erexp2_ssa7_cl = erexp15_ssa2_low
                        erexp19_product = erexp2_ssa7_ch
                        erexp19_error = erexp2_ssa7_ch
                        erexp19_a_split = pl.mul(erexp2_ssa7_ch, 4097.0)
                        erexp19_a_delta = pl.sub(erexp19_a_split, erexp2_ssa7_ch)
                        erexp19_a_hi = pl.sub(erexp19_a_split, erexp19_a_delta)
                        erexp19_a_lo = pl.sub(erexp2_ssa7_ch, erexp19_a_hi)
                        erexp19_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp19_b_delta = pl.sub(erexp19_b_split, erexp2_ssa5_rh)
                        erexp19_b_hi = pl.sub(erexp19_b_split, erexp19_b_delta)
                        erexp19_b_lo = pl.sub(erexp2_ssa5_rh, erexp19_b_hi)
                        erexp19_product = pl.mul(erexp2_ssa7_ch, erexp2_ssa5_rh)
                        erexp19_high_product = pl.mul(erexp19_a_hi, erexp19_b_hi)
                        erexp19_error = pl.sub(erexp19_high_product, erexp19_product)
                        erexp19_high_low = pl.mul(erexp19_a_hi, erexp19_b_lo)
                        erexp19_error = pl.add(erexp19_error, erexp19_high_low)
                        erexp19_low_high = pl.mul(erexp19_a_lo, erexp19_b_hi)
                        erexp19_error = pl.add(erexp19_error, erexp19_low_high)
                        erexp19_low_product = pl.mul(erexp19_a_lo, erexp19_b_lo)
                        erexp19_error = pl.add(erexp19_error, erexp19_low_product)
                        erexp18_ssa1_high = erexp19_product
                        erexp18_ssa1_error = erexp19_error
                        erexp18_cross1 = pl.mul(erexp2_ssa7_ch, erexp2_ssa5_rl)
                        erexp18_cross2 = pl.mul(erexp2_ssa7_cl, erexp2_ssa5_rh)
                        erexp18_cross3 = pl.mul(erexp2_ssa7_cl, erexp2_ssa5_rl)
                        erexp18_low = erexp18_ssa1_error
                        erexp18_low = pl.add(erexp18_ssa1_error, erexp18_cross1)
                        erexp18_low = pl.add(erexp18_low, erexp18_cross2)
                        erexp18_low = pl.add(erexp18_low, erexp18_cross3)
                        erexp20_total = erexp18_ssa1_high
                        erexp20_error = erexp18_ssa1_high
                        erexp20_total = pl.add(erexp18_ssa1_high, erexp18_low)
                        erexp20_z = pl.sub(erexp20_total, erexp18_ssa1_high)
                        erexp20_total_minus_z = pl.sub(erexp20_total, erexp20_z)
                        erexp20_a_error = pl.sub(erexp18_ssa1_high, erexp20_total_minus_z)
                        erexp20_b_error = pl.sub(erexp18_low, erexp20_z)
                        erexp20_error = pl.add(erexp20_a_error, erexp20_b_error)
                        erexp18_ssa2_high = erexp20_total
                        erexp18_ssa2_low = erexp20_error
                        erexp2_ssa8_ch = erexp18_ssa2_high
                        erexp2_ssa8_cl = erexp18_ssa2_low
                        erexp2_coefficient_hi14 = erexp1_zero
                        erexp2_coefficient_lo14 = erexp1_zero
                        erexp2_coefficient_hi14 = pl.add(erexp1_zero, 1.147074536050896e-11)
                        erexp2_coefficient_lo14 = pl.add(erexp1_zero, 2.372207689231238e-19)
                        erexp22_total = erexp2_ssa8_ch
                        erexp22_error = erexp2_ssa8_ch
                        erexp22_total = pl.add(erexp2_ssa8_ch, erexp2_coefficient_hi14)
                        erexp22_z = pl.sub(erexp22_total, erexp2_ssa8_ch)
                        erexp22_total_minus_z = pl.sub(erexp22_total, erexp22_z)
                        erexp22_a_error = pl.sub(erexp2_ssa8_ch, erexp22_total_minus_z)
                        erexp22_b_error = pl.sub(erexp2_coefficient_hi14, erexp22_z)
                        erexp22_error = pl.add(erexp22_a_error, erexp22_b_error)
                        erexp21_ssa1_total = erexp22_total
                        erexp21_ssa1_error = erexp22_error
                        erexp21_low = erexp2_ssa8_cl
                        erexp21_low = pl.add(erexp2_ssa8_cl, erexp2_coefficient_lo14)
                        erexp21_tail = erexp21_ssa1_error
                        erexp21_tail = pl.add(erexp21_ssa1_error, erexp21_low)
                        erexp23_total = erexp21_ssa1_total
                        erexp23_error = erexp21_ssa1_total
                        erexp23_total = pl.add(erexp21_ssa1_total, erexp21_tail)
                        erexp23_z = pl.sub(erexp23_total, erexp21_ssa1_total)
                        erexp23_total_minus_z = pl.sub(erexp23_total, erexp23_z)
                        erexp23_a_error = pl.sub(erexp21_ssa1_total, erexp23_total_minus_z)
                        erexp23_b_error = pl.sub(erexp21_tail, erexp23_z)
                        erexp23_error = pl.add(erexp23_a_error, erexp23_b_error)
                        erexp21_ssa2_high = erexp23_total
                        erexp21_ssa2_low = erexp23_error
                        erexp2_ssa9_ch = erexp21_ssa2_high
                        erexp2_ssa9_cl = erexp21_ssa2_low
                        erexp25_product = erexp2_ssa9_ch
                        erexp25_error = erexp2_ssa9_ch
                        erexp25_a_split = pl.mul(erexp2_ssa9_ch, 4097.0)
                        erexp25_a_delta = pl.sub(erexp25_a_split, erexp2_ssa9_ch)
                        erexp25_a_hi = pl.sub(erexp25_a_split, erexp25_a_delta)
                        erexp25_a_lo = pl.sub(erexp2_ssa9_ch, erexp25_a_hi)
                        erexp25_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp25_b_delta = pl.sub(erexp25_b_split, erexp2_ssa5_rh)
                        erexp25_b_hi = pl.sub(erexp25_b_split, erexp25_b_delta)
                        erexp25_b_lo = pl.sub(erexp2_ssa5_rh, erexp25_b_hi)
                        erexp25_product = pl.mul(erexp2_ssa9_ch, erexp2_ssa5_rh)
                        erexp25_high_product = pl.mul(erexp25_a_hi, erexp25_b_hi)
                        erexp25_error = pl.sub(erexp25_high_product, erexp25_product)
                        erexp25_high_low = pl.mul(erexp25_a_hi, erexp25_b_lo)
                        erexp25_error = pl.add(erexp25_error, erexp25_high_low)
                        erexp25_low_high = pl.mul(erexp25_a_lo, erexp25_b_hi)
                        erexp25_error = pl.add(erexp25_error, erexp25_low_high)
                        erexp25_low_product = pl.mul(erexp25_a_lo, erexp25_b_lo)
                        erexp25_error = pl.add(erexp25_error, erexp25_low_product)
                        erexp24_ssa1_high = erexp25_product
                        erexp24_ssa1_error = erexp25_error
                        erexp24_cross1 = pl.mul(erexp2_ssa9_ch, erexp2_ssa5_rl)
                        erexp24_cross2 = pl.mul(erexp2_ssa9_cl, erexp2_ssa5_rh)
                        erexp24_cross3 = pl.mul(erexp2_ssa9_cl, erexp2_ssa5_rl)
                        erexp24_low = erexp24_ssa1_error
                        erexp24_low = pl.add(erexp24_ssa1_error, erexp24_cross1)
                        erexp24_low = pl.add(erexp24_low, erexp24_cross2)
                        erexp24_low = pl.add(erexp24_low, erexp24_cross3)
                        erexp26_total = erexp24_ssa1_high
                        erexp26_error = erexp24_ssa1_high
                        erexp26_total = pl.add(erexp24_ssa1_high, erexp24_low)
                        erexp26_z = pl.sub(erexp26_total, erexp24_ssa1_high)
                        erexp26_total_minus_z = pl.sub(erexp26_total, erexp26_z)
                        erexp26_a_error = pl.sub(erexp24_ssa1_high, erexp26_total_minus_z)
                        erexp26_b_error = pl.sub(erexp24_low, erexp26_z)
                        erexp26_error = pl.add(erexp26_a_error, erexp26_b_error)
                        erexp24_ssa2_high = erexp26_total
                        erexp24_ssa2_low = erexp26_error
                        erexp2_ssa10_ch = erexp24_ssa2_high
                        erexp2_ssa10_cl = erexp24_ssa2_low
                        erexp2_coefficient_hi13 = erexp1_zero
                        erexp2_coefficient_lo13 = erexp1_zero
                        erexp2_coefficient_hi13 = pl.add(erexp1_zero, 1.6059044372074283e-10)
                        erexp2_coefficient_lo13 = pl.add(erexp1_zero, -5.352526511562726e-18)
                        erexp28_total = erexp2_ssa10_ch
                        erexp28_error = erexp2_ssa10_ch
                        erexp28_total = pl.add(erexp2_ssa10_ch, erexp2_coefficient_hi13)
                        erexp28_z = pl.sub(erexp28_total, erexp2_ssa10_ch)
                        erexp28_total_minus_z = pl.sub(erexp28_total, erexp28_z)
                        erexp28_a_error = pl.sub(erexp2_ssa10_ch, erexp28_total_minus_z)
                        erexp28_b_error = pl.sub(erexp2_coefficient_hi13, erexp28_z)
                        erexp28_error = pl.add(erexp28_a_error, erexp28_b_error)
                        erexp27_ssa1_total = erexp28_total
                        erexp27_ssa1_error = erexp28_error
                        erexp27_low = erexp2_ssa10_cl
                        erexp27_low = pl.add(erexp2_ssa10_cl, erexp2_coefficient_lo13)
                        erexp27_tail = erexp27_ssa1_error
                        erexp27_tail = pl.add(erexp27_ssa1_error, erexp27_low)
                        erexp29_total = erexp27_ssa1_total
                        erexp29_error = erexp27_ssa1_total
                        erexp29_total = pl.add(erexp27_ssa1_total, erexp27_tail)
                        erexp29_z = pl.sub(erexp29_total, erexp27_ssa1_total)
                        erexp29_total_minus_z = pl.sub(erexp29_total, erexp29_z)
                        erexp29_a_error = pl.sub(erexp27_ssa1_total, erexp29_total_minus_z)
                        erexp29_b_error = pl.sub(erexp27_tail, erexp29_z)
                        erexp29_error = pl.add(erexp29_a_error, erexp29_b_error)
                        erexp27_ssa2_high = erexp29_total
                        erexp27_ssa2_low = erexp29_error
                        erexp2_ssa11_ch = erexp27_ssa2_high
                        erexp2_ssa11_cl = erexp27_ssa2_low
                        erexp31_product = erexp2_ssa11_ch
                        erexp31_error = erexp2_ssa11_ch
                        erexp31_a_split = pl.mul(erexp2_ssa11_ch, 4097.0)
                        erexp31_a_delta = pl.sub(erexp31_a_split, erexp2_ssa11_ch)
                        erexp31_a_hi = pl.sub(erexp31_a_split, erexp31_a_delta)
                        erexp31_a_lo = pl.sub(erexp2_ssa11_ch, erexp31_a_hi)
                        erexp31_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp31_b_delta = pl.sub(erexp31_b_split, erexp2_ssa5_rh)
                        erexp31_b_hi = pl.sub(erexp31_b_split, erexp31_b_delta)
                        erexp31_b_lo = pl.sub(erexp2_ssa5_rh, erexp31_b_hi)
                        erexp31_product = pl.mul(erexp2_ssa11_ch, erexp2_ssa5_rh)
                        erexp31_high_product = pl.mul(erexp31_a_hi, erexp31_b_hi)
                        erexp31_error = pl.sub(erexp31_high_product, erexp31_product)
                        erexp31_high_low = pl.mul(erexp31_a_hi, erexp31_b_lo)
                        erexp31_error = pl.add(erexp31_error, erexp31_high_low)
                        erexp31_low_high = pl.mul(erexp31_a_lo, erexp31_b_hi)
                        erexp31_error = pl.add(erexp31_error, erexp31_low_high)
                        erexp31_low_product = pl.mul(erexp31_a_lo, erexp31_b_lo)
                        erexp31_error = pl.add(erexp31_error, erexp31_low_product)
                        erexp30_ssa1_high = erexp31_product
                        erexp30_ssa1_error = erexp31_error
                        erexp30_cross1 = pl.mul(erexp2_ssa11_ch, erexp2_ssa5_rl)
                        erexp30_cross2 = pl.mul(erexp2_ssa11_cl, erexp2_ssa5_rh)
                        erexp30_cross3 = pl.mul(erexp2_ssa11_cl, erexp2_ssa5_rl)
                        erexp30_low = erexp30_ssa1_error
                        erexp30_low = pl.add(erexp30_ssa1_error, erexp30_cross1)
                        erexp30_low = pl.add(erexp30_low, erexp30_cross2)
                        erexp30_low = pl.add(erexp30_low, erexp30_cross3)
                        erexp32_total = erexp30_ssa1_high
                        erexp32_error = erexp30_ssa1_high
                        erexp32_total = pl.add(erexp30_ssa1_high, erexp30_low)
                        erexp32_z = pl.sub(erexp32_total, erexp30_ssa1_high)
                        erexp32_total_minus_z = pl.sub(erexp32_total, erexp32_z)
                        erexp32_a_error = pl.sub(erexp30_ssa1_high, erexp32_total_minus_z)
                        erexp32_b_error = pl.sub(erexp30_low, erexp32_z)
                        erexp32_error = pl.add(erexp32_a_error, erexp32_b_error)
                        erexp30_ssa2_high = erexp32_total
                        erexp30_ssa2_low = erexp32_error
                        erexp2_ssa12_ch = erexp30_ssa2_high
                        erexp2_ssa12_cl = erexp30_ssa2_low
                        erexp2_coefficient_hi12 = erexp1_zero
                        erexp2_coefficient_lo12 = erexp1_zero
                        erexp2_coefficient_hi12 = pl.add(erexp1_zero, 2.0876755879584152e-09)
                        erexp2_coefficient_lo12 = pl.add(erexp1_zero, 1.1082839147459852e-16)
                        erexp34_total = erexp2_ssa12_ch
                        erexp34_error = erexp2_ssa12_ch
                        erexp34_total = pl.add(erexp2_ssa12_ch, erexp2_coefficient_hi12)
                        erexp34_z = pl.sub(erexp34_total, erexp2_ssa12_ch)
                        erexp34_total_minus_z = pl.sub(erexp34_total, erexp34_z)
                        erexp34_a_error = pl.sub(erexp2_ssa12_ch, erexp34_total_minus_z)
                        erexp34_b_error = pl.sub(erexp2_coefficient_hi12, erexp34_z)
                        erexp34_error = pl.add(erexp34_a_error, erexp34_b_error)
                        erexp33_ssa1_total = erexp34_total
                        erexp33_ssa1_error = erexp34_error
                        erexp33_low = erexp2_ssa12_cl
                        erexp33_low = pl.add(erexp2_ssa12_cl, erexp2_coefficient_lo12)
                        erexp33_tail = erexp33_ssa1_error
                        erexp33_tail = pl.add(erexp33_ssa1_error, erexp33_low)
                        erexp35_total = erexp33_ssa1_total
                        erexp35_error = erexp33_ssa1_total
                        erexp35_total = pl.add(erexp33_ssa1_total, erexp33_tail)
                        erexp35_z = pl.sub(erexp35_total, erexp33_ssa1_total)
                        erexp35_total_minus_z = pl.sub(erexp35_total, erexp35_z)
                        erexp35_a_error = pl.sub(erexp33_ssa1_total, erexp35_total_minus_z)
                        erexp35_b_error = pl.sub(erexp33_tail, erexp35_z)
                        erexp35_error = pl.add(erexp35_a_error, erexp35_b_error)
                        erexp33_ssa2_high = erexp35_total
                        erexp33_ssa2_low = erexp35_error
                        erexp2_ssa13_ch = erexp33_ssa2_high
                        erexp2_ssa13_cl = erexp33_ssa2_low
                        erexp37_product = erexp2_ssa13_ch
                        erexp37_error = erexp2_ssa13_ch
                        erexp37_a_split = pl.mul(erexp2_ssa13_ch, 4097.0)
                        erexp37_a_delta = pl.sub(erexp37_a_split, erexp2_ssa13_ch)
                        erexp37_a_hi = pl.sub(erexp37_a_split, erexp37_a_delta)
                        erexp37_a_lo = pl.sub(erexp2_ssa13_ch, erexp37_a_hi)
                        erexp37_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp37_b_delta = pl.sub(erexp37_b_split, erexp2_ssa5_rh)
                        erexp37_b_hi = pl.sub(erexp37_b_split, erexp37_b_delta)
                        erexp37_b_lo = pl.sub(erexp2_ssa5_rh, erexp37_b_hi)
                        erexp37_product = pl.mul(erexp2_ssa13_ch, erexp2_ssa5_rh)
                        erexp37_high_product = pl.mul(erexp37_a_hi, erexp37_b_hi)
                        erexp37_error = pl.sub(erexp37_high_product, erexp37_product)
                        erexp37_high_low = pl.mul(erexp37_a_hi, erexp37_b_lo)
                        erexp37_error = pl.add(erexp37_error, erexp37_high_low)
                        erexp37_low_high = pl.mul(erexp37_a_lo, erexp37_b_hi)
                        erexp37_error = pl.add(erexp37_error, erexp37_low_high)
                        erexp37_low_product = pl.mul(erexp37_a_lo, erexp37_b_lo)
                        erexp37_error = pl.add(erexp37_error, erexp37_low_product)
                        erexp36_ssa1_high = erexp37_product
                        erexp36_ssa1_error = erexp37_error
                        erexp36_cross1 = pl.mul(erexp2_ssa13_ch, erexp2_ssa5_rl)
                        erexp36_cross2 = pl.mul(erexp2_ssa13_cl, erexp2_ssa5_rh)
                        erexp36_cross3 = pl.mul(erexp2_ssa13_cl, erexp2_ssa5_rl)
                        erexp36_low = erexp36_ssa1_error
                        erexp36_low = pl.add(erexp36_ssa1_error, erexp36_cross1)
                        erexp36_low = pl.add(erexp36_low, erexp36_cross2)
                        erexp36_low = pl.add(erexp36_low, erexp36_cross3)
                        erexp38_total = erexp36_ssa1_high
                        erexp38_error = erexp36_ssa1_high
                        erexp38_total = pl.add(erexp36_ssa1_high, erexp36_low)
                        erexp38_z = pl.sub(erexp38_total, erexp36_ssa1_high)
                        erexp38_total_minus_z = pl.sub(erexp38_total, erexp38_z)
                        erexp38_a_error = pl.sub(erexp36_ssa1_high, erexp38_total_minus_z)
                        erexp38_b_error = pl.sub(erexp36_low, erexp38_z)
                        erexp38_error = pl.add(erexp38_a_error, erexp38_b_error)
                        erexp36_ssa2_high = erexp38_total
                        erexp36_ssa2_low = erexp38_error
                        erexp2_ssa14_ch = erexp36_ssa2_high
                        erexp2_ssa14_cl = erexp36_ssa2_low
                        erexp2_coefficient_hi11 = erexp1_zero
                        erexp2_coefficient_lo11 = erexp1_zero
                        erexp2_coefficient_hi11 = pl.add(erexp1_zero, 2.5052107943679403e-08)
                        erexp2_coefficient_lo11 = pl.add(erexp1_zero, 4.4176230446483665e-16)
                        erexp40_total = erexp2_ssa14_ch
                        erexp40_error = erexp2_ssa14_ch
                        erexp40_total = pl.add(erexp2_ssa14_ch, erexp2_coefficient_hi11)
                        erexp40_z = pl.sub(erexp40_total, erexp2_ssa14_ch)
                        erexp40_total_minus_z = pl.sub(erexp40_total, erexp40_z)
                        erexp40_a_error = pl.sub(erexp2_ssa14_ch, erexp40_total_minus_z)
                        erexp40_b_error = pl.sub(erexp2_coefficient_hi11, erexp40_z)
                        erexp40_error = pl.add(erexp40_a_error, erexp40_b_error)
                        erexp39_ssa1_total = erexp40_total
                        erexp39_ssa1_error = erexp40_error
                        erexp39_low = erexp2_ssa14_cl
                        erexp39_low = pl.add(erexp2_ssa14_cl, erexp2_coefficient_lo11)
                        erexp39_tail = erexp39_ssa1_error
                        erexp39_tail = pl.add(erexp39_ssa1_error, erexp39_low)
                        erexp41_total = erexp39_ssa1_total
                        erexp41_error = erexp39_ssa1_total
                        erexp41_total = pl.add(erexp39_ssa1_total, erexp39_tail)
                        erexp41_z = pl.sub(erexp41_total, erexp39_ssa1_total)
                        erexp41_total_minus_z = pl.sub(erexp41_total, erexp41_z)
                        erexp41_a_error = pl.sub(erexp39_ssa1_total, erexp41_total_minus_z)
                        erexp41_b_error = pl.sub(erexp39_tail, erexp41_z)
                        erexp41_error = pl.add(erexp41_a_error, erexp41_b_error)
                        erexp39_ssa2_high = erexp41_total
                        erexp39_ssa2_low = erexp41_error
                        erexp2_ssa15_ch = erexp39_ssa2_high
                        erexp2_ssa15_cl = erexp39_ssa2_low
                        erexp43_product = erexp2_ssa15_ch
                        erexp43_error = erexp2_ssa15_ch
                        erexp43_a_split = pl.mul(erexp2_ssa15_ch, 4097.0)
                        erexp43_a_delta = pl.sub(erexp43_a_split, erexp2_ssa15_ch)
                        erexp43_a_hi = pl.sub(erexp43_a_split, erexp43_a_delta)
                        erexp43_a_lo = pl.sub(erexp2_ssa15_ch, erexp43_a_hi)
                        erexp43_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp43_b_delta = pl.sub(erexp43_b_split, erexp2_ssa5_rh)
                        erexp43_b_hi = pl.sub(erexp43_b_split, erexp43_b_delta)
                        erexp43_b_lo = pl.sub(erexp2_ssa5_rh, erexp43_b_hi)
                        erexp43_product = pl.mul(erexp2_ssa15_ch, erexp2_ssa5_rh)
                        erexp43_high_product = pl.mul(erexp43_a_hi, erexp43_b_hi)
                        erexp43_error = pl.sub(erexp43_high_product, erexp43_product)
                        erexp43_high_low = pl.mul(erexp43_a_hi, erexp43_b_lo)
                        erexp43_error = pl.add(erexp43_error, erexp43_high_low)
                        erexp43_low_high = pl.mul(erexp43_a_lo, erexp43_b_hi)
                        erexp43_error = pl.add(erexp43_error, erexp43_low_high)
                        erexp43_low_product = pl.mul(erexp43_a_lo, erexp43_b_lo)
                        erexp43_error = pl.add(erexp43_error, erexp43_low_product)
                        erexp42_ssa1_high = erexp43_product
                        erexp42_ssa1_error = erexp43_error
                        erexp42_cross1 = pl.mul(erexp2_ssa15_ch, erexp2_ssa5_rl)
                        erexp42_cross2 = pl.mul(erexp2_ssa15_cl, erexp2_ssa5_rh)
                        erexp42_cross3 = pl.mul(erexp2_ssa15_cl, erexp2_ssa5_rl)
                        erexp42_low = erexp42_ssa1_error
                        erexp42_low = pl.add(erexp42_ssa1_error, erexp42_cross1)
                        erexp42_low = pl.add(erexp42_low, erexp42_cross2)
                        erexp42_low = pl.add(erexp42_low, erexp42_cross3)
                        erexp44_total = erexp42_ssa1_high
                        erexp44_error = erexp42_ssa1_high
                        erexp44_total = pl.add(erexp42_ssa1_high, erexp42_low)
                        erexp44_z = pl.sub(erexp44_total, erexp42_ssa1_high)
                        erexp44_total_minus_z = pl.sub(erexp44_total, erexp44_z)
                        erexp44_a_error = pl.sub(erexp42_ssa1_high, erexp44_total_minus_z)
                        erexp44_b_error = pl.sub(erexp42_low, erexp44_z)
                        erexp44_error = pl.add(erexp44_a_error, erexp44_b_error)
                        erexp42_ssa2_high = erexp44_total
                        erexp42_ssa2_low = erexp44_error
                        erexp2_ssa16_ch = erexp42_ssa2_high
                        erexp2_ssa16_cl = erexp42_ssa2_low
                        erexp2_coefficient_hi10 = erexp1_zero
                        erexp2_coefficient_lo10 = erexp1_zero
                        erexp2_coefficient_hi10 = pl.add(erexp1_zero, 2.755731998149713e-07)
                        erexp2_coefficient_lo10 = pl.add(erexp1_zero, -7.575112209051195e-15)
                        erexp46_total = erexp2_ssa16_ch
                        erexp46_error = erexp2_ssa16_ch
                        erexp46_total = pl.add(erexp2_ssa16_ch, erexp2_coefficient_hi10)
                        erexp46_z = pl.sub(erexp46_total, erexp2_ssa16_ch)
                        erexp46_total_minus_z = pl.sub(erexp46_total, erexp46_z)
                        erexp46_a_error = pl.sub(erexp2_ssa16_ch, erexp46_total_minus_z)
                        erexp46_b_error = pl.sub(erexp2_coefficient_hi10, erexp46_z)
                        erexp46_error = pl.add(erexp46_a_error, erexp46_b_error)
                        erexp45_ssa1_total = erexp46_total
                        erexp45_ssa1_error = erexp46_error
                        erexp45_low = erexp2_ssa16_cl
                        erexp45_low = pl.add(erexp2_ssa16_cl, erexp2_coefficient_lo10)
                        erexp45_tail = erexp45_ssa1_error
                        erexp45_tail = pl.add(erexp45_ssa1_error, erexp45_low)
                        erexp47_total = erexp45_ssa1_total
                        erexp47_error = erexp45_ssa1_total
                        erexp47_total = pl.add(erexp45_ssa1_total, erexp45_tail)
                        erexp47_z = pl.sub(erexp47_total, erexp45_ssa1_total)
                        erexp47_total_minus_z = pl.sub(erexp47_total, erexp47_z)
                        erexp47_a_error = pl.sub(erexp45_ssa1_total, erexp47_total_minus_z)
                        erexp47_b_error = pl.sub(erexp45_tail, erexp47_z)
                        erexp47_error = pl.add(erexp47_a_error, erexp47_b_error)
                        erexp45_ssa2_high = erexp47_total
                        erexp45_ssa2_low = erexp47_error
                        erexp2_ssa17_ch = erexp45_ssa2_high
                        erexp2_ssa17_cl = erexp45_ssa2_low
                        erexp49_product = erexp2_ssa17_ch
                        erexp49_error = erexp2_ssa17_ch
                        erexp49_a_split = pl.mul(erexp2_ssa17_ch, 4097.0)
                        erexp49_a_delta = pl.sub(erexp49_a_split, erexp2_ssa17_ch)
                        erexp49_a_hi = pl.sub(erexp49_a_split, erexp49_a_delta)
                        erexp49_a_lo = pl.sub(erexp2_ssa17_ch, erexp49_a_hi)
                        erexp49_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp49_b_delta = pl.sub(erexp49_b_split, erexp2_ssa5_rh)
                        erexp49_b_hi = pl.sub(erexp49_b_split, erexp49_b_delta)
                        erexp49_b_lo = pl.sub(erexp2_ssa5_rh, erexp49_b_hi)
                        erexp49_product = pl.mul(erexp2_ssa17_ch, erexp2_ssa5_rh)
                        erexp49_high_product = pl.mul(erexp49_a_hi, erexp49_b_hi)
                        erexp49_error = pl.sub(erexp49_high_product, erexp49_product)
                        erexp49_high_low = pl.mul(erexp49_a_hi, erexp49_b_lo)
                        erexp49_error = pl.add(erexp49_error, erexp49_high_low)
                        erexp49_low_high = pl.mul(erexp49_a_lo, erexp49_b_hi)
                        erexp49_error = pl.add(erexp49_error, erexp49_low_high)
                        erexp49_low_product = pl.mul(erexp49_a_lo, erexp49_b_lo)
                        erexp49_error = pl.add(erexp49_error, erexp49_low_product)
                        erexp48_ssa1_high = erexp49_product
                        erexp48_ssa1_error = erexp49_error
                        erexp48_cross1 = pl.mul(erexp2_ssa17_ch, erexp2_ssa5_rl)
                        erexp48_cross2 = pl.mul(erexp2_ssa17_cl, erexp2_ssa5_rh)
                        erexp48_cross3 = pl.mul(erexp2_ssa17_cl, erexp2_ssa5_rl)
                        erexp48_low = erexp48_ssa1_error
                        erexp48_low = pl.add(erexp48_ssa1_error, erexp48_cross1)
                        erexp48_low = pl.add(erexp48_low, erexp48_cross2)
                        erexp48_low = pl.add(erexp48_low, erexp48_cross3)
                        erexp50_total = erexp48_ssa1_high
                        erexp50_error = erexp48_ssa1_high
                        erexp50_total = pl.add(erexp48_ssa1_high, erexp48_low)
                        erexp50_z = pl.sub(erexp50_total, erexp48_ssa1_high)
                        erexp50_total_minus_z = pl.sub(erexp50_total, erexp50_z)
                        erexp50_a_error = pl.sub(erexp48_ssa1_high, erexp50_total_minus_z)
                        erexp50_b_error = pl.sub(erexp48_low, erexp50_z)
                        erexp50_error = pl.add(erexp50_a_error, erexp50_b_error)
                        erexp48_ssa2_high = erexp50_total
                        erexp48_ssa2_low = erexp50_error
                        erexp2_ssa18_ch = erexp48_ssa2_high
                        erexp2_ssa18_cl = erexp48_ssa2_low
                        erexp2_coefficient_hi9 = erexp1_zero
                        erexp2_coefficient_lo9 = erexp1_zero
                        erexp2_coefficient_hi9 = pl.add(erexp1_zero, 2.7557318844628753e-06)
                        erexp2_coefficient_lo9 = pl.add(erexp1_zero, 3.793571224297229e-14)
                        erexp52_total = erexp2_ssa18_ch
                        erexp52_error = erexp2_ssa18_ch
                        erexp52_total = pl.add(erexp2_ssa18_ch, erexp2_coefficient_hi9)
                        erexp52_z = pl.sub(erexp52_total, erexp2_ssa18_ch)
                        erexp52_total_minus_z = pl.sub(erexp52_total, erexp52_z)
                        erexp52_a_error = pl.sub(erexp2_ssa18_ch, erexp52_total_minus_z)
                        erexp52_b_error = pl.sub(erexp2_coefficient_hi9, erexp52_z)
                        erexp52_error = pl.add(erexp52_a_error, erexp52_b_error)
                        erexp51_ssa1_total = erexp52_total
                        erexp51_ssa1_error = erexp52_error
                        erexp51_low = erexp2_ssa18_cl
                        erexp51_low = pl.add(erexp2_ssa18_cl, erexp2_coefficient_lo9)
                        erexp51_tail = erexp51_ssa1_error
                        erexp51_tail = pl.add(erexp51_ssa1_error, erexp51_low)
                        erexp53_total = erexp51_ssa1_total
                        erexp53_error = erexp51_ssa1_total
                        erexp53_total = pl.add(erexp51_ssa1_total, erexp51_tail)
                        erexp53_z = pl.sub(erexp53_total, erexp51_ssa1_total)
                        erexp53_total_minus_z = pl.sub(erexp53_total, erexp53_z)
                        erexp53_a_error = pl.sub(erexp51_ssa1_total, erexp53_total_minus_z)
                        erexp53_b_error = pl.sub(erexp51_tail, erexp53_z)
                        erexp53_error = pl.add(erexp53_a_error, erexp53_b_error)
                        erexp51_ssa2_high = erexp53_total
                        erexp51_ssa2_low = erexp53_error
                        erexp2_ssa19_ch = erexp51_ssa2_high
                        erexp2_ssa19_cl = erexp51_ssa2_low
                        erexp55_product = erexp2_ssa19_ch
                        erexp55_error = erexp2_ssa19_ch
                        erexp55_a_split = pl.mul(erexp2_ssa19_ch, 4097.0)
                        erexp55_a_delta = pl.sub(erexp55_a_split, erexp2_ssa19_ch)
                        erexp55_a_hi = pl.sub(erexp55_a_split, erexp55_a_delta)
                        erexp55_a_lo = pl.sub(erexp2_ssa19_ch, erexp55_a_hi)
                        erexp55_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp55_b_delta = pl.sub(erexp55_b_split, erexp2_ssa5_rh)
                        erexp55_b_hi = pl.sub(erexp55_b_split, erexp55_b_delta)
                        erexp55_b_lo = pl.sub(erexp2_ssa5_rh, erexp55_b_hi)
                        erexp55_product = pl.mul(erexp2_ssa19_ch, erexp2_ssa5_rh)
                        erexp55_high_product = pl.mul(erexp55_a_hi, erexp55_b_hi)
                        erexp55_error = pl.sub(erexp55_high_product, erexp55_product)
                        erexp55_high_low = pl.mul(erexp55_a_hi, erexp55_b_lo)
                        erexp55_error = pl.add(erexp55_error, erexp55_high_low)
                        erexp55_low_high = pl.mul(erexp55_a_lo, erexp55_b_hi)
                        erexp55_error = pl.add(erexp55_error, erexp55_low_high)
                        erexp55_low_product = pl.mul(erexp55_a_lo, erexp55_b_lo)
                        erexp55_error = pl.add(erexp55_error, erexp55_low_product)
                        erexp54_ssa1_high = erexp55_product
                        erexp54_ssa1_error = erexp55_error
                        erexp54_cross1 = pl.mul(erexp2_ssa19_ch, erexp2_ssa5_rl)
                        erexp54_cross2 = pl.mul(erexp2_ssa19_cl, erexp2_ssa5_rh)
                        erexp54_cross3 = pl.mul(erexp2_ssa19_cl, erexp2_ssa5_rl)
                        erexp54_low = erexp54_ssa1_error
                        erexp54_low = pl.add(erexp54_ssa1_error, erexp54_cross1)
                        erexp54_low = pl.add(erexp54_low, erexp54_cross2)
                        erexp54_low = pl.add(erexp54_low, erexp54_cross3)
                        erexp56_total = erexp54_ssa1_high
                        erexp56_error = erexp54_ssa1_high
                        erexp56_total = pl.add(erexp54_ssa1_high, erexp54_low)
                        erexp56_z = pl.sub(erexp56_total, erexp54_ssa1_high)
                        erexp56_total_minus_z = pl.sub(erexp56_total, erexp56_z)
                        erexp56_a_error = pl.sub(erexp54_ssa1_high, erexp56_total_minus_z)
                        erexp56_b_error = pl.sub(erexp54_low, erexp56_z)
                        erexp56_error = pl.add(erexp56_a_error, erexp56_b_error)
                        erexp54_ssa2_high = erexp56_total
                        erexp54_ssa2_low = erexp56_error
                        erexp2_ssa20_ch = erexp54_ssa2_high
                        erexp2_ssa20_cl = erexp54_ssa2_low
                        erexp2_coefficient_hi8 = erexp1_zero
                        erexp2_coefficient_lo8 = erexp1_zero
                        erexp2_coefficient_hi8 = pl.add(erexp1_zero, 2.4801587642286904e-05)
                        erexp2_coefficient_lo8 = pl.add(erexp1_zero, -3.40699609366682e-13)
                        erexp58_total = erexp2_ssa20_ch
                        erexp58_error = erexp2_ssa20_ch
                        erexp58_total = pl.add(erexp2_ssa20_ch, erexp2_coefficient_hi8)
                        erexp58_z = pl.sub(erexp58_total, erexp2_ssa20_ch)
                        erexp58_total_minus_z = pl.sub(erexp58_total, erexp58_z)
                        erexp58_a_error = pl.sub(erexp2_ssa20_ch, erexp58_total_minus_z)
                        erexp58_b_error = pl.sub(erexp2_coefficient_hi8, erexp58_z)
                        erexp58_error = pl.add(erexp58_a_error, erexp58_b_error)
                        erexp57_ssa1_total = erexp58_total
                        erexp57_ssa1_error = erexp58_error
                        erexp57_low = erexp2_ssa20_cl
                        erexp57_low = pl.add(erexp2_ssa20_cl, erexp2_coefficient_lo8)
                        erexp57_tail = erexp57_ssa1_error
                        erexp57_tail = pl.add(erexp57_ssa1_error, erexp57_low)
                        erexp59_total = erexp57_ssa1_total
                        erexp59_error = erexp57_ssa1_total
                        erexp59_total = pl.add(erexp57_ssa1_total, erexp57_tail)
                        erexp59_z = pl.sub(erexp59_total, erexp57_ssa1_total)
                        erexp59_total_minus_z = pl.sub(erexp59_total, erexp59_z)
                        erexp59_a_error = pl.sub(erexp57_ssa1_total, erexp59_total_minus_z)
                        erexp59_b_error = pl.sub(erexp57_tail, erexp59_z)
                        erexp59_error = pl.add(erexp59_a_error, erexp59_b_error)
                        erexp57_ssa2_high = erexp59_total
                        erexp57_ssa2_low = erexp59_error
                        erexp2_ssa21_ch = erexp57_ssa2_high
                        erexp2_ssa21_cl = erexp57_ssa2_low
                        erexp61_product = erexp2_ssa21_ch
                        erexp61_error = erexp2_ssa21_ch
                        erexp61_a_split = pl.mul(erexp2_ssa21_ch, 4097.0)
                        erexp61_a_delta = pl.sub(erexp61_a_split, erexp2_ssa21_ch)
                        erexp61_a_hi = pl.sub(erexp61_a_split, erexp61_a_delta)
                        erexp61_a_lo = pl.sub(erexp2_ssa21_ch, erexp61_a_hi)
                        erexp61_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp61_b_delta = pl.sub(erexp61_b_split, erexp2_ssa5_rh)
                        erexp61_b_hi = pl.sub(erexp61_b_split, erexp61_b_delta)
                        erexp61_b_lo = pl.sub(erexp2_ssa5_rh, erexp61_b_hi)
                        erexp61_product = pl.mul(erexp2_ssa21_ch, erexp2_ssa5_rh)
                        erexp61_high_product = pl.mul(erexp61_a_hi, erexp61_b_hi)
                        erexp61_error = pl.sub(erexp61_high_product, erexp61_product)
                        erexp61_high_low = pl.mul(erexp61_a_hi, erexp61_b_lo)
                        erexp61_error = pl.add(erexp61_error, erexp61_high_low)
                        erexp61_low_high = pl.mul(erexp61_a_lo, erexp61_b_hi)
                        erexp61_error = pl.add(erexp61_error, erexp61_low_high)
                        erexp61_low_product = pl.mul(erexp61_a_lo, erexp61_b_lo)
                        erexp61_error = pl.add(erexp61_error, erexp61_low_product)
                        erexp60_ssa1_high = erexp61_product
                        erexp60_ssa1_error = erexp61_error
                        erexp60_cross1 = pl.mul(erexp2_ssa21_ch, erexp2_ssa5_rl)
                        erexp60_cross2 = pl.mul(erexp2_ssa21_cl, erexp2_ssa5_rh)
                        erexp60_cross3 = pl.mul(erexp2_ssa21_cl, erexp2_ssa5_rl)
                        erexp60_low = erexp60_ssa1_error
                        erexp60_low = pl.add(erexp60_ssa1_error, erexp60_cross1)
                        erexp60_low = pl.add(erexp60_low, erexp60_cross2)
                        erexp60_low = pl.add(erexp60_low, erexp60_cross3)
                        erexp62_total = erexp60_ssa1_high
                        erexp62_error = erexp60_ssa1_high
                        erexp62_total = pl.add(erexp60_ssa1_high, erexp60_low)
                        erexp62_z = pl.sub(erexp62_total, erexp60_ssa1_high)
                        erexp62_total_minus_z = pl.sub(erexp62_total, erexp62_z)
                        erexp62_a_error = pl.sub(erexp60_ssa1_high, erexp62_total_minus_z)
                        erexp62_b_error = pl.sub(erexp60_low, erexp62_z)
                        erexp62_error = pl.add(erexp62_a_error, erexp62_b_error)
                        erexp60_ssa2_high = erexp62_total
                        erexp60_ssa2_low = erexp62_error
                        erexp2_ssa22_ch = erexp60_ssa2_high
                        erexp2_ssa22_cl = erexp60_ssa2_low
                        erexp2_coefficient_hi7 = erexp1_zero
                        erexp2_coefficient_lo7 = erexp1_zero
                        erexp2_coefficient_hi7 = pl.add(erexp1_zero, 0.00019841270113829523)
                        erexp2_coefficient_lo7 = pl.add(erexp1_zero, -2.725596874933456e-12)
                        erexp64_total = erexp2_ssa22_ch
                        erexp64_error = erexp2_ssa22_ch
                        erexp64_total = pl.add(erexp2_ssa22_ch, erexp2_coefficient_hi7)
                        erexp64_z = pl.sub(erexp64_total, erexp2_ssa22_ch)
                        erexp64_total_minus_z = pl.sub(erexp64_total, erexp64_z)
                        erexp64_a_error = pl.sub(erexp2_ssa22_ch, erexp64_total_minus_z)
                        erexp64_b_error = pl.sub(erexp2_coefficient_hi7, erexp64_z)
                        erexp64_error = pl.add(erexp64_a_error, erexp64_b_error)
                        erexp63_ssa1_total = erexp64_total
                        erexp63_ssa1_error = erexp64_error
                        erexp63_low = erexp2_ssa22_cl
                        erexp63_low = pl.add(erexp2_ssa22_cl, erexp2_coefficient_lo7)
                        erexp63_tail = erexp63_ssa1_error
                        erexp63_tail = pl.add(erexp63_ssa1_error, erexp63_low)
                        erexp65_total = erexp63_ssa1_total
                        erexp65_error = erexp63_ssa1_total
                        erexp65_total = pl.add(erexp63_ssa1_total, erexp63_tail)
                        erexp65_z = pl.sub(erexp65_total, erexp63_ssa1_total)
                        erexp65_total_minus_z = pl.sub(erexp65_total, erexp65_z)
                        erexp65_a_error = pl.sub(erexp63_ssa1_total, erexp65_total_minus_z)
                        erexp65_b_error = pl.sub(erexp63_tail, erexp65_z)
                        erexp65_error = pl.add(erexp65_a_error, erexp65_b_error)
                        erexp63_ssa2_high = erexp65_total
                        erexp63_ssa2_low = erexp65_error
                        erexp2_ssa23_ch = erexp63_ssa2_high
                        erexp2_ssa23_cl = erexp63_ssa2_low
                        erexp67_product = erexp2_ssa23_ch
                        erexp67_error = erexp2_ssa23_ch
                        erexp67_a_split = pl.mul(erexp2_ssa23_ch, 4097.0)
                        erexp67_a_delta = pl.sub(erexp67_a_split, erexp2_ssa23_ch)
                        erexp67_a_hi = pl.sub(erexp67_a_split, erexp67_a_delta)
                        erexp67_a_lo = pl.sub(erexp2_ssa23_ch, erexp67_a_hi)
                        erexp67_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp67_b_delta = pl.sub(erexp67_b_split, erexp2_ssa5_rh)
                        erexp67_b_hi = pl.sub(erexp67_b_split, erexp67_b_delta)
                        erexp67_b_lo = pl.sub(erexp2_ssa5_rh, erexp67_b_hi)
                        erexp67_product = pl.mul(erexp2_ssa23_ch, erexp2_ssa5_rh)
                        erexp67_high_product = pl.mul(erexp67_a_hi, erexp67_b_hi)
                        erexp67_error = pl.sub(erexp67_high_product, erexp67_product)
                        erexp67_high_low = pl.mul(erexp67_a_hi, erexp67_b_lo)
                        erexp67_error = pl.add(erexp67_error, erexp67_high_low)
                        erexp67_low_high = pl.mul(erexp67_a_lo, erexp67_b_hi)
                        erexp67_error = pl.add(erexp67_error, erexp67_low_high)
                        erexp67_low_product = pl.mul(erexp67_a_lo, erexp67_b_lo)
                        erexp67_error = pl.add(erexp67_error, erexp67_low_product)
                        erexp66_ssa1_high = erexp67_product
                        erexp66_ssa1_error = erexp67_error
                        erexp66_cross1 = pl.mul(erexp2_ssa23_ch, erexp2_ssa5_rl)
                        erexp66_cross2 = pl.mul(erexp2_ssa23_cl, erexp2_ssa5_rh)
                        erexp66_cross3 = pl.mul(erexp2_ssa23_cl, erexp2_ssa5_rl)
                        erexp66_low = erexp66_ssa1_error
                        erexp66_low = pl.add(erexp66_ssa1_error, erexp66_cross1)
                        erexp66_low = pl.add(erexp66_low, erexp66_cross2)
                        erexp66_low = pl.add(erexp66_low, erexp66_cross3)
                        erexp68_total = erexp66_ssa1_high
                        erexp68_error = erexp66_ssa1_high
                        erexp68_total = pl.add(erexp66_ssa1_high, erexp66_low)
                        erexp68_z = pl.sub(erexp68_total, erexp66_ssa1_high)
                        erexp68_total_minus_z = pl.sub(erexp68_total, erexp68_z)
                        erexp68_a_error = pl.sub(erexp66_ssa1_high, erexp68_total_minus_z)
                        erexp68_b_error = pl.sub(erexp66_low, erexp68_z)
                        erexp68_error = pl.add(erexp68_a_error, erexp68_b_error)
                        erexp66_ssa2_high = erexp68_total
                        erexp66_ssa2_low = erexp68_error
                        erexp2_ssa24_ch = erexp66_ssa2_high
                        erexp2_ssa24_cl = erexp66_ssa2_low
                        erexp2_coefficient_hi6 = erexp1_zero
                        erexp2_coefficient_lo6 = erexp1_zero
                        erexp2_coefficient_hi6 = pl.add(erexp1_zero, 0.0013888889225199819)
                        erexp2_coefficient_lo6 = pl.add(erexp1_zero, -3.3631094437103215e-11)
                        erexp70_total = erexp2_ssa24_ch
                        erexp70_error = erexp2_ssa24_ch
                        erexp70_total = pl.add(erexp2_ssa24_ch, erexp2_coefficient_hi6)
                        erexp70_z = pl.sub(erexp70_total, erexp2_ssa24_ch)
                        erexp70_total_minus_z = pl.sub(erexp70_total, erexp70_z)
                        erexp70_a_error = pl.sub(erexp2_ssa24_ch, erexp70_total_minus_z)
                        erexp70_b_error = pl.sub(erexp2_coefficient_hi6, erexp70_z)
                        erexp70_error = pl.add(erexp70_a_error, erexp70_b_error)
                        erexp69_ssa1_total = erexp70_total
                        erexp69_ssa1_error = erexp70_error
                        erexp69_low = erexp2_ssa24_cl
                        erexp69_low = pl.add(erexp2_ssa24_cl, erexp2_coefficient_lo6)
                        erexp69_tail = erexp69_ssa1_error
                        erexp69_tail = pl.add(erexp69_ssa1_error, erexp69_low)
                        erexp71_total = erexp69_ssa1_total
                        erexp71_error = erexp69_ssa1_total
                        erexp71_total = pl.add(erexp69_ssa1_total, erexp69_tail)
                        erexp71_z = pl.sub(erexp71_total, erexp69_ssa1_total)
                        erexp71_total_minus_z = pl.sub(erexp71_total, erexp71_z)
                        erexp71_a_error = pl.sub(erexp69_ssa1_total, erexp71_total_minus_z)
                        erexp71_b_error = pl.sub(erexp69_tail, erexp71_z)
                        erexp71_error = pl.add(erexp71_a_error, erexp71_b_error)
                        erexp69_ssa2_high = erexp71_total
                        erexp69_ssa2_low = erexp71_error
                        erexp2_ssa25_ch = erexp69_ssa2_high
                        erexp2_ssa25_cl = erexp69_ssa2_low
                        erexp73_product = erexp2_ssa25_ch
                        erexp73_error = erexp2_ssa25_ch
                        erexp73_a_split = pl.mul(erexp2_ssa25_ch, 4097.0)
                        erexp73_a_delta = pl.sub(erexp73_a_split, erexp2_ssa25_ch)
                        erexp73_a_hi = pl.sub(erexp73_a_split, erexp73_a_delta)
                        erexp73_a_lo = pl.sub(erexp2_ssa25_ch, erexp73_a_hi)
                        erexp73_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp73_b_delta = pl.sub(erexp73_b_split, erexp2_ssa5_rh)
                        erexp73_b_hi = pl.sub(erexp73_b_split, erexp73_b_delta)
                        erexp73_b_lo = pl.sub(erexp2_ssa5_rh, erexp73_b_hi)
                        erexp73_product = pl.mul(erexp2_ssa25_ch, erexp2_ssa5_rh)
                        erexp73_high_product = pl.mul(erexp73_a_hi, erexp73_b_hi)
                        erexp73_error = pl.sub(erexp73_high_product, erexp73_product)
                        erexp73_high_low = pl.mul(erexp73_a_hi, erexp73_b_lo)
                        erexp73_error = pl.add(erexp73_error, erexp73_high_low)
                        erexp73_low_high = pl.mul(erexp73_a_lo, erexp73_b_hi)
                        erexp73_error = pl.add(erexp73_error, erexp73_low_high)
                        erexp73_low_product = pl.mul(erexp73_a_lo, erexp73_b_lo)
                        erexp73_error = pl.add(erexp73_error, erexp73_low_product)
                        erexp72_ssa1_high = erexp73_product
                        erexp72_ssa1_error = erexp73_error
                        erexp72_cross1 = pl.mul(erexp2_ssa25_ch, erexp2_ssa5_rl)
                        erexp72_cross2 = pl.mul(erexp2_ssa25_cl, erexp2_ssa5_rh)
                        erexp72_cross3 = pl.mul(erexp2_ssa25_cl, erexp2_ssa5_rl)
                        erexp72_low = erexp72_ssa1_error
                        erexp72_low = pl.add(erexp72_ssa1_error, erexp72_cross1)
                        erexp72_low = pl.add(erexp72_low, erexp72_cross2)
                        erexp72_low = pl.add(erexp72_low, erexp72_cross3)
                        erexp74_total = erexp72_ssa1_high
                        erexp74_error = erexp72_ssa1_high
                        erexp74_total = pl.add(erexp72_ssa1_high, erexp72_low)
                        erexp74_z = pl.sub(erexp74_total, erexp72_ssa1_high)
                        erexp74_total_minus_z = pl.sub(erexp74_total, erexp74_z)
                        erexp74_a_error = pl.sub(erexp72_ssa1_high, erexp74_total_minus_z)
                        erexp74_b_error = pl.sub(erexp72_low, erexp74_z)
                        erexp74_error = pl.add(erexp74_a_error, erexp74_b_error)
                        erexp72_ssa2_high = erexp74_total
                        erexp72_ssa2_low = erexp74_error
                        erexp2_ssa26_ch = erexp72_ssa2_high
                        erexp2_ssa26_cl = erexp72_ssa2_low
                        erexp2_coefficient_hi5 = erexp1_zero
                        erexp2_coefficient_lo5 = erexp1_zero
                        erexp2_coefficient_hi5 = pl.add(erexp1_zero, 0.008333333767950535)
                        erexp2_coefficient_lo5 = pl.add(erexp1_zero, -4.34617203337595e-10)
                        erexp76_total = erexp2_ssa26_ch
                        erexp76_error = erexp2_ssa26_ch
                        erexp76_total = pl.add(erexp2_ssa26_ch, erexp2_coefficient_hi5)
                        erexp76_z = pl.sub(erexp76_total, erexp2_ssa26_ch)
                        erexp76_total_minus_z = pl.sub(erexp76_total, erexp76_z)
                        erexp76_a_error = pl.sub(erexp2_ssa26_ch, erexp76_total_minus_z)
                        erexp76_b_error = pl.sub(erexp2_coefficient_hi5, erexp76_z)
                        erexp76_error = pl.add(erexp76_a_error, erexp76_b_error)
                        erexp75_ssa1_total = erexp76_total
                        erexp75_ssa1_error = erexp76_error
                        erexp75_low = erexp2_ssa26_cl
                        erexp75_low = pl.add(erexp2_ssa26_cl, erexp2_coefficient_lo5)
                        erexp75_tail = erexp75_ssa1_error
                        erexp75_tail = pl.add(erexp75_ssa1_error, erexp75_low)
                        erexp77_total = erexp75_ssa1_total
                        erexp77_error = erexp75_ssa1_total
                        erexp77_total = pl.add(erexp75_ssa1_total, erexp75_tail)
                        erexp77_z = pl.sub(erexp77_total, erexp75_ssa1_total)
                        erexp77_total_minus_z = pl.sub(erexp77_total, erexp77_z)
                        erexp77_a_error = pl.sub(erexp75_ssa1_total, erexp77_total_minus_z)
                        erexp77_b_error = pl.sub(erexp75_tail, erexp77_z)
                        erexp77_error = pl.add(erexp77_a_error, erexp77_b_error)
                        erexp75_ssa2_high = erexp77_total
                        erexp75_ssa2_low = erexp77_error
                        erexp2_ssa27_ch = erexp75_ssa2_high
                        erexp2_ssa27_cl = erexp75_ssa2_low
                        erexp79_product = erexp2_ssa27_ch
                        erexp79_error = erexp2_ssa27_ch
                        erexp79_a_split = pl.mul(erexp2_ssa27_ch, 4097.0)
                        erexp79_a_delta = pl.sub(erexp79_a_split, erexp2_ssa27_ch)
                        erexp79_a_hi = pl.sub(erexp79_a_split, erexp79_a_delta)
                        erexp79_a_lo = pl.sub(erexp2_ssa27_ch, erexp79_a_hi)
                        erexp79_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp79_b_delta = pl.sub(erexp79_b_split, erexp2_ssa5_rh)
                        erexp79_b_hi = pl.sub(erexp79_b_split, erexp79_b_delta)
                        erexp79_b_lo = pl.sub(erexp2_ssa5_rh, erexp79_b_hi)
                        erexp79_product = pl.mul(erexp2_ssa27_ch, erexp2_ssa5_rh)
                        erexp79_high_product = pl.mul(erexp79_a_hi, erexp79_b_hi)
                        erexp79_error = pl.sub(erexp79_high_product, erexp79_product)
                        erexp79_high_low = pl.mul(erexp79_a_hi, erexp79_b_lo)
                        erexp79_error = pl.add(erexp79_error, erexp79_high_low)
                        erexp79_low_high = pl.mul(erexp79_a_lo, erexp79_b_hi)
                        erexp79_error = pl.add(erexp79_error, erexp79_low_high)
                        erexp79_low_product = pl.mul(erexp79_a_lo, erexp79_b_lo)
                        erexp79_error = pl.add(erexp79_error, erexp79_low_product)
                        erexp78_ssa1_high = erexp79_product
                        erexp78_ssa1_error = erexp79_error
                        erexp78_cross1 = pl.mul(erexp2_ssa27_ch, erexp2_ssa5_rl)
                        erexp78_cross2 = pl.mul(erexp2_ssa27_cl, erexp2_ssa5_rh)
                        erexp78_cross3 = pl.mul(erexp2_ssa27_cl, erexp2_ssa5_rl)
                        erexp78_low = erexp78_ssa1_error
                        erexp78_low = pl.add(erexp78_ssa1_error, erexp78_cross1)
                        erexp78_low = pl.add(erexp78_low, erexp78_cross2)
                        erexp78_low = pl.add(erexp78_low, erexp78_cross3)
                        erexp80_total = erexp78_ssa1_high
                        erexp80_error = erexp78_ssa1_high
                        erexp80_total = pl.add(erexp78_ssa1_high, erexp78_low)
                        erexp80_z = pl.sub(erexp80_total, erexp78_ssa1_high)
                        erexp80_total_minus_z = pl.sub(erexp80_total, erexp80_z)
                        erexp80_a_error = pl.sub(erexp78_ssa1_high, erexp80_total_minus_z)
                        erexp80_b_error = pl.sub(erexp78_low, erexp80_z)
                        erexp80_error = pl.add(erexp80_a_error, erexp80_b_error)
                        erexp78_ssa2_high = erexp80_total
                        erexp78_ssa2_low = erexp80_error
                        erexp2_ssa28_ch = erexp78_ssa2_high
                        erexp2_ssa28_cl = erexp78_ssa2_low
                        erexp2_coefficient_hi4 = erexp1_zero
                        erexp2_coefficient_lo4 = erexp1_zero
                        erexp2_coefficient_hi4 = pl.add(erexp1_zero, 0.0416666679084301)
                        erexp2_coefficient_lo4 = pl.add(erexp1_zero, -1.2417634698280722e-09)
                        erexp82_total = erexp2_ssa28_ch
                        erexp82_error = erexp2_ssa28_ch
                        erexp82_total = pl.add(erexp2_ssa28_ch, erexp2_coefficient_hi4)
                        erexp82_z = pl.sub(erexp82_total, erexp2_ssa28_ch)
                        erexp82_total_minus_z = pl.sub(erexp82_total, erexp82_z)
                        erexp82_a_error = pl.sub(erexp2_ssa28_ch, erexp82_total_minus_z)
                        erexp82_b_error = pl.sub(erexp2_coefficient_hi4, erexp82_z)
                        erexp82_error = pl.add(erexp82_a_error, erexp82_b_error)
                        erexp81_ssa1_total = erexp82_total
                        erexp81_ssa1_error = erexp82_error
                        erexp81_low = erexp2_ssa28_cl
                        erexp81_low = pl.add(erexp2_ssa28_cl, erexp2_coefficient_lo4)
                        erexp81_tail = erexp81_ssa1_error
                        erexp81_tail = pl.add(erexp81_ssa1_error, erexp81_low)
                        erexp83_total = erexp81_ssa1_total
                        erexp83_error = erexp81_ssa1_total
                        erexp83_total = pl.add(erexp81_ssa1_total, erexp81_tail)
                        erexp83_z = pl.sub(erexp83_total, erexp81_ssa1_total)
                        erexp83_total_minus_z = pl.sub(erexp83_total, erexp83_z)
                        erexp83_a_error = pl.sub(erexp81_ssa1_total, erexp83_total_minus_z)
                        erexp83_b_error = pl.sub(erexp81_tail, erexp83_z)
                        erexp83_error = pl.add(erexp83_a_error, erexp83_b_error)
                        erexp81_ssa2_high = erexp83_total
                        erexp81_ssa2_low = erexp83_error
                        erexp2_ssa29_ch = erexp81_ssa2_high
                        erexp2_ssa29_cl = erexp81_ssa2_low
                        erexp85_product = erexp2_ssa29_ch
                        erexp85_error = erexp2_ssa29_ch
                        erexp85_a_split = pl.mul(erexp2_ssa29_ch, 4097.0)
                        erexp85_a_delta = pl.sub(erexp85_a_split, erexp2_ssa29_ch)
                        erexp85_a_hi = pl.sub(erexp85_a_split, erexp85_a_delta)
                        erexp85_a_lo = pl.sub(erexp2_ssa29_ch, erexp85_a_hi)
                        erexp85_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp85_b_delta = pl.sub(erexp85_b_split, erexp2_ssa5_rh)
                        erexp85_b_hi = pl.sub(erexp85_b_split, erexp85_b_delta)
                        erexp85_b_lo = pl.sub(erexp2_ssa5_rh, erexp85_b_hi)
                        erexp85_product = pl.mul(erexp2_ssa29_ch, erexp2_ssa5_rh)
                        erexp85_high_product = pl.mul(erexp85_a_hi, erexp85_b_hi)
                        erexp85_error = pl.sub(erexp85_high_product, erexp85_product)
                        erexp85_high_low = pl.mul(erexp85_a_hi, erexp85_b_lo)
                        erexp85_error = pl.add(erexp85_error, erexp85_high_low)
                        erexp85_low_high = pl.mul(erexp85_a_lo, erexp85_b_hi)
                        erexp85_error = pl.add(erexp85_error, erexp85_low_high)
                        erexp85_low_product = pl.mul(erexp85_a_lo, erexp85_b_lo)
                        erexp85_error = pl.add(erexp85_error, erexp85_low_product)
                        erexp84_ssa1_high = erexp85_product
                        erexp84_ssa1_error = erexp85_error
                        erexp84_cross1 = pl.mul(erexp2_ssa29_ch, erexp2_ssa5_rl)
                        erexp84_cross2 = pl.mul(erexp2_ssa29_cl, erexp2_ssa5_rh)
                        erexp84_cross3 = pl.mul(erexp2_ssa29_cl, erexp2_ssa5_rl)
                        erexp84_low = erexp84_ssa1_error
                        erexp84_low = pl.add(erexp84_ssa1_error, erexp84_cross1)
                        erexp84_low = pl.add(erexp84_low, erexp84_cross2)
                        erexp84_low = pl.add(erexp84_low, erexp84_cross3)
                        erexp86_total = erexp84_ssa1_high
                        erexp86_error = erexp84_ssa1_high
                        erexp86_total = pl.add(erexp84_ssa1_high, erexp84_low)
                        erexp86_z = pl.sub(erexp86_total, erexp84_ssa1_high)
                        erexp86_total_minus_z = pl.sub(erexp86_total, erexp86_z)
                        erexp86_a_error = pl.sub(erexp84_ssa1_high, erexp86_total_minus_z)
                        erexp86_b_error = pl.sub(erexp84_low, erexp86_z)
                        erexp86_error = pl.add(erexp86_a_error, erexp86_b_error)
                        erexp84_ssa2_high = erexp86_total
                        erexp84_ssa2_low = erexp86_error
                        erexp2_ssa30_ch = erexp84_ssa2_high
                        erexp2_ssa30_cl = erexp84_ssa2_low
                        erexp2_coefficient_hi3 = erexp1_zero
                        erexp2_coefficient_lo3 = erexp1_zero
                        erexp2_coefficient_hi3 = pl.add(erexp1_zero, 0.1666666716337204)
                        erexp2_coefficient_lo3 = pl.add(erexp1_zero, -4.967053879312289e-09)
                        erexp88_total = erexp2_ssa30_ch
                        erexp88_error = erexp2_ssa30_ch
                        erexp88_total = pl.add(erexp2_ssa30_ch, erexp2_coefficient_hi3)
                        erexp88_z = pl.sub(erexp88_total, erexp2_ssa30_ch)
                        erexp88_total_minus_z = pl.sub(erexp88_total, erexp88_z)
                        erexp88_a_error = pl.sub(erexp2_ssa30_ch, erexp88_total_minus_z)
                        erexp88_b_error = pl.sub(erexp2_coefficient_hi3, erexp88_z)
                        erexp88_error = pl.add(erexp88_a_error, erexp88_b_error)
                        erexp87_ssa1_total = erexp88_total
                        erexp87_ssa1_error = erexp88_error
                        erexp87_low = erexp2_ssa30_cl
                        erexp87_low = pl.add(erexp2_ssa30_cl, erexp2_coefficient_lo3)
                        erexp87_tail = erexp87_ssa1_error
                        erexp87_tail = pl.add(erexp87_ssa1_error, erexp87_low)
                        erexp89_total = erexp87_ssa1_total
                        erexp89_error = erexp87_ssa1_total
                        erexp89_total = pl.add(erexp87_ssa1_total, erexp87_tail)
                        erexp89_z = pl.sub(erexp89_total, erexp87_ssa1_total)
                        erexp89_total_minus_z = pl.sub(erexp89_total, erexp89_z)
                        erexp89_a_error = pl.sub(erexp87_ssa1_total, erexp89_total_minus_z)
                        erexp89_b_error = pl.sub(erexp87_tail, erexp89_z)
                        erexp89_error = pl.add(erexp89_a_error, erexp89_b_error)
                        erexp87_ssa2_high = erexp89_total
                        erexp87_ssa2_low = erexp89_error
                        erexp2_ssa31_ch = erexp87_ssa2_high
                        erexp2_ssa31_cl = erexp87_ssa2_low
                        erexp91_product = erexp2_ssa31_ch
                        erexp91_error = erexp2_ssa31_ch
                        erexp91_a_split = pl.mul(erexp2_ssa31_ch, 4097.0)
                        erexp91_a_delta = pl.sub(erexp91_a_split, erexp2_ssa31_ch)
                        erexp91_a_hi = pl.sub(erexp91_a_split, erexp91_a_delta)
                        erexp91_a_lo = pl.sub(erexp2_ssa31_ch, erexp91_a_hi)
                        erexp91_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp91_b_delta = pl.sub(erexp91_b_split, erexp2_ssa5_rh)
                        erexp91_b_hi = pl.sub(erexp91_b_split, erexp91_b_delta)
                        erexp91_b_lo = pl.sub(erexp2_ssa5_rh, erexp91_b_hi)
                        erexp91_product = pl.mul(erexp2_ssa31_ch, erexp2_ssa5_rh)
                        erexp91_high_product = pl.mul(erexp91_a_hi, erexp91_b_hi)
                        erexp91_error = pl.sub(erexp91_high_product, erexp91_product)
                        erexp91_high_low = pl.mul(erexp91_a_hi, erexp91_b_lo)
                        erexp91_error = pl.add(erexp91_error, erexp91_high_low)
                        erexp91_low_high = pl.mul(erexp91_a_lo, erexp91_b_hi)
                        erexp91_error = pl.add(erexp91_error, erexp91_low_high)
                        erexp91_low_product = pl.mul(erexp91_a_lo, erexp91_b_lo)
                        erexp91_error = pl.add(erexp91_error, erexp91_low_product)
                        erexp90_ssa1_high = erexp91_product
                        erexp90_ssa1_error = erexp91_error
                        erexp90_cross1 = pl.mul(erexp2_ssa31_ch, erexp2_ssa5_rl)
                        erexp90_cross2 = pl.mul(erexp2_ssa31_cl, erexp2_ssa5_rh)
                        erexp90_cross3 = pl.mul(erexp2_ssa31_cl, erexp2_ssa5_rl)
                        erexp90_low = erexp90_ssa1_error
                        erexp90_low = pl.add(erexp90_ssa1_error, erexp90_cross1)
                        erexp90_low = pl.add(erexp90_low, erexp90_cross2)
                        erexp90_low = pl.add(erexp90_low, erexp90_cross3)
                        erexp92_total = erexp90_ssa1_high
                        erexp92_error = erexp90_ssa1_high
                        erexp92_total = pl.add(erexp90_ssa1_high, erexp90_low)
                        erexp92_z = pl.sub(erexp92_total, erexp90_ssa1_high)
                        erexp92_total_minus_z = pl.sub(erexp92_total, erexp92_z)
                        erexp92_a_error = pl.sub(erexp90_ssa1_high, erexp92_total_minus_z)
                        erexp92_b_error = pl.sub(erexp90_low, erexp92_z)
                        erexp92_error = pl.add(erexp92_a_error, erexp92_b_error)
                        erexp90_ssa2_high = erexp92_total
                        erexp90_ssa2_low = erexp92_error
                        erexp2_ssa32_ch = erexp90_ssa2_high
                        erexp2_ssa32_cl = erexp90_ssa2_low
                        erexp2_coefficient_hi2 = erexp1_zero
                        erexp2_coefficient_lo2 = erexp1_zero
                        erexp2_coefficient_hi2 = pl.add(erexp1_zero, 0.5)
                        erexp2_coefficient_lo2 = pl.add(erexp1_zero, 0.0)
                        erexp94_total = erexp2_ssa32_ch
                        erexp94_error = erexp2_ssa32_ch
                        erexp94_total = pl.add(erexp2_ssa32_ch, erexp2_coefficient_hi2)
                        erexp94_z = pl.sub(erexp94_total, erexp2_ssa32_ch)
                        erexp94_total_minus_z = pl.sub(erexp94_total, erexp94_z)
                        erexp94_a_error = pl.sub(erexp2_ssa32_ch, erexp94_total_minus_z)
                        erexp94_b_error = pl.sub(erexp2_coefficient_hi2, erexp94_z)
                        erexp94_error = pl.add(erexp94_a_error, erexp94_b_error)
                        erexp93_ssa1_total = erexp94_total
                        erexp93_ssa1_error = erexp94_error
                        erexp93_low = erexp2_ssa32_cl
                        erexp93_low = pl.add(erexp2_ssa32_cl, erexp2_coefficient_lo2)
                        erexp93_tail = erexp93_ssa1_error
                        erexp93_tail = pl.add(erexp93_ssa1_error, erexp93_low)
                        erexp95_total = erexp93_ssa1_total
                        erexp95_error = erexp93_ssa1_total
                        erexp95_total = pl.add(erexp93_ssa1_total, erexp93_tail)
                        erexp95_z = pl.sub(erexp95_total, erexp93_ssa1_total)
                        erexp95_total_minus_z = pl.sub(erexp95_total, erexp95_z)
                        erexp95_a_error = pl.sub(erexp93_ssa1_total, erexp95_total_minus_z)
                        erexp95_b_error = pl.sub(erexp93_tail, erexp95_z)
                        erexp95_error = pl.add(erexp95_a_error, erexp95_b_error)
                        erexp93_ssa2_high = erexp95_total
                        erexp93_ssa2_low = erexp95_error
                        erexp2_ssa33_ch = erexp93_ssa2_high
                        erexp2_ssa33_cl = erexp93_ssa2_low
                        erexp97_product = erexp2_ssa33_ch
                        erexp97_error = erexp2_ssa33_ch
                        erexp97_a_split = pl.mul(erexp2_ssa33_ch, 4097.0)
                        erexp97_a_delta = pl.sub(erexp97_a_split, erexp2_ssa33_ch)
                        erexp97_a_hi = pl.sub(erexp97_a_split, erexp97_a_delta)
                        erexp97_a_lo = pl.sub(erexp2_ssa33_ch, erexp97_a_hi)
                        erexp97_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp97_b_delta = pl.sub(erexp97_b_split, erexp2_ssa5_rh)
                        erexp97_b_hi = pl.sub(erexp97_b_split, erexp97_b_delta)
                        erexp97_b_lo = pl.sub(erexp2_ssa5_rh, erexp97_b_hi)
                        erexp97_product = pl.mul(erexp2_ssa33_ch, erexp2_ssa5_rh)
                        erexp97_high_product = pl.mul(erexp97_a_hi, erexp97_b_hi)
                        erexp97_error = pl.sub(erexp97_high_product, erexp97_product)
                        erexp97_high_low = pl.mul(erexp97_a_hi, erexp97_b_lo)
                        erexp97_error = pl.add(erexp97_error, erexp97_high_low)
                        erexp97_low_high = pl.mul(erexp97_a_lo, erexp97_b_hi)
                        erexp97_error = pl.add(erexp97_error, erexp97_low_high)
                        erexp97_low_product = pl.mul(erexp97_a_lo, erexp97_b_lo)
                        erexp97_error = pl.add(erexp97_error, erexp97_low_product)
                        erexp96_ssa1_high = erexp97_product
                        erexp96_ssa1_error = erexp97_error
                        erexp96_cross1 = pl.mul(erexp2_ssa33_ch, erexp2_ssa5_rl)
                        erexp96_cross2 = pl.mul(erexp2_ssa33_cl, erexp2_ssa5_rh)
                        erexp96_cross3 = pl.mul(erexp2_ssa33_cl, erexp2_ssa5_rl)
                        erexp96_low = erexp96_ssa1_error
                        erexp96_low = pl.add(erexp96_ssa1_error, erexp96_cross1)
                        erexp96_low = pl.add(erexp96_low, erexp96_cross2)
                        erexp96_low = pl.add(erexp96_low, erexp96_cross3)
                        erexp98_total = erexp96_ssa1_high
                        erexp98_error = erexp96_ssa1_high
                        erexp98_total = pl.add(erexp96_ssa1_high, erexp96_low)
                        erexp98_z = pl.sub(erexp98_total, erexp96_ssa1_high)
                        erexp98_total_minus_z = pl.sub(erexp98_total, erexp98_z)
                        erexp98_a_error = pl.sub(erexp96_ssa1_high, erexp98_total_minus_z)
                        erexp98_b_error = pl.sub(erexp96_low, erexp98_z)
                        erexp98_error = pl.add(erexp98_a_error, erexp98_b_error)
                        erexp96_ssa2_high = erexp98_total
                        erexp96_ssa2_low = erexp98_error
                        erexp2_ssa34_ch = erexp96_ssa2_high
                        erexp2_ssa34_cl = erexp96_ssa2_low
                        erexp2_coefficient_hi1 = erexp1_zero
                        erexp2_coefficient_lo1 = erexp1_zero
                        erexp2_coefficient_hi1 = pl.add(erexp1_zero, 1.0)
                        erexp2_coefficient_lo1 = pl.add(erexp1_zero, 0.0)
                        erexp100_total = erexp2_ssa34_ch
                        erexp100_error = erexp2_ssa34_ch
                        erexp100_total = pl.add(erexp2_ssa34_ch, erexp2_coefficient_hi1)
                        erexp100_z = pl.sub(erexp100_total, erexp2_ssa34_ch)
                        erexp100_total_minus_z = pl.sub(erexp100_total, erexp100_z)
                        erexp100_a_error = pl.sub(erexp2_ssa34_ch, erexp100_total_minus_z)
                        erexp100_b_error = pl.sub(erexp2_coefficient_hi1, erexp100_z)
                        erexp100_error = pl.add(erexp100_a_error, erexp100_b_error)
                        erexp99_ssa1_total = erexp100_total
                        erexp99_ssa1_error = erexp100_error
                        erexp99_low = erexp2_ssa34_cl
                        erexp99_low = pl.add(erexp2_ssa34_cl, erexp2_coefficient_lo1)
                        erexp99_tail = erexp99_ssa1_error
                        erexp99_tail = pl.add(erexp99_ssa1_error, erexp99_low)
                        erexp101_total = erexp99_ssa1_total
                        erexp101_error = erexp99_ssa1_total
                        erexp101_total = pl.add(erexp99_ssa1_total, erexp99_tail)
                        erexp101_z = pl.sub(erexp101_total, erexp99_ssa1_total)
                        erexp101_total_minus_z = pl.sub(erexp101_total, erexp101_z)
                        erexp101_a_error = pl.sub(erexp99_ssa1_total, erexp101_total_minus_z)
                        erexp101_b_error = pl.sub(erexp99_tail, erexp101_z)
                        erexp101_error = pl.add(erexp101_a_error, erexp101_b_error)
                        erexp99_ssa2_high = erexp101_total
                        erexp99_ssa2_low = erexp101_error
                        erexp2_ssa35_ch = erexp99_ssa2_high
                        erexp2_ssa35_cl = erexp99_ssa2_low
                        erexp103_product = erexp2_ssa35_ch
                        erexp103_error = erexp2_ssa35_ch
                        erexp103_a_split = pl.mul(erexp2_ssa35_ch, 4097.0)
                        erexp103_a_delta = pl.sub(erexp103_a_split, erexp2_ssa35_ch)
                        erexp103_a_hi = pl.sub(erexp103_a_split, erexp103_a_delta)
                        erexp103_a_lo = pl.sub(erexp2_ssa35_ch, erexp103_a_hi)
                        erexp103_b_split = pl.mul(erexp2_ssa5_rh, 4097.0)
                        erexp103_b_delta = pl.sub(erexp103_b_split, erexp2_ssa5_rh)
                        erexp103_b_hi = pl.sub(erexp103_b_split, erexp103_b_delta)
                        erexp103_b_lo = pl.sub(erexp2_ssa5_rh, erexp103_b_hi)
                        erexp103_product = pl.mul(erexp2_ssa35_ch, erexp2_ssa5_rh)
                        erexp103_high_product = pl.mul(erexp103_a_hi, erexp103_b_hi)
                        erexp103_error = pl.sub(erexp103_high_product, erexp103_product)
                        erexp103_high_low = pl.mul(erexp103_a_hi, erexp103_b_lo)
                        erexp103_error = pl.add(erexp103_error, erexp103_high_low)
                        erexp103_low_high = pl.mul(erexp103_a_lo, erexp103_b_hi)
                        erexp103_error = pl.add(erexp103_error, erexp103_low_high)
                        erexp103_low_product = pl.mul(erexp103_a_lo, erexp103_b_lo)
                        erexp103_error = pl.add(erexp103_error, erexp103_low_product)
                        erexp102_ssa1_high = erexp103_product
                        erexp102_ssa1_error = erexp103_error
                        erexp102_cross1 = pl.mul(erexp2_ssa35_ch, erexp2_ssa5_rl)
                        erexp102_cross2 = pl.mul(erexp2_ssa35_cl, erexp2_ssa5_rh)
                        erexp102_cross3 = pl.mul(erexp2_ssa35_cl, erexp2_ssa5_rl)
                        erexp102_low = erexp102_ssa1_error
                        erexp102_low = pl.add(erexp102_ssa1_error, erexp102_cross1)
                        erexp102_low = pl.add(erexp102_low, erexp102_cross2)
                        erexp102_low = pl.add(erexp102_low, erexp102_cross3)
                        erexp104_total = erexp102_ssa1_high
                        erexp104_error = erexp102_ssa1_high
                        erexp104_total = pl.add(erexp102_ssa1_high, erexp102_low)
                        erexp104_z = pl.sub(erexp104_total, erexp102_ssa1_high)
                        erexp104_total_minus_z = pl.sub(erexp104_total, erexp104_z)
                        erexp104_a_error = pl.sub(erexp102_ssa1_high, erexp104_total_minus_z)
                        erexp104_b_error = pl.sub(erexp102_low, erexp104_z)
                        erexp104_error = pl.add(erexp104_a_error, erexp104_b_error)
                        erexp102_ssa2_high = erexp104_total
                        erexp102_ssa2_low = erexp104_error
                        erexp2_ssa36_ch = erexp102_ssa2_high
                        erexp2_ssa36_cl = erexp102_ssa2_low
                        erexp2_coefficient_hi0 = erexp1_zero
                        erexp2_coefficient_lo0 = erexp1_zero
                        erexp2_coefficient_hi0 = pl.add(erexp1_zero, 1.0)
                        erexp2_coefficient_lo0 = pl.add(erexp1_zero, 0.0)
                        erexp106_total = erexp2_ssa36_ch
                        erexp106_error = erexp2_ssa36_ch
                        erexp106_total = pl.add(erexp2_ssa36_ch, erexp2_coefficient_hi0)
                        erexp106_z = pl.sub(erexp106_total, erexp2_ssa36_ch)
                        erexp106_total_minus_z = pl.sub(erexp106_total, erexp106_z)
                        erexp106_a_error = pl.sub(erexp2_ssa36_ch, erexp106_total_minus_z)
                        erexp106_b_error = pl.sub(erexp2_coefficient_hi0, erexp106_z)
                        erexp106_error = pl.add(erexp106_a_error, erexp106_b_error)
                        erexp105_ssa1_total = erexp106_total
                        erexp105_ssa1_error = erexp106_error
                        erexp105_low = erexp2_ssa36_cl
                        erexp105_low = pl.add(erexp2_ssa36_cl, erexp2_coefficient_lo0)
                        erexp105_tail = erexp105_ssa1_error
                        erexp105_tail = pl.add(erexp105_ssa1_error, erexp105_low)
                        erexp107_total = erexp105_ssa1_total
                        erexp107_error = erexp105_ssa1_total
                        erexp107_total = pl.add(erexp105_ssa1_total, erexp105_tail)
                        erexp107_z = pl.sub(erexp107_total, erexp105_ssa1_total)
                        erexp107_total_minus_z = pl.sub(erexp107_total, erexp107_z)
                        erexp107_a_error = pl.sub(erexp105_ssa1_total, erexp107_total_minus_z)
                        erexp107_b_error = pl.sub(erexp105_tail, erexp107_z)
                        erexp107_error = pl.add(erexp107_a_error, erexp107_b_error)
                        erexp105_ssa2_high = erexp107_total
                        erexp105_ssa2_low = erexp107_error
                        erexp2_ssa37_ch = erexp105_ssa2_high
                        erexp2_ssa37_cl = erexp105_ssa2_low
                        erexp2_reduced = erexp2_ssa37_ch
                        erexp2_reduced = pl.add(erexp2_ssa37_ch, erexp2_ssa37_cl)
                        erexp1_one = pl.add(erexp1_zero, 1.0)
                        erexp2_remaining = pl.neg(erexp2_n)
                        erexp2_power = erexp1_one
                        erexp2_take64 = pl.sub(erexp2_remaining, 63.0)
                        erexp2_take64 = pl.maximum(erexp2_take64, 0.0)
                        erexp2_take64 = pl.minimum(erexp2_take64, 1.0)
                        erexp2_used64 = pl.mul(erexp2_take64, 64.0)
                        erexp2_remaining = pl.sub(erexp2_remaining, erexp2_used64)
                        erexp2_not64 = pl.sub(erexp1_one, erexp2_take64)
                        erexp2_small64 = pl.mul(erexp2_take64, 5.421010862427522e-20)
                        erexp2_factor64 = pl.add(erexp2_not64, erexp2_small64)
                        erexp2_power = pl.mul(erexp2_power, erexp2_factor64)
                        erexp2_take32 = pl.sub(erexp2_remaining, 31.0)
                        erexp2_take32 = pl.maximum(erexp2_take32, 0.0)
                        erexp2_take32 = pl.minimum(erexp2_take32, 1.0)
                        erexp2_used32 = pl.mul(erexp2_take32, 32.0)
                        erexp2_remaining = pl.sub(erexp2_remaining, erexp2_used32)
                        erexp2_not32 = pl.sub(erexp1_one, erexp2_take32)
                        erexp2_small32 = pl.mul(erexp2_take32, 2.3283064365386963e-10)
                        erexp2_factor32 = pl.add(erexp2_not32, erexp2_small32)
                        erexp2_power = pl.mul(erexp2_power, erexp2_factor32)
                        erexp2_take16 = pl.sub(erexp2_remaining, 15.0)
                        erexp2_take16 = pl.maximum(erexp2_take16, 0.0)
                        erexp2_take16 = pl.minimum(erexp2_take16, 1.0)
                        erexp2_used16 = pl.mul(erexp2_take16, 16.0)
                        erexp2_remaining = pl.sub(erexp2_remaining, erexp2_used16)
                        erexp2_not16 = pl.sub(erexp1_one, erexp2_take16)
                        erexp2_small16 = pl.mul(erexp2_take16, 1.52587890625e-05)
                        erexp2_factor16 = pl.add(erexp2_not16, erexp2_small16)
                        erexp2_power = pl.mul(erexp2_power, erexp2_factor16)
                        erexp2_take8 = pl.sub(erexp2_remaining, 7.0)
                        erexp2_take8 = pl.maximum(erexp2_take8, 0.0)
                        erexp2_take8 = pl.minimum(erexp2_take8, 1.0)
                        erexp2_used8 = pl.mul(erexp2_take8, 8.0)
                        erexp2_remaining = pl.sub(erexp2_remaining, erexp2_used8)
                        erexp2_not8 = pl.sub(erexp1_one, erexp2_take8)
                        erexp2_small8 = pl.mul(erexp2_take8, 0.00390625)
                        erexp2_factor8 = pl.add(erexp2_not8, erexp2_small8)
                        erexp2_power = pl.mul(erexp2_power, erexp2_factor8)
                        erexp2_take4 = pl.sub(erexp2_remaining, 3.0)
                        erexp2_take4 = pl.maximum(erexp2_take4, 0.0)
                        erexp2_take4 = pl.minimum(erexp2_take4, 1.0)
                        erexp2_used4 = pl.mul(erexp2_take4, 4.0)
                        erexp2_remaining = pl.sub(erexp2_remaining, erexp2_used4)
                        erexp2_not4 = pl.sub(erexp1_one, erexp2_take4)
                        erexp2_small4 = pl.mul(erexp2_take4, 0.0625)
                        erexp2_factor4 = pl.add(erexp2_not4, erexp2_small4)
                        erexp2_power = pl.mul(erexp2_power, erexp2_factor4)
                        erexp2_take2 = pl.sub(erexp2_remaining, 1.0)
                        erexp2_take2 = pl.maximum(erexp2_take2, 0.0)
                        erexp2_take2 = pl.minimum(erexp2_take2, 1.0)
                        erexp2_used2 = pl.mul(erexp2_take2, 2.0)
                        erexp2_remaining = pl.sub(erexp2_remaining, erexp2_used2)
                        erexp2_not2 = pl.sub(erexp1_one, erexp2_take2)
                        erexp2_small2 = pl.mul(erexp2_take2, 0.25)
                        erexp2_factor2 = pl.add(erexp2_not2, erexp2_small2)
                        erexp2_power = pl.mul(erexp2_power, erexp2_factor2)
                        erexp2_take1 = pl.sub(erexp2_remaining, 0.0)
                        erexp2_take1 = pl.maximum(erexp2_take1, 0.0)
                        erexp2_take1 = pl.minimum(erexp2_take1, 1.0)
                        erexp2_used1 = pl.mul(erexp2_take1, 1.0)
                        erexp2_remaining = pl.sub(erexp2_remaining, erexp2_used1)
                        erexp2_not1 = pl.sub(erexp1_one, erexp2_take1)
                        erexp2_small1 = pl.mul(erexp2_take1, 0.5)
                        erexp2_factor1 = pl.add(erexp2_not1, erexp2_small1)
                        erexp2_power = pl.mul(erexp2_power, erexp2_factor1)
                        erexp2_positive_remaining = erexp2_n
                        erexp2_positive_take64 = pl.minimum(pl.maximum(pl.sub(erexp2_positive_remaining, 63.0), 0.0), 1.0)
                        erexp2_positive_remaining = pl.sub(erexp2_positive_remaining, pl.mul(erexp2_positive_take64, 64.0))
                        erexp2_positive_factor64 = pl.add(erexp1_one, pl.mul(erexp2_positive_take64, 1.8446744073709552e+19))
                        erexp2_power = pl.mul(erexp2_power, erexp2_positive_factor64)
                        erexp2_positive_take32 = pl.minimum(pl.maximum(pl.sub(erexp2_positive_remaining, 31.0), 0.0), 1.0)
                        erexp2_positive_remaining = pl.sub(erexp2_positive_remaining, pl.mul(erexp2_positive_take32, 32.0))
                        erexp2_positive_factor32 = pl.add(erexp1_one, pl.mul(erexp2_positive_take32, 4294967295.0))
                        erexp2_power = pl.mul(erexp2_power, erexp2_positive_factor32)
                        erexp2_positive_take16 = pl.minimum(pl.maximum(pl.sub(erexp2_positive_remaining, 15.0), 0.0), 1.0)
                        erexp2_positive_remaining = pl.sub(erexp2_positive_remaining, pl.mul(erexp2_positive_take16, 16.0))
                        erexp2_positive_factor16 = pl.add(erexp1_one, pl.mul(erexp2_positive_take16, 65535.0))
                        erexp2_power = pl.mul(erexp2_power, erexp2_positive_factor16)
                        erexp2_positive_take8 = pl.minimum(pl.maximum(pl.sub(erexp2_positive_remaining, 7.0), 0.0), 1.0)
                        erexp2_positive_remaining = pl.sub(erexp2_positive_remaining, pl.mul(erexp2_positive_take8, 8.0))
                        erexp2_positive_factor8 = pl.add(erexp1_one, pl.mul(erexp2_positive_take8, 255.0))
                        erexp2_power = pl.mul(erexp2_power, erexp2_positive_factor8)
                        erexp2_positive_take4 = pl.minimum(pl.maximum(pl.sub(erexp2_positive_remaining, 3.0), 0.0), 1.0)
                        erexp2_positive_remaining = pl.sub(erexp2_positive_remaining, pl.mul(erexp2_positive_take4, 4.0))
                        erexp2_positive_factor4 = pl.add(erexp1_one, pl.mul(erexp2_positive_take4, 15.0))
                        erexp2_power = pl.mul(erexp2_power, erexp2_positive_factor4)
                        erexp2_positive_take2 = pl.minimum(pl.maximum(pl.sub(erexp2_positive_remaining, 1.0), 0.0), 1.0)
                        erexp2_positive_remaining = pl.sub(erexp2_positive_remaining, pl.mul(erexp2_positive_take2, 2.0))
                        erexp2_positive_factor2 = pl.add(erexp1_one, pl.mul(erexp2_positive_take2, 3.0))
                        erexp2_power = pl.mul(erexp2_power, erexp2_positive_factor2)
                        erexp2_positive_take1 = pl.minimum(pl.maximum(pl.sub(erexp2_positive_remaining, 0.0), 0.0), 1.0)
                        erexp2_positive_remaining = pl.sub(erexp2_positive_remaining, pl.mul(erexp2_positive_take1, 1.0))
                        erexp2_positive_factor1 = pl.add(erexp1_one, pl.mul(erexp2_positive_take1, 1.0))
                        erexp2_power = pl.mul(erexp2_power, erexp2_positive_factor1)
                        erexp1_reduced = erexp2_reduced
                        erexp1_low = erexp2_ssa37_cl
                        erexp1_power = erexp2_power
                        erexp1_value = pl.mul(erexp1_reduced, erexp1_power)
                        erexp1_fallback = pl.exp(pl.neg(gate_f))
                        erexp1_inside = pl.minimum(pl.maximum(pl.mul(pl.neg(pl.sub(pl.abs(gate_f), 80.00000762939453)), 131072.0), 0.0), 1.0)
                        erexp1_outside = pl.sub(erexp1_one, erexp1_inside)
                        erexp1_result = pl.add(pl.mul(erexp1_value, erexp1_inside), pl.mul(erexp1_fallback, erexp1_outside))
                        sg_fix_exponential = erexp1_result
                        sg_fix_denominator = pl.add(sg_fix_exponential, 1.0)
                        sg_fix_original = pl.recip(sg_fix_denominator)
                        sg_fix_bounded = pl.minimum(sg_fix_denominator, 4294967296.0)
                        sg_fix_estimate = sg_fix_original
                        sg_fix_product = sg_fix_estimate
                        sg_fix_error = sg_fix_estimate
                        sg_fix_a_split = pl.mul(sg_fix_estimate, 4097.0)
                        sg_fix_a_delta = pl.sub(sg_fix_a_split, sg_fix_estimate)
                        sg_fix_a_hi = pl.sub(sg_fix_a_split, sg_fix_a_delta)
                        sg_fix_a_lo = pl.sub(sg_fix_estimate, sg_fix_a_hi)
                        sg_fix_b_split = pl.mul(sg_fix_bounded, 4097.0)
                        sg_fix_b_delta = pl.sub(sg_fix_b_split, sg_fix_bounded)
                        sg_fix_b_hi = pl.sub(sg_fix_b_split, sg_fix_b_delta)
                        sg_fix_b_lo = pl.sub(sg_fix_bounded, sg_fix_b_hi)
                        sg_fix_product = pl.mul(sg_fix_estimate, sg_fix_bounded)
                        sg_fix_high_product = pl.mul(sg_fix_a_hi, sg_fix_b_hi)
                        sg_fix_error = pl.sub(sg_fix_high_product, sg_fix_product)
                        sg_fix_high_low = pl.mul(sg_fix_a_hi, sg_fix_b_lo)
                        sg_fix_error = pl.add(sg_fix_error, sg_fix_high_low)
                        sg_fix_low_high = pl.mul(sg_fix_a_lo, sg_fix_b_hi)
                        sg_fix_error = pl.add(sg_fix_error, sg_fix_low_high)
                        sg_fix_low_product = pl.mul(sg_fix_a_lo, sg_fix_b_lo)
                        sg_fix_error = pl.add(sg_fix_error, sg_fix_low_product)
                        sg_fix_one = pl.full([ROW_HALF, MOE_INTER], dtype=pl.FP32, value=1.0)
                        sg_fix_residual_hi = pl.sub(sg_fix_one, sg_fix_product)
                        sg_fix_residual = pl.sub(sg_fix_residual_hi, sg_fix_error)
                        sg_fix_correction = pl.div(sg_fix_residual, sg_fix_bounded)
                        sg_fix_corrected = pl.add(sg_fix_estimate, sg_fix_correction)
                        sg_fix_overflow = pl.sub(sg_fix_denominator, 4294967296.0)
                        sg_fix_overflow = pl.mul(sg_fix_overflow, 0.001953125)
                        sg_fix_overflow = pl.minimum(pl.maximum(sg_fix_overflow, 0.0), 1.0)
                        sg_fix_inside = pl.sub(sg_fix_one, sg_fix_overflow)
                        sg_fix_fixed_inside = pl.mul(sg_fix_corrected, sg_fix_inside)
                        sg_fix_retained = pl.mul(sg_fix_original, sg_fix_overflow)
                        sg_fix_fixed = pl.add(sg_fix_fixed_inside, sg_fix_retained)
                        sigmoid = sg_fix_fixed
                        h_raw = pl.mul(pl.mul(gate_f, sigmoid), up_f)
                        # valid_shape rides the matmul's A operand through the c2v push
                        # and the per-lane pop, so h_raw already carries this lane's
                        # row extent; re-marking it is rejected ("cannot re-narrow a
                        # popped tile"). Mask the tail, then re-widen so the narrowed
                        # extent does not leave the pad rows of y undefined.
                        h_pad = pl.fillpad(h_raw, pad_value=pl.PadValue.zero)
                        h_f32 = pl.set_validshape(h_pad, ROW_HALF, FUSED_N_TILE)

                        h_abs = pl.maximum(h_f32, pl.neg(h_f32))
                        h_amax = pl.maximum(
                            pl.full([1, ROW_HALF], dtype=pl.FP32, value=INT8_AMAX_EPS),
                            pl.reshape(pl.row_max(h_abs), [1, ROW_HALF]),
                        )
                        qd_epb_numerator = pl.full([1, ROW_HALF], dtype=pl.FP32, value=INT8_SCALE_MAX)
                        qd_epb_work = pl.mul(h_amax, 5.421010862427522e-20)
                        qd_epb_estimate = pl.div(qd_epb_numerator, qd_epb_work)
                        qd_epb_left_split = pl.mul(qd_epb_estimate, 4097.0)
                        qd_epb_left_delta = pl.sub(qd_epb_left_split, qd_epb_estimate)
                        qd_epb_left_hi = pl.sub(qd_epb_left_split, qd_epb_left_delta)
                        qd_epb_left_lo = pl.sub(qd_epb_estimate, qd_epb_left_hi)
                        qd_epb_right_split = pl.mul(qd_epb_work, 4097.0)
                        qd_epb_right_delta = pl.sub(qd_epb_right_split, qd_epb_work)
                        qd_epb_right_hi = pl.sub(qd_epb_right_split, qd_epb_right_delta)
                        qd_epb_right_lo = pl.sub(qd_epb_work, qd_epb_right_hi)
                        qd_epb_product = pl.mul(qd_epb_estimate, qd_epb_work)
                        qd_epb_hi_product = pl.mul(qd_epb_left_hi, qd_epb_right_hi)
                        qd_epb_error = pl.sub(qd_epb_hi_product, qd_epb_product)
                        qd_epb_hi_lo = pl.mul(qd_epb_left_hi, qd_epb_right_lo)
                        qd_epb_error = pl.add(qd_epb_error, qd_epb_hi_lo)
                        qd_epb_lo_hi = pl.mul(qd_epb_left_lo, qd_epb_right_hi)
                        qd_epb_error = pl.add(qd_epb_error, qd_epb_lo_hi)
                        qd_epb_lo_product = pl.mul(qd_epb_left_lo, qd_epb_right_lo)
                        qd_epb_error = pl.add(qd_epb_error, qd_epb_lo_product)
                        qd_epb_residual_hi = pl.sub(qd_epb_numerator, qd_epb_product)
                        qd_epb_residual = pl.sub(qd_epb_residual_hi, qd_epb_error)
                        qd_epb_correction = pl.div(qd_epb_residual, qd_epb_work)
                        qd_epb_corrected_work = pl.add(qd_epb_estimate, qd_epb_correction)
                        sq_row = pl.mul(qd_epb_corrected_work, 5.421010862427522e-20)
                        dqfix_ones_row = pl.full([1, ROW_HALF], dtype=pl.FP32, value=1.0)
                        dqfix_ef_work = pl.mul(sq_row, 65536.0)
                        dqfix_ef_estimate = pl.div(dqfix_ones_row, dqfix_ef_work)
                        dqfix_ef_left_split = pl.mul(dqfix_ef_estimate, 4097.0)
                        dqfix_ef_left_delta = pl.sub(dqfix_ef_left_split, dqfix_ef_estimate)
                        dqfix_ef_left_hi = pl.sub(dqfix_ef_left_split, dqfix_ef_left_delta)
                        dqfix_ef_left_lo = pl.sub(dqfix_ef_estimate, dqfix_ef_left_hi)
                        dqfix_ef_right_split = pl.mul(dqfix_ef_work, 4097.0)
                        dqfix_ef_right_delta = pl.sub(dqfix_ef_right_split, dqfix_ef_work)
                        dqfix_ef_right_hi = pl.sub(dqfix_ef_right_split, dqfix_ef_right_delta)
                        dqfix_ef_right_lo = pl.sub(dqfix_ef_work, dqfix_ef_right_hi)
                        dqfix_ef_product = pl.mul(dqfix_ef_estimate, dqfix_ef_work)
                        dqfix_ef_hi_product = pl.mul(dqfix_ef_left_hi, dqfix_ef_right_hi)
                        dqfix_ef_error = pl.sub(dqfix_ef_hi_product, dqfix_ef_product)
                        dqfix_ef_hi_lo = pl.mul(dqfix_ef_left_hi, dqfix_ef_right_lo)
                        dqfix_ef_error = pl.add(dqfix_ef_error, dqfix_ef_hi_lo)
                        dqfix_ef_lo_hi = pl.mul(dqfix_ef_left_lo, dqfix_ef_right_hi)
                        dqfix_ef_error = pl.add(dqfix_ef_error, dqfix_ef_lo_hi)
                        dqfix_ef_lo_product = pl.mul(dqfix_ef_left_lo, dqfix_ef_right_lo)
                        dqfix_ef_error = pl.add(dqfix_ef_error, dqfix_ef_lo_product)
                        dqfix_ef_residual_hi = pl.sub(dqfix_ones_row, dqfix_ef_product)
                        dqfix_ef_residual = pl.sub(dqfix_ef_residual_hi, dqfix_ef_error)
                        dqfix_ef_correction = pl.div(dqfix_ef_residual, dqfix_ef_work)
                        dqfix_ef_corrected_work = pl.add(dqfix_ef_estimate, dqfix_ef_correction)
                        dqfix_scale_dequant_row = pl.mul(dqfix_ef_corrected_work, 65536.0)
                        h_scale_dq = pl.reshape(dqfix_scale_dequant_row, [ROW_HALF, 1])
                        sq_col = pl.reshape(sq_row, [ROW_HALF, 1])
                        q_scaled = pl.row_expand_mul(h_f32, sq_col)
                        q_i32 = pl.cast(q_scaled, target_type=pl.INT32, mode="rint")
                        q_f16 = pl.cast(q_i32, target_type=pl.FP16, mode="round")
                        h_i8 = pl.cast(q_f16, target_type=pl.INT8, mode="trunc")

                        row_scale = pl.mul(h_scale_dq, w_col)

                        # V->C: the two lanes' row bands are stitched back into one
                        # full RECV_TILE operand in L1 for the down matmul.
                        h_gathered = pl.aic_gather(h_i8)

                    for db in pl.pipeline(D // FUSED_Y_TILE, stage=2):
                        d0 = db * FUSED_Y_TILE
                        w2_k = w2_2d[e_w2 + d0 : e_w2 + d0 + FUSED_Y_TILE, 0:MOE_INTER]
                        y_acc = pl.matmul(h_gathered, w2_k, b_trans=True, out_dtype=pl.INT32)
                        w2_sc = routed_w2_scale[eid : eid + 1, d0 : d0 + FUSED_Y_TILE]

                        # ---- vector phase 2: dequant + store ----
                        # Same row split, so this lane's row_scale from phase 1 lines
                        # up with this lane's half of the accumulator.
                        for aiv_id2 in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                            y_off = flat_t0 + pl.cast(aiv_id2 * ROW_HALF, pl.INDEX)
                            y_sh = pl.aiv_shard(y_acc)
                            y_f = pl.cast(y_sh, target_type=pl.FP32, mode="none")
                            y_f = pl.col_expand_mul(pl.row_expand_mul(y_f, row_scale), w2_sc)
                            recv_y_flat[y_off : y_off + ROW_HALF, d0 : d0 + FUSED_Y_TILE] = \
                                pl.cast(y_f, target_type=pl.BF16, mode="rint")

    return recv_y


@pl.jit.inline(auto_scope=False)
def expert_routed_persistent_balanced(
    recv_x: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    routed_w1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_BANK, D], pl.FP32],
    recv_y: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.BF16],
):
    work_slot = pl.create_tensor([WORK_SLOTS_PAD], dtype=pl.INT32)
    work_rows = pl.create_tensor([WORK_SLOTS_PAD], dtype=pl.INT32)

    with pl.spmd(1, name_hint="exp_balance_plan", allow_early_resolve=True) as plan_tid:
        plan_core = pl.tile.get_block_idx()
        zero_i32 = pl.cast(0, pl.INT32)
        for pad in pl.range(plan_core, N_SLOTS_B):
            pl.write(work_slot, [pad], zero_i32)
            pl.write(work_rows, [pad], zero_i32)
        n_live = pl.cast(0, pl.INDEX)
        for scan in pl.range(plan_core, N_SLOTS_B):
            scan_rows = pl.read(recv_expert_count, [scan, 0])
            if scan_rows > 0:
                pl.write(work_slot, [n_live], pl.cast(scan, pl.INT32))
                pl.write(work_rows, [n_live], scan_rows)
                n_live = n_live + 1

    expert_routed_persistent_planned(
        recv_x, recv_scale_dq, recv_weights, work_slot, work_rows, slot_expert,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale, recv_y,
    )
    return recv_y


@pl.jit
def expert_routed_persistent_balanced_test(
    recv_x: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    routed_w1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_BANK, D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.BF16]],
):
    expert_routed_persistent_balanced(
        recv_x, recv_scale_dq, recv_weights, recv_expert_count, slot_expert,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )
    return recv_y


if __name__ == "__main__":
    import argparse
    from decode_shapes import TP
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tp", type=int, default=TP, choices=[1, 2, 4, 8], help="tensor-parallel degree; config freezes it at import")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None,
                        help="dir with cached in/{name}.pt + out/{name}.pt; reuses them "
                             "instead of regenerating inputs + recomputing golden.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=4, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        compile_only=args.compile_only,
        fn=expert_routed_persistent_balanced_test,
        specs=build_tensor_specs(merged=True),
        golden_fn=golden_expert_routed,
        golden_data=args.golden_data,
        save_data=args.save_data,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.platform.removesuffix("sim"),
            execution_mode=args.platform.endswith("sim"),
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "recv_y": compare_recv_y,
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
