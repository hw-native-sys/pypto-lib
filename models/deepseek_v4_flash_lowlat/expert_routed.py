# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim  # A2/A3 device validation; simulator CI is deferred for this model.
"""DeepSeek-V4 MoE routed expert compute over a TP slice of the intermediate dim.

Every rank holds every expert and a 1/TP slice of the intermediate dim, so
``recv_y`` is a partial sum that ``moe`` all-reduces across the TP group. Rows
arrive already grouped: slot ``i`` holds the rows routed to expert
``slot_expert[i]``, and inactive slots carry a zero count.
"""

import pypto.language as pl

from config import FLASH as M, INT8_SCALE_MAX, INT8_AMAX_EPS
from decode_shapes import DECODE_BATCH, DECODE_SEQ, TP


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
TOPK = M.num_experts_per_tok
MOE_INTER = M.moe_intermediate_size // TP  # this rank's slice of the intermediate dim
SWIGLU_LIMIT = M.swiglu_limit
N_EXPERTS = M.n_routed_experts

# Grouped-route layout: one padded row tile per active expert. A step routes
# T * TOPK pairs, so no more than that many experts can be active at once.
RECV_TILE = 16
N_SLOTS = min(N_EXPERTS, T * TOPK)
RECV_MAX = RECV_TILE
IDX_PAD = 8  # INT32 tile width: a vector fill needs a 32-byte row

# The shared expert is just another expert: every rank sees the same tokens, it
# runs the same INT8 SwiGLU FFN over the same intermediate slice, and its output
# is summed into the same TP partial. So it becomes expert ``SHARED_EID`` of an
# extended weight bank, routed to one extra slot past the routing table's last
# with every row weighted 1.0 -- and a separate shared-expert task chain
# disappears. Routing can fill at most ``T * TOPK`` slots, so ``SH_SLOT`` is
# never claimed by a routed expert.
SHARED_EID = N_EXPERTS
N_BANK = N_EXPERTS + 1
SH_SLOT = N_SLOTS
N_SLOTS_B = N_SLOTS + 1
# The shared slot carries every active token in ONE row tile.
assert T <= RECV_MAX, f"the shared-expert slot holds all {T} tokens in one tile of {RECV_MAX} rows"

# tiling
K_TILE = 512
INTER_K = min(512, MOE_INTER // 2)  # >= 2 K steps: a single-trip accumulate loop breaks L0C reuse
# Four cube blocks per submit, not two: 62 blocks over 24 AIC cores packed
# raggedly (min 2 / max 3 per core). Blocks are free -- one spmd submits once.
MM_INTER_TILE = min(256, max(64, MOE_INTER // 4))
MM_GATE_INNER = 1
ACT_INTER_TILE = min(128, MOE_INTER // 2)
ACT_GATE_INNER = 1
D_OUT_TILE = 256
# h_tile_i8 store innermost = QUANT_TILE bytes (int8); 512 hits the a2a3 L2 cache
# line (perf_hint PH001 flagged the prior 256B store as sub-line).
QUANT_TILE = min(512, MOE_INTER)
D_OUT_TILE_ACT = 512
W2_INNER = 4
W2_ACT_INNER = 8

# One submit covers SLOT_GROUP expert slots for the two cheap stages only: the AICPU
# pays per submit, not per block, but a grouped task is also a barrier over its
# whole group, so the two heavy matmuls keep one submit per active slot and only
# the quantize / dequantize epilogues are fanned out of a shared submit. Each
# grouped block re-reads its own row count and runs a 0-or-1-trip loop around its
# body, which is the empty-slot skip the per-slot loop used to do on the AICPU.
SLOT_GROUP = max(g for g in (12, 8, 6, 4, 3, 2, 1) if N_SLOTS % g == 0)


@pl.jit.inline(auto_scope=False)
def expert_routed(
    recv_x: pl.Tensor[[N_SLOTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_SLOTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_SLOTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_SLOTS, IDX_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS, IDX_PAD], pl.INT32],
    routed_w1: pl.Tensor[[N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    recv_y: pl.Tensor[[N_SLOTS, RECV_MAX, D], pl.BF16],
):
    recv_y_flat = pl.reshape(recv_y, [N_SLOTS * RECV_MAX, D])
    recv_x_flat = pl.reshape(recv_x, [N_SLOTS * RECV_MAX, D])

    with pl.scope():
        # Keep only the requantized SwiGLU result across the W1/W3 and W2 phases.
        # The full INT32 gate/up tensors would occupy 512 MiB at EP8; h_i8 and its
        # per-row dequant scale occupy about 64 MiB instead.
        h_i8 = pl.create_tensor([N_SLOTS * RECV_MAX, MOE_INTER], dtype=pl.INT8)
        h_scale_dq = pl.create_tensor([N_SLOTS * RECV_MAX, 1], dtype=pl.FP32, manual_dep=True)
        # Slot-major so a grouped quantize block can address its own rows.
        # manual_dep: the pool's only real edge is gate_up -> h_q, restated below
        # per slot group, so the scheduler skips creator retention and the
        # tensormap overlap walk over a 768-row tensor.
        # Split variant: the cube task writes INT32 gate/up here so it stays
        # AIC-only instead of reserving its two paired AIV cores for the whole
        # matmul. 1.5 MiB at the decode shape; scales with N_SLOTS*RECV_MAX.
        gate_i32 = pl.create_tensor([N_SLOTS * RECV_MAX, MOE_INTER], dtype=pl.INT32, manual_dep=True)
        up_i32 = pl.create_tensor([N_SLOTS * RECV_MAX, MOE_INTER], dtype=pl.INT32, manual_dep=True)
        h_fp32 = pl.create_tensor([N_SLOTS * RECV_MAX, MOE_INTER], dtype=pl.FP32, manual_dep=True)
        gate_up_tids = pl.array.create(N_SLOTS, pl.TASK_ID)

        for local_i in pl.parallel(N_SLOTS):
            flat_base = local_i * RECV_MAX
            expert_i = pl.cast(pl.read(slot_expert, [local_i, 0]), pl.INDEX)

            n_rows = pl.read(recv_expert_count, [local_i, 0])
            n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE

            for t in pl.parallel(n_tiles):
                t0 = t * RECV_TILE
                flat_t0 = flat_base + t0
                valid_rows = pl.min(RECV_TILE, n_rows - t0)

                # Cube only: gate and up share the x tile and the K loop, so one
                # spmd covers both. The SwiGLU epilogue is a separate AIV task, so
                # this one never reserves the paired vector cores.
                with pl.spmd(MOE_INTER // MM_INTER_TILE, name_hint="exp_gate_up_mm") as gate_up_tid:
                    nb_idx = pl.tile.get_block_idx()
                    n0 = nb_idx * MM_INTER_TILE

                    gate_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                        x_k = recv_x_flat[flat_t0 : flat_t0 + RECV_TILE, k0 : k0 + K_TILE]
                        w1_k = routed_w1[
                            expert_i : expert_i + 1,
                            n0 : n0 + MM_INTER_TILE,
                            k0 : k0 + K_TILE,
                        ]
                        if k0 == 0:
                            gate_acc = pl.matmul(x_k, w1_k, b_trans=True, out_dtype=pl.INT32)
                        else:
                            gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)

                    up_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                    for uk0 in pl.pipeline(0, D, K_TILE, stage=2):
                        x_u = recv_x_flat[flat_t0 : flat_t0 + RECV_TILE, uk0 : uk0 + K_TILE]
                        w3_k = routed_w3[
                            expert_i : expert_i + 1,
                            n0 : n0 + MM_INTER_TILE,
                            uk0 : uk0 + K_TILE,
                        ]
                        if uk0 == 0:
                            up_acc = pl.matmul(x_u, w3_k, b_trans=True, out_dtype=pl.INT32)
                        else:
                            up_acc = pl.matmul_acc(up_acc, x_u, w3_k, b_trans=True)

                    gate_i32[flat_t0 : flat_t0 + RECV_TILE, n0 : n0 + MM_INTER_TILE] = pl.reshape(
                        gate_acc, [RECV_TILE, MM_INTER_TILE]
                    )
                    up_i32[flat_t0 : flat_t0 + RECV_TILE, n0 : n0 + MM_INTER_TILE] = pl.reshape(
                        up_acc, [RECV_TILE, MM_INTER_TILE]
                    )
                gate_up_tids[local_i] = gate_up_tid

        # AIV -> AIV with no other consumer: the SwiGLU epilogue and the
        # requantize are one task. `gated` stays in UB (RECV_TILE x MOE_INTER
        # requantize are one task, so the epilogue chain costs one submit, not two.
        for ga in pl.parallel(N_SLOTS // SLOT_GROUP):
            with pl.spmd(
                SLOT_GROUP,
                name_hint="exp_act_h_q",
                deps=[gate_up_tids[ga * SLOT_GROUP + s_] for s_ in range(SLOT_GROUP)],
            ):
                a_sub = pl.tile.get_block_idx()
                a_slot = ga * SLOT_GROUP + a_sub
                flat_a0 = a_slot * RECV_MAX
                a_expert = pl.cast(pl.read(slot_expert, [a_slot, 0]), pl.INDEX)
                a_rows = pl.read(recv_expert_count, [a_slot, 0])
                for _a in pl.range((a_rows + RECV_TILE - 1) // RECV_TILE):
                    a_valid = pl.min(RECV_TILE, a_rows)
                    a_scale = pl.reshape(recv_scale_dq[a_slot : a_slot + 1, 0:RECV_TILE], [RECV_TILE, 1])
                    eh_amax = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    for ag in pl.pipeline(MOE_INTER // ACT_INTER_TILE, stage=2):
                        a0 = ag * ACT_INTER_TILE
                        g_i32 = gate_i32[flat_a0 : flat_a0 + RECV_TILE, a0 : a0 + ACT_INTER_TILE]
                        u_i32 = up_i32[flat_a0 : flat_a0 + RECV_TILE, a0 : a0 + ACT_INTER_TILE]
                        w1_sc = routed_w1_scale[a_expert : a_expert + 1, a0 : a0 + ACT_INTER_TILE]
                        w3_sc = routed_w3_scale[a_expert : a_expert + 1, a0 : a0 + ACT_INTER_TILE]
                        g2 = pl.cast(g_i32, target_type=pl.FP32, mode="none")
                        u2 = pl.cast(u_i32, target_type=pl.FP32, mode="none")
                        g2 = pl.col_expand_mul(pl.row_expand_mul(g2, a_scale), w1_sc)
                        u2 = pl.col_expand_mul(pl.row_expand_mul(u2, a_scale), w3_sc)
                        if SWIGLU_LIMIT > 0.0:
                            g2 = pl.minimum(g2, SWIGLU_LIMIT)
                            u2 = pl.maximum(pl.minimum(u2, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                        expert_exp_argument = pl.neg(g2)
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
                        erexp1_fallback = pl.exp(pl.neg(g2))
                        erexp1_inside = pl.minimum(pl.maximum(pl.mul(pl.neg(pl.sub(pl.abs(g2), 80.00000762939453)), 131072.0), 0.0), 1.0)
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
                        sg_fix_one = pl.full([RECV_TILE, ACT_INTER_TILE], dtype=pl.FP32, value=1.0)
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
                        sig = sg_fix_fixed
                        gd = pl.mul(pl.mul(g2, sig), u2)
                        gd_valid = pl.set_validshape(gd, a_valid, ACT_INTER_TILE)
                        gd_pad = pl.fillpad(gd_valid, pad_value=pl.PadValue.zero)
                        h_fp32[flat_a0 : flat_a0 + RECV_TILE, a0 : a0 + ACT_INTER_TILE] = gd_pad
                        eh_amax = pl.maximum(
                            eh_amax,
                            pl.reshape(pl.row_max(pl.maximum(gd_pad, pl.neg(gd_pad))), [1, RECV_TILE]),
                        )
                    qd_erh_numerator = pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    qd_erh_work = pl.mul(eh_amax, 5.421010862427522e-20)
                    qd_erh_estimate = pl.div(qd_erh_numerator, qd_erh_work)
                    qd_erh_left_split = pl.mul(qd_erh_estimate, 4097.0)
                    qd_erh_left_delta = pl.sub(qd_erh_left_split, qd_erh_estimate)
                    qd_erh_left_hi = pl.sub(qd_erh_left_split, qd_erh_left_delta)
                    qd_erh_left_lo = pl.sub(qd_erh_estimate, qd_erh_left_hi)
                    qd_erh_right_split = pl.mul(qd_erh_work, 4097.0)
                    qd_erh_right_delta = pl.sub(qd_erh_right_split, qd_erh_work)
                    qd_erh_right_hi = pl.sub(qd_erh_right_split, qd_erh_right_delta)
                    qd_erh_right_lo = pl.sub(qd_erh_work, qd_erh_right_hi)
                    qd_erh_product = pl.mul(qd_erh_estimate, qd_erh_work)
                    qd_erh_hi_product = pl.mul(qd_erh_left_hi, qd_erh_right_hi)
                    qd_erh_error = pl.sub(qd_erh_hi_product, qd_erh_product)
                    qd_erh_hi_lo = pl.mul(qd_erh_left_hi, qd_erh_right_lo)
                    qd_erh_error = pl.add(qd_erh_error, qd_erh_hi_lo)
                    qd_erh_lo_hi = pl.mul(qd_erh_left_lo, qd_erh_right_hi)
                    qd_erh_error = pl.add(qd_erh_error, qd_erh_lo_hi)
                    qd_erh_lo_product = pl.mul(qd_erh_left_lo, qd_erh_right_lo)
                    qd_erh_error = pl.add(qd_erh_error, qd_erh_lo_product)
                    qd_erh_residual_hi = pl.sub(qd_erh_numerator, qd_erh_product)
                    qd_erh_residual = pl.sub(qd_erh_residual_hi, qd_erh_error)
                    qd_erh_correction = pl.div(qd_erh_residual, qd_erh_work)
                    qd_erh_corrected_work = pl.add(qd_erh_estimate, qd_erh_correction)
                    eh_sq_row = pl.mul(qd_erh_corrected_work, 5.421010862427522e-20)
                    h_scale_dq[flat_a0 : flat_a0 + RECV_TILE, 0:1] = pl.reshape(
                        pl.recip(eh_sq_row), [RECV_TILE, 1]
                    )
                    eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                    for k1 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                        eh_q_scaled = pl.row_expand_mul(
                            h_fp32[flat_a0 : flat_a0 + RECV_TILE, k1 : k1 + QUANT_TILE], eh_sq_col
                        )
                        eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode="rint")
                        eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode="round")
                        h_i8[flat_a0 : flat_a0 + RECV_TILE, k1 : k1 + QUANT_TILE] = pl.cast(
                            eh_q_half, target_type=pl.INT8, mode="trunc"
                        )

        with pl.scope():
            y_i32 = pl.create_tensor([N_SLOTS * RECV_MAX, D], dtype=pl.INT32, manual_dep=True)
            w2_mm_tids = pl.array.create(N_SLOTS, pl.TASK_ID)

            for local_e in pl.parallel(N_SLOTS):
                e_flat_base = local_e * RECV_MAX
                expert_e = pl.cast(pl.read(slot_expert, [local_e, 0]), pl.INDEX)

                e_rows = pl.read(recv_expert_count, [local_e, 0])
                e_tiles = (e_rows + RECV_TILE - 1) // RECV_TILE

                for tt in pl.parallel(e_tiles):
                    tt0 = tt * RECV_TILE
                    flat_tt0 = e_flat_base + tt0
                    h_tile_i8 = h_i8[flat_tt0 : flat_tt0 + RECV_TILE]
                    y_tile_i32 = y_i32[flat_tt0 : flat_tt0 + RECV_TILE]

                    with pl.spmd(
                        D // (W2_INNER * D_OUT_TILE),
                        name_hint="exp_w2_mm",
                        allow_early_resolve=True,
                    ) as w2_mm_tid:
                        wb_idx = pl.tile.get_block_idx()
                        d_base = wb_idx * (W2_INNER * D_OUT_TILE)
                        for dg in pl.range(W2_INNER):
                            d0 = d_base + dg * D_OUT_TILE
                            y_acc = pl.create_tensor([1, RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
                            for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                                h_k = h_tile_i8[:, k0 : k0 + INTER_K]
                                w2_k = routed_w2[
                                    expert_e : expert_e + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K
                                ]
                                if k0 == 0:
                                    y_acc = pl.matmul(h_k, w2_k, b_trans=True, out_dtype=pl.INT32)
                                else:
                                    y_acc = pl.matmul_acc(y_acc, h_k, w2_k, b_trans=True)
                            y_tile_i32[:, d0 : d0 + D_OUT_TILE] = pl.reshape(y_acc, [RECV_TILE, D_OUT_TILE])
                    w2_mm_tids[local_e] = w2_mm_tid

            for ga in pl.parallel(N_SLOTS // SLOT_GROUP):
                # Restate the y_i32 RAW edge the manual_dep pool no longer infers.
                with pl.spmd(
                    SLOT_GROUP * (D // (W2_ACT_INNER * D_OUT_TILE_ACT)),
                    name_hint="exp_w2_act",
                    allow_early_resolve=True,
                    deps=[w2_mm_tids[ga * SLOT_GROUP + s] for s in range(SLOT_GROUP)],
                ):
                    act_blk = pl.tile.get_block_idx()
                    act_sub = act_blk // (D // (W2_ACT_INNER * D_OUT_TILE_ACT))
                    db_idx = act_blk % (D // (W2_ACT_INNER * D_OUT_TILE_ACT))
                    act_slot = ga * SLOT_GROUP + act_sub
                    flat_at0 = act_slot * RECV_MAX
                    act_rows = pl.read(recv_expert_count, [act_slot, 0])
                    for _a in pl.range((act_rows + RECV_TILE - 1) // RECV_TILE):
                        expert_a = pl.cast(pl.read(slot_expert, [act_slot, 0]), pl.INDEX)
                        act_d_base = db_idx * (W2_ACT_INNER * D_OUT_TILE_ACT)
                        w_col_blk = pl.reshape(
                            recv_weights[act_slot : act_slot + 1, 0:RECV_TILE],
                            [RECV_TILE, 1],
                        )
                        row_scale_blk = pl.mul(h_scale_dq[flat_at0 : flat_at0 + RECV_TILE, 0:1], w_col_blk)
                        for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                            act_d0 = act_d_base + dg * D_OUT_TILE_ACT
                            y_2d_i32 = y_i32[
                                flat_at0 : flat_at0 + RECV_TILE, act_d0 : act_d0 + D_OUT_TILE_ACT
                            ]
                            w2_scale_chunk = routed_w2_scale[
                                expert_a : expert_a + 1, act_d0 : act_d0 + D_OUT_TILE_ACT
                            ]
                            y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                            y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, row_scale_blk), w2_scale_chunk)
                            recv_y_flat[flat_at0 : flat_at0 + RECV_TILE, act_d0 : act_d0 + D_OUT_TILE_ACT] = (
                                pl.cast(y_2d, target_type=pl.BF16, mode="rint")
                            )

    return recv_y


@pl.jit
def expert_routed_test(
    recv_x: pl.Tensor[[N_SLOTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_SLOTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_SLOTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_SLOTS, IDX_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS, IDX_PAD], pl.INT32],
    routed_w1: pl.Tensor[[N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_SLOTS, RECV_MAX, D], pl.BF16]],
):
    expert_routed(
        recv_x,
        recv_scale_dq,
        recv_weights,
        recv_expert_count,
        slot_expert,
        routed_w1,
        routed_w1_scale,
        routed_w3,
        routed_w3_scale,
        routed_w2,
        routed_w2_scale,
        recv_y,
    )
    return recv_y


def golden_expert_routed(tensors):
    """Integer-accumulate W8A8 reference for one TP intermediate partition."""
    import torch
    from utils import int8_quant_per_row

    counts = tensors["recv_expert_count"]
    recv_y = torch.zeros_like(tensors["recv_y"], dtype=torch.float32)
    for slot in range(counts.shape[0]):
        n_rows = int(counts[slot, 0])
        if not 0 <= n_rows <= tensors["recv_x"].shape[1]:
            raise ValueError("Expert row count exceeds the receive buffer")
        if n_rows == 0:
            continue
        expert = int(tensors["slot_expert"][slot, 0])
        x_i32 = tensors["recv_x"][slot, :n_rows].to(torch.int32)
        row_scale = tensors["recv_scale_dq"][slot, :n_rows].float().unsqueeze(-1)
        route_weight = tensors["recv_weights"][slot, :n_rows].float().unsqueeze(-1)
        gate_acc = x_i32 @ tensors["routed_w1"][expert].to(torch.int32).T
        up_acc = x_i32 @ tensors["routed_w3"][expert].to(torch.int32).T
        gate = (gate_acc.float() * row_scale) * tensors["routed_w1_scale"][expert].float()
        up = (up_acc.float() * row_scale) * tensors["routed_w3_scale"][expert].float()
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        sigmoid_exp = torch.exp(-gate.double()).float()
        sigmoid = torch.reciprocal(sigmoid_exp + 1.0)
        hidden = (gate * sigmoid) * up
        hidden_i8, hidden_scale = int8_quant_per_row(hidden, direct_div=True)
        down_acc = hidden_i8.to(torch.int32) @ tensors["routed_w2"][expert].to(torch.int32).T
        row_scale = hidden_scale * route_weight
        recv_y[slot, :n_rows] = (down_acc.float() * row_scale) * tensors["routed_w2_scale"][expert].float()
    tensors["recv_y"].copy_(recv_y.to(torch.bfloat16))


def gen_routed_weight(shape, dequant_std):
    """Synthesize a routed-expert per-channel-symmetric INT8 weight + FP32 scale by
    simulating the real DeepSeek-V4-Flash MXFP4 routed-expert quant grid (e2m1, per-32-group
    E8M0 scale), then re-quantizing per-output-channel. A plain ``randn`` INT8 is wrong:
    routed collapses onto ~37 discrete levels with an ~11.6% zero spike (the FP4 grid) and a
    per-channel scale CV ~0.09 (the fine group scale flattens it). Per-output-channel INT8 is
    scale-invariant, so the level structure / zero spike emerge from the grid alone and
    ``dequant_std`` only sets the absolute scale magnitude. (shared experts use a different
    grid -- see utils.gen_shared_weight.)

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel
    scale shape ([E, out, in] -> scale [E, out]).
    """
    import torch

    FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # nearest-grid bounds
    FP4_MAX, TINY = 6.0, 1e-20
    GROUP = 32
    CHUNK_ELEMS = 1 << 25  # elements per pass; unchunked this walks GiB-sized temporaries

    *lead, out, inn = shape
    n_lead = 1
    for dim in lead:
        n_lead *= dim

    W = torch.randn(*shape).reshape(n_lead, out, inn)
    w_i8 = torch.empty(n_lead, out, inn, dtype=torch.int8)
    scale = torch.empty(n_lead, out, 1, dtype=torch.float32)

    # e2m1 + per-32-group E8M0 (round-up) scale on the in dim, then per-output-channel
    # INT8. Chunked over the leading dim: every reduction here is confined to one
    # (out, in) row, so the chunk boundary cannot change a result.
    step = max(1, CHUNK_ELEMS // (out * inn))
    for i0 in range(0, n_lead, step):
        w = W[i0 : i0 + step]
        wg = w.reshape(-1, out, inn // GROUP, GROUP)
        absw = wg.abs()
        grp_scale = torch.exp2(
            torch.ceil(torch.log2((absw.amax(-1, keepdim=True) / FP4_MAX).clamp_min(TINY)))
        )
        idx = torch.bucketize(absw.div_(grp_scale), FP4_MID).clamp_max_(7)
        wq = (torch.sign(wg) * FP4_MAG[idx]).mul_(grp_scale).reshape(w.shape)
        amax = wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        chan_scale = amax / INT8_SCALE_MAX
        w_i8[i0 : i0 + step] = (
            torch.round(wq.div_(chan_scale)).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        )
        scale[i0 : i0 + step] = chan_scale
    del W

    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8.reshape(*shape), scale.reshape(*lead, out)


def compare_recv_y(actual, expected, *, inputs, **kwargs):
    """Compare only the input-declared active rows of each expert slot."""
    import torch
    from golden import ratio_reldiff

    if actual.shape != expected.shape or actual.ndim != 3:
        return False, "Expert output must have matching [slots, rows, channels] shapes"
    counts = inputs.get("recv_expert_count")
    if counts is None or counts.ndim != 2 or counts.shape[0] != actual.shape[0] or counts.shape[1] < 1:
        return False, "Missing or incompatible recv_expert_count input"
    if counts.dtype not in (torch.int32, torch.int64):
        return False, "Expert row counts must have an integer dtype"
    rows = counts.cpu()[:, 0].to(torch.int64)
    if bool(((rows < 0) | (rows > actual.shape[1])).any()):
        return False, "Expert row count is outside the output capacity"
    valid = torch.arange(actual.shape[1]).unsqueeze(0) < rows.unsqueeze(1)
    if not bool(valid.any()):
        return True, ""
    return ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)(
        actual.cpu()[valid], expected.cpu()[valid], inputs=inputs, **kwargs,
    )



def build_tensor_specs(merged=False):
    """Fixture for the routed expert.

    ``merged=True`` builds the shared-expert-merged form: ``N_SLOTS_B`` slots
    whose last one carries every token at weight 1.0, and an ``N_BANK`` bank
    whose last expert is a shared-expert weight set (a different quant grid --
    MXFP8, not MXFP4 -- so the merge does not silently change the numerics the
    shared path is validated against).
    """
    from utils import int8_quant_per_row
    import torch
    from golden import TensorSpec

    # Across-layer-mean dequant std (typical layer) of the real DeepSeek-V4-Flash MXFP4
    # routed experts; gen_routed_weight simulates the FP4 grid (see its docstring).
    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}
    SHARED_DEQUANT_STD = {"w1": 2.31e-2, "w2": 2.39e-2, "w3": 2.30e-2}

    n_slots = N_SLOTS_B if merged else N_SLOTS
    n_bank = N_BANK if merged else N_EXPERTS

    # One slot per active expert: T * TOPK routed pairs spread over N_SLOTS slots,
    # each slot holding at most T rows. Slot experts are distinct global ids.
    total = T * TOPK
    counts = torch.bincount(torch.randint(0, N_SLOTS, (total,)), minlength=N_SLOTS)
    counts = counts.clamp(max=T).to(torch.int32)
    slot_expert_1d = torch.randperm(N_EXPERTS)[:N_SLOTS].to(torch.int32)
    if merged:
        # The shared slot holds every token, at expert id SHARED_EID.
        counts = torch.cat([counts, torch.tensor([T], dtype=torch.int32)])
        slot_expert_1d = torch.cat([slot_expert_1d, torch.tensor([SHARED_EID], dtype=torch.int32)])
    counts_2d = torch.zeros(n_slots, IDX_PAD, dtype=torch.int32)
    counts_2d[:, 0] = counts
    slot_expert_2d = torch.zeros(n_slots, IDX_PAD, dtype=torch.int32)
    slot_expert_2d[:, 0] = slot_expert_1d

    # Build a consistent INT8 recv_x + per-row dequant scale (dispatch is
    # responsible for per-token quantization). Invalid tail rows go to INT8 0
    # with scale 0 so dequant produces 0.
    x_bf16 = torch.randn(n_slots, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(n_slots, 1, 1)
    recv_x_i8_pre, recv_scale_dq_pre = int8_quant_per_row(x_bf16)
    recv_x_i8_pre = torch.where(valid_mask_3d, recv_x_i8_pre, torch.zeros_like(recv_x_i8_pre))
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    recv_scale_dq_pre = torch.where(
        valid_mask_2d,
        recv_scale_dq_pre.squeeze(-1),
        torch.zeros_like(recv_scale_dq_pre.squeeze(-1)),
    )

    def init_recv_x():
        return recv_x_i8_pre

    def init_recv_scale_dq():
        return recv_scale_dq_pre.float()

    def init_recv_expert_count():
        return counts_2d

    def init_slot_expert():
        return slot_expert_2d

    # Per-row routing weight in [0, 1); tail rows (slot >= count) stay 0 so
    # they don't perturb the BF16 round-trip in expert_routed.
    recv_weights_pre = torch.rand(n_slots, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre))
    if merged:
        # The shared expert is not gated: its rows come in at weight 1.0.
        recv_weights_pre[SH_SLOT, :] = torch.where(
            valid_mask_2d[SH_SLOT, :], torch.ones(RECV_MAX), torch.zeros(RECV_MAX)
        )

    def init_recv_weights():
        return recv_weights_pre

    # Only the slot experts are ever read, so synthesize N_SLOTS weight sets and
    # scatter them into the full expert bank; the rest stay zero.
    def scatter_bank(shape_tail, dequant_std, shared_dequant_std, chan_cv):
        from utils import gen_shared_weight

        w_i8, w_s = gen_routed_weight((N_SLOTS, *shape_tail), dequant_std)
        bank_i8 = torch.zeros(n_bank, *shape_tail, dtype=torch.int8)
        bank_s = torch.zeros(n_bank, shape_tail[0], dtype=torch.float32)
        routed_eids = slot_expert_1d[:N_SLOTS].long()
        bank_i8[routed_eids] = w_i8
        bank_s[routed_eids] = w_s
        if merged:
            sh_i8, sh_s = gen_shared_weight(shape_tail, shared_dequant_std, chan_cv=chan_cv)
            bank_i8[SHARED_EID] = sh_i8
            bank_s[SHARED_EID] = sh_s
        return bank_i8, bank_s

    w1_i8, w1_s = scatter_bank((MOE_INTER, D), ROUTED_DEQUANT_STD["w1"], SHARED_DEQUANT_STD["w1"], 0.50)
    w3_i8, w3_s = scatter_bank((MOE_INTER, D), ROUTED_DEQUANT_STD["w3"], SHARED_DEQUANT_STD["w3"], 0.50)
    w2_i8, w2_s = scatter_bank((D, MOE_INTER), ROUTED_DEQUANT_STD["w2"], SHARED_DEQUANT_STD["w2"], 0.33)

    return [
        TensorSpec("recv_x", [n_slots, RECV_MAX, D], torch.int8, init_value=init_recv_x),
        TensorSpec("recv_scale_dq", [n_slots, RECV_MAX], torch.float32, init_value=init_recv_scale_dq),
        TensorSpec("recv_weights", [n_slots, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [n_slots, IDX_PAD], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("slot_expert", [n_slots, IDX_PAD], torch.int32, init_value=init_slot_expert),
        TensorSpec("routed_w1", [n_bank, MOE_INTER, D], torch.int8, init_value=lambda: w1_i8),
        TensorSpec("routed_w1_scale", [n_bank, MOE_INTER], torch.float32, init_value=lambda: w1_s),
        TensorSpec("routed_w3", [n_bank, MOE_INTER, D], torch.int8, init_value=lambda: w3_i8),
        TensorSpec("routed_w3_scale", [n_bank, MOE_INTER], torch.float32, init_value=lambda: w3_s),
        TensorSpec("routed_w2", [n_bank, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("routed_w2_scale", [n_bank, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("recv_y", [n_slots, RECV_MAX, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--tp",
        type=int,
        default=TP,
        choices=[1, 2, 4, 8],
        help="tensor-parallel degree; config freezes it at import",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument(
        "--golden-data",
        type=str,
        default=None,
        help="dir with cached in/{name}.pt + out/{name}.pt; reuses them "
        "instead of regenerating inputs + recomputing golden.",
    )
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=4, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        compile_only=args.compile_only,
        fn=expert_routed_test,
        specs=build_tensor_specs(),
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
            # BF16 recv_y, ~1 ULP. Gen weights reproduce real(L21): 0.016% vs 0.015% of points > 1e-3.
            "recv_y": compare_recv_y,
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
