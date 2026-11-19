# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill persistent routed and shared expert computation in sixteen-row tiles."""

import pypto.language as pl
from .config import INT8_SCALE_MAX, INT8_AMAX_EPS
from .prefill_moe_geometry import D, IDX_PAD, MOE_INTER, N_BANK, N_SLOTS_B, RECV_MAX, RECV_TILE
from .prefill_moe_geometry import SWIGLU_LIMIT, WORK_SLOTS_PAD

NUM_CORES = 24
AIV_LANES = 2
ROW_HALF = RECV_TILE // AIV_LANES
FUSED_N_TILE = min(256, MOE_INTER)
FUSED_K_TILE = max(64, min(512, 131072 // FUSED_N_TILE))
FUSED_Y_TILE = max(64, min(512, 131072 // MOE_INTER))
if 4 * RECV_TILE * FUSED_N_TILE * 4 > 131072:
    raise ValueError("Prefill expert accumulators exceed L0C")


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


    w1_2d = pl.reshape(routed_w1, [N_BANK * MOE_INTER, D])
    w3_2d = pl.reshape(routed_w3, [N_BANK * MOE_INTER, D])
    w2_2d = pl.reshape(routed_w2, [N_BANK * D, MOE_INTER])

    with pl.scope():

        grid_count = NUM_CORES
        if MOE_INTER > FUSED_N_TILE:
            grid_count = pl.cast(pl.system.available_cluster_count(), pl.INDEX)
        with pl.spmd(
            grid_count, name_hint="exp_routed_balanced", sync_start=True,
            allow_early_resolve=True,
        ):
            core = pl.tile.get_block_idx()
            num_cores = NUM_CORES
            if MOE_INTER > FUSED_N_TILE:
                num_cores = pl.cast(pl.tile.get_block_num(), pl.INDEX)

            for w in pl.range(core, N_SLOTS_B, num_cores):
                s = pl.cast(pl.read(work_slot, [w]), pl.INDEX)
                eid = pl.cast(pl.read(slot_expert, [s, 0]), pl.INDEX)
                n_rows = pl.read(work_rows, [w])
                e_w1 = eid * MOE_INTER
                e_w2 = eid * D
                slot_base = s * RECV_MAX


                for t in pl.range((n_rows + RECV_TILE - 1) // RECV_TILE):
                    t0 = t * RECV_TILE
                    flat_t0 = slot_base + t0
                    valid_rows = pl.min(RECV_TILE, n_rows - t0)

                    h_full = pl.create_tensor([ROW_HALF, MOE_INTER], dtype=pl.FP32)
                    h_gathered = pl.create_tensor([RECV_TILE, FUSED_N_TILE], dtype=pl.INT8)
                    h_gathered_full = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8)
                    row_scale = pl.create_tensor([ROW_HALF, 1], dtype=pl.FP32)

                    for inter_start in pl.unroll(0, MOE_INTER, FUSED_N_TILE):
                        inter_base = e_w1 + inter_start
                        gate_acc = pl.create_tensor([RECV_TILE, FUSED_N_TILE], dtype=pl.INT32)
                        for kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                            k0 = kb * FUSED_K_TILE
                            x_k = pl.slice(recv_x_flat, [RECV_TILE, FUSED_K_TILE], [flat_t0, k0],
                                           valid_shape=[valid_rows, FUSED_K_TILE])
                            w1_k = w1_2d[inter_base : inter_base + FUSED_N_TILE, k0 : k0 + FUSED_K_TILE]
                            gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True, init_cond=(kb == 0))

                        up_acc = pl.create_tensor([RECV_TILE, FUSED_N_TILE], dtype=pl.INT32)
                        for ukb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                            uk0 = ukb * FUSED_K_TILE
                            x_u = pl.slice(recv_x_flat, [RECV_TILE, FUSED_K_TILE], [flat_t0, uk0],
                                           valid_shape=[valid_rows, FUSED_K_TILE])
                            w3_k = w3_2d[inter_base : inter_base + FUSED_N_TILE, uk0 : uk0 + FUSED_K_TILE]
                            up_acc = pl.matmul_acc(up_acc, x_u, w3_k, b_trans=True, init_cond=(ukb == 0))

                        w1_sc = routed_w1_scale[eid : eid + 1, inter_start : inter_start + FUSED_N_TILE]
                        w3_sc = routed_w3_scale[eid : eid + 1, inter_start : inter_start + FUSED_N_TILE]


                        for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):


                            lane_r0 = pl.cast(aiv_id * ROW_HALF, pl.INDEX)
                            if MOE_INTER > FUSED_N_TILE:
                                if inter_start == 0:
                                    h_full = pl.full([ROW_HALF, MOE_INTER], dtype=pl.FP32, value=0.0)

                            x_sc = pl.reshape(
                                recv_scale_dq[s : s + 1, t0 + lane_r0 : t0 + lane_r0 + ROW_HALF],
                                [ROW_HALF, 1])
                            w_col = pl.reshape(
                                recv_weights[s : s + 1, t0 + lane_r0 : t0 + lane_r0 + ROW_HALF],
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
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_f)), 1.0))
                            h_raw = pl.mul(pl.mul(gate_f, sigmoid), up_f)
                            if MOE_INTER == FUSED_N_TILE:


                                h_pad = pl.fillpad(h_raw, pad_value=pl.PadValue.zero)
                                h_f32 = pl.set_validshape(h_pad, ROW_HALF, FUSED_N_TILE)

                                h_abs = pl.maximum(h_f32, pl.neg(h_f32))
                                h_amax = pl.maximum(
                                    pl.full([1, ROW_HALF], dtype=pl.FP32, value=INT8_AMAX_EPS),
                                    pl.reshape(pl.row_max(h_abs), [1, ROW_HALF]),
                                )
                                sq_row = pl.div(
                                    pl.full([1, ROW_HALF], dtype=pl.FP32, value=INT8_SCALE_MAX), h_amax
                                )
                                h_scale_dq = pl.reshape(pl.recip(sq_row), [ROW_HALF, 1])
                                sq_col = pl.reshape(sq_row, [ROW_HALF, 1])
                                q_scaled = pl.row_expand_mul(h_f32, sq_col)
                                q_i32 = pl.cast(q_scaled, target_type=pl.INT32, mode="rint")
                                q_f16 = pl.cast(q_i32, target_type=pl.FP16, mode="round")
                                h_i8 = pl.cast(q_f16, target_type=pl.INT8, mode="trunc")

                                row_scale = pl.mul(h_scale_dq, w_col)


                                h_gathered = pl.aic_gather(h_i8)
                            else:
                                h_full[:, inter_start : inter_start + FUSED_N_TILE] = h_raw

                    if MOE_INTER > FUSED_N_TILE:

                        for aiv_quant in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                            lane_r0 = pl.cast(aiv_quant * ROW_HALF, pl.INDEX)
                            w_col = pl.reshape(recv_weights[s : s + 1, t0 + lane_r0 : t0 + lane_r0 + ROW_HALF], [ROW_HALF, 1])
                            h_amax = pl.full([1, ROW_HALF], dtype=pl.FP32, value=INT8_AMAX_EPS)
                            for amax_start in pl.range(0, MOE_INTER, FUSED_N_TILE):
                                h_amax_chunk = h_full[:, amax_start : amax_start + FUSED_N_TILE]
                                h_abs = pl.maximum(h_amax_chunk, pl.neg(h_amax_chunk))
                                h_chunk_max = pl.reshape(pl.row_max(h_abs), [1, ROW_HALF])
                                h_amax = pl.maximum(h_amax, h_chunk_max)
                            sq_row = pl.div(
                                pl.full([1, ROW_HALF], dtype=pl.FP32, value=INT8_SCALE_MAX), h_amax
                            )
                            h_scale_dq = pl.reshape(pl.recip(sq_row), [ROW_HALF, 1])
                            sq_col = pl.reshape(sq_row, [ROW_HALF, 1])
                            h_f16 = pl.create_tensor([ROW_HALF, MOE_INTER], dtype=pl.FP16)
                            for q_start in pl.range(0, MOE_INTER, FUSED_N_TILE):
                                h_chunk = h_full[:, q_start : q_start + FUSED_N_TILE]
                                q_scaled = pl.row_expand_mul(h_chunk, sq_col)
                                q_i32 = pl.cast(q_scaled, target_type=pl.INT32, mode="rint")
                                q_f16 = pl.cast(q_i32, target_type=pl.FP16, mode="round")
                                h_f16[:, q_start : q_start + FUSED_N_TILE] = q_f16

                            h_i8_full = pl.cast(h_f16, target_type=pl.INT8, mode="trunc")
                            row_scale = pl.mul(h_scale_dq, w_col)


                            h_gathered_full = pl.aic_gather(h_i8_full)

                    for db in pl.pipeline(D // FUSED_Y_TILE, stage=2):
                        d0 = db * FUSED_Y_TILE
                        if MOE_INTER == FUSED_N_TILE:
                            w2_k = w2_2d[e_w2 + d0 : e_w2 + d0 + FUSED_Y_TILE, 0:FUSED_N_TILE]
                            y_acc = pl.matmul(h_gathered, w2_k, b_trans=True, out_dtype=pl.INT32)
                        else:
                            w2_full = w2_2d[e_w2 + d0 : e_w2 + d0 + FUSED_Y_TILE, 0:MOE_INTER]
                            y_acc = pl.matmul(h_gathered_full, w2_full, b_trans=True, out_dtype=pl.INT32)
                        w2_sc = routed_w2_scale[eid : eid + 1, d0 : d0 + FUSED_Y_TILE]


                        for aiv_id2 in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                            y_off = flat_t0 + pl.cast(aiv_id2 * ROW_HALF, pl.INDEX)
                            y_sh = pl.aiv_shard(y_acc)
                            y_f = pl.cast(y_sh, target_type=pl.FP32, mode="none")
                            y_f = pl.col_expand_mul(pl.row_expand_mul(y_f, row_scale), w2_sc)
                            recv_y_flat[y_off : y_off + ROW_HALF, d0 : d0 + FUSED_Y_TILE] = \
                                pl.cast(y_f, target_type=pl.BF16, mode="rint")

    return recv_y
