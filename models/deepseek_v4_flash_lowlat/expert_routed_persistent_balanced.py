# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Persistent routed and shared MoE expert execution.

The planned entry consumes compact slot and row-count tables produced by route
grouping. The balanced wrapper constructs those tables for standalone fixtures
with arbitrary active slots. Each grid block processes complete expert chains,
with gate/up matmuls, SwiGLU, activation quantization, down matmul and dequantization
kept in one mixed task. The grid uses the runtime cluster count. Each cube core uses two vector lanes through UP_DOWN
regions; intermediate activations stay on chip.
"""


import pypto.language as pl
if __package__:
    from .config import INT8_SCALE_MAX, INT8_AMAX_EPS
else:
    from config import INT8_SCALE_MAX, INT8_AMAX_EPS
if __package__:
    from .expert_routed import (
        D,
        IDX_PAD,
        MOE_INTER,
        N_BANK,
        N_SLOTS_B,
        RECV_MAX,
        RECV_TILE,
        SWIGLU_LIMIT,
        build_tensor_specs,
        golden_expert_routed,
        compare_recv_y,
    )
else:
    from expert_routed import D, IDX_PAD, MOE_INTER, N_BANK, N_SLOTS_B, RECV_MAX, RECV_TILE, SWIGLU_LIMIT, build_tensor_specs, golden_expert_routed, compare_recv_y


NUM_CORES = 24
WORK_SLOTS_PAD = (N_SLOTS_B + IDX_PAD - 1) // IDX_PAD * IDX_PAD

# One cluster = 1 cube core + 2 buddy vector cores.
AIV_LANES = 2
ROW_HALF = RECV_TILE // AIV_LANES

# Whole-shard tiles preserve the guarded native TP8 Cube contract. Wider
# shards retain all columns and quantize only after a full-shard maximum.
FUSED_N_TILE = min(256, MOE_INTER)
# Two double-buffered INT32 accumulators share the 128 KiB L0C.
assert 4 * RECV_TILE * FUSED_N_TILE * 4 <= 131072, "gate/up accumulator tiles exceed L0C"
FUSED_K_TILE = max(64, min(512, 131072 // FUSED_N_TILE))
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
        # Keep the native whole-shard cohort; wide tiles use available clusters.
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

                # 0-or-1 trip: the zero case IS the empty-slot skip.
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
                            if MOE_INTER > FUSED_N_TILE:
                                if inter_start == 0:
                                    h_full = pl.full([ROW_HALF, MOE_INTER], dtype=pl.FP32, value=0.0)

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
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_f)), 1.0))
                            h_raw = pl.mul(pl.mul(gate_f, sigmoid), up_f)
                            if MOE_INTER == FUSED_N_TILE:
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

                                # V->C: the two lanes' row bands are stitched back into one
                                # full RECV_TILE operand in L1 for the down matmul.
                                h_gathered = pl.aic_gather(h_i8)
                            else:
                                h_full[:, inter_start : inter_start + FUSED_N_TILE] = h_raw

                    if MOE_INTER > FUSED_N_TILE:
                        # Reduce the whole shard before quantizing any column tile.
                        for aiv_quant in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                            lane_r0 = pl.cast(aiv_quant * ROW_HALF, pl.INDEX)
                            w_col = pl.reshape(recv_weights[s : s + 1, lane_r0 : lane_r0 + ROW_HALF], [ROW_HALF, 1])
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

                            # V->C: the two lanes' row bands are stitched back into one
                            # full RECV_TILE operand in L1 for the down matmul.
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

    if __package__:
        from .decode_shapes import TP
    else:
        from decode_shapes import TP
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("--sim", action="store_true", default=False)
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
            execution_mode=args.sim or args.platform.endswith("sim"),
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
