# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A/B partner of ``expert_routed.py``: the chain as THREE persistent grid-stride
kernels instead of ~124 per-slot task instances.

Ported from ``deepseek_v4_flash_mtp/expert_routed_tp_persistent.py``, which runs
the whole chain as ONE ``pl.spmd(NUM_CORES)`` task with every intermediate on
chip. That form does not build here: a cast to INT8 inside a fused mixed
cube+vector scope emits a TCVT destination view with a (0, 0) valid shape and
incore compilation fails, for every cast spelling tried (see the local
known-issues log). Only the INT8 quantize is therefore lifted into its own
vector-only task; the two matmul phases keep their vector epilogues fused, which
is more fusion than ``expert_routed.py`` has, not less.

    p1  mixed   gate mm + up mm + dequant + SwiGLU + per-row amax  -> q_scaled FP32
    p2  vector  q_scaled -> INT8                                   (the forced split)
    p3  mixed   down mm + per-row/per-channel dequant              -> recv_y BF16

Each phase is one submit of NUM_CORES blocks that grid-strides the slot table,
against ``expert_routed.py``'s one submit per (slot, tile) for the two matmuls.
Slot ``i`` owns rows ``[i * RECV_MAX, i * RECV_MAX + recv_expert_count[i])`` and
``RECV_MAX == RECV_TILE``, so the slot table is already the work pool.

The work item is the expert slot, so a phase costs ``ceil(n_active / NUM_CORES)``
rounds; at this tree's decode shape ``n_active`` is ~31 of 48 slots, i.e. two
rounds for a 1.3x workload.
"""


import pypto.language as pl

from config import (INT8_SCALE_MAX, INT8_AMAX_EPS)
from expert_routed import (
    D, IDX_PAD, MOE_INTER, N_EXPERTS, N_SLOTS, RECV_MAX, RECV_TILE, SWIGLU_LIMIT,
    build_tensor_specs, golden_expert_routed,
)


# Persistent blocks == the a2a3 AIC count. No barrier lives inside a phase, so a
# mismatch on another SoC is a scheduling fact, not a hang.
NUM_CORES = 24

# tiling
FUSED_N_TILE = MOE_INTER
FUSED_K_TILE = 512
FUSED_Y_TILE = 512
QUANT_TILE = MOE_INTER


@pl.jit.inline(auto_scope=False)
def expert_routed_persistent(
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
    recv_x_flat = pl.reshape(recv_x, [N_SLOTS * RECV_MAX, D])
    recv_y_flat = pl.reshape(recv_y, [N_SLOTS * RECV_MAX, D])

    # Fold the expert axis into the row axis so every matmul operand is 2-D and
    # the accumulator comes back 2-D (no [1, RECV_TILE, TILE] reshape unwrap).
    w1_2d = pl.reshape(routed_w1, [N_EXPERTS * MOE_INTER, D])
    w3_2d = pl.reshape(routed_w3, [N_EXPERTS * MOE_INTER, D])
    w2_2d = pl.reshape(routed_w2, [N_EXPERTS * D, MOE_INTER])

    with pl.scope():
        # Only what has to cross a phase boundary: the pre-scaled activation and
        # the row scale that carries the routing weight into the down dequant.
        q_scaled_gm = pl.create_tensor(
            [N_SLOTS * RECV_MAX, MOE_INTER], dtype=pl.FP32, manual_dep=True
        )
        h_i8 = pl.create_tensor(
            [N_SLOTS * RECV_MAX, MOE_INTER], dtype=pl.INT8, manual_dep=True
        )
        row_scale_gm = pl.create_tensor(
            [N_SLOTS * RECV_MAX, 1], dtype=pl.FP32, manual_dep=True
        )

        with pl.spmd(
            NUM_CORES, name_hint="exp_p1_gate_up_act", allow_early_resolve=True
        ) as p1_tid:
            core = pl.tile.get_block_idx()  # 0 .. NUM_CORES-1

            for s in pl.range(core, N_SLOTS, NUM_CORES):
                eid = pl.cast(pl.read(slot_expert, [s, 0]), pl.INDEX)
                n_rows = pl.read(recv_expert_count, [s, 0])
                e_w1 = eid * MOE_INTER
                slot_base = s * RECV_MAX

                # 0-or-1 trip: the zero case IS the empty-slot skip.
                for t in pl.range((n_rows + RECV_TILE - 1) // RECV_TILE):
                    t0 = t * RECV_TILE
                    flat_t0 = slot_base + t0

                    gate_acc = pl.create_tensor([RECV_TILE, FUSED_N_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, FUSED_K_TILE, stage=2):
                        x_k = recv_x_flat[flat_t0 : flat_t0 + RECV_TILE, k0 : k0 + FUSED_K_TILE]
                        w1_k = w1_2d[e_w1 : e_w1 + FUSED_N_TILE, k0 : k0 + FUSED_K_TILE]
                        if k0 == 0:
                            gate_acc = pl.matmul(x_k, w1_k, b_trans=True, out_dtype=pl.INT32)
                        else:
                            gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)

                    up_acc = pl.create_tensor([RECV_TILE, FUSED_N_TILE], dtype=pl.INT32)
                    for uk0 in pl.pipeline(0, D, FUSED_K_TILE, stage=2):
                        x_u = recv_x_flat[flat_t0 : flat_t0 + RECV_TILE, uk0 : uk0 + FUSED_K_TILE]
                        w3_k = w3_2d[e_w1 : e_w1 + FUSED_N_TILE, uk0 : uk0 + FUSED_K_TILE]
                        if uk0 == 0:
                            up_acc = pl.matmul(x_u, w3_k, b_trans=True, out_dtype=pl.INT32)
                        else:
                            up_acc = pl.matmul_acc(up_acc, x_u, w3_k, b_trans=True)

                    x_sc = pl.reshape(recv_scale_dq[s : s + 1, 0:RECV_TILE], [RECV_TILE, 1])
                    w1_sc = routed_w1_scale[eid : eid + 1, 0:FUSED_N_TILE]
                    w3_sc = routed_w3_scale[eid : eid + 1, 0:FUSED_N_TILE]
                    gate_f = pl.cast(gate_acc, target_type=pl.FP32, mode="none")
                    up_f = pl.cast(up_acc, target_type=pl.FP32, mode="none")
                    gate_f = pl.col_expand_mul(pl.row_expand_mul(gate_f, x_sc), w1_sc)
                    up_f = pl.col_expand_mul(pl.row_expand_mul(up_f, x_sc), w3_sc)
                    if SWIGLU_LIMIT > 0.0:
                        gate_f = pl.minimum(gate_f, SWIGLU_LIMIT)
                        up_f = pl.maximum(pl.minimum(up_f, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_f)), 1.0))
                    # A pad row's dequant scale is 0, so gate_f and up_f are
                    # already exactly 0 there and no tail mask is needed.
                    h_f32 = pl.mul(pl.mul(gate_f, sigmoid), up_f)

                    h_abs = pl.maximum(h_f32, pl.neg(h_f32))
                    h_amax = pl.maximum(
                        pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS),
                        pl.reshape(pl.row_max(h_abs), [1, RECV_TILE]),
                    )
                    sq_row = pl.div(
                        pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), h_amax
                    )
                    h_scale_dq = pl.reshape(pl.recip(sq_row), [RECV_TILE, 1])
                    sq_col = pl.reshape(sq_row, [RECV_TILE, 1])
                    q_scaled_gm[flat_t0 : flat_t0 + RECV_TILE, 0:MOE_INTER] = \
                        pl.row_expand_mul(h_f32, sq_col)

                    w_col = pl.reshape(recv_weights[s : s + 1, 0:RECV_TILE], [RECV_TILE, 1])
                    row_scale_gm[flat_t0 : flat_t0 + RECV_TILE, 0:1] = \
                        pl.mul(h_scale_dq, w_col)

        # Vector-only, and only because the INT8 cast cannot live in a mixed scope.
        with pl.spmd(
            NUM_CORES, name_hint="exp_p2_quant", allow_early_resolve=True, deps=[p1_tid]
        ) as p2_tid:
            q_core = pl.tile.get_block_idx()

            for qs in pl.range(q_core, N_SLOTS, NUM_CORES):
                q_rows = pl.read(recv_expert_count, [qs, 0])
                q_base = qs * RECV_MAX

                for qt in pl.range((q_rows + RECV_TILE - 1) // RECV_TILE):
                    q_t0 = q_base + qt * RECV_TILE
                    for q0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                        q_chunk = q_scaled_gm[q_t0 : q_t0 + RECV_TILE, q0 : q0 + QUANT_TILE]
                        q_i32 = pl.cast(q_chunk, target_type=pl.INT32, mode="rint")
                        q_f16 = pl.cast(q_i32, target_type=pl.FP16, mode="round")
                        h_i8[q_t0 : q_t0 + RECV_TILE, q0 : q0 + QUANT_TILE] = pl.cast(
                            q_f16, target_type=pl.INT8, mode="trunc"
                        )

        with pl.spmd(
            NUM_CORES, name_hint="exp_p3_down_act", allow_early_resolve=True, deps=[p2_tid]
        ) as _p3_tid:
            d_core = pl.tile.get_block_idx()

            for ds in pl.range(d_core, N_SLOTS, NUM_CORES):
                d_eid = pl.cast(pl.read(slot_expert, [ds, 0]), pl.INDEX)
                d_rows = pl.read(recv_expert_count, [ds, 0])
                e_w2 = d_eid * D
                d_base = ds * RECV_MAX

                for dt in pl.range((d_rows + RECV_TILE - 1) // RECV_TILE):
                    d_t0 = d_base + dt * RECV_TILE
                    h_tile = h_i8[d_t0 : d_t0 + RECV_TILE, 0:MOE_INTER]
                    row_scale = row_scale_gm[d_t0 : d_t0 + RECV_TILE, 0:1]

                    for db in pl.range(D // FUSED_Y_TILE):
                        d0 = db * FUSED_Y_TILE
                        w2_k = w2_2d[e_w2 + d0 : e_w2 + d0 + FUSED_Y_TILE, 0:MOE_INTER]
                        y_acc = pl.matmul(h_tile, w2_k, b_trans=True, out_dtype=pl.INT32)
                        w2_sc = routed_w2_scale[d_eid : d_eid + 1, d0 : d0 + FUSED_Y_TILE]
                        y_f = pl.cast(y_acc, target_type=pl.FP32, mode="none")
                        y_f = pl.col_expand_mul(pl.row_expand_mul(y_f, row_scale), w2_sc)
                        recv_y_flat[d_t0 : d_t0 + RECV_TILE, d0 : d0 + FUSED_Y_TILE] = \
                            pl.cast(y_f, target_type=pl.BF16, mode="rint")

    return recv_y


@pl.jit
def expert_routed_persistent_test(
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
    expert_routed_persistent(
        recv_x, recv_scale_dq, recv_weights, recv_expert_count, slot_expert,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )
    return recv_y


if __name__ == "__main__":
    import argparse
    from config import TP
    from golden import ratio_reldiff, run_jit

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

    result = run_jit(
        compile_only=args.compile_only,
        fn=expert_routed_persistent_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
