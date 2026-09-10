# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""REVERTED A/B: the routed expert scattering straight onto the token rows.

Same single persistent 24-block task as
``expert_routed_persistent_balanced.py`` -- same hand-written ``pl.split_aiv``
regions, same ``exp_balance_plan`` pre-task -- but the down-matmul epilogue
scatter-accumulates each VALID row onto its destination token with
``pl.assemble(..., atomic=pl.AtomicType.Add)`` instead of storing a per-slot
``recv_y`` tile. That folds ``moe.combine_local`` into the expert: only valid
rows are written (a slot holds one or two of 16 at decode) and the per-token
TOPK sum happens in the atomic rather than in a downstream gather.

It is SLOWER here. moe.py --tp 8, 8 cards, 20 rounds / 3 warmup, fastest-rank
mean, interleaved against the committed form: 244.5 -> 269.0 us (+10.0 %), 3/3
pairs. The scatter costs more inside the kernel than the gather cost outside it:
standalone the same kernel goes 129.7 -> 136.8 us (+7.1 us), because the epilogue
emits ROW_HALF predicated 1 x FUSED_Y_TILE atomic stores per lane per D chunk
(2 lanes x 8 chunks x 8 rows = 128 sites per slot) where the tile form emitted 16
coalesced stores -- and at T=8 a slot has one or two live rows, so nearly all of
those sites are dead predicates and lane 1 is empty outright. Deleting
combine_local gives back only its 6.2 us span and one hop, and the accumulator
needs two new tasks (``ffn_zero``, ``sh_scatter``) on top.

The sign flips as rows-per-slot grows: the reference this is ported from
(deepseek_v4_flash_mtp/expert_routed_tp_persistent.py) runs 64 gathered rows over
~32 experts, i.e. 12 rows per group, where the tile store is mostly padding and
the gather is 8x larger. Re-measure if T or TOPK grows.

``ffn_partial`` is an ACCUMULATOR, so its test binding is ``pl.InOut`` with a
zero ``init_value``: a pure ``pl.Out`` buffer is allocator residue that never
reaches the device, and the atomics would land on garbage. The harness exits 1
with no message if you get that wrong.

Numerics: the routed contribution stays FP32 end to end instead of rounding
through a BF16 ``recv_y``, and the TOPK sum reassociates run to run because
atomic-add order across cores is not fixed. Both were accepted deliberately.
"""

import pypto.language as pl

from config import INT8_SCALE_MAX, INT8_AMAX_EPS
from expert_routed import (
    D,
    IDX_PAD,
    MOE_INTER,
    N_EXPERTS,
    N_SLOTS,
    RECV_MAX,
    RECV_TILE,
    SWIGLU_LIMIT,
    T,
    build_tensor_specs,
)


# Persistent blocks == the a2a3 AIC count. No barrier lives in this kernel, so a
# mismatch on another SoC is a scheduling fact, not a hang.
NUM_CORES = 24

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
def expert_routed_scatter(
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
    plan_row_token: pl.Tensor[[N_SLOTS, RECV_MAX], pl.INT32],
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
):
    recv_x_flat = pl.reshape(recv_x, [N_SLOTS * RECV_MAX, D])

    # Fold the expert axis into the row axis so every matmul operand is 2-D and
    # the accumulator comes back 2-D (no [1, RECV_TILE, TILE] reshape unwrap).
    w1_2d = pl.reshape(routed_w1, [N_EXPERTS * MOE_INTER, D])
    w3_2d = pl.reshape(routed_w3, [N_EXPERTS * MOE_INTER, D])
    w2_2d = pl.reshape(routed_w2, [N_EXPERTS * D, MOE_INTER])

    with pl.scope():
        # One core owns the scan: the compaction is a serial prefix, and 48
        # scalar reads cost less than the barrier a parallel scan would need.
        # The list is PADDED to N_SLOTS rather than bounded by a live count: a
        # runtime pool bound would have to be pl.read at orchestration level,
        # which parks the whole task graph behind the pre-task's completion.
        # Padding entries carry rows = 0, so they land on the same zero-trip
        # skip an empty slot already takes, and the bound stays static.
        work_slot = pl.create_tensor([N_SLOTS], dtype=pl.INT32)
        work_rows = pl.create_tensor([N_SLOTS], dtype=pl.INT32)

        with pl.spmd(1, name_hint="exp_balance_plan") as plan_tid:
            plan_core = pl.tile.get_block_idx()
            zero_i32 = pl.cast(0, pl.INT32)
            for pad in pl.range(plan_core, N_SLOTS):
                pl.write(work_slot, [pad], zero_i32)
                pl.write(work_rows, [pad], zero_i32)
            n_live = pl.cast(0, pl.INDEX)
            for scan in pl.range(plan_core, N_SLOTS):
                scan_rows = pl.read(recv_expert_count, [scan, 0])
                if scan_rows > 0:
                    pl.write(work_slot, [n_live], pl.cast(scan, pl.INT32))
                    pl.write(work_rows, [n_live], scan_rows)
                    n_live = n_live + 1

        with pl.spmd(
            NUM_CORES,
            name_hint="exp_routed_scatter",
            allow_early_resolve=True,
            deps=[plan_tid],
        ):
            core = pl.tile.get_block_idx()  # 0 .. NUM_CORES-1

            for w in pl.range(core, N_SLOTS, NUM_CORES):
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

                    x_k0 = pl.slice(
                        recv_x_flat,
                        [RECV_TILE, FUSED_K_TILE],
                        [flat_t0, 0],
                        valid_shape=[valid_rows, FUSED_K_TILE],
                    )
                    w1_k0 = w1_2d[e_w1 : e_w1 + FUSED_N_TILE, 0:FUSED_K_TILE]
                    gate_acc = pl.matmul(x_k0, w1_k0, b_trans=True, out_dtype=pl.INT32)
                    for kb in pl.pipeline(1, D // FUSED_K_TILE, stage=2):
                        k0 = kb * FUSED_K_TILE
                        x_k = pl.slice(
                            recv_x_flat,
                            [RECV_TILE, FUSED_K_TILE],
                            [flat_t0, k0],
                            valid_shape=[valid_rows, FUSED_K_TILE],
                        )
                        w1_k = w1_2d[e_w1 : e_w1 + FUSED_N_TILE, k0 : k0 + FUSED_K_TILE]
                        gate_acc = pl.matmul_acc(gate_acc, x_k, w1_k, b_trans=True)

                    x_u0 = pl.slice(
                        recv_x_flat,
                        [RECV_TILE, FUSED_K_TILE],
                        [flat_t0, 0],
                        valid_shape=[valid_rows, FUSED_K_TILE],
                    )
                    w3_k0 = w3_2d[e_w1 : e_w1 + FUSED_N_TILE, 0:FUSED_K_TILE]
                    up_acc = pl.matmul(x_u0, w3_k0, b_trans=True, out_dtype=pl.INT32)
                    for ukb in pl.pipeline(1, D // FUSED_K_TILE, stage=2):
                        uk0 = ukb * FUSED_K_TILE
                        x_u = pl.slice(
                            recv_x_flat,
                            [RECV_TILE, FUSED_K_TILE],
                            [flat_t0, uk0],
                            valid_shape=[valid_rows, FUSED_K_TILE],
                        )
                        w3_k = w3_2d[e_w1 : e_w1 + FUSED_N_TILE, uk0 : uk0 + FUSED_K_TILE]
                        up_acc = pl.matmul_acc(up_acc, x_u, w3_k, b_trans=True)

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

                        gate_sh = pl.aiv_shard(gate_acc)
                        up_sh = pl.aiv_shard(up_acc)
                        x_sc = pl.reshape(
                            recv_scale_dq[s : s + 1, lane_r0 : lane_r0 + ROW_HALF], [ROW_HALF, 1]
                        )
                        gate_f = pl.cast(gate_sh, target_type=pl.FP32, mode="none")
                        up_f = pl.cast(up_sh, target_type=pl.FP32, mode="none")
                        gate_f = pl.col_expand_mul(pl.row_expand_mul(gate_f, x_sc), w1_sc)
                        up_f = pl.col_expand_mul(pl.row_expand_mul(up_f, x_sc), w3_sc)
                        if SWIGLU_LIMIT > 0.0:
                            gate_f = pl.minimum(gate_f, SWIGLU_LIMIT)
                            up_f = pl.maximum(pl.minimum(up_f, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_f)), 1.0))
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
                        sq_row = pl.div(pl.full([1, ROW_HALF], dtype=pl.FP32, value=INT8_SCALE_MAX), h_amax)
                        h_scale_dq = pl.reshape(pl.recip(sq_row), [ROW_HALF, 1])
                        sq_col = pl.reshape(sq_row, [ROW_HALF, 1])
                        q_scaled = pl.row_expand_mul(h_f32, sq_col)
                        q_i32 = pl.cast(q_scaled, target_type=pl.INT32, mode="rint")
                        q_f16 = pl.cast(q_i32, target_type=pl.FP16, mode="round")
                        h_i8 = pl.cast(q_f16, target_type=pl.INT8, mode="trunc")

                        w_col = pl.reshape(
                            recv_weights[s : s + 1, lane_r0 : lane_r0 + ROW_HALF], [ROW_HALF, 1]
                        )
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
                            y_r0 = pl.cast(aiv_id2 * ROW_HALF, pl.INDEX)
                            y_valid = pl.min(ROW_HALF, pl.max(n_rows - y_r0, 0))
                            y_sh = pl.aiv_shard(y_acc)
                            y_f = pl.cast(y_sh, target_type=pl.FP32, mode="none")
                            y_f = pl.col_expand_mul(pl.row_expand_mul(y_f, row_scale), w2_sc)
                            # Unrolled and predicated because the destination row
                            # is a per-row runtime value: a dynamic row index into
                            # an on-chip tile is not expressible, a static one is.
                            for r in pl.unroll(ROW_HALF):
                                if r < y_valid:
                                    dst = pl.cast(pl.read(plan_row_token, [s, y_r0 + r]), pl.INDEX)
                                    ffn_partial = pl.assemble(
                                        ffn_partial,
                                        y_f[r : r + 1, :],
                                        [dst, d0],
                                        atomic=pl.AtomicType.Add,
                                    )

    return ffn_partial


@pl.jit
def expert_routed_scatter_test(
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
    plan_row_token: pl.Tensor[[N_SLOTS, RECV_MAX], pl.INT32],
    # InOut, not Out: the kernel accumulates, so the host zeros must reach the
    # device -- a pure Out buffer is allocator residue.
    ffn_partial: pl.InOut[pl.Tensor[[T, D], pl.FP32]],
):
    expert_routed_scatter(
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
        plan_row_token,
        ffn_partial,
    )
    return ffn_partial


def build_tensor_specs_scatter():
    """expert_routed's fixture with the recv_y output swapped for the scatter
    plan and its zero-seeded accumulator. Row r of a slot comes from token r,
    which is what real routing produces: an expert takes at most one row per
    token, so a slot's rows carry distinct tokens."""
    import torch
    from golden import TensorSpec

    specs = [s for s in build_tensor_specs() if s.name != "recv_y"]
    row_token = torch.arange(RECV_MAX, dtype=torch.int32).repeat(N_SLOTS, 1)
    specs.append(TensorSpec("plan_row_token", [N_SLOTS, RECV_MAX], torch.int32, init_value=lambda: row_token))
    specs.append(
        TensorSpec("ffn_partial", [T, D], torch.float32, is_output=True, init_value=lambda: torch.zeros(T, D))
    )
    return specs


def golden_expert_routed_scatter(tensors):
    """Per-route math kept in FP32 -- the kernel no longer rounds through a BF16
    recv_y -- then the scatter the kernel folds into its store."""
    from utils import int8_quant_per_row
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)

    x_i8 = tensors["recv_x"]
    x_sd = tensors["recv_scale_dq"].float()
    w_rt = tensors["recv_weights"].float()
    counts = tensors["recv_expert_count"][:, 0]
    slot_expert = tensors["slot_expert"][:, 0]
    rt = tensors["plan_row_token"]
    w1 = dequant_w(tensors["routed_w1"], tensors["routed_w1_scale"].float())
    w3 = dequant_w(tensors["routed_w3"], tensors["routed_w3_scale"].float())
    w2 = dequant_w(tensors["routed_w2"], tensors["routed_w2_scale"].float())

    out = torch.zeros(T, D, dtype=torch.float32)
    for slot in range(N_SLOTS):
        n = int(counts[slot].item())
        if n == 0:
            continue
        e = int(slot_expert[slot].item())
        x_q = x_i8[slot, :n, :].float() * x_sd[slot, :n].reshape(-1, 1)
        gate = x_q @ w1[e].T
        up = x_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        h_i8, h_sd = int8_quant_per_row(h)
        h = h_i8.float() * (h_sd * w_rt[slot, :n].reshape(-1, 1))
        y = h @ w2[e].T
        for r in range(n):
            out[int(rt[slot, r].item())] += y[r]
    tensors["ffn_partial"][:] = out


if __name__ == "__main__":
    import argparse
    from config import TP
    from golden import ratio_reldiff, run_jit

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

    result = run_jit(
        compile_only=args.compile_only,
        fn=expert_routed_scatter_test,
        specs=build_tensor_specs_scatter(),
        golden_fn=golden_expert_routed_scatter,
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
            # FP32 now, and the TOPK sum reassociates across cores run to run,
            # so this is looser than the BF16 tile form's threshold.
            "ffn_partial": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
