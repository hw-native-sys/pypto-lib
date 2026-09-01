# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE routed expert compute over a TP slice of the intermediate dim.

Every rank holds every expert and a 1/TP slice of the intermediate dim, so
``recv_y`` is a partial sum that ``moe`` all-reduces across the TP group. Rows
arrive already grouped: slot ``i`` holds the rows routed to expert
``slot_expert[i]``, and inactive slots carry a zero count.
"""


import pypto.language as pl

from config import (FLASH as M, DECODE_BATCH, DECODE_SEQ, TP, INT8_SCALE_MAX, INT8_AMAX_EPS)


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
        h_scale_dq = pl.create_tensor(
            [N_SLOTS * RECV_MAX, 1], dtype=pl.FP32, manual_dep=True
        )
        # Slot-major so a grouped quantize block can address its own rows.
        # manual_dep: the pool's only real edge is gate_up -> h_q, restated below
        # per slot group, so the scheduler skips creator retention and the
        # tensormap overlap walk over a 768-row tensor.
        # Split variant: the cube task writes INT32 gate/up here so it stays
        # AIC-only instead of reserving its two paired AIV cores for the whole
        # matmul. 1.5 MiB at the decode shape; scales with N_SLOTS*RECV_MAX.
        gate_i32 = pl.create_tensor(
            [N_SLOTS * RECV_MAX, MOE_INTER], dtype=pl.INT32, manual_dep=True
        )
        up_i32 = pl.create_tensor(
            [N_SLOTS * RECV_MAX, MOE_INTER], dtype=pl.INT32, manual_dep=True
        )
        h_fp32 = pl.create_tensor(
            [N_SLOTS * RECV_MAX, MOE_INTER], dtype=pl.FP32, manual_dep=True
        )
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
                with pl.spmd(
                    MOE_INTER // MM_INTER_TILE, name_hint="exp_gate_up_mm"
                ) as gate_up_tid:
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

                    gate_i32[flat_t0 : flat_t0 + RECV_TILE, n0 : n0 + MM_INTER_TILE] = \
                        pl.reshape(gate_acc, [RECV_TILE, MM_INTER_TILE])
                    up_i32[flat_t0 : flat_t0 + RECV_TILE, n0 : n0 + MM_INTER_TILE] = \
                        pl.reshape(up_acc, [RECV_TILE, MM_INTER_TILE])
                gate_up_tids[local_i] = gate_up_tid

        # AIV -> AIV with no other consumer: the SwiGLU epilogue and the
        # requantize are one task. `gated` stays in UB (RECV_TILE x MOE_INTER
        # requantize are one task, so the epilogue chain costs one submit, not two.
        for ga in pl.parallel(N_SLOTS // SLOT_GROUP):
            with pl.spmd(
                SLOT_GROUP,
                name_hint="exp_act_h_q",
                deps=[gate_up_tids[ga * SLOT_GROUP + s_] for s_ in range(SLOT_GROUP)],
            ) as _act_h_q_tid:
                a_sub = pl.tile.get_block_idx()
                a_slot = ga * SLOT_GROUP + a_sub
                flat_a0 = a_slot * RECV_MAX
                a_expert = pl.cast(pl.read(slot_expert, [a_slot, 0]), pl.INDEX)
                a_rows = pl.read(recv_expert_count, [a_slot, 0])
                for _a in pl.range((a_rows + RECV_TILE - 1) // RECV_TILE):
                    a_valid = pl.min(RECV_TILE, a_rows)
                    a_scale = pl.reshape(
                        recv_scale_dq[a_slot : a_slot + 1, 0:RECV_TILE], [RECV_TILE, 1]
                    )
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
                        sig = pl.recip(pl.add(pl.exp(pl.neg(g2)), 1.0))
                        gd = pl.mul(pl.mul(g2, sig), u2)
                        gd_valid = pl.set_validshape(gd, a_valid, ACT_INTER_TILE)
                        gd_pad = pl.fillpad(gd_valid, pad_value=pl.PadValue.zero)
                        h_fp32[flat_a0 : flat_a0 + RECV_TILE, a0 : a0 + ACT_INTER_TILE] = gd_pad
                        eh_amax = pl.maximum(
                            eh_amax,
                            pl.reshape(pl.row_max(pl.maximum(gd_pad, pl.neg(gd_pad))), [1, RECV_TILE]),
                        )
                    eh_sq_row = pl.div(
                        pl.full([1, RECV_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), eh_amax
                    )
                    h_scale_dq[flat_a0 : flat_a0 + RECV_TILE, 0:1] = pl.reshape(
                        pl.recip(eh_sq_row), [RECV_TILE, 1]
                    )
                    eh_sq_col = pl.reshape(eh_sq_row, [RECV_TILE, 1])
                    for k1 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                        eh_q_scaled = pl.row_expand_mul(h_fp32[flat_a0 : flat_a0 + RECV_TILE, k1 : k1 + QUANT_TILE], eh_sq_col)
                        eh_q_i32 = pl.cast(eh_q_scaled, target_type=pl.INT32, mode="rint")
                        eh_q_half = pl.cast(eh_q_i32, target_type=pl.FP16, mode="round")
                        h_i8[flat_a0 : flat_a0 + RECV_TILE, k1 : k1 + QUANT_TILE] = pl.cast(
                            eh_q_half, target_type=pl.INT8, mode="trunc"
                        )

        with pl.scope():
            y_i32 = pl.create_tensor(
                [N_SLOTS * RECV_MAX, D], dtype=pl.INT32, manual_dep=True
            )
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
                                w2_k = routed_w2[expert_e : expert_e + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
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
                ) as _w2_act_tid:
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
                            recv_weights[act_slot : act_slot + 1, 0 : RECV_TILE],
                            [RECV_TILE, 1],
                        )
                        row_scale_blk = pl.mul(
                            h_scale_dq[flat_at0 : flat_at0 + RECV_TILE, 0:1], w_col_blk
                        )
                        for dg in pl.pipeline(W2_ACT_INNER, stage=2):
                            act_d0 = act_d_base + dg * D_OUT_TILE_ACT
                            y_2d_i32 = y_i32[flat_at0 : flat_at0 + RECV_TILE, act_d0 : act_d0 + D_OUT_TILE_ACT]
                            w2_scale_chunk = routed_w2_scale[expert_a : expert_a + 1, act_d0 : act_d0 + D_OUT_TILE_ACT]
                            y_2d = pl.cast(y_2d_i32, target_type=pl.FP32, mode="none")
                            y_2d = pl.col_expand_mul(pl.row_expand_mul(y_2d, row_scale_blk), w2_scale_chunk)
                            recv_y_flat[
                                flat_at0 : flat_at0 + RECV_TILE, act_d0 : act_d0 + D_OUT_TILE_ACT
                            ] = pl.cast(y_2d, target_type=pl.BF16, mode="rint")

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
        recv_x, recv_scale_dq, recv_weights, recv_expert_count, slot_expert,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )
    return recv_y


def golden_expert_routed(tensors):
    """Torch reference for the routed expert over one TP intermediate slice.

    Slot ``i`` holds the ``recv_expert_count[i]`` rows routed to expert
    ``slot_expert[i]``; ``recv_y[i, cnt:, :]`` stays at zero."""
    from utils import int8_quant_per_row
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)

    recv_x_i8 = tensors["recv_x"]  # INT8, pre-quantized in dispatch
    recv_scale_dq = tensors["recv_scale_dq"].float()  # [N_SLOTS, RECV_MAX]
    recv_weights = tensors["recv_weights"].float()  # [N_SLOTS, RECV_MAX]
    recv_expert_count = tensors["recv_expert_count"]  # [N_SLOTS, IDX_PAD] int32
    slot_expert = tensors["slot_expert"]  # [N_SLOTS, IDX_PAD] int32
    w1 = dequant_w(tensors["routed_w1"], tensors["routed_w1_scale"].float())
    w3 = dequant_w(tensors["routed_w3"], tensors["routed_w3_scale"].float())
    w2 = dequant_w(tensors["routed_w2"], tensors["routed_w2_scale"].float())

    recv_y = torch.zeros(N_SLOTS, RECV_MAX, D)
    for slot in range(N_SLOTS):
        n_rows = int(recv_expert_count[slot, 0].item())
        if n_rows == 0:
            continue
        e = int(slot_expert[slot, 0].item())
        x_sub_i8 = recv_x_i8[slot, :n_rows, :]
        x_sub_sd = recv_scale_dq[slot, :n_rows].reshape(-1, 1)
        x_sub_q = x_sub_i8.float() * x_sub_sd
        w_per_row = recv_weights[slot, :n_rows].reshape(-1, 1)

        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        # A8 requant before w2 matmul; the amax spans this rank's slice only.
        h_i8, h_sd = int8_quant_per_row(h)
        h = h_i8.float() * (h_sd * w_per_row)
        recv_y[slot, :n_rows, :] = h @ w2[e].T

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def gen_routed_weight(shape, dequant_std):
    """Synthesize a routed-expert per-channel-symmetric INT8 weight + FP32 scale by
    simulating the real DeepSeek-V4-Flash MXFP4 routed-expert quant grid (e2m1, per-32-group
    E8M0 scale), then re-quantizing per-output-channel. A plain ``randn`` INT8 is wrong:
    routed collapses onto ~37 discrete levels with an ~11.6% zero spike (the FP4 grid) and a
    per-channel scale CV ~0.09 (the fine group scale flattens it). Per-output-channel INT8 is
    scale-invariant, so the level structure / zero spike emerge from the grid alone and
    ``dequant_std`` only sets the absolute scale magnitude. (shared experts use a different
    grid -- see expert_shared.gen_shared_weight.)

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel
    scale shape ([E, out, in] -> scale [E, out]).
    """
    import torch

    FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # nearest-grid bounds
    FP4_MAX, TINY = 6.0, 1e-20
    GROUP = 32
    CHUNK_ELEMS = 1 << 25    # elements per pass; unchunked this walks GiB-sized temporaries

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
        w = W[i0:i0 + step]
        wg = w.reshape(-1, out, inn // GROUP, GROUP)
        absw = wg.abs()
        grp_scale = torch.exp2(torch.ceil(torch.log2((absw.amax(-1, keepdim=True) / FP4_MAX).clamp_min(TINY))))
        idx = torch.bucketize(absw.div_(grp_scale), FP4_MID).clamp_max_(7)
        wq = (torch.sign(wg) * FP4_MAG[idx]).mul_(grp_scale).reshape(w.shape)
        amax = wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        chan_scale = amax / INT8_SCALE_MAX
        w_i8[i0:i0 + step] = torch.round(wq.div_(chan_scale)).clamp_(
            -INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        scale[i0:i0 + step] = chan_scale
    del W

    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8.reshape(*shape), scale.reshape(*lead, out)


def build_tensor_specs():
    from utils import int8_quant_per_row
    import torch
    from golden import TensorSpec

    # Across-layer-mean dequant std (typical layer) of the real DeepSeek-V4-Flash MXFP4
    # routed experts; gen_routed_weight simulates the FP4 grid (see its docstring).
    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

    # One slot per active expert: T * TOPK routed pairs spread over N_SLOTS slots,
    # each slot holding at most T rows. Slot experts are distinct global ids.
    total = T * TOPK
    counts = torch.bincount(torch.randint(0, N_SLOTS, (total,)), minlength=N_SLOTS)
    counts = counts.clamp(max=T).to(torch.int32)
    counts_2d = torch.zeros(N_SLOTS, IDX_PAD, dtype=torch.int32)
    counts_2d[:, 0] = counts
    slot_expert_1d = torch.randperm(N_EXPERTS)[:N_SLOTS].to(torch.int32)
    slot_expert_2d = torch.zeros(N_SLOTS, IDX_PAD, dtype=torch.int32)
    slot_expert_2d[:, 0] = slot_expert_1d

    # Build a consistent INT8 recv_x + per-row dequant scale (dispatch is
    # responsible for per-token quantization). Invalid tail rows go to INT8 0
    # with scale 0 so dequant produces 0.
    x_bf16 = torch.randn(N_SLOTS, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_SLOTS, 1, 1)
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
    recv_weights_pre = torch.rand(N_SLOTS, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre))

    def init_recv_weights():
        return recv_weights_pre

    # Only the slot experts are ever read, so synthesize N_SLOTS weight sets and
    # scatter them into the full expert bank; the rest stay zero.
    def scatter_bank(shape_tail, dequant_std):
        w_i8, w_s = gen_routed_weight((N_SLOTS, *shape_tail), dequant_std)
        bank_i8 = torch.zeros(N_EXPERTS, *shape_tail, dtype=torch.int8)
        bank_s = torch.zeros(N_EXPERTS, shape_tail[0], dtype=torch.float32)
        bank_i8[slot_expert_1d.long()] = w_i8
        bank_s[slot_expert_1d.long()] = w_s
        return bank_i8, bank_s

    w1_i8, w1_s = scatter_bank((MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
    w3_i8, w3_s = scatter_bank((MOE_INTER, D), ROUTED_DEQUANT_STD["w3"])
    w2_i8, w2_s = scatter_bank((D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])

    return [
        TensorSpec("recv_x", [N_SLOTS, RECV_MAX, D], torch.int8, init_value=init_recv_x),
        TensorSpec("recv_scale_dq", [N_SLOTS, RECV_MAX], torch.float32, init_value=init_recv_scale_dq),
        TensorSpec("recv_weights", [N_SLOTS, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_SLOTS, IDX_PAD], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("slot_expert", [N_SLOTS, IDX_PAD], torch.int32, init_value=init_slot_expert),
        TensorSpec("routed_w1", [N_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w1_i8),
        TensorSpec("routed_w1_scale", [N_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w1_s),
        TensorSpec("routed_w3", [N_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w3_i8),
        TensorSpec("routed_w3_scale", [N_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w3_s),
        TensorSpec("routed_w2", [N_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("routed_w2_scale", [N_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("recv_y", [N_SLOTS, RECV_MAX, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
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
        fn=expert_routed_test,
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
            # BF16 recv_y, ~1 ULP. Gen weights reproduce real(L21): 0.016% vs 0.015% of points > 1e-3.
            "recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
