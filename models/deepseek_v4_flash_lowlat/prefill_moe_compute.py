# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill MoE preparation, expert compute, and ordered local combine."""

import pypto.language as pl
from .config import INT8_SCALE_MAX, INT8_AMAX_EPS
from .prefill_moe_geometry import D, IDX_PAD, MOE_INTER, N_BANK, N_SLOTS_B, RECV_MAX, RECV_TILE
from .prefill_moe_geometry import SWIGLU_LIMIT, WORK_SLOTS_PAD
import pypto.language.distributed as pld
from .prefill_moe_geometry import exchange_gate_columns
from .prefill_moe_geometry import PREFILL_MOE_QUANT_GRID, PREFILL_MOE_OWNER_PREPARE, PREFILL_MOE_SHARE_GATE, GATE_WINDOW_ROWS, N_RANKS
from .config import FLASH as M, FP32_NEG_INF
from .prefill_moe_geometry import T, PREFILL_MOE_FUSED_NORM_QUANT
from .prefill_moe_geometry import OWNER_WINDOW_ROWS, PREP_OWNER_ROWS
from .prefill_moe_geometry import exchange_preparation
from .prefill_moe_geometry import parallel_route_metadata
from .prefill_moe_geometry import PREFILL_MOE_PARALLEL_ROUTE
from .hc_pre import hc_pre_with_input_ready as hc_pre
from .prefill_moe_geometry import (
    HC_DIM, HC_MULT, MIX_HC, N_EXPERTS, N_ROUTES,
    PREFILL_MOE_EXPERT_AFFINITY, EXPERT_IDS_PAD, SHARED_EID, SHARED_TILES, SH_SLOT, TOPK, VOCAB, PREFILL_MOE_BATCHED_GATHER, GATHER_TOKEN_TILE, SHARED_GATHER_T_TILE, PREFILL_MOE_COMBINE_ROWS, PREFILL_MOE_LOCAL_ROUTE_STATE,
)


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


GATE_PREP_ROWS = pl.dynamic("LOWLAT_PREFILL_GATE_PREPARATION_ROWS")

NORM_EPS = M.rms_norm_eps
ROUTE_SCALE = M.routed_scaling_factor
N_HASH_LAYERS = M.num_hash_layers
T_TILE = 8
GATE_T_TILE = 1
ROUTE_ROW_PAD = 16
GATE_M_TILE = 16
GATE_N_TILE = 16
T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
D_TILE = 256
ROW_PAD = 8
FFN_REDUCE_TILE = D // ROW_PAD
GATE_D_TILE = 2048
QUANT_TILE = 256
SCORE_PAD = 256
TOPK_PAD = 8
SORT_PAD = TOPK_PAD * 2

if N_EXPERTS % GATE_N_TILE or D % ROW_PAD or D % GATE_D_TILE or TOPK > TOPK_PAD:
    raise ValueError("Unsupported prefill router numerical tile geometry")


@pl.jit.inline(auto_scope=False)
def _router_scores(route_logits: pl.Tensor[[GATE_T_TILE, N_EXPERTS], pl.FP32]) -> pl.Tensor:
    gp_relu = pl.maximum(route_logits, 0.0)
    gp_abs = pl.abs(route_logits)
    gp_softplus_log = pl.add(gp_relu, pl.log(pl.add(pl.exp(pl.neg(gp_abs)), 1.0)))
    gp_neg_floor_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(route_logits), 10.0), 0.0), 1.0)
    gp_neg_floor = pl.mul(gp_neg_floor_mask, pl.exp(pl.minimum(route_logits, 0.0)))
    gp_softplus = pl.maximum(gp_softplus_log, gp_neg_floor)
    gp_score = pl.sqrt(gp_softplus)
    return gp_score


@pl.jit.inline(auto_scope=False)
def gate(
    x_mixed: pl.Tensor[[GATE_PREP_ROWS, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[GATE_PREP_ROWS], pl.INT64],
    x_norm_i8: pl.Tensor[[GATE_PREP_ROWS, D], pl.INT8],
    x_norm_scale: pl.Tensor[[GATE_PREP_ROWS, 1], pl.FP32],
    indices: pl.Tensor[[GATE_PREP_ROWS, ROUTE_ROW_PAD], pl.INT32],
    weights: pl.Tensor[[GATE_PREP_ROWS, ROUTE_ROW_PAD], pl.FP32],
    gate_window: pld.DistributedTensor[[GATE_WINDOW_ROWS, N_EXPERTS], pl.FP32],
    gate_signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
    gate_rank: pl.Scalar[pl.INT32],
    gate_epoch: pl.Scalar[pl.INT32],
    previous_gate: pl.Scalar[pl.TASK_ID],
):


    gate_rows = pl.tensor.dim(x_mixed, 0)
    gate_padded_rows = ((gate_rows + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
    owner_tasks = pl.array.create(2, pl.TASK_ID)
    owner_tasks[0] = previous_gate
    owner_tasks[1] = previous_gate
    xg_buf = pl.create_tensor([gate_padded_rows, D], dtype=pl.FP32)
    inv_rms_buf = pl.create_tensor([gate_padded_rows, 1], dtype=pl.FP32)


    xn_scale_buf = pl.create_tensor([gate_padded_rows, 1], dtype=pl.FP32)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > gate_rows:
        active_tokens = pl.cast(gate_rows, pl.INDEX)
    active_gate_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    active_gate_tokens = active_gate_tiles * GATE_M_TILE
    if active_gate_tokens > gate_rows:
        active_gate_tokens = pl.cast(gate_rows, pl.INDEX)


    norm_w_2d = pl.reshape(norm_w, [1, D])
    if active_gate_tokens > 0:
        for tok in pl.spmd(active_gate_tokens, name_hint="ffn_norm", allow_early_resolve=True):
            rms_x = pl.cast(pl.tile.load(x_mixed, [tok, 0], [1, D]), pl.FP32)
            rms_w = pl.cast(pl.tile.load(norm_w_2d, [0, 0], [1, D]), pl.FP32)
            xg = pl.mul(rms_x, rms_w)
            pl.tile.store(xg, [tok, 0], xg_buf, shapes=[1, D])

            sq_rows = pl.reshape(pl.mul(rms_x, rms_x), [ROW_PAD, FFN_REDUCE_TILE])
            sq_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
            sq_partial = pl.row_sum(sq_rows, sq_partial_tmp)
            sq_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            sq_reduce[0:1, :] = pl.reshape(sq_partial, [1, ROW_PAD])
            sq_reduce = pl.set_validshape(sq_reduce, 1, ROW_PAD)
            sq_sum_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            sq_sum = pl.row_sum(sq_reduce, sq_sum_tmp)
            sq_sum = pl.set_validshape(pl.reshape(sq_sum, [1, ROW_PAD]), 1, 1)
            inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)))
            pl.tile.store(inv_rms, [tok, 0], inv_rms_buf, shapes=[1, 1])

            xg_abs_rows = pl.reshape(pl.abs(xg), [ROW_PAD, FFN_REDUCE_TILE])
            amax_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
            amax_partial = pl.row_max(xg_abs_rows, amax_partial_tmp)
            amax_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            amax_reduce[0:1, :] = pl.reshape(amax_partial, [1, ROW_PAD])
            amax_reduce = pl.set_validshape(amax_reduce, 1, ROW_PAD)
            amax_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            xg_amax = pl.row_max(amax_reduce, amax_tmp)
            xg_amax = pl.set_validshape(pl.reshape(xg_amax, [1, ROW_PAD]), 1, 1)
            xg_amax = pl.maximum(xg_amax, INT8_AMAX_EPS)

            scale_max = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_SCALE_MAX)
            scale_max = pl.set_validshape(scale_max, 1, 1)
            xg_sq = pl.div(scale_max, xg_amax)
            xg_dequant_scale = pl.mul(xg_amax, 1.0 / INT8_SCALE_MAX)
            x_norm_dequant_scale = pl.mul(xg_dequant_scale, inv_rms)
            pl.tile.store(x_norm_dequant_scale, [tok, 0], x_norm_scale, shapes=[1, 1])
            if PREFILL_MOE_FUSED_NORM_QUANT == 0:
                pl.tile.store(xg_sq, [tok, 0], xn_scale_buf, shapes=[1, 1])
            else:
                fused_xn_sq_scalar = pl.read(xg_sq, [0, 0])
                fused_xn_q_scaled = pl.mul(xg, fused_xn_sq_scalar)
                fused_xn_q_i32 = pl.cast(fused_xn_q_scaled, pl.INT32, mode="rint")
                fused_xn_q_half = pl.cast(fused_xn_q_i32, pl.FP16, mode="round")
                fused_xn_q_i8 = pl.cast(fused_xn_q_half, pl.INT8, mode="trunc", saturation_mode="on")
                pl.tile.store(fused_xn_q_i8, [tok, 0], x_norm_i8, shapes=[1, D])

    if PREFILL_MOE_QUANT_GRID != 0:
        with pl.spmd((active_gate_tokens + T_TILE - 1) // T_TILE, name_hint="x_norm_quant_owner", allow_early_resolve=True) as owner_quantized:
            owner_t0 = pl.tile.get_block_idx() * T_TILE
            owner_quant_rows = pl.min(T_TILE, active_gate_tokens - owner_t0)
            owner_xn_sq_col = pl.slice(xn_scale_buf, [T_TILE, 1], [owner_t0, 0], valid_shape=[owner_quant_rows, 1])
            for owner_xq_b_k in pl.pipeline(0, D, QUANT_TILE, stage=2):
                owner_xn_input = pl.slice(
                    xg_buf, [T_TILE, QUANT_TILE], [owner_t0, owner_xq_b_k], valid_shape=[owner_quant_rows, QUANT_TILE]
                )
                owner_xn_q_scaled = pl.row_expand_mul(owner_xn_input, owner_xn_sq_col)
                owner_xn_q_i32 = pl.cast(owner_xn_q_scaled, pl.INT32, mode="rint")
                owner_xn_q_half = pl.cast(owner_xn_q_i32, pl.FP16, mode="round")
                owner_xn_q_i8 = pl.cast(owner_xn_q_half, pl.INT8, mode="trunc", saturation_mode="on")
                owner_xn_q_valid = pl.set_validshape(owner_xn_q_i8, owner_quant_rows, QUANT_TILE)
                x_norm_i8[owner_t0 : owner_t0 + T_TILE, owner_xq_b_k : owner_xq_b_k + QUANT_TILE] = owner_xn_q_valid

        owner_tasks[0] = owner_quantized
    else:
        if PREFILL_MOE_FUSED_NORM_QUANT == 0:
            seed_dummy = pl.system.task_dummy(deps=[])


            for t0 in pl.parallel(0, active_gate_tokens, T_TILE):
                with pl.at(
                    level=pl.Level.CORE_GROUP,
                    name_hint="x_norm_quant",
                    deps=[seed_dummy],
                    allow_early_resolve=True,
                ):
                    quant_rows = pl.min(T_TILE, active_gate_tokens - t0)
                    xn_sq_col = pl.slice(xn_scale_buf, [T_TILE, 1], [t0, 0], valid_shape=[quant_rows, 1])
                    for xq_b_k in pl.pipeline(0, D, QUANT_TILE, stage=2):
                        xn_input = pl.slice(
                            xg_buf, [T_TILE, QUANT_TILE], [t0, xq_b_k], valid_shape=[quant_rows, QUANT_TILE]
                        )
                        xn_q_scaled = pl.row_expand_mul(xn_input, xn_sq_col)
                        xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode="rint")
                        xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode="round")
                        xn_q_i8 = pl.cast(xn_q_half, pl.INT8, mode="trunc", saturation_mode="on")
                        xn_q_valid = pl.set_validshape(xn_q_i8, quant_rows, QUANT_TILE)
                        x_norm_i8[t0 : t0 + T_TILE, xq_b_k : xq_b_k + QUANT_TILE] = xn_q_valid

    gate_logits_buf = pl.create_tensor([gate_padded_rows, SCORE_PAD], dtype=pl.FP32)
    GATE_N_BLOCKS = N_EXPERTS // GATE_N_TILE
    if PREFILL_MOE_SHARE_GATE != 0:
        GATE_N_BLOCKS = GATE_N_BLOCKS // N_RANKS
    gate_tasks = pl.array.create(1, pl.TASK_ID)
    gate_tasks[0] = previous_gate
    if active_gate_tiles > 0:
        with pl.spmd(active_gate_tiles * GATE_N_BLOCKS, name_hint="gate_mm", allow_early_resolve=True) as gate_computed:
            gb_idx = pl.tile.get_block_idx()
            tg = gb_idx // GATE_N_BLOCKS
            nb = gb_idx % GATE_N_BLOCKS
            t1 = tg * GATE_M_TILE
            n0 = nb * GATE_N_TILE
            if PREFILL_MOE_SHARE_GATE != 0:
                n0 = (pl.cast(gate_rank, pl.INDEX) * GATE_N_BLOCKS + nb) * GATE_N_TILE
            gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(0, D // GATE_D_TILE, stage=2):
                gd_kd = kb * GATE_D_TILE
                gd_x = xg_buf[t1 : t1 + GATE_M_TILE, gd_kd : gd_kd + GATE_D_TILE]
                gd_w = gate_w[n0 : n0 + GATE_N_TILE, gd_kd : gd_kd + GATE_D_TILE]
                gate_logits_tile = pl.matmul_acc(
                    gate_logits_tile, gd_x, gd_w, b_trans=True, init_cond=(kb == 0)
                )
            gate_logits_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gate_logits_tile

        if PREFILL_MOE_SHARE_GATE != 0:
            gate_prior_scalar = gate_tasks[0]
            gate_copied = exchange_gate_columns(gate_logits_buf, gate_window, gate_signal, gate_rank, gate_epoch, active_gate_tokens, gate_computed, gate_prior_scalar)
            gate_tasks[0] = gate_copied

    gate_route_ready = gate_tasks[0]
    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE

    if active_route_tiles > 0:
        if layer_id < N_HASH_LAYERS:
            with pl.spmd(active_route_tiles, name_hint="route_hash", allow_early_resolve=True, deps=[gate_route_ready]) as owner_hashed:
                th_idx = pl.tile.get_block_idx()
                t1 = th_idx * GATE_T_TILE
                route_logits = pl.slice(gate_logits_buf, [GATE_T_TILE, N_EXPERTS], [t1, 0])
                route_inv_rms = pl.read(inv_rms_buf, [t1, 0])
                route_logits = pl.mul(route_logits, route_inv_rms)
                gp_score = _router_scores(route_logits)

                hs_token = pl.cast(pl.read(input_ids, [t1]), pl.INDEX)
                hs_idx_tile = pl.slice(
                    tid2eid, [GATE_T_TILE, TOPK_PAD], [hs_token, 0], valid_shape=[GATE_T_TILE, TOPK]
                )

                gather_all = pl.gather(gp_score, dim=-1, index=hs_idx_tile)
                hs_reduce = pl.create_tensor([ROW_PAD, TOPK_PAD], dtype=pl.FP32)
                hs_reduce[0:GATE_T_TILE, :] = gather_all
                hs_reduce = pl.set_validshape(hs_reduce, GATE_T_TILE, TOPK)
                hs_denom = pl.row_sum(hs_reduce)
                hs_weight_mat = pl.mul(pl.row_expand_div(hs_reduce, hs_denom), ROUTE_SCALE)
                hs_indices_valid = pl.set_validshape(hs_idx_tile, GATE_T_TILE, TOPK)
                indices = pl.assemble(indices, hs_indices_valid, [t1, 0])
                weights = pl.assemble(weights, hs_weight_mat, [t1, 0])
            if PREFILL_MOE_OWNER_PREPARE != 0:
                owner_tasks[1] = owner_hashed
        else:
            with pl.spmd(active_route_tiles, name_hint="route_sort", allow_early_resolve=True, deps=[gate_route_ready]) as owner_sorted:
                ts_idx = pl.tile.get_block_idx()
                t1 = ts_idx * GATE_T_TILE
                route_logits = pl.slice(gate_logits_buf, [GATE_T_TILE, N_EXPERTS], [t1, 0])
                route_inv_rms = pl.read(inv_rms_buf, [t1, 0])
                route_logits = pl.mul(route_logits, route_inv_rms)
                gp_score = _router_scores(route_logits)
                gp_bias_row = pl.reshape(gate_bias, [1, N_EXPERTS])
                gp_biased = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
                gp_biased[:, :] = pl.full([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
                gp_biased[:, 0:N_EXPERTS] = pl.add(gp_score, gp_bias_row)


                topk_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)


                for sr_tt in pl.range(GATE_T_TILE):
                    sr_row = gp_biased[sr_tt : sr_tt + 1, :]
                    sr_idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                    sr_sorted = pl.sort32(sr_row, sr_idx_init)
                    sr_sorted = pl.mrgsort(sr_sorted, block_len=64)
                    sr_sorted = pl.mrgsort(sr_sorted[:, 0:256], sr_sorted[:, 256:512])
                    sr_pairs = sr_sorted[:, 0:SORT_PAD]
                    sr_i = pl.gather(sr_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i

                gather_all = pl.gather(gp_score, dim=-1, index=topk_idx_tile)
                nm_reduce = pl.create_tensor([ROW_PAD, TOPK_PAD], dtype=pl.FP32)
                nm_reduce[0:GATE_T_TILE, :] = gather_all
                nm_reduce = pl.set_validshape(nm_reduce, GATE_T_TILE, TOPK)
                nm_denom = pl.row_sum(nm_reduce)
                nm_weight_mat = pl.mul(pl.row_expand_div(nm_reduce, nm_denom), ROUTE_SCALE)
                nm_indices_valid = pl.set_validshape(topk_idx_tile, GATE_T_TILE, TOPK)
                indices = pl.assemble(indices, nm_indices_valid, [t1, 0])
                weights = pl.assemble(weights, nm_weight_mat, [t1, 0])

            if PREFILL_MOE_OWNER_PREPARE != 0:
                owner_tasks[1] = owner_sorted

    if PREFILL_MOE_OWNER_PREPARE != 0:
        owner_quant_scalar = owner_tasks[0]
        owner_route_scalar = owner_tasks[1]
        owner_gate_done = pl.system.task_dummy(deps=[owner_quant_scalar, owner_route_scalar])
        gate_tasks[0] = owner_gate_done
    return weights, gate_tasks[0]


@pl.jit.inline
def route_group(
    indices: pl.Tensor[[T, ROUTE_ROW_PAD], pl.INT32],
    weights: pl.Tensor[[T, ROUTE_ROW_PAD], pl.FP32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],

    recv_x: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.INT8],
    recv_scale: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_w: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    work_slot: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    work_rows: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    preparation_ready: pl.Scalar[pl.TASK_ID],
):
    """Group routes into bounded sixteen-row expert tiles and shared tiles."""
    recv_x_flat = pl.reshape(recv_x, [N_SLOTS_B * RECV_MAX, D])
    expert_slot = pl.create_tensor([1, N_EXPERTS], dtype=pl.INT32, manual_dep=True)
    work_slot_2d = pl.reshape(work_slot, [1, WORK_SLOTS_PAD])
    work_rows_2d = pl.reshape(work_rows, [1, WORK_SLOTS_PAD])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_table_init", allow_early_resolve=True) as init_tid:
        expert_slot[:, :] = pl.full([1, N_EXPERTS], dtype=pl.INT32, value=-1)
        work_slot_2d[:, :] = pl.full([1, WORK_SLOTS_PAD], dtype=pl.INT32, value=0)
        work_rows_2d[:, :] = pl.full([1, WORK_SLOTS_PAD], dtype=pl.INT32, value=0)
        for slot_t0 in pl.range(0, N_SLOTS_B, 64):
            slot_rows = pl.min(64, N_SLOTS_B - slot_t0)
            zero_experts = pl.tile.full([64, IDX_PAD], dtype=pl.INT32, value=0)
            valid_experts = pl.set_validshape(zero_experts, slot_rows, IDX_PAD)
            pl.store(valid_experts, [slot_t0, 0], slot_expert)
            zero_scales = pl.tile.full([64, RECV_MAX], dtype=pl.FP32, value=0.0)
            valid_scales = pl.set_validshape(zero_scales, slot_rows, RECV_MAX)
            pl.store(valid_scales, [slot_t0, 0], recv_scale)
            zero_weights = pl.tile.full([64, RECV_MAX], dtype=pl.FP32, value=0.0)
            valid_weights = pl.set_validshape(zero_weights, slot_rows, RECV_MAX)
            pl.store(valid_weights, [slot_t0, 0], recv_w)
        for route_t0 in pl.range(0, N_ROUTES, 64):
            route_rows = pl.min(64, N_ROUTES - route_t0)
            zero_routes = pl.tile.full([64, IDX_PAD], dtype=pl.INT32, value=0)
            valid_routes = pl.set_validshape(zero_routes, route_rows, IDX_PAD)
            pl.store(valid_routes, [route_t0, 0], route_slot_row)

        slot_expert[SH_SLOT : SH_SLOT + SHARED_TILES, :] = pl.full(
            [SHARED_TILES, IDX_PAD], dtype=pl.INT32, value=SHARED_EID)

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)


    route_completed = pl.array.create(1, pl.TASK_ID)
    route_completed[0] = init_tid
    if PREFILL_MOE_PARALLEL_ROUTE != 0:
        parallel_done = parallel_route_metadata(indices, weights, x_norm_scale, recv_scale, recv_w, work_slot, work_rows, slot_expert, route_slot_row, active_tokens, init_tid, preparation_ready)
        route_completed[0] = parallel_done
    else:
        with pl.spmd(1, name_hint="route_group_scalar", allow_early_resolve=True, deps=[init_tid, preparation_ready]) as legacy_scalar_tid:
            scalar_core = pl.tile.get_block_idx()
            if PREFILL_MOE_LOCAL_ROUTE_STATE != 0:
                local_experts = pl.tile.full([1, N_EXPERTS], dtype=pl.INT32, value=-1)
                local_rows = pl.tile.full([1, WORK_SLOTS_PAD], dtype=pl.INT32, value=0)
                local_slots = pl.tile.full([1, WORK_SLOTS_PAD], dtype=pl.INT32, value=0)
                next_slot = pl.cast(0, pl.INDEX)
                for token in pl.range(scalar_core, active_tokens):
                    token_scale = pl.read(x_norm_scale, [token, 0])
                    for k in pl.range(TOPK):
                        expert_id = pl.read(indices, [token, k])
                        expert_col = pl.cast(expert_id, pl.INDEX)
                        slot = pl.read(local_experts, [0, expert_col])
                        if slot >= 0:
                            occupied = pl.read(local_rows, [0, pl.cast(slot, pl.INDEX)])
                            if occupied >= RECV_MAX:
                                slot = pl.const(-1, pl.INT32)
                        if slot < 0:
                            slot = pl.cast(next_slot, pl.INT32)
                            pl.write(local_experts, [0, expert_col], slot)
                            pl.write(slot_expert, [next_slot, 0], expert_id)
                            pl.write(local_slots, [0, next_slot], slot)
                            next_slot = next_slot + 1
                        slot_row = pl.cast(slot, pl.INDEX)
                        count = pl.read(local_rows, [0, slot_row])
                        count_col = pl.cast(count, pl.INDEX)
                        pl.write(local_rows, [0, slot_row], pl.cast(count + 1, pl.INT32))
                        pl.write(recv_scale, [slot_row, count_col], token_scale)
                        pl.write(recv_w, [slot_row, count_col], pl.read(weights, [token, k]))
                        route = token * TOPK + k
                        pl.write(route_slot_row, [route, 0], pl.cast(slot * RECV_MAX + count, pl.INT32))
                    shared_slot = SH_SLOT + token // RECV_MAX
                    shared_row = token % RECV_MAX
                    pl.write(recv_scale, [shared_slot, shared_row], token_scale)
                    pl.write(recv_w, [shared_slot, shared_row], pl.cast(1.0, pl.FP32))
                for shared_tile in pl.range(SHARED_TILES):
                    shared_count = pl.min(RECV_MAX, pl.max(active_tokens - shared_tile * RECV_MAX, 0))
                    pl.write(local_slots, [0, next_slot + shared_tile], pl.cast(SH_SLOT + shared_tile, pl.INT32))
                    pl.write(local_rows, [0, next_slot + shared_tile], pl.cast(shared_count, pl.INT32))
                pl.store(local_slots, [0, 0], work_slot_2d)
                pl.store(local_rows, [0, 0], work_rows_2d)
            else:
                next_slot = pl.cast(0, pl.INDEX)
                for token in pl.range(scalar_core, active_tokens):
                    token_scale = pl.read(x_norm_scale, [token, 0])
                    for k in pl.range(TOPK):
                        expert_id = pl.read(indices, [token, k])
                        expert_col = pl.cast(expert_id, pl.INDEX)
                        slot = pl.read(expert_slot, [0, expert_col])
                        if slot >= 0:
                            occupied = pl.read(work_rows, [pl.cast(slot, pl.INDEX)])
                            if occupied >= RECV_MAX:
                                slot = pl.const(-1, pl.INT32)
                        if slot < 0:
                            slot = pl.cast(next_slot, pl.INT32)
                            pl.write(expert_slot, [0, expert_col], slot)
                            pl.write(slot_expert, [next_slot, 0], expert_id)
                            pl.write(work_slot, [next_slot], slot)
                            next_slot = next_slot + 1
                        slot_row = pl.cast(slot, pl.INDEX)
                        count = pl.read(work_rows, [slot_row])
                        count_col = pl.cast(count, pl.INDEX)
                        pl.write(work_rows, [slot_row], pl.cast(count + 1, pl.INT32))
                        pl.write(recv_scale, [slot_row, count_col], token_scale)
                        pl.write(recv_w, [slot_row, count_col], pl.read(weights, [token, k]))
                        route = token * TOPK + k
                        pl.write(route_slot_row, [route, 0], pl.cast(slot * RECV_MAX + count, pl.INT32))
    
                    shared_slot = SH_SLOT + token // RECV_MAX
                    shared_row = token % RECV_MAX
                    pl.write(recv_scale, [shared_slot, shared_row], token_scale)
                    pl.write(recv_w, [shared_slot, shared_row], pl.cast(1.0, pl.FP32))
                for shared_tile in pl.range(SHARED_TILES):
                    shared_count = pl.min(RECV_MAX, pl.max(active_tokens - shared_tile * RECV_MAX, 0))
                    pl.write(work_slot, [next_slot + shared_tile], pl.cast(SH_SLOT + shared_tile, pl.INT32))
                    pl.write(work_rows, [next_slot + shared_tile], pl.cast(shared_count, pl.INT32))


        route_completed[0] = legacy_scalar_tid
    scalar_tid = route_completed[0]

    if PREFILL_MOE_BATCHED_GATHER != 0:
        routed_blocks = (T + GATHER_TOKEN_TILE - 1) // GATHER_TOKEN_TILE
        shared_blocks = (T + SHARED_GATHER_T_TILE - 1) // SHARED_GATHER_T_TILE
        with pl.spmd(routed_blocks + shared_blocks, name_hint="route_gather_batched", allow_early_resolve=True, deps=[scalar_tid]):
            gather_block = pl.tile.get_block_idx()
            if gather_block < routed_blocks:
                token_begin = gather_block * GATHER_TOKEN_TILE
                token_end = pl.min(token_begin + GATHER_TOKEN_TILE, active_tokens)
                for gather_token in pl.range(token_begin, token_end):
                    source_row = x_norm_i8[gather_token:gather_token + 1, :]
                    for route_index in pl.range(TOPK):
                        route_id = gather_token * TOPK + route_index
                        destination = pl.cast(pl.read(route_slot_row, [route_id, 0]), pl.INDEX)
                        recv_x_flat[destination:destination + 1, :] = source_row
            else:
                shared_begin = (gather_block - routed_blocks) * SHARED_GATHER_T_TILE
                shared_rows = pl.min(SHARED_GATHER_T_TILE, T - shared_begin)
                shared_value = pl.slice(x_norm_i8, [SHARED_GATHER_T_TILE, D], [shared_begin, 0], valid_shape=[shared_rows, D])
                shared_destination = SH_SLOT * RECV_MAX + shared_begin
                recv_x_flat[shared_destination:shared_destination + SHARED_GATHER_T_TILE, :] = shared_value
    else:
        with pl.spmd(T + 1, name_hint="route_gather", allow_early_resolve=True, deps=[scalar_tid]) as _gather_tid:
            blk = pl.tile.get_block_idx()
            if blk < T:
                if blk < active_tokens:
                    for k in pl.range(TOPK):
                        dst_row = pl.cast(pl.read(route_slot_row, [blk * TOPK + k, 0]), pl.INDEX)
                        recv_x_flat[dst_row : dst_row + 1, :] = x_norm_i8[blk : blk + 1, :]
            else:
                sh_base = pl.cast(SH_SLOT * RECV_MAX, pl.INDEX)
                for shared_t0 in pl.range(0, T, 8):
                    recv_x_flat[sh_base + shared_t0 : sh_base + shared_t0 + 8, :] = x_norm_i8[shared_t0 : shared_t0 + 8, :]


@pl.jit.inline
def combine_local(
    recv_y: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.BF16],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """ffn_partial[t] = recv_y[SH_SLOT, t] + Sigma_k recv_y[route_slot_row[t * TOPK + k]].

    The shared expert's row is just the first summand: it came out of the same
    kernel, in the same layout, so it needs no separate operand."""
    recv_y_flat = pl.reshape(recv_y, [N_SLOTS_B * RECV_MAX, D])

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    combine_blocks = (T + PREFILL_MOE_COMBINE_ROWS - 1) // PREFILL_MOE_COMBINE_ROWS
    with pl.spmd(combine_blocks, name_hint="shared_routed", allow_early_resolve=True):
        combine_block = pl.tile.get_block_idx()
        token_begin = combine_block * PREFILL_MOE_COMBINE_ROWS
        token_end = pl.min(token_begin + PREFILL_MOE_COMBINE_ROWS, T)
        for t in pl.range(token_begin, token_end):
            if t < active_tokens:
                sh_row = pl.cast(SH_SLOT * RECV_MAX + t, pl.INDEX)
                acc = pl.cast(recv_y_flat[sh_row : sh_row + 1, :], target_type=pl.FP32)
                for k in pl.range(TOPK):
                    src_row = pl.cast(pl.read(route_slot_row, [t * TOPK + k, 0]), pl.INDEX)
                    acc = pl.add(acc, pl.cast(recv_y_flat[src_row : src_row + 1, :], target_type=pl.FP32))
                ffn_partial[t : t + 1, :] = acc
            else:
                ffn_partial[t : t + 1, :] = pl.full([1, D], dtype=pl.FP32, value=0.0)

@pl.jit.inline(auto_scope=False)
def prepare_moe(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    gate_window: pld.DistributedTensor[[GATE_WINDOW_ROWS, N_EXPERTS], pl.FP32],
    gate_signal: pld.DistributedTensor[[N_RANKS, 16], pl.INT32],
    gate_rank: pl.Scalar[pl.INT32],
    gate_epoch: pl.Scalar[pl.INT32],
    previous_gate: pl.Scalar[pl.TASK_ID],
    owner_norm_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, D], pl.INT8],
    owner_scale_window: pld.DistributedTensor[[1, OWNER_WINDOW_ROWS], pl.FP32],
    owner_post_window: pld.DistributedTensor[[1, OWNER_WINDOW_ROWS * HC_MULT], pl.FP32],
    owner_comb_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, HC_MULT * HC_MULT], pl.FP32],
    owner_indices_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, 16], pl.INT32],
    owner_weights_window: pld.DistributedTensor[[OWNER_WINDOW_ROWS, 16], pl.FP32],
    residual_ready: pl.Scalar[pl.TASK_ID] = None,
):
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32, manual_dep=True)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, ROUTE_ROW_PAD], dtype=pl.INT32)
    weights = pl.create_tensor([T, ROUTE_ROW_PAD], dtype=pl.FP32)
    prepared = pl.array.create(1, pl.TASK_ID)
    prepared[0] = previous_gate
    if PREFILL_MOE_OWNER_PREPARE != 0:
        owner_offset = pl.cast(gate_rank, pl.INDEX) * PREP_OWNER_ROWS
        owner_hc = pl.slice(x_hc, [PREP_OWNER_ROWS, HC_MULT, D], [owner_offset, 0, 0])
        owner_ids = pl.slice(input_ids, [PREP_OWNER_ROWS], [owner_offset])
        owner_mixed = pl.create_tensor([PREP_OWNER_ROWS, D], dtype=pl.BF16)
        owner_post = pl.create_tensor([PREP_OWNER_ROWS, HC_MULT], dtype=pl.FP32)
        owner_comb = pl.create_tensor([PREP_OWNER_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
        owner_norm = pl.create_tensor([PREP_OWNER_ROWS, D], dtype=pl.INT8)
        owner_scale = pl.create_tensor([PREP_OWNER_ROWS, 1], dtype=pl.FP32)
        owner_indices = pl.create_tensor([PREP_OWNER_ROWS, ROUTE_ROW_PAD], dtype=pl.INT32)
        owner_weights = pl.create_tensor([PREP_OWNER_ROWS, ROUTE_ROW_PAD], dtype=pl.FP32)
        hc_pre(owner_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, owner_mixed, owner_post, owner_comb, residual_ready)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_owner_route_padding_init"):
            owner_indices[:, :] = pl.full([PREP_OWNER_ROWS, ROUTE_ROW_PAD], dtype=pl.INT32, value=0)
            owner_weights[:, :] = pl.full([PREP_OWNER_ROWS, ROUTE_ROW_PAD], dtype=pl.FP32, value=0.0)
        owner_gate_weights, owner_gate_done = gate(owner_mixed, norm_w, gate_w, gate_bias, layer_id, pl.const(PREP_OWNER_ROWS, pl.INT32), tid2eid, owner_ids, owner_norm, owner_scale, owner_indices, owner_weights, gate_window, gate_signal, gate_rank, gate_epoch, previous_gate)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_owner_hc_completion") as owner_hc_done:
            pl.write(owner_post, [0, 0], pl.read(owner_post, [0, 0]))
            pl.write(owner_comb, [0, 0], pl.read(owner_comb, [0, 0]))
        owner_fields_done = pl.system.task_dummy(deps=[owner_gate_done, owner_hc_done])
        owner_collected = exchange_preparation(owner_norm, owner_scale, owner_post, owner_comb, owner_indices, owner_weights, x_norm_i8, x_norm_scale, post_ffn, comb_ffn, indices, weights, owner_norm_window, owner_scale_window, owner_post_window, owner_comb_window, owner_indices_window, owner_weights_window, gate_signal, gate_rank, gate_epoch, owner_fields_done, previous_gate)
        prepared[0] = owner_collected
    else:
        x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
        hc_pre(
            x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            x_mixed, post_ffn, comb_ffn, residual_ready,
        )

        gate_weights, legacy_gate_finished = gate(
            x_mixed, norm_w, gate_w, gate_bias,
            layer_id, num_tokens, tid2eid, input_ids,
            x_norm_i8, x_norm_scale, indices, weights,
            gate_window, gate_signal, gate_rank, gate_epoch, previous_gate,
        )

        prepared[0] = legacy_gate_finished
    gate_finished = prepared[0]

    recv_x = pl.create_tensor([N_SLOTS_B, RECV_MAX, D], dtype=pl.INT8)
    recv_scale = pl.create_tensor([N_SLOTS_B, RECV_MAX], dtype=pl.FP32)
    recv_w = pl.create_tensor([N_SLOTS_B, RECV_MAX], dtype=pl.FP32)
    work_slot = pl.create_tensor([WORK_SLOTS_PAD], dtype=pl.INT32)
    work_rows = pl.create_tensor([WORK_SLOTS_PAD], dtype=pl.INT32)
    slot_expert = pl.create_tensor([N_SLOTS_B, IDX_PAD], dtype=pl.INT32)
    route_slot_row = pl.create_tensor([N_ROUTES, IDX_PAD], dtype=pl.INT32)
    route_group(
        indices, weights, x_norm_i8, x_norm_scale,
        recv_x, recv_scale, recv_w, work_slot, work_rows, slot_expert, route_slot_row,
        num_tokens, gate_finished,
    )
    return post_ffn, comb_ffn, recv_x, recv_scale, recv_w, work_slot, work_rows, slot_expert, route_slot_row, gate_finished


@pl.jit.inline

def _prefill_expert_affinity_plan(
    work_slot: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    work_rows: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
):
    """Group independent physical tiles by expert without changing their rows."""
    heads = pl.create_tensor([1, EXPERT_IDS_PAD], dtype=pl.INT32)
    tails = pl.create_tensor([1, EXPERT_IDS_PAD], dtype=pl.INT32)
    counts = pl.create_tensor([1, EXPERT_IDS_PAD], dtype=pl.INT32)
    links = pl.create_tensor([1, WORK_SLOTS_PAD], dtype=pl.INT32)
    slots = pl.create_tensor([1, WORK_SLOTS_PAD], dtype=pl.INT32)
    rows = pl.create_tensor([1, WORK_SLOTS_PAD], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_expert_affinity_plan"):
        for expert_init in pl.range(EXPERT_IDS_PAD):
            pl.write(heads, [0, expert_init], pl.const(-1, pl.INT32))
            pl.write(tails, [0, expert_init], pl.const(-1, pl.INT32))
            pl.write(counts, [0, expert_init], pl.const(0, pl.INT32))
        for work_init in pl.range(WORK_SLOTS_PAD):
            pl.write(links, [0, work_init], pl.const(-1, pl.INT32))
            pl.write(slots, [0, work_init], pl.const(0, pl.INT32))
            pl.write(rows, [0, work_init], pl.const(0, pl.INT32))
        for work in pl.range(WORK_SLOTS_PAD):
            active_rows = pl.read(work_rows, [work])
            if active_rows > 0:
                slot = pl.cast(pl.read(work_slot, [work]), pl.INDEX)
                expert = pl.cast(pl.read(slot_expert, [slot, 0]), pl.INDEX)
                tail = pl.read(tails, [0, expert])
                if tail < 0:
                    pl.write(heads, [0, expert], pl.cast(work, pl.INT32))
                else:
                    pl.write(links, [0, pl.cast(tail, pl.INDEX)], pl.cast(work, pl.INT32))
                pl.write(tails, [0, expert], pl.cast(work, pl.INT32))
                pl.write(counts, [0, expert], pl.cast(pl.read(counts, [0, expert]) + 1, pl.INT32))
        output_work = pl.cast(0, pl.INDEX)
        for expert in pl.range(N_BANK):
            cursor = pl.cast(pl.read(heads, [0, expert]), pl.INDEX)
            tiles = pl.cast(pl.read(counts, [0, expert]), pl.INDEX)
            for step in pl.range(tiles):
                pl.write(slots, [0, output_work], pl.read(work_slot, [cursor]))
                pl.write(rows, [0, output_work], pl.read(work_rows, [cursor]))
                cursor = pl.cast(pl.read(links, [0, cursor]), pl.INDEX)
                output_work = output_work + 1
    return pl.reshape(slots, [WORK_SLOTS_PAD]), pl.reshape(rows, [WORK_SLOTS_PAD])


@pl.jit.inline(auto_scope=False)
def moe_local(
    recv_x: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.INT8],
    recv_scale: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_w: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    work_slot: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    work_rows: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    routed_w1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_BANK, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    recv_y = pl.create_tensor([N_SLOTS_B, RECV_MAX, D], dtype=pl.BF16)
    planned_slot = work_slot
    planned_rows = work_rows
    if PREFILL_MOE_EXPERT_AFFINITY != 0:
        planned_slot, planned_rows = _prefill_expert_affinity_plan(work_slot, work_rows, slot_expert)
    expert_routed_persistent_planned(
        recv_x, recv_scale, recv_w, planned_slot, planned_rows, slot_expert,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )

    ffn_partial = pl.create_tensor([T, D], dtype=pl.FP32)
    combine_local(recv_y, route_slot_row, ffn_partial, num_tokens)
    return ffn_partial
