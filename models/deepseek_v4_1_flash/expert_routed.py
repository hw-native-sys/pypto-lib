# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 routed expert compute with distributed scatter."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import FLASH as M

from config import EP_SIZE as EP_WORLD_SIZE, MOE_RECV_MAX as RECV_MAX
from config import ROUTE_T_DYN, N_LOCAL_EXPERTS


DECODE_BATCH = 8
DECODE_SEQ = 1


# model config
B = DECODE_BATCH
S = DECODE_SEQ
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
MX_GROUP = 32
K_SCALE = D // MX_GROUP
H_SCALE = MOE_INTER // MX_GROUP
EXPECTED_EXPERT_ROWS = (RECV_MAX * M.num_experts_per_tok + M.n_routed_experts - 1) // M.n_routed_experts

# Routed-expert across-layer mean dequantization standard deviations.
ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

# tiling
# Row blocks divide the receive stride; vector subtiles contain 16 rows.
RECV_TILE = min(
    (tile for tile in (16, 32, 48, 64, 80) if RECV_MAX % tile == 0),
    key=lambda tile: ((EXPECTED_EXPERT_ROWS + tile - 1) // tile, tile),
)
ACT_T_TILE = 16
MX_K_TILE = 256
MX_MM_INTER_TILE = 256
# Three N fragments per FP4 decoding task.
MX_MM_TASK_TILE = 768
GATE_UP_TASK_TILE = 256
MX_W2_D_OUT_TILE = 256
MX_W2_TASK_TILE = 1024
MX_PACKED_LANE_ROWS = MX_K_TILE // 2
MX_PACKED_LANE_COLS = MX_MM_INTER_TILE // 2
MX_W1_PACKED_ROWS = (2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE)) * MX_PACKED_LANE_ROWS
MX_W3_PACKED_ROWS = (2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE)) * MX_PACKED_LANE_ROWS
MX_W2_PACKED_ROWS = (2 * (MOE_INTER // MX_K_TILE) * (D // MX_W2_D_OUT_TILE)) * MX_PACKED_LANE_ROWS
ROUTE_D_OUT_TILE = 1024
ACT_INTER_TILE = 256
QUANT_TILE = 768

SWIGLU_SCALE_COL_TILE = QUANT_TILE // 32
SWIGLU_SCALE_TMP_TILE = ((64 + (ACT_T_TILE // 16) * (QUANT_TILE // 32) + 31) // 32) * 32


@pl.jit.inline(auto_scope=False)
def expert_routed_scatter(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    recv_route: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_meta: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
    input_ready: pl.Scalar[pl.TASK_ID],
):
    hidden_completion_tids = pl.array.create(N_LOCAL_EXPERTS, pl.TASK_ID)
    expert_completion_tids = pl.array.create(N_LOCAL_EXPERTS, pl.TASK_ID)
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    recv_mx_scale_view = pl.tensor.view(recv_mx_scale, [N_LOCAL_EXPERTS * RECV_MAX, K_SCALE], layout=pl.MX_A_ZZ)
    with pl.scope():
        h_mx = pl.create_tensor([N_LOCAL_EXPERTS * RECV_MAX, MOE_INTER], dtype=pl.FP8E4M3FN)
        h_scale_backing = pl.create_tensor([1, N_LOCAL_EXPERTS * RECV_MAX * H_SCALE], dtype=pl.FP8E8M0)
        for local_i in pl.parallel(N_LOCAL_EXPERTS):
            tile_completion_tids = pl.array.create((RECV_MAX // RECV_TILE), pl.TASK_ID)
            for tile in pl.range((RECV_MAX // RECV_TILE)):
                tile_completion_tids[tile] = pl.system.task_dummy(deps=[])
            flat_base = local_i * RECV_MAX
            routed_w1_packed_e = routed_w1_packed[local_i, :, :]
            routed_w3_packed_e = routed_w3_packed[local_i, :, :]

            n_rows = pl.read(recv_expert_count, [local_i, 0])
            if n_rows > 0:
                n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
                with pl.scope():

                    w1_mxfp8 = pl.create_tensor([D, MOE_INTER], dtype=pl.FP8E4M3FN)
                    with pl.spmd(2 * (MOE_INTER // MX_MM_TASK_TILE), name_hint='exp_gate_mxfp4_aiv') as gate_mxfp4_aiv_tid:
                        block_idx = pl.tile.get_block_idx()
                        nb_idx = block_idx // 2
                        w1_lane = block_idx % 2
                        n_base = nb_idx * MX_MM_TASK_TILE
                        w1_lut_tile = pl.load(mxfp4_pair_lut, [w1_lane, 0], [1, 256], target_memory=pl.Mem.Vec)
                        for ng in pl.range(MX_MM_TASK_TILE // MX_MM_INTER_TILE):
                            n0 = n_base + ng * MX_MM_INTER_TILE
                            for k0 in pl.pipeline(0, D, MX_K_TILE, stage=2):
                                w1_tile_id = (
                                    n0 // MX_MM_INTER_TILE * (D // MX_K_TILE) + k0 // MX_K_TILE
                                )
                                w1_packed_row = (
                                    w1_tile_id * 2 * MX_PACKED_LANE_ROWS + w1_lane * MX_PACKED_LANE_ROWS
                                )
                                w1_packed_tile = pl.load(
                                    routed_w1_packed_e, [w1_packed_row, 0], [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS],
                                    target_memory=pl.Mem.Vec,
                                )
                                w1_packed_u16 = pl.cast(w1_packed_tile, target_type=pl.UINT16)
                                w1_packed_indices = pl.reinterpret_view(w1_packed_u16, pl.INT16)
                                w1_gather_tmp = pl.create_tile(
                                    [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS], dtype=pl.INT16, target_memory=pl.Mem.Vec,
                                )
                                w1_pair_codes = pl.tile.gather(w1_lut_tile, w1_packed_indices, w1_gather_tmp)
                                w1_bytes = pl.reinterpret_view(w1_pair_codes, pl.INT8, shape=[MX_PACKED_LANE_ROWS, MX_MM_INTER_TILE])
                                w1_mxfp8_reinterpret_view = pl.reinterpret_view(w1_bytes, pl.FP8E4M3FN)
                                w1_mxfp8 = pl.store(w1_mxfp8_reinterpret_view, [k0 + w1_lane * MX_PACKED_LANE_ROWS, n0], w1_mxfp8)

                    w3_mxfp8 = pl.create_tensor([D, MOE_INTER], dtype=pl.FP8E4M3FN)
                    with pl.spmd(2 * (MOE_INTER // MX_MM_TASK_TILE), name_hint='exp_up_mxfp4_aiv') as up_mxfp4_aiv_tid:
                        block_idx = pl.tile.get_block_idx()
                        nb_idx = block_idx // 2
                        w3_lane = block_idx % 2
                        n_base = nb_idx * MX_MM_TASK_TILE
                        w3_lut_tile = pl.load(mxfp4_pair_lut, [w3_lane, 0], [1, 256], target_memory=pl.Mem.Vec)
                        for ng in pl.range(MX_MM_TASK_TILE // MX_MM_INTER_TILE):
                            n0 = n_base + ng * MX_MM_INTER_TILE
                            for k0 in pl.pipeline(0, D, MX_K_TILE, stage=2):
                                w3_tile_id = (
                                    n0 // MX_MM_INTER_TILE * (D // MX_K_TILE) + k0 // MX_K_TILE
                                )
                                w3_packed_row = (
                                    w3_tile_id * 2 * MX_PACKED_LANE_ROWS + w3_lane * MX_PACKED_LANE_ROWS
                                )
                                w3_packed_tile = pl.load(
                                    routed_w3_packed_e, [w3_packed_row, 0], [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS],
                                    target_memory=pl.Mem.Vec,
                                )
                                w3_packed_u16 = pl.cast(w3_packed_tile, target_type=pl.UINT16)
                                w3_packed_indices = pl.reinterpret_view(w3_packed_u16, pl.INT16)
                                w3_gather_tmp = pl.create_tile(
                                    [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS], dtype=pl.INT16, target_memory=pl.Mem.Vec,
                                )
                                w3_pair_codes = pl.tile.gather(w3_lut_tile, w3_packed_indices, w3_gather_tmp)
                                w3_bytes = pl.reinterpret_view(w3_pair_codes, pl.INT8, shape=[MX_PACKED_LANE_ROWS, MX_MM_INTER_TILE])
                                w3_mxfp8_reinterpret_view = pl.reinterpret_view(w3_bytes, pl.FP8E4M3FN)
                                w3_mxfp8 = pl.store(w3_mxfp8_reinterpret_view, [k0 + w3_lane * MX_PACKED_LANE_ROWS, n0], w3_mxfp8)

                    for t in pl.parallel(n_tiles):
                        t0 = t * RECV_TILE
                        flat_t0 = flat_base + t0
                        valid_rows = pl.min(RECV_TILE, n_rows - t0)

                        with pl.scope():
                            gate_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
                            up_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)

                            with pl.spmd(MOE_INTER // GATE_UP_TASK_TILE, name_hint='exp_gate_up_mxfp4_aic', deps=[gate_mxfp4_aiv_tid, up_mxfp4_aiv_tid, input_ready]) as gate_up_mxfp4_aic_tid:
                                nb_idx = pl.tile.get_block_idx()
                                n_base = nb_idx * GATE_UP_TASK_TILE
                                for ng in pl.range(GATE_UP_TASK_TILE // MX_MM_INTER_TILE):
                                    n0 = n_base + ng * MX_MM_INTER_TILE
                                    scale_row_base = local_i * K_SCALE
                                    gate_acc = pl.create_tile([RECV_TILE, MX_MM_INTER_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                                    up_acc = pl.create_tile([RECV_TILE, MX_MM_INTER_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                                    for k0 in pl.pipeline(0, D, MX_K_TILE, stage=2):
                                        ks = k0 // MX_GROUP
                                        xs_k = pl.load(recv_x_flat, [flat_t0, k0], [RECV_TILE, MX_K_TILE])
                                        xs_scale_k = pl.load(
                                            recv_mx_scale_view, [flat_t0, ks], [RECV_TILE, MX_K_TILE // MX_GROUP],
                                            target_memory=pl.Mem.Mat,
                                        )
                                        w1_tile = pl.load(w1_mxfp8, [k0, n0], [MX_K_TILE, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)
                                        w1_scale_k = pl.load(
                                            routed_w1_scale, [scale_row_base + ks, n0], [MX_K_TILE // MX_GROUP, MX_MM_INTER_TILE],
                                            target_memory=pl.Mem.Mat,
                                        )
                                        w3_tile = pl.load(w3_mxfp8, [k0, n0], [MX_K_TILE, MX_MM_INTER_TILE], target_memory=pl.Mem.Mat)
                                        w3_scale_k = pl.load(
                                            routed_w3_scale, [scale_row_base + ks, n0], [MX_K_TILE // MX_GROUP, MX_MM_INTER_TILE],
                                            target_memory=pl.Mem.Mat,
                                        )
                                        if k0 == 0:
                                            gate_acc = pl.matmul_mx(xs_k, xs_scale_k, w1_tile, w1_scale_k)
                                            up_acc = pl.matmul_mx(xs_k, xs_scale_k, w3_tile, w3_scale_k)
                                        else:
                                            gate_acc = pl.matmul_mx_acc(gate_acc, xs_k, xs_scale_k, w1_tile, w1_scale_k)
                                            up_acc = pl.matmul_mx_acc(up_acc, xs_k, xs_scale_k, w3_tile, w3_scale_k)
                                    gate_tile_fp32 = pl.store(gate_acc, [0, n0], gate_tile_fp32)
                                    up_tile_fp32 = pl.store(up_acc, [0, n0], up_tile_fp32)

                            h_tile_mx = h_mx[flat_t0 : flat_t0 + RECV_TILE]
                            h_tile_scale_backing = h_scale_backing[:, flat_t0 * H_SCALE:(flat_t0 + RECV_TILE) * H_SCALE]
                            with pl.spmd(MOE_INTER // QUANT_TILE, name_hint='exp_gate_up_act_quant', deps=[gate_up_mxfp4_aic_tid]) as gate_up_act_quant_tid:
                                ab_idx = pl.tile.get_block_idx()
                                a_base = ab_idx * QUANT_TILE
                                for act_t0 in pl.range(0, valid_rows, ACT_T_TILE):
                                    act_rows_max = pl.max(valid_rows - act_t0, 0)
                                    act_rows = pl.min(ACT_T_TILE, act_rows_max)
                                    h_fp32 = pl.tile.full([ACT_T_TILE, QUANT_TILE], dtype=pl.FP32, value=0.0)
                                    route_weight = pl.load(recv_weights, [local_i, t0 + act_t0], [1, ACT_T_TILE])
                                    route_weight = pl.reshape(route_weight, [ACT_T_TILE, 1])
                                    for ag in pl.pipeline((QUANT_TILE // ACT_INTER_TILE), stage=2):
                                        a0 = a_base + ag * ACT_INTER_TILE
                                        h_a0 = ag * ACT_INTER_TILE
                                        gate_2d = pl.load(gate_tile_fp32, [act_t0, a0], [ACT_T_TILE, ACT_INTER_TILE])
                                        up_2d = pl.load(up_tile_fp32, [act_t0, a0], [ACT_T_TILE, ACT_INTER_TILE])
                                        # BF16 W1/W3 rounding before SwiGLU.
                                        gate_2d_cast = pl.cast(gate_2d, pl.BF16, mode='rint')
                                        gate_2d = pl.cast(gate_2d_cast, pl.FP32)
                                        up_2d_cast = pl.cast(up_2d, pl.BF16, mode='rint')
                                        up_2d = pl.cast(up_2d_cast, pl.FP32)
                                        if SWIGLU_LIMIT > 0.0:
                                            gate_2d = pl.minimum(gate_2d, SWIGLU_LIMIT)
                                            up_2d_minimum = pl.minimum(up_2d, SWIGLU_LIMIT)
                                            up_2d = pl.maximum(up_2d_minimum, -SWIGLU_LIMIT)
                                        sigmoid_neg = pl.neg(gate_2d)
                                        sigmoid_exp = pl.exp(sigmoid_neg)
                                        sigmoid_add = pl.add(sigmoid_exp, 1.0)
                                        sigmoid = pl.recip(sigmoid_add)
                                        silu = pl.mul(gate_2d, sigmoid)
                                        gated_mul = pl.mul(silu, up_2d)
                                        gated = pl.row_expand_mul(gated_mul, route_weight)
                                        gated_cast = pl.cast(gated, pl.BF16, mode='rint')
                                        gated = pl.cast(gated_cast, pl.FP32)
                                        h_fp32[0:ACT_T_TILE, h_a0:h_a0 + ACT_INTER_TILE] = gated
                                    h_fp32_valid = pl.set_validshape(h_fp32, act_rows, QUANT_TILE)
                                    h_fp32_padded = pl.fillpad(h_fp32_valid, pad_value=pl.PadValue.zero)
                                    # Upward-rounded group scales.
                                    sq_input = pl.set_validshape(h_fp32_padded, ACT_T_TILE, QUANT_TILE)
                                    sq_values = pl.reshape(sq_input, [ACT_T_TILE * (QUANT_TILE // 32), 32])
                                    sq_reduce_tmp = pl.create_tile([ACT_T_TILE * (QUANT_TILE // 32), 32], dtype=pl.FP32)
                                    sq_maximum_abs = pl.abs(sq_values)
                                    sq_maximum_row_max = pl.row_max(sq_maximum_abs, tmp_tile=sq_reduce_tmp)
                                    sq_maximum = pl.maximum(sq_maximum_row_max, 0.0001)
                                    sq_bits_mul = pl.mul(sq_maximum, 1.0 / 448.0)
                                    sq_bits = pl.reinterpret_view(sq_bits_mul, pl.INT32)
                                    sq_exponent_add = pl.add(sq_bits, 8388607)
                                    sq_exponent = pl.shrs(sq_exponent_add, 23)
                                    sq_scale_shls = pl.shls(sq_exponent, 23)
                                    sq_scale = pl.reinterpret_view(sq_scale_shls, pl.FP32)
                                    sq_normalized = pl.row_expand_div(sq_values, sq_scale)
                                    sq_clipped_maximum = pl.maximum(sq_normalized, -448.0)
                                    sq_clipped = pl.minimum(sq_clipped_maximum, 448.0)
                                    sq_quantized = pl.cast(sq_clipped, pl.FP8E4M3FN, mode="rint")
                                    h_quant = pl.reshape(sq_quantized, [ACT_T_TILE, QUANT_TILE])
                                    sq_signed_exponent_shrs = pl.shrs(sq_exponent, 7)
                                    sq_signed_exponent_mul = pl.mul(sq_signed_exponent_shrs, 256)
                                    sq_signed_exponent = pl.sub(sq_exponent, sq_signed_exponent_mul)
                                    sq_codes_cast = pl.cast(sq_signed_exponent, pl.INT8)
                                    sq_codes = pl.reinterpret_view(sq_codes_cast, pl.UINT8)
                                    sq_flat = pl.reshape(sq_codes, [1, ACT_T_TILE * (QUANT_TILE // 32)])
                                    sq_temporary = pl.create_tile([1, SWIGLU_SCALE_TMP_TILE], dtype=pl.UINT8)
                                    sq_packed = pl.tmov_x2zz(sq_flat, sq_temporary, group_axis=1, dst_rows=ACT_T_TILE, dst_cols=SWIGLU_SCALE_COL_TILE)
                                    h_scale = pl.reinterpret_view(sq_packed, pl.FP8E8M0)
                                    h_tile_mx = pl.store(h_quant, [act_t0, a_base], h_tile_mx)
                                    scale_offset = act_t0 * H_SCALE + ab_idx * ACT_T_TILE * (QUANT_TILE // MX_GROUP)
                                    h_tile_scale_backing_reshape = pl.reshape(h_scale, [1, ACT_T_TILE * (QUANT_TILE // MX_GROUP)])
                                    h_tile_scale_backing = pl.store(h_tile_scale_backing_reshape, [0, scale_offset], h_tile_scale_backing)
                            tile_completion_tids[t] = gate_up_act_quant_tid
            hidden_completion_tids[local_i] = pl.system.task_dummy(deps=[tile_completion_tids])
        with pl.scope():
            for local_e in pl.parallel(N_LOCAL_EXPERTS):
                tile_completion_tids = pl.array.create((RECV_MAX // RECV_TILE), pl.TASK_ID)
                for tile in pl.range((RECV_MAX // RECV_TILE)):
                    tile_completion_tids[tile] = pl.system.task_dummy(deps=[])
                e_flat_base = local_e * RECV_MAX
                routed_w2_packed_e = routed_w2_packed[local_e, :, :]

                e_rows = pl.read(recv_expert_count, [local_e, 0])
                if e_rows > 0:
                    e_tiles = (e_rows + RECV_TILE - 1) // RECV_TILE
                    with pl.scope():

                        w2_mxfp8 = pl.create_tensor([MOE_INTER, D], dtype=pl.FP8E4M3FN)
                        with pl.spmd(2 * (D // MX_W2_TASK_TILE), name_hint='exp_w2_mxfp4_aiv') as w2_mxfp4_aiv_tid:
                            block_idx = pl.tile.get_block_idx()
                            wb_idx = block_idx // 2
                            w2_lane = block_idx % 2
                            d_base = wb_idx * MX_W2_TASK_TILE
                            w2_lut_tile = pl.load(mxfp4_pair_lut, [w2_lane, 0], [1, 256], target_memory=pl.Mem.Vec)
                            for dg in pl.range(MX_W2_TASK_TILE // MX_W2_D_OUT_TILE):
                                d0 = d_base + dg * MX_W2_D_OUT_TILE
                                for k0 in pl.pipeline(0, MOE_INTER, MX_K_TILE, stage=2):
                                    w2_tile_id = (
                                        d0 // MX_W2_D_OUT_TILE * (MOE_INTER // MX_K_TILE) + k0 // MX_K_TILE
                                    )
                                    w2_packed_row = (
                                        w2_tile_id * 2 * MX_PACKED_LANE_ROWS + w2_lane * MX_PACKED_LANE_ROWS
                                    )
                                    w2_packed_tile = pl.load(
                                        routed_w2_packed_e, [w2_packed_row, 0], [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS],
                                        target_memory=pl.Mem.Vec,
                                    )
                                    w2_packed_u16 = pl.cast(w2_packed_tile, target_type=pl.UINT16)
                                    w2_packed_indices = pl.reinterpret_view(w2_packed_u16, pl.INT16)
                                    w2_gather_tmp = pl.create_tile(
                                        [MX_PACKED_LANE_ROWS, MX_PACKED_LANE_COLS], dtype=pl.INT16, target_memory=pl.Mem.Vec,
                                    )
                                    w2_pair_codes = pl.tile.gather(w2_lut_tile, w2_packed_indices, w2_gather_tmp)
                                    w2_bytes = pl.reinterpret_view(w2_pair_codes, pl.INT8, shape=[MX_PACKED_LANE_ROWS, MX_W2_D_OUT_TILE])
                                    w2_mxfp8_reinterpret_view = pl.reinterpret_view(w2_bytes, pl.FP8E4M3FN)
                                    w2_mxfp8 = pl.store(w2_mxfp8_reinterpret_view, [k0 + w2_lane * MX_PACKED_LANE_ROWS, d0], w2_mxfp8)

                        for tt in pl.parallel(e_tiles):
                            tt0 = tt * RECV_TILE
                            flat_tt0 = e_flat_base + tt0
                            valid_rows = pl.min(RECV_TILE, e_rows - tt0)
                            h_tile_mx = h_mx[flat_tt0 : flat_tt0 + RECV_TILE]
                            h_tile_scale_backing = h_scale_backing[
                                :, flat_tt0 * H_SCALE : (flat_tt0 + RECV_TILE) * H_SCALE
                            ]
                            h_tile_scale_mx = pl.tensor.view(h_tile_scale_backing, [RECV_TILE, H_SCALE], layout=pl.MX_A_ZZ)
                            recv_y_tile_fp32 = pl.create_tensor([RECV_TILE, D], dtype=pl.FP32)
                            with pl.spmd(D // MX_W2_TASK_TILE, name_hint='exp_w2_mxfp4_aic', deps=[w2_mxfp4_aiv_tid, hidden_completion_tids[local_e]]) as w2_mxfp4_aic_tid:
                                wb_idx = pl.tile.get_block_idx()
                                d_base = wb_idx * MX_W2_TASK_TILE
                                for dg in pl.range(MX_W2_TASK_TILE // MX_W2_D_OUT_TILE):
                                    d0 = d_base + dg * MX_W2_D_OUT_TILE
                                    w2_scale_row_base = local_e * H_SCALE
                                    y_acc = pl.create_tile([RECV_TILE, MX_W2_D_OUT_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                                    for k0 in pl.pipeline(0, MOE_INTER, MX_K_TILE, stage=2):
                                        ks = k0 // MX_GROUP
                                        w2_tile = pl.load(w2_mxfp8, [k0, d0], [MX_K_TILE, MX_W2_D_OUT_TILE], target_memory=pl.Mem.Mat)
                                        h_k = pl.load(h_tile_mx, [0, k0], [RECV_TILE, MX_K_TILE])
                                        h_scale_k = pl.load(
                                            h_tile_scale_mx, [0, ks], [RECV_TILE, MX_K_TILE // MX_GROUP],
                                            target_memory=pl.Mem.Mat,
                                        )
                                        w2_scale_k = pl.load(
                                            routed_w2_scale, [w2_scale_row_base + ks, d0], [MX_K_TILE // MX_GROUP, MX_W2_D_OUT_TILE],
                                            target_memory=pl.Mem.Mat,
                                        )
                                        if k0 == 0:
                                            y_acc = pl.matmul_mx(h_k, h_scale_k, w2_tile, w2_scale_k)
                                        else:
                                            y_acc = pl.matmul_mx_acc(y_acc, h_k, h_scale_k, w2_tile, w2_scale_k)
                                    recv_y_tile_fp32 = pl.store(y_acc, [0, d0], recv_y_tile_fp32)

                            recv_y_tile = pl.create_tensor([RECV_TILE, D], dtype=pl.BF16)
                            with pl.at(level=pl.Level.CORE_GROUP, name_hint='expert_tile_scatter', deps=[w2_mxfp4_aic_tid], no_dep_args=[routed_output]) as scatter_tid:
                                for out_t0 in pl.range(0, valid_rows, ACT_T_TILE):
                                    out_rows_max = pl.max(valid_rows - out_t0, 0)
                                    out_rows = pl.min(ACT_T_TILE, out_rows_max)
                                    for d0 in pl.range(0, D, ROUTE_D_OUT_TILE):
                                        y_fp32 = pl.load(recv_y_tile_fp32, [out_t0, d0], [ACT_T_TILE, ROUTE_D_OUT_TILE])
                                        y_valid = pl.set_validshape(y_fp32, out_rows, ROUTE_D_OUT_TILE)
                                        y_padded = pl.fillpad(y_valid, pad_value=pl.PadValue.zero)
                                        recv_y_tile_cast = pl.cast(y_padded, target_type=pl.BF16, mode='rint')
                                        recv_y_tile = pl.store(recv_y_tile_cast, [out_t0, d0], recv_y_tile)
                                tile_end = tt0 + valid_rows
                                source_begin = pl.cast(0, pl.INDEX)
                                for src in pl.range(EP_WORLD_SIZE):
                                    source_rows_read = pl.read(recv_meta, [src, local_e])
                                    source_rows = pl.cast(source_rows_read, pl.INDEX)
                                    source_end = source_begin + source_rows
                                    scatter_begin = pl.max(tt0, source_begin)
                                    scatter_end = pl.min(tile_end, source_end)
                                    for compact_row in pl.range(scatter_begin, scatter_end):
                                        route_read = pl.read(recv_route, [local_e, compact_row])
                                        route = pl.cast(route_read, pl.INDEX)
                                        pld.tensor.put(dst=routed_output, peer=src, src=recv_y_tile, dst_offsets=[route, 0], src_offsets=[compact_row - tt0, 0], shape=[1, D])
                                    source_begin = source_end
                            tile_completion_tids[tt] = scatter_tid
                expert_completion_tids[local_e] = pl.system.task_dummy(deps=[tile_completion_tids])

    scatter_done = pl.system.task_dummy(deps=[expert_completion_tids[local_e] for local_e in range(N_LOCAL_EXPERTS)])
    return scatter_done


def golden_expert_routed(tensors):
    """Torch reference for MXFP4-grid W1/W3/W2 with MXFP8 activations."""
    import torch
    import torch.nn.functional as F

    from quant import (
        decode_e8m0_codes_v41 as decode_e8m0_codes,
        host_quant_swiglu_mxfp8_v41 as host_quant_swiglu_mxfp8,
        matmul_mx_golden_v41 as matmul_mx_golden,
        unpack_mxfp4_weight_tiles,
    )

    recv_x = tensors["recv_x"]
    recv_scale_codes = tensors["recv_mx_scale"].reshape(N_LOCAL_EXPERTS * RECV_MAX, K_SCALE)
    recv_scale_decoded = decode_e8m0_codes(recv_scale_codes, side="a")
    recv_mx_scale = recv_scale_decoded.reshape(N_LOCAL_EXPERTS, RECV_MAX, K_SCALE)
    recv_weights = tensors["recv_weights"].float()
    recv_expert_count = tensors["recv_expert_count"]
    w1_scale = decode_e8m0_codes(tensors['routed_w1_scale'], side='b').reshape(N_LOCAL_EXPERTS, K_SCALE, MOE_INTER)
    w3_scale = decode_e8m0_codes(tensors['routed_w3_scale'], side='b').reshape(N_LOCAL_EXPERTS, K_SCALE, MOE_INTER)
    w2_scale = decode_e8m0_codes(tensors['routed_w2_scale'], side='b').reshape(N_LOCAL_EXPERTS, H_SCALE, D)

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub = recv_x[e, :n_rows, :]
        x_scale = recv_mx_scale[e, :n_rows, :]
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)
        w1_packed = tensors['routed_w1_packed'][e].reshape(
            2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE), MX_K_TILE * MX_MM_INTER_TILE // 4,
        )
        w1_fp8 = unpack_mxfp4_weight_tiles(w1_packed, D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE)
        w3_packed = tensors['routed_w3_packed'][e].reshape(
            2 * (D // MX_K_TILE) * (MOE_INTER // MX_MM_INTER_TILE), MX_K_TILE * MX_MM_INTER_TILE // 4,
        )
        w3_fp8 = unpack_mxfp4_weight_tiles(w3_packed, D, MOE_INTER, MX_K_TILE, MX_MM_INTER_TILE)
        w2_packed = tensors['routed_w2_packed'][e].reshape(
            2 * (MOE_INTER // MX_K_TILE) * (D // MX_W2_D_OUT_TILE), MX_K_TILE * MX_W2_D_OUT_TILE // 4,
        )
        w2_fp8 = unpack_mxfp4_weight_tiles(w2_packed, MOE_INTER, D, MX_K_TILE, MX_W2_D_OUT_TILE)

        gate = matmul_mx_golden(x_sub, x_scale, w1_fp8, w1_scale[e]).to(torch.bfloat16).float()
        up = matmul_mx_golden(x_sub, x_scale, w3_fp8, w3_scale[e]).to(torch.bfloat16).float()
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = (F.silu(gate) * up * w_per_row).bfloat16().float()
        h_fp8, h_scale_codes = host_quant_swiglu_mxfp8(h, return_e8m0=True)
        y = matmul_mx_golden(h_fp8, h_scale_codes, w2_fp8, w2_scale[e]).to(torch.bfloat16).float()
        recv_y[e, :n_rows, :] = y

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def gen_routed_mxfp4_weights(n_experts, dequant_std, seed_base=0):
    """Generate checkpoint-style packed MXFP4 routed weights and scales."""
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from quant import pack_mx_b_scale, pack_mxfp4_weight_tiles, quantize_mxfp4_weight

    w1_weights = torch.empty((n_experts, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS), dtype=torch.uint8)
    w3_weights = torch.empty_like(w1_weights)
    w2_weights = torch.empty((n_experts, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS), dtype=torch.uint8)
    w1_scales = torch.empty((n_experts, D // MX_GROUP, MOE_INTER), dtype=torch.uint8)
    w3_scales = torch.empty_like(w1_scales)
    w2_scales = torch.empty((n_experts, MOE_INTER // MX_GROUP, D), dtype=torch.uint8)

    def generate_expert(e):
        seed = seed_base + e * 3
        generator = torch.Generator().manual_seed(seed)
        w1, w1s = quantize_mxfp4_weight(torch.randn(MOE_INTER, D, generator=generator) * dequant_std['w1'])
        w3, w3s = quantize_mxfp4_weight(torch.randn(MOE_INTER, D, generator=generator) * dequant_std['w3'])
        w2, w2s = quantize_mxfp4_weight(torch.randn(D, MOE_INTER, generator=generator) * dequant_std['w2'])
        w1_tiles = pack_mxfp4_weight_tiles(w1, MX_K_TILE, MX_MM_INTER_TILE)
        w1_weights[e] = w1_tiles.reshape(MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w3_tiles = pack_mxfp4_weight_tiles(w3, MX_K_TILE, MX_MM_INTER_TILE)
        w3_weights[e] = w3_tiles.reshape(MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w2_tiles = pack_mxfp4_weight_tiles(w2, MX_K_TILE, MX_W2_D_OUT_TILE)
        w2_weights[e] = w2_tiles.reshape(MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS)
        w1_scales[e] = w1s.transpose(0, 1).contiguous().view(torch.uint8)
        w3_scales[e] = w3s.transpose(0, 1).contiguous().view(torch.uint8)
        w2_scales[e] = w2s.transpose(0, 1).contiguous().view(torch.uint8)

    with ThreadPoolExecutor(max_workers=min(4, n_experts)) as pool:
        list(pool.map(generate_expert, range(n_experts)))

    def pack_expert_scales(scales):
        packed = pack_mx_b_scale(scales.flatten(0, 1))
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        return packed.view(e8m0) if e8m0 is not None else packed

    return (
        w1_weights, pack_expert_scales(w1_scales),
        w3_weights, pack_expert_scales(w3_scales),
        w2_weights, pack_expert_scales(w2_scales),
    )


# Single-device validation with local scatter destinations.
WORLD_SIZE = 1
TOPK = M.num_experts_per_tok
ROUTE_ROWS = B * S * TOPK


@pl.jit(auto_scope=False)
def l2_expert_routed(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2_packed: pl.Tensor[[N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    recv_route: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_meta: pl.Tensor[[EP_WORLD_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    routed_y: pl.Out[pl.Tensor[[ROUTE_ROWS, D], pl.BF16]],
    routed_output: pld.DistributedTensor[[ROUTE_T_DYN, D], pl.BF16],
):
    input_ready = pl.system.task_dummy(deps=[])
    scatter_done = expert_routed_scatter(
        recv_x, recv_mx_scale, recv_weights, recv_expert_count,
        routed_w1_packed, routed_w1_scale,
        routed_w3_packed, routed_w3_scale,
        routed_w2_packed, routed_w2_scale,
        mxfp4_pair_lut, recv_route, recv_meta,
        routed_output, input_ready,
    )
    with pl.spmd(ROUTE_ROWS, name_hint="routed_output_copy", deps=[scatter_done]):
        row = pl.tile.get_block_idx()
        value = pl.load(routed_output, [row, 0], [1, D])
        routed_y = pl.store(value, [row, 0], routed_y)
    return routed_y


@pl.jit.host
def l3_expert_routed(
    recv_x: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[WORLD_SIZE, 1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1_packed: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w1_scale: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w3_packed: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w3_scale: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0],
    routed_w2_packed: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], pl.UINT8],
    routed_w2_scale: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0],
    mxfp4_pair_lut: pl.Tensor[[WORLD_SIZE, 2, 256], pl.INT16],
    recv_route: pl.Tensor[[WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX], pl.INT32],
    recv_meta: pl.Tensor[[WORLD_SIZE, EP_WORLD_SIZE, N_LOCAL_EXPERTS], pl.INT32],
    routed_y: pl.Out[pl.Tensor[[WORLD_SIZE, ROUTE_ROWS, D], pl.BF16]],
):
    routed_output_buf = pld.alloc_window_buffer([ROUTE_ROWS, D], dtype=pl.BF16)
    for r in pl.range(pld.world_size()):
        routed_output = pld.window(routed_output_buf, [ROUTE_ROWS, D], dtype=pl.BF16)
        # Per-rank MX_B_NN scale views.
        routed_w1_scale_r: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w1_scale[r]
        routed_w3_scale_r: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w3_scale[r]
        routed_w2_scale_r: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN] = routed_w2_scale[r]
        l2_expert_routed(
            recv_x[r], recv_mx_scale[r], recv_weights[r], recv_expert_count[r],
            routed_w1_packed[r], routed_w1_scale_r,
            routed_w3_packed[r], routed_w3_scale_r,
            routed_w2_packed[r], routed_w2_scale_r,
            mxfp4_pair_lut[r], recv_route[r], recv_meta[r],
            routed_y[r], routed_output, device=r,
        )


def golden_expert_routed_scatter(tensors):
    """Place each expert's golden rows at the window row named by ``recv_route``."""
    import torch

    rank_tensors = {name: value[0] for name, value in tensors.items() if name != "routed_y"}
    rank_tensors["recv_y"] = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    golden_expert_routed(rank_tensors)

    counts = rank_tensors["recv_expert_count"].reshape(-1)
    routed = torch.zeros(ROUTE_ROWS, D, dtype=torch.bfloat16)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(counts[e].item())
        routes = rank_tensors["recv_route"][e, :n_rows].to(torch.int64)
        routed[routes] = rank_tensors["recv_y"][e, :n_rows]
    tensors["routed_y"][0] = routed


def build_tensor_specs():
    from functools import cache

    import torch
    from golden.spec import TensorSpec
    from quant import build_mxfp4_pair_lut, host_quant_mxfp8_v41 as host_quant_mxfp8, pack_mx_a_scale as pack_a_scale

    total = B * S * M.num_experts_per_tok
    counts = torch.bincount(torch.randint(0, N_LOCAL_EXPERTS, (total,)), minlength=N_LOCAL_EXPERTS).to(torch.int32)
    counts_2d = counts.reshape(N_LOCAL_EXPERTS, 1)

    x_bf16 = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = (
        torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1)
    )
    # Zero-initialized FP8 byte carrier.
    recv_x_pre = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.uint8).view(torch.float8_e4m3fn)
    recv_scale_codes = torch.zeros(N_LOCAL_EXPERTS * RECV_MAX, K_SCALE, dtype=torch.uint8)
    for e in range(N_LOCAL_EXPERTS):
        n = int(counts_2d[e, 0].item())
        if n > 0:
            xf, xs = host_quant_mxfp8(x_bf16[e, :n, :], return_e8m0=True)
            recv_x_pre[e, :n, :] = xf
            recv_scale_codes[e * RECV_MAX : e * RECV_MAX + n, :] = xs.view(torch.uint8)
    recv_mx_scale_pre = pack_a_scale(recv_scale_codes).view(torch.float8_e8m0fnu)
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    # FP8 padding mask on the UINT8 carrier.
    recv_x_bytes = recv_x_pre.view(torch.uint8)
    recv_x_zero = torch.zeros_like(recv_x_bytes)
    recv_x_bytes = torch.where(valid_mask_3d, recv_x_bytes, recv_x_zero)
    recv_x_pre = recv_x_bytes.view(torch.float8_e4m3fn)

    def init_recv_x():
        return recv_x_pre

    def init_recv_mx_scale():
        return recv_mx_scale_pre.reshape(N_LOCAL_EXPERTS * RECV_MAX, K_SCALE)

    def init_recv_expert_count():
        return counts_2d

    recv_weights_pre = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre))

    def init_recv_weights():
        return recv_weights_pre

    # Cached packed weights and matching scales.
    @cache
    def routed_weights():
        return gen_routed_mxfp4_weights(N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=10)

    mxfp4_pair_lut = build_mxfp4_pair_lut()

    # Distinct scatter destinations for active compact rows.
    destinations = torch.randperm(total).to(torch.int32)
    recv_route_pre = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.int32)
    offset = 0
    for e in range(N_LOCAL_EXPERTS):
        n = int(counts[e].item())
        recv_route_pre[e, :n] = destinations[offset : offset + n]
        offset += n
    # Local source rows and zero remote counts.
    recv_meta_pre = torch.zeros(EP_WORLD_SIZE, N_LOCAL_EXPERTS, dtype=torch.int32)
    recv_meta_pre[0] = counts

    fp8 = torch.float8_e4m3fn
    fp8_e8m0 = torch.float8_e8m0fnu

    specs = [
        TensorSpec('recv_x', [WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX, D], fp8, init_value=lambda: init_recv_x().unsqueeze(0)),
        TensorSpec('recv_mx_scale', [WORLD_SIZE, 1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], fp8_e8m0, init_value=lambda: init_recv_mx_scale().reshape(1, 1, -1)),
        TensorSpec('recv_weights', [WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=lambda: init_recv_weights().unsqueeze(0)),
        TensorSpec('recv_expert_count', [WORLD_SIZE, N_LOCAL_EXPERTS, 1], torch.int32, init_value=lambda: init_recv_expert_count().unsqueeze(0)),
        TensorSpec('routed_w1_packed', [WORLD_SIZE, N_LOCAL_EXPERTS, MX_W1_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[0].unsqueeze(0)),
        TensorSpec('routed_w1_scale', [WORLD_SIZE, N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], fp8_e8m0, init_value=lambda: routed_weights()[1].unsqueeze(0)),
        TensorSpec('routed_w3_packed', [WORLD_SIZE, N_LOCAL_EXPERTS, MX_W3_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[2].unsqueeze(0)),
        TensorSpec('routed_w3_scale', [WORLD_SIZE, N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], fp8_e8m0, init_value=lambda: routed_weights()[3].unsqueeze(0)),
        TensorSpec('routed_w2_packed', [WORLD_SIZE, N_LOCAL_EXPERTS, MX_W2_PACKED_ROWS, MX_PACKED_LANE_COLS], torch.uint8, init_value=lambda: routed_weights()[4].unsqueeze(0)),
        TensorSpec('routed_w2_scale', [WORLD_SIZE, N_LOCAL_EXPERTS * H_SCALE, D], fp8_e8m0, init_value=lambda: routed_weights()[5].unsqueeze(0)),
        TensorSpec('mxfp4_pair_lut', [WORLD_SIZE, 2, 256], torch.int16, init_value=lambda: mxfp4_pair_lut.unsqueeze(0)),
        TensorSpec('recv_route', [WORLD_SIZE, N_LOCAL_EXPERTS, RECV_MAX], torch.int32, init_value=lambda: recv_route_pre.unsqueeze(0)),
        TensorSpec('recv_meta', [WORLD_SIZE, EP_WORLD_SIZE, N_LOCAL_EXPERTS], torch.int32, init_value=lambda: recv_meta_pre.unsqueeze(0)),
        TensorSpec("routed_y", [WORLD_SIZE, ROUTE_ROWS, D], torch.bfloat16),
    ]
    for spec in specs:
        spec.resident = "stacked"
    return specs


if __name__ == "__main__":
    import argparse

    from golden import ratio_reldiff, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    result = run(
        fn=l3_expert_routed,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed_scatter,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=[args.device],
                num_sub_workers=0,
            ),
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            enable_chip_swimlane=args.enable_chip_swimlane,
            # Heap capacity for all local experts and activation tiles.
            ring_heap=1_073_741_824,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # BF16 routed-row comparison.
            "routed_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
