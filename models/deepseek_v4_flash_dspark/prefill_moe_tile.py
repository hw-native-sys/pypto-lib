# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark prefill expert-major exchanges and count-driven routed experts."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import FLASH as M, EP, MOE_TOKENS, INT8_AMAX_EPS, INT8_SCALE_MAX
from gate import gate
from expert_shared import expert_shared

# model config
T = MOE_TOKENS
D = M.hidden_size
TOPK = M.num_experts_per_tok
VOCAB = M.vocab_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit
N_RANKS = EP
N_EXPERTS_GLOBAL = M.n_routed_experts
N_LOCAL = N_EXPERTS_GLOBAL // N_RANKS

# tiling
RECV_TILE = 16
K_TILE = 512
INTER_K_TILE = 512
MM_INTER_TILE = 256
ACT_INTER_TILE = 128
D_OUT_TILE = 256
QUANT_TILE = 512
D_OUT_ACT_TILE = 512
W2_INNER = 1
W2_ACT_INNER = 1
PREFILL_MOE_SCALE_PAD = 16
PREFILL_MOE_EXPERT_SCALE_PAD = 16
PREFILL_MOE_ROUTE_MAP_PAD = 16
PREFILL_MOE_FINALIZE_TOKEN_TILE = 16
PREFILL_MOE_GROUPED_EXPERT_TILE = 16
PREFILL_MOE_ROUTES_PER_SRC = T * TOPK
PREFILL_MOE_PEER_CAP = PREFILL_MOE_ROUTES_PER_SRC
PREFILL_MOE_TOTAL_CAP = N_RANKS * PREFILL_MOE_PEER_CAP
PREFILL_MOE_GROUPED_TOTAL_CAP = PREFILL_MOE_TOTAL_CAP + N_LOCAL * (RECV_TILE - 1)
PREFILL_MOE_SEND_SCALE_PAD = 16
SOURCE_PACK_BLOCKS = N_RANKS
RESORT_BLOCKS_PER_EXPERT = 1
COUNT_TOKEN_TILE = min(T, 128)
COUNT_SHARDS = (T + COUNT_TOKEN_TILE - 1) // COUNT_TOKEN_TILE
DISPATCH_ROW_TILE = 1
REVERSE_ROW_TILE = 1


def _make_expert_gate_up_quant_tile(grouped_capacity: int, row_tile: int):
    """Build a gate/up projection, SwiGLU, and INT8 quantization tile function."""
    ROW_TILE = row_tile
    MM_COL_TILE = 128 if row_tile == 192 else MM_INTER_TILE
    VEC_ROW_TILE = RECV_TILE
    ACT_TILE = ACT_INTER_TILE
    H_QUANT_TILE = QUANT_TILE

    @pl.jit.inline(auto_scope=False)
    def expert_gate_up_quant_tile(
        expert_x: pl.Tensor[[grouped_capacity, D], pl.INT8],
        expert_scale: pl.Tensor[[grouped_capacity, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        h_i8: pl.Tensor[[grouped_capacity, MOE_INTER], pl.INT8],
        h_scale_dq: pl.Tensor[[grouped_capacity, 1], pl.FP32],
        tile_row_start: pl.Scalar[pl.INDEX],
        local_expert_id: pl.Scalar[pl.INDEX],
        valid_rows: pl.Scalar[pl.INDEX],
        layout_tid: pl.Scalar[pl.TASK_ID],
        routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
        routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
        routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    ):
        """Compute gate/up projections, apply SwiGLU, and quantize a token tile."""
        weight_e = local_expert_id
        with pl.scope():
            w13_tile_i32 = pl.create_tensor([ROW_TILE, 2 * MOE_INTER], dtype=pl.INT32)

            with pl.spmd((2 * MOE_INTER) // MM_COL_TILE, name_hint="prefill_exp_w13_mm", deps=[layout_tid]) as w13_tid:
                block = pl.tile.get_block_idx()
                n0 = block * MM_COL_TILE
                if n0 < MOE_INTER:
                    w13_acc = pl.create_tensor([1, ROW_TILE, MM_COL_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                        x_chunk = expert_x[tile_row_start : tile_row_start + ROW_TILE, k0 : k0 + K_TILE]
                        w13_chunk = routed_w1[weight_e : weight_e + 1, n0 : n0 + MM_COL_TILE, k0 : k0 + K_TILE]
                        w13_acc = pl.matmul_acc(w13_acc, x_chunk, w13_chunk, b_trans=True, init_cond=k0 == 0)
                    w13_tile_i32[:, n0 : n0 + MM_COL_TILE] = pl.reshape(w13_acc, [ROW_TILE, MM_COL_TILE])
                else:
                    up_n0 = (block % (MOE_INTER // MM_COL_TILE)) * MM_COL_TILE
                    w13_acc = pl.create_tensor([1, ROW_TILE, MM_COL_TILE], dtype=pl.INT32)
                    for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                        x_chunk = expert_x[tile_row_start : tile_row_start + ROW_TILE, k0 : k0 + K_TILE]
                        w13_chunk = routed_w3[weight_e : weight_e + 1, up_n0 : up_n0 + MM_COL_TILE, k0 : k0 + K_TILE]
                        w13_acc = pl.matmul_acc(w13_acc, x_chunk, w13_chunk, b_trans=True, init_cond=k0 == 0)
                    w13_tile_i32[:, n0 : n0 + MM_COL_TILE] = pl.reshape(w13_acc, [ROW_TILE, MM_COL_TILE])

            h_tile_fp32 = pl.create_tensor([ROW_TILE, MOE_INTER], dtype=pl.FP32)
            h_tile_i8 = h_i8[tile_row_start : tile_row_start + ROW_TILE]
            h_tile_scale_dq = h_scale_dq[tile_row_start : tile_row_start + ROW_TILE]
            with pl.spmd((ROW_TILE // VEC_ROW_TILE), name_hint="prefill_exp_gate_up_quant", deps=[w13_tid]):
                vector_row = pl.tile.get_block_idx() * VEC_ROW_TILE
                vector_flat_row = tile_row_start + vector_row
                vector_valid_rows = pl.min(VEC_ROW_TILE, pl.max(valid_rows - vector_row, 0))
                x_scale_padded = expert_scale[vector_flat_row : vector_flat_row + VEC_ROW_TILE, 0:PREFILL_MOE_EXPERT_SCALE_PAD]
                x_scale_transposed = pl.transpose(x_scale_padded, axis1=0, axis2=1)
                x_scale_tile = pl.reshape(x_scale_transposed[0:1, :], [VEC_ROW_TILE, 1])
                row_amax = pl.full([1, VEC_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                for inter0 in pl.pipeline(0, MOE_INTER, ACT_TILE, stage=2):
                    gate_i32 = w13_tile_i32[vector_row : vector_row + VEC_ROW_TILE, inter0 : inter0 + ACT_TILE]
                    up_i32 = w13_tile_i32[vector_row : vector_row + VEC_ROW_TILE, MOE_INTER + inter0 : MOE_INTER + inter0 + ACT_TILE]
                    gate_fp32 = pl.cast(gate_i32, pl.FP32, mode="none")
                    gate_scaled = pl.row_expand_mul(gate_fp32, x_scale_tile)
                    gate_weight_scale = routed_w1_scale[local_expert_id : local_expert_id + 1, inter0 : inter0 + ACT_TILE]
                    gate_fp32 = pl.col_expand_mul(gate_scaled, gate_weight_scale)
                    up_fp32 = pl.cast(up_i32, pl.FP32, mode="none")
                    up_scaled = pl.row_expand_mul(up_fp32, x_scale_tile)
                    up_weight_scale = routed_w3_scale[local_expert_id : local_expert_id + 1, inter0 : inter0 + ACT_TILE]
                    up_fp32 = pl.col_expand_mul(up_scaled, up_weight_scale)
                    if SWIGLU_LIMIT > 0.0:
                        gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                        up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_fp32)), 1.0))
                    activated_chunk = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                    h_valid = pl.set_validshape(activated_chunk, vector_valid_rows, ACT_TILE)
                    h_padded = pl.fillpad(h_valid, pad_value=pl.PadValue.zero)
                    h_tile_fp32[vector_row : vector_row + VEC_ROW_TILE, inter0 : inter0 + ACT_TILE] = h_padded
                    h_abs = pl.maximum(h_padded, pl.neg(h_padded))
                    row_amax = pl.maximum(row_amax, pl.reshape(pl.row_max(h_abs), [1, VEC_ROW_TILE]))
                quant_scale = pl.div(pl.full([1, VEC_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), row_amax)
                h_tile_scale_dq[vector_row : vector_row + VEC_ROW_TILE, :] = pl.reshape(pl.recip(quant_scale), [VEC_ROW_TILE, 1])
                quant_scale_col = pl.reshape(quant_scale, [VEC_ROW_TILE, 1])
                for k0 in pl.pipeline(0, MOE_INTER, H_QUANT_TILE, stage=2):
                    h_quant_chunk = h_tile_fp32[vector_row : vector_row + VEC_ROW_TILE, k0 : k0 + H_QUANT_TILE]
                    h_scaled = pl.row_expand_mul(h_quant_chunk, quant_scale_col)
                    h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                    h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                    h_tile_i8[vector_row : vector_row + VEC_ROW_TILE, k0 : k0 + H_QUANT_TILE] = pl.cast(h_fp16, target_type=pl.INT8, mode="trunc")

    return expert_gate_up_quant_tile


def _make_expert_down_proj_tile(grouped_capacity: int, row_tile: int):
    """Build an expert down-projection tile function with routing weights."""
    ROW_TILE = row_tile
    OUTPUT_MM_TILE = 128 if row_tile == 192 else D_OUT_TILE
    VEC_ROW_TILE = RECV_TILE
    OUTPUT_ACT_TILE = D_OUT_ACT_TILE

    @pl.jit.inline(auto_scope=False)
    def expert_down_proj_tile(
        expert_scale: pl.Tensor[[grouped_capacity, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        h_i8: pl.Tensor[[grouped_capacity, MOE_INTER], pl.INT8],
        h_scale_dq: pl.Tensor[[grouped_capacity, 1], pl.FP32],
        tile_row_start: pl.Scalar[pl.INDEX],
        local_expert_id: pl.Scalar[pl.INDEX],
        routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
        routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
        expert_y: pl.Tensor[[grouped_capacity, D], pl.BF16],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Write routing-weighted BF16 down-projection output and return its TaskId."""
        weight_e = local_expert_id
        h_tile_i8 = h_i8[tile_row_start : tile_row_start + ROW_TILE]
        h_tile_scale_dq = h_scale_dq[tile_row_start : tile_row_start + ROW_TILE]

        y_i32 = pl.create_tensor([ROW_TILE, D], dtype=pl.INT32)
        with pl.spmd(D // (W2_INNER * OUTPUT_MM_TILE), name_hint="prefill_exp_w2_mm", allow_early_resolve=True) as w2_tid:
            block = pl.tile.get_block_idx()
            d_base = block * (W2_INNER * OUTPUT_MM_TILE)
            for inner in pl.range(W2_INNER):
                d0 = d_base + inner * OUTPUT_MM_TILE
                y_acc = pl.create_tensor([1, ROW_TILE, OUTPUT_MM_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, MOE_INTER, INTER_K_TILE, stage=2):
                    h_w2_chunk = h_tile_i8[:, k0 : k0 + INTER_K_TILE]
                    w2_chunk = routed_w2[weight_e : weight_e + 1, d0 : d0 + OUTPUT_MM_TILE, k0 : k0 + INTER_K_TILE]
                    y_acc = pl.matmul_acc(y_acc, h_w2_chunk, w2_chunk, b_trans=True, init_cond=k0 == 0)
                y_i32[:, d0 : d0 + OUTPUT_MM_TILE] = pl.reshape(y_acc, [ROW_TILE, OUTPUT_MM_TILE])

        expert_y_tile = expert_y[tile_row_start : tile_row_start + ROW_TILE, :]
        with pl.spmd(
            (ROW_TILE // VEC_ROW_TILE) * (D // (W2_ACT_INNER * OUTPUT_ACT_TILE)),
            name_hint="prefill_exp_w2_act",
            deps=[w2_tid],
            allow_early_resolve=False,
        ) as w2_act_tid:
            grid_block = pl.tile.get_block_idx()
            vector_row = (grid_block // (D // (W2_ACT_INNER * OUTPUT_ACT_TILE))) * VEC_ROW_TILE
            block = grid_block % (D // (W2_ACT_INNER * OUTPUT_ACT_TILE))
            vector_flat_row = tile_row_start + vector_row
            d_base = block * (W2_ACT_INNER * OUTPUT_ACT_TILE)
            scale_padded = expert_scale[vector_flat_row : vector_flat_row + VEC_ROW_TILE, 0:PREFILL_MOE_EXPERT_SCALE_PAD]
            scale_transposed = pl.transpose(scale_padded, axis1=0, axis2=1)
            weight_col = pl.reshape(scale_transposed[1:2, :], [VEC_ROW_TILE, 1])
            row_scale = pl.mul(h_tile_scale_dq[vector_row : vector_row + VEC_ROW_TILE, :], weight_col)
            for inner in pl.pipeline(W2_ACT_INNER, stage=2):
                d0 = d_base + inner * OUTPUT_ACT_TILE
                y_fp32 = pl.cast(y_i32[vector_row : vector_row + VEC_ROW_TILE, d0 : d0 + OUTPUT_ACT_TILE], target_type=pl.FP32, mode="none")
                y_scaled = pl.row_expand_mul(y_fp32, row_scale)
                output_weight_scale = routed_w2_scale[local_expert_id : local_expert_id + 1, d0 : d0 + OUTPUT_ACT_TILE]
                y_fp32 = pl.col_expand_mul(y_scaled, output_weight_scale)
                y_bf16 = pl.cast(y_fp32, pl.BF16, mode="rint")
                expert_y_tile[vector_row : vector_row + VEC_ROW_TILE, d0 : d0 + OUTPUT_ACT_TILE] = y_bf16
        return w2_act_tid

    return expert_down_proj_tile


def make_prefill_expert_grouped(grouped_capacity: int):
    """Specialize count-driven expert compute for an aligned scratch capacity."""
    if grouped_capacity <= 0 or grouped_capacity % RECV_TILE:
        raise ValueError("grouped expert capacity must be a positive multiple of RECV_TILE")
    grouped_tile_capacity = grouped_capacity // RECV_TILE
    gate_up_quant_m192 = _make_expert_gate_up_quant_tile(grouped_capacity, 192)
    gate_up_quant_m128 = _make_expert_gate_up_quant_tile(grouped_capacity, 128)
    gate_up_quant_m96 = _make_expert_gate_up_quant_tile(grouped_capacity, 96)
    gate_up_quant_m64 = _make_expert_gate_up_quant_tile(grouped_capacity, 64)
    gate_up_quant_m32 = _make_expert_gate_up_quant_tile(grouped_capacity, 32)
    gate_up_quant_m16 = _make_expert_gate_up_quant_tile(grouped_capacity, RECV_TILE)
    down_proj_m192 = _make_expert_down_proj_tile(grouped_capacity, 192)
    down_proj_m128 = _make_expert_down_proj_tile(grouped_capacity, 128)
    down_proj_m96 = _make_expert_down_proj_tile(grouped_capacity, 96)
    down_proj_m64 = _make_expert_down_proj_tile(grouped_capacity, 64)
    down_proj_m32 = _make_expert_down_proj_tile(grouped_capacity, 32)
    down_proj_m16 = _make_expert_down_proj_tile(grouped_capacity, RECV_TILE)

    @pl.jit.inline(auto_scope=False)
    def prefill_expert_grouped(
        expert_x: pl.Tensor[[grouped_capacity, D], pl.INT8],
        expert_scale: pl.Tensor[[grouped_capacity, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
        expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
        routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
        routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
        routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
        routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
        routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
        expert_y: pl.Tensor[[grouped_capacity, D], pl.BF16],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Compute aligned expert slabs with count-driven row tiles and weighted BF16 outputs."""
        expert_bases = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
        h_i8 = pl.create_tensor([grouped_capacity, MOE_INTER], dtype=pl.INT8)
        h_scale_dq = pl.create_tensor([grouped_capacity, 1], dtype=pl.FP32, manual_dep=True)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_exp_group_layout", allow_early_resolve=True) as layout_tid:
            grouped_base = pl.cast(0, pl.INDEX)
            for local_expert_id in pl.range(N_LOCAL):
                pl.write(expert_bases, [local_expert_id, 0], pl.cast(grouped_base, pl.INT32))
                rows = pl.cast(pl.read(expert_counts, [local_expert_id, 0]), pl.INDEX)
                aligned_rows = ((rows + RECV_TILE - 1) // RECV_TILE) * RECV_TILE
                grouped_base = grouped_base + aligned_rows

        expert_completion_tids = pl.array.create(N_LOCAL, pl.TASK_ID)

        with pl.scope():
            for local_expert_id in pl.parallel(N_LOCAL):
                flat_base = pl.cast(pl.read(expert_bases, [local_expert_id, 0]), pl.INDEX)
                n_rows = pl.read(expert_counts, [local_expert_id, 0])
                tile192 = pl.cast(((n_rows + RECV_TILE - 1) // RECV_TILE) * RECV_TILE == 192, pl.INDEX)
                main_tiles = (n_rows // 128) * (1 - tile192)
                aligned_tail = (((n_rows % 128) + RECV_TILE - 1) // RECV_TILE) * RECV_TILE * (1 - tile192)
                tail96 = pl.cast(aligned_tail == 96, pl.INDEX)
                remaining_tail = aligned_tail - tail96 * 96
                tail64 = remaining_tail // 64
                tail32 = (remaining_tail % 64) // 32
                tail16 = (remaining_tail % 32) // RECV_TILE
                for tile in pl.parallel(tile192):
                    gate_up_quant_m192(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        flat_base, local_expert_id, pl.cast(n_rows, pl.INDEX), layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(main_tiles):
                    tile_row_start = flat_base + tile * 128
                    gate_up_quant_m128(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, pl.cast(128, pl.INDEX), layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail96):
                    tile_row = main_tiles * 128 + tile * 96
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(96, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m96(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail64):
                    tile_row = main_tiles * 128 + tail96 * 96 + tile * 64
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(64, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m64(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail32):
                    tile_row = main_tiles * 128 + tail96 * 96 + tail64 * 64 + tile * 32
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(32, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m32(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )
                for tile in pl.parallel(tail16):
                    tile_row = main_tiles * 128 + tail96 * 96 + tail64 * 64 + tail32 * 32 + tile * 16
                    tile_row_start = flat_base + tile_row
                    valid_rows = pl.cast(pl.min(16, n_rows - tile_row), pl.INDEX)
                    gate_up_quant_m16(
                        expert_x, expert_scale, h_i8, h_scale_dq,
                        tile_row_start, local_expert_id, valid_rows, layout_tid,
                        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                    )

            for local_expert_id in pl.parallel(N_LOCAL):
                w2_act_tids = pl.array.create(grouped_tile_capacity, pl.TASK_ID)
                flat_base = pl.cast(pl.read(expert_bases, [local_expert_id, 0]), pl.INDEX)
                n_rows = pl.read(expert_counts, [local_expert_id, 0])
                tile192 = pl.cast(((n_rows + RECV_TILE - 1) // RECV_TILE) * RECV_TILE == 192, pl.INDEX)
                main_tiles = (n_rows // 128) * (1 - tile192)
                aligned_tail = (((n_rows % 128) + RECV_TILE - 1) // RECV_TILE) * RECV_TILE * (1 - tile192)
                tail96 = pl.cast(aligned_tail == 96, pl.INDEX)
                remaining_tail = aligned_tail - tail96 * 96
                tail64 = remaining_tail // 64
                tail32 = (remaining_tail % 64) // 32
                tail16 = (remaining_tail % 32) // RECV_TILE
                for tile in pl.parallel(tile192):
                    w2_act_tid = down_proj_m192(
                        expert_scale, h_i8, h_scale_dq, flat_base, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y,
                    )
                    w2_act_tids[tile] = w2_act_tid
                for tile in pl.parallel(main_tiles):
                    tile_row_start = flat_base + tile * 128
                    w2_act_tid = down_proj_m128(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y,
                    )
                    w2_act_tids[tile192 + tile] = w2_act_tid
                for tile in pl.parallel(tail96):
                    tile_row_start = flat_base + main_tiles * 128 + tile * 96
                    w2_act_tid = down_proj_m96(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[tile192 + main_tiles + tile] = w2_act_tid
                for tile in pl.parallel(tail64):
                    tile_row_start = flat_base + main_tiles * 128 + tail96 * 96 + tile * 64
                    w2_act_tid = down_proj_m64(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[tile192 + main_tiles + tail96 + tile] = w2_act_tid
                for tile in pl.parallel(tail32):
                    tile_row_start = flat_base + main_tiles * 128 + tail96 * 96 + tail64 * 64 + tile * 32
                    w2_act_tid = down_proj_m32(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[tile192 + main_tiles + tail96 + tail64 + tile] = w2_act_tid
                for tile in pl.parallel(tail16):
                    tile_row_start = flat_base + main_tiles * 128 + tail96 * 96 + tail64 * 64 + tail32 * 32 + tile * 16
                    w2_act_tid = down_proj_m16(
                        expert_scale, h_i8, h_scale_dq, tile_row_start, local_expert_id,
                        routed_w2, routed_w2_scale, expert_y)
                    w2_act_tids[tile192 + main_tiles + tail96 + tail64 + tail32 + tile] = w2_act_tid

                expert_completion_tid = pl.system.task_dummy(deps=[w2_act_tids])
                expert_completion_tids[local_expert_id] = expert_completion_tid

        completion_tid = pl.system.task_dummy(deps=[expert_completion_tids[local_expert_id] for local_expert_id in range(N_LOCAL)])

        return completion_tid

    return prefill_expert_grouped


prefill_expert_grouped = make_prefill_expert_grouped(PREFILL_MOE_GROUPED_TOTAL_CAP)


@pl.jit.inline
def prefill_moe_dispatch(
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
    expert_x_out: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, D], pl.INT8],
    expert_scale_out: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32],
    expert_counts_out: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_expert_counts_out: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    send_counts_out: pl.Tensor[[N_RANKS, 1], pl.INT32],
    route_to_packed_out: pl.Tensor[[PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD], pl.INT32],
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    scale_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    prior_dep: pl.Scalar[pl.TASK_ID],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Pack active routes, exchange counts and payloads, and form aligned expert slabs."""

    send_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)

    x_send = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, D], dtype=pl.INT8, manual_dep=True)
    scale_send = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_SEND_SCALE_PAD], dtype=pl.FP32, manual_dep=True,)

    partial_counts = pl.create_tensor([COUNT_SHARDS, N_RANKS * N_LOCAL], dtype=pl.INT32)
    packed_offsets = pl.create_tensor([COUNT_SHARDS + 1, N_RANKS * N_LOCAL], dtype=pl.INT32)
    with pl.spmd(COUNT_SHARDS, name_hint="prefill_moe_route_count", allow_early_resolve=True) as partial_count_tid:
        shard = pl.tile.get_block_idx()
        token_begin = shard * COUNT_TOKEN_TILE
        active_tokens = pl.min(T, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
        token_rows = pl.min(COUNT_TOKEN_TILE, pl.max(active_tokens - token_begin, 0))
        counts = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
        for expert_id in pl.range(N_RANKS * N_LOCAL):
            counts[expert_id] = 0
        if token_rows > 0:
            route_indices = pl.tile.load(indices, [token_begin, 0], [COUNT_TOKEN_TILE, 8], valid_shape=[token_rows, TOPK])
            for token in pl.range(token_rows):
                for topk in pl.range(TOPK):
                    route_expert_id = pl.tile.read(route_indices, [token, topk])
                    counts[route_expert_id] = counts[route_expert_id] + 1
        partial_count_row = pl.tile.full([1, N_RANKS * N_LOCAL], dtype=pl.INT32, value=0)
        for expert_id in pl.range(N_RANKS * N_LOCAL):
            pl.tile.write(partial_count_row, [0, expert_id], counts[expert_id])
        pl.tile.store(partial_count_row, [shard, 0], partial_counts)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_route_prefix",
               deps=[partial_count_tid], allow_early_resolve=True) as prefix_tid:
        counts_tile = pl.tile.load(partial_counts, [0, 0], [COUNT_SHARDS, N_RANKS * N_LOCAL])
        offsets_tile = pl.tile.full([COUNT_SHARDS + 1, N_RANKS * N_LOCAL], dtype=pl.INT32, value=0)
        packed_prefix = pl.const(0, pl.INT32)
        for dst in pl.range(N_RANKS):
            rank_total = pl.const(0, pl.INT32)
            for local_e in pl.range(N_LOCAL):
                expert_id = dst * N_LOCAL + local_e
                pl.tile.write(offsets_tile, [COUNT_SHARDS, expert_id], packed_prefix)
                expert_total = pl.const(0, pl.INT32)
                for shard in pl.range(COUNT_SHARDS):
                    pl.tile.write(offsets_tile, [shard, expert_id], packed_prefix)
                    count = pl.tile.read(counts_tile, [shard, expert_id])
                    packed_prefix = packed_prefix + count
                    expert_total = expert_total + count
                pl.write(send_expert_counts, [dst, local_e], expert_total)
                rank_total = rank_total + expert_total
            pl.write(send_counts_out, [dst, 0], rank_total)
        pl.tile.store(offsets_tile, [0, 0], packed_offsets)

    with pl.spmd(COUNT_SHARDS, name_hint="prefill_moe_route_positions", deps=[prefix_tid], allow_early_resolve=True) as count_tid:
        shard = pl.tile.get_block_idx()
        token_begin = shard * COUNT_TOKEN_TILE
        active_tokens = pl.min(T, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
        token_rows = pl.min(COUNT_TOKEN_TILE, pl.max(active_tokens - token_begin, 0))
        if token_rows > 0:
            shard_offsets = pl.tile.load(packed_offsets, [shard, 0], [1, N_RANKS * N_LOCAL])
            expert_starts = pl.tile.load(packed_offsets, [COUNT_SHARDS, 0], [1, N_RANKS * N_LOCAL])
            packed_cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
            destination_prefix = pl.array.create(N_RANKS, pl.INT32)
            for expert_id in pl.range(N_RANKS * N_LOCAL):
                packed_cursor[expert_id] = pl.tile.read(shard_offsets, [0, expert_id])
            for dst in pl.range(N_RANKS):
                destination_prefix[dst] = pl.tile.read(expert_starts, [0, dst * N_LOCAL])
            route_indices = pl.tile.load(indices, [token_begin, 0], [COUNT_TOKEN_TILE, 8], valid_shape=[token_rows, TOPK])
            route_positions = pl.tile.full([COUNT_TOKEN_TILE * TOPK, PREFILL_MOE_ROUTE_MAP_PAD], dtype=pl.INT32, value=0)
            for token in pl.range(token_rows):
                for topk in pl.range(TOPK):
                    route_expert_id = pl.tile.read(route_indices, [token, topk])
                    packed_slot = packed_cursor[route_expert_id]
                    destination = route_expert_id // N_LOCAL
                    return_row = destination * PREFILL_MOE_PEER_CAP + packed_slot - destination_prefix[destination]
                    pl.tile.write(route_positions, [token * TOPK + topk, 0], packed_slot)
                    pl.tile.write(route_positions, [token * TOPK + topk, 1], pl.cast(return_row, pl.INT32))
                    packed_cursor[route_expert_id] = packed_slot + 1
            valid_positions = pl.set_validshape(route_positions, token_rows * TOPK, PREFILL_MOE_ROUTE_MAP_PAD)
            pl.tile.store(valid_positions, [token_begin * TOPK, 0], route_to_packed_out)

    token_scales_row = pl.reshape(x_norm_scale, [1, T])
    with pl.spmd(SOURCE_PACK_BLOCKS, name_hint="prefill_moe_dispatch_pack", deps=[count_tid], allow_early_resolve=True,) as pack_tid:
        pack_block = pl.tile.get_block_idx()
        active_tokens = pl.cast(num_tokens, pl.INDEX)
        if active_tokens < 0:
            active_tokens = pl.cast(0, pl.INDEX)
        if active_tokens > T:
            active_tokens = pl.cast(T, pl.INDEX)
        scale_row = pl.tile.full([1, PREFILL_MOE_SEND_SCALE_PAD], dtype=pl.FP32, value=0.0)
        token_scales = pl.tile.load(token_scales_row, [0, 0], [1, T])
        route_weights = pl.tile.load(weights, [0, 0], [T, 8], valid_shape=[T, TOPK])
        route_map = pl.tile.load(route_to_packed_out, [0, 0], [PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD])
        for token in pl.range(pack_block, active_tokens, SOURCE_PACK_BLOCKS):
            hidden_row = x_norm_i8[token:token + 1, :]
            token_scale = pl.tile.read(token_scales, [0, token])
            for topk in pl.range(TOPK):
                route = token * TOPK + topk
                packed_slot = pl.tile.read(route_map, [route, 0])
                packed_row = pl.cast(packed_slot, pl.INDEX)
                x_send[packed_row:packed_row + 1, :] = hidden_row
                pl.tile.write(scale_row, [0, 0], token_scale)
                pl.tile.write(scale_row, [0, 1], pl.tile.read(route_weights, [token, topk]))
                pl.tile.store(scale_row, [packed_row, 0], scale_send)

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_moe_dispatch_count_exchange",
        deps=[count_tid, prior_dep],
    ) as count_publish_tid:
        for dest in pl.range(N_RANKS):
            pld.tensor.put(
                dst=count_target, peer=dest, src=send_expert_counts,
                dst_offsets=[my_rank, 0], src_offsets=[dest, 0], shape=[1, N_LOCAL],
            )
            if dest != my_rank:
                pld.system.notify(target=count_signal, peer=dest, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_count_wait") as count_wait_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=count_signal, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_count_finalize", deps=[count_publish_tid, count_wait_tid]) as count_exchange_tid:
        for local_e in pl.range(N_LOCAL):
            expert_total = pl.const(0, pl.INT32)
            for src in pl.range(N_RANKS):
                count = pl.read(count_target, [src, local_e])
                pl.write(recv_expert_counts_out, [src, local_e], count)
                expert_total = expert_total + count
            pl.write(expert_counts_out, [local_e, 0], expert_total)

    with pl.spmd(
        N_RANKS * 4,
        name_hint="prefill_moe_dispatch_payload_put",
        deps=[pack_tid, count_exchange_tid],
        allow_early_resolve=False,
    ) as payload_put_tid:
        transfer_block = pl.tile.get_block_idx()
        dest = transfer_block // 4
        row_group = transfer_block % 4
        n_rows = pl.cast(pl.read(send_counts_out, [dest, 0]), pl.INDEX)
        if n_rows < 0:
            n_rows = pl.cast(0, pl.INDEX)
        if n_rows > PREFILL_MOE_PEER_CAP:
            n_rows = pl.cast(PREFILL_MOE_PEER_CAP, pl.INDEX)
        src_base = pl.cast(0, pl.INDEX)
        for prior_dst in pl.range(dest):
            src_base = src_base + pl.cast(pl.read(send_counts_out, [prior_dst, 0]), pl.INDEX)
        dst_base = pl.cast(my_rank, pl.INDEX) * PREFILL_MOE_PEER_CAP
        full_rows = (n_rows // DISPATCH_ROW_TILE) * DISPATCH_ROW_TILE
        for row in pl.range(row_group * DISPATCH_ROW_TILE, full_rows, 4 * DISPATCH_ROW_TILE):
            pld.tensor.put(
                dst=x_target,
                peer=dest,
                src=x_send,
                dst_offsets=[dst_base + row, 0],
                src_offsets=[src_base + row, 0],
                shape=[DISPATCH_ROW_TILE, D],
            )
            pld.tensor.put(
                dst=scale_target,
                peer=dest,
                src=scale_send,
                dst_offsets=[dst_base + row, 0],
                src_offsets=[src_base + row, 0],
                shape=[DISPATCH_ROW_TILE, PREFILL_MOE_SCALE_PAD],
            )
        for row in pl.range(full_rows + row_group, n_rows, 4):
            pld.tensor.put(
                dst=x_target,
                peer=dest,
                src=x_send,
                dst_offsets=[dst_base + row, 0],
                src_offsets=[src_base + row, 0],
                shape=[1, D],
            )
            pld.tensor.put(
                dst=scale_target,
                peer=dest,
                src=scale_send,
                dst_offsets=[dst_base + row, 0],
                src_offsets=[src_base + row, 0],
                shape=[1, PREFILL_MOE_SCALE_PAD],
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_dispatch_payload_notify", deps=[payload_put_tid]) as payload_notify_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(target=x_signal, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_dispatch_payload_wait", deps=[payload_notify_tid]) as payload_exchange_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=x_signal, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    with pl.spmd(
        N_LOCAL * RESORT_BLOCKS_PER_EXPERT,
        name_hint="prefill_moe_dispatch_resort",
        deps=[count_exchange_tid, payload_exchange_tid],
        allow_early_resolve=False,
    ) as resort_tid:
        resort_block = pl.tile.get_block_idx()
        local_e = resort_block // RESORT_BLOCKS_PER_EXPERT
        row_group = resort_block % RESORT_BLOCKS_PER_EXPERT
        if pl.read(expert_counts_out, [local_e, 0]) > 0:
            expert_scale_row = pl.tile.full([1, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32, value=0.0)

            expert_base = pl.cast(0, pl.INDEX)
            for prior_e in pl.range(local_e):
                prior_rows = pl.cast(pl.read(expert_counts_out, [prior_e, 0]), pl.INDEX)
                prior_tiles = (prior_rows + PREFILL_MOE_GROUPED_EXPERT_TILE - 1) // PREFILL_MOE_GROUPED_EXPERT_TILE
                expert_base = expert_base + prior_tiles * PREFILL_MOE_GROUPED_EXPERT_TILE

            source_prefix = pl.cast(0, pl.INDEX)
            for src in pl.range(N_RANKS):
                source_expert_base = pl.cast(0, pl.INDEX)
                for prior_e in pl.range(local_e):
                    prior_count = pl.cast(pl.read(recv_expert_counts_out, [src, prior_e]), pl.INDEX)
                    source_expert_base = source_expert_base + prior_count
                n_rows = pl.cast(pl.read(recv_expert_counts_out, [src, local_e]), pl.INDEX)
                input_base = src * PREFILL_MOE_PEER_CAP + source_expert_base
                output_base = expert_base + source_prefix
                full_rows = (n_rows // 16) * 16
                for row in pl.range(row_group * 16, full_rows, RESORT_BLOCKS_PER_EXPERT * 16):
                    input_row = input_base + row
                    output_row = output_base + row
                    expert_x_out[output_row : output_row + 16, :] = x_target[input_row : input_row + 16, :]
                    for inner in pl.range(16):
                        pl.tile.write(expert_scale_row, [0, 0], pl.read(scale_target, [input_row + inner, 0]))
                        pl.tile.write(expert_scale_row, [0, 1], pl.read(scale_target, [input_row + inner, 1]))
                        pl.tile.store(expert_scale_row, [output_row + inner, 0], expert_scale_out)
                for row in pl.range(full_rows + row_group, n_rows, RESORT_BLOCKS_PER_EXPERT):
                    input_row = input_base + row
                    output_row = output_base + row
                    expert_x_out[output_row : output_row + 1, :] = x_target[input_row : input_row + 1, :]
                    pl.tile.write(expert_scale_row, [0, 0], pl.read(scale_target, [input_row, 0]))
                    pl.tile.write(expert_scale_row, [0, 1], pl.read(scale_target, [input_row, 1]))
                    pl.tile.store(expert_scale_row, [output_row, 0], expert_scale_out)
                source_prefix = source_prefix + n_rows

            expert_rows = pl.cast(pl.read(expert_counts_out, [local_e, 0]), pl.INDEX)
            expert_tiles = (expert_rows + PREFILL_MOE_GROUPED_EXPERT_TILE - 1) // PREFILL_MOE_GROUPED_EXPERT_TILE
            aligned_rows = expert_tiles * PREFILL_MOE_GROUPED_EXPERT_TILE
            zero_x = pl.cast(pl.full([1, D], dtype=pl.FP16, value=0.0), pl.INT8, mode="trunc")
            zero_scale = pl.full([1, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32, value=0.0)
            for row in pl.range(expert_rows + row_group, aligned_rows, RESORT_BLOCKS_PER_EXPERT):
                output_row = expert_base + row
                expert_x_out[output_row : output_row + 1, :] = zero_x
                expert_scale_out[output_row : output_row + 1, :] = zero_scale

    return resort_tid


@pl.jit.inline
def _prefill_moe_reverse_exchange(
    expert_y: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, D], pl.BF16],
    expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_expert_counts: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    reverse_send_counts: pl.Tensor[[N_RANKS, 1], pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    expert_tid: pl.Scalar[pl.TASK_ID],
    counts_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    with pl.spmd(
        N_RANKS * N_LOCAL,
        name_hint="prefill_moe_combine_y_put",
        deps=[expert_tid, counts_tid],
        allow_early_resolve=False,
    ) as reverse_put_tid:
        transfer_block = pl.tile.get_block_idx()
        dest = transfer_block // N_LOCAL
        local_e = transfer_block % N_LOCAL
        destination_rows = pl.cast(pl.read(reverse_send_counts, [dest, 0]), pl.INDEX)
        destination_rows = pl.min(PREFILL_MOE_PEER_CAP, pl.max(destination_rows, 0))
        dst_base = pl.cast(my_rank, pl.INDEX) * PREFILL_MOE_PEER_CAP
        send_prefix = pl.cast(0, pl.INDEX)
        expert_base = pl.cast(0, pl.INDEX)
        for prior_e in pl.range(local_e):
            send_prefix = send_prefix + pl.cast(pl.read(recv_expert_counts, [dest, prior_e]), pl.INDEX)
            expert_rows = pl.cast(pl.read(expert_counts, [prior_e, 0]), pl.INDEX)
            expert_tiles = (expert_rows + PREFILL_MOE_GROUPED_EXPERT_TILE - 1) // PREFILL_MOE_GROUPED_EXPERT_TILE
            expert_base = expert_base + expert_tiles * PREFILL_MOE_GROUPED_EXPERT_TILE
        source_prefix = pl.cast(0, pl.INDEX)
        for prior_src in pl.range(dest):
            source_prefix = source_prefix + pl.cast(pl.read(recv_expert_counts, [prior_src, local_e]), pl.INDEX)
        n_rows = pl.cast(pl.read(recv_expert_counts, [dest, local_e]), pl.INDEX)
        n_rows = pl.min(n_rows, pl.max(destination_rows - send_prefix, 0))
        input_base = expert_base + source_prefix
        output_base = dst_base + send_prefix
        bulk_rows = (n_rows // REVERSE_ROW_TILE) * REVERSE_ROW_TILE
        for row in pl.range(0, bulk_rows, REVERSE_ROW_TILE):
            pld.tensor.put(
                dst=reverse_target,
                peer=dest,
                src=expert_y,
                dst_offsets=[output_base + row, 0],
                src_offsets=[input_base + row, 0],
                shape=[REVERSE_ROW_TILE, D],
            )
        for row in pl.range(bulk_rows, n_rows):
            pld.tensor.put(
                dst=reverse_target,
                peer=dest,
                src=expert_y,
                dst_offsets=[output_base + row, 0],
                src_offsets=[input_base + row, 0],
                shape=[1, D],
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_combine_y_notify", deps=[reverse_put_tid]) as reverse_notify_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(target=reverse_signal, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_combine_y_wait", deps=[reverse_notify_tid]) as reverse_exchange_tid:
        for src in pl.range(N_RANKS):
            if src != my_rank:
                pld.system.defer_wait(signal=reverse_signal, offsets=[src, 0], expected=epoch, cmp=pld.WaitCmp.Ge)

    return reverse_exchange_tid


@pl.jit.inline
def prefill_moe_combine(
    expert_y: pl.Tensor[[PREFILL_MOE_GROUPED_TOTAL_CAP, D], pl.BF16],
    expert_counts: pl.Tensor[[N_LOCAL, 1], pl.INT32],
    recv_expert_counts: pl.Tensor[[N_RANKS, N_LOCAL], pl.INT32],
    route_to_packed: pl.Tensor[[PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD], pl.INT32],
    shared_y: pl.Tensor[[T, D], pl.BF16],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    dispatch_tid: pl.Scalar[pl.TASK_ID],
    expert_tid: pl.Scalar[pl.TASK_ID],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Return weighted expert rows and combine shared output in top-k order."""

    reverse_send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32, manual_dep=True)

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_moe_combine_counts",
        deps=[dispatch_tid],
        allow_early_resolve=True,
    ) as reverse_counts_tid:
        for src in pl.range(N_RANKS):
            source_total = pl.const(0, pl.INT32)
            for local_e in pl.range(N_LOCAL):
                source_total = source_total + pl.read(recv_expert_counts, [src, local_e])
            pl.write(reverse_send_counts, [src, 0], source_total)

    reverse_exchange_tid = _prefill_moe_reverse_exchange(
        expert_y, expert_counts, recv_expert_counts,
        reverse_target, reverse_signal, reverse_send_counts,
        epoch, my_rank, expert_tid, reverse_counts_tid,
    )

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    with pl.spmd(
        T // PREFILL_MOE_FINALIZE_TOKEN_TILE,
        name_hint="prefill_moe_combine_finalize",
        deps=[reverse_exchange_tid],
    ) as finalize_tid:
        token0 = pl.tile.get_block_idx() * PREFILL_MOE_FINALIZE_TOKEN_TILE
        for lane in pl.range(PREFILL_MOE_FINALIZE_TOKEN_TILE):
            token = token0 + lane
            if token < active_tokens:
                acc = pl.cast(shared_y[token : token + 1, :], pl.FP32)
                for topk in pl.range(TOPK):
                    route = token * TOPK + topk
                    return_row = pl.cast(pl.read(route_to_packed, [route, 1]), pl.INDEX)
                    route_y = pl.cast(reverse_target[return_row : return_row + 1, 0:D], pl.FP32)
                    acc = pl.add(acc, route_y)
                ffn_out[token : token + 1, :] = pl.cast(acc, pl.BF16, mode="rint")
            else:
                ffn_out[token : token + 1, :] = shared_y[token : token + 1, :]

    return finalize_tid


@pl.jit.inline(auto_scope=False)
def prefill_moe_tile(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8, pl.NZ],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8, pl.NZ],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8, pl.NZ],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8, pl.NZ],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prior_dep: pl.Scalar[pl.TASK_ID],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one DSpark prefill wave with compact expert-major exchanges."""
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(x_mixed, norm_w, gate_w, gate_bias, layer_id, num_tokens, tid2eid, input_ids, x_norm_i8, x_norm_scale, indices, weights,)

    shared_out = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(x_norm_i8, x_norm_scale, shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale, shared_out,)

    expert_counts = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_expert_counts = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)
    send_counts = pl.create_tensor([N_RANKS, 1], dtype=pl.INT32, manual_dep=True)
    route_to_packed = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, PREFILL_MOE_ROUTE_MAP_PAD], dtype=pl.INT32, manual_dep=True)
    grouped_y = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, D], dtype=pl.BF16)
    dispatch_tids = pl.array.create(1, pl.TASK_ID)
    expert_tids = pl.array.create(1, pl.TASK_ID)
    with pl.scope():
        grouped_x = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, D], dtype=pl.INT8)
        grouped_scale = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32)
        dispatch_tids[0] = prefill_moe_dispatch(
            indices, x_norm_i8, x_norm_scale, weights,
            grouped_x, grouped_scale, expert_counts, recv_expert_counts, send_counts, route_to_packed,
            recv_meta, arrived, recv_x, data_arrived, recv_aux,
            num_tokens, my_rank, prior_dep, moe_epoch,
        )
        expert_tids[0] = prefill_expert_grouped(
            grouped_x, grouped_scale, expert_counts,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
            grouped_y,
        )
    dispatch_tid = pl.system.task_dummy(deps=[dispatch_tids])
    expert_tid = pl.system.task_dummy(deps=[expert_tids])
    completion_tid = prefill_moe_combine(
        grouped_y, expert_counts, recv_expert_counts, route_to_packed, shared_out, ffn_out,
        routed_y_buf, combine_arrived, num_tokens, my_rank, dispatch_tid, expert_tid, moe_epoch,
    )
    return completion_tid
