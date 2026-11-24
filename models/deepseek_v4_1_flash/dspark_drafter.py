# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""DeepSeek V4.1 DSpark projection, metadata, and three-layer drafter."""

import argparse
import inspect

import pypto.language as pl
import pypto.language.distributed as pld
import torch

import config as C
from config import (
    BLOCK_SIZE, B_DYN, EP_SIZE, FLASH, MOE_TOKENS, TABLE_DYN, WINDOW_CACHE_BYTES,
    DSPARK_DRAFT_LAYERS, DSPARK_N_ROUTED_EXPERTS, DSPARK_NUM_EXPERTS_PER_TOK,
    DSPARK_NOISE_TOKEN_ID, DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH,
)
from rmsnorm import rms_norm
from mega_moe import FP4_PANEL_ROWS, forward_routed_weight_spec, use_packed_decode_weights, workspace_bytes
from expert_routed import (
    MX_PACKED_LANE_COLS, MX_W1_PACKED_ROWS, MX_W2_PACKED_ROWS, MX_W3_PACKED_ROWS,
)
from input_pack import pack_x_hc
from decode_moe import BALANCED_ROUTING, initialize_balanced_routes, make_decode_moe

from dspark_attn import dspark_attn, dspark_context_kv
from qkv_proj_rope import qkv_quant


# Dynamic shape variables.
T_DYN = pl.dynamic("DSPARK_CONTEXT_T_DYN")
DSPARK_QUERY_T_DYN = pl.dynamic("DSPARK_QUERY_T_DYN")

_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--draft-experts-per-rank", type=int, default=DSPARK_N_ROUTED_EXPERTS // EP_SIZE)
_shape_parser.add_argument("--packed-routed-weights", action=argparse.BooleanOptionalAction, default=False)
_shape_parser.add_argument("--runtime-dir", default=None)
_shape_options = _shape_parser.parse_known_args()[0]
if _shape_options.draft_experts_per_rank <= 0 or EP_SIZE * _shape_options.draft_experts_per_rank < DSPARK_NUM_EXPERTS_PER_TOK:
    _shape_parser.error("--draft-experts-per-rank must give at least draft TopK routed experts")


# model config
D = FLASH.hidden_size
HC_MULT = FLASH.hc_mult
HC_DIM = FLASH.hc_dim
MIX_HC = FLASH.mix_hc
HEAD_DIM = FLASH.head_dim
ROPE_DIM = FLASH.qk_rope_head_dim
Q_LORA = FLASH.q_lora_rank
LOCAL_H = FLASH.num_attention_heads
LOCAL_O_GROUPS = FLASH.o_groups
O_LORA = FLASH.o_lora_rank
O_GROUP_IN = FLASH.num_attention_heads * HEAD_DIM // FLASH.o_groups
LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA
VOCAB = FLASH.vocab_size
N_LOCAL_EXPERTS = _shape_options.draft_experts_per_rank
N_EXPERTS = EP_SIZE * N_LOCAL_EXPERTS
MOE_INTER = FLASH.moe_intermediate_size
MX_GROUP = FLASH.weight_block_size[0]
TOPK = DSPARK_NUM_EXPERTS_PER_TOK

MAIN_IN = DSPARK_DRAFT_LAYERS * D
DSPARK_WINDOW = FLASH.sliding_window


# Projection tiles.
PROJ_M_TILE = 80
PROJ_ROWS_PER_LAUNCH = 32 * PROJ_M_TILE
PROJ_N_TILE = 160
PROJ_K_TILE = 256
PROJ_CUBE_K_TILE = 128
PROJ_SCALE_K_TILE = PROJ_CUBE_K_TILE // MX_GROUP
PROJ_L1_BUFFERS = 4
PROJ_L0_BUFFERS = 2
QUANT_M_TILE = 16
QUANT_K_TILE = 1280


MOE_WORKSPACE_BYTES = workspace_bytes(EP_SIZE, N_LOCAL_EXPERTS, MOE_TOKENS, TOPK, D, MOE_INTER)
PACK_ROUTED_WEIGHTS = use_packed_decode_weights(MOE_TOKENS, N_LOCAL_EXPERTS, TOPK) and (
    _shape_options.packed_routed_weights or _shape_options.runtime_dir is not None
)
PANEL_ROWS = FP4_PANEL_ROWS if PACK_ROUTED_WEIGHTS else 1
WEIGHT_PANEL_BYTES = PANEL_ROWS * MX_PACKED_LANE_COLS
W1_DEVICE_ROWS = MX_W1_PACKED_ROWS // PANEL_ROWS
W2_DEVICE_ROWS = MX_W2_PACKED_ROWS // PANEL_ROWS
W3_DEVICE_ROWS = MX_W3_PACKED_ROWS // PANEL_ROWS
draft_moe = make_decode_moe(N_LOCAL_EXPERTS, TOPK, packed_routed_weights=PACK_ROUTED_WEIGHTS)


@pl.jit.inline
def _accumulate_dspark_payload(
    acc: pl.Tile[[PROJ_M_TILE, PROJ_N_TILE], pl.FP32, pl.Mem.Acc],
    a_l1: pl.Tile[[PROJ_M_TILE, PROJ_K_TILE], pl.FP8E4M3FN, pl.Mem.Mat],
    b_l1: pl.Tile[[PROJ_K_TILE, PROJ_N_TILE], pl.FP8E4M3FN, pl.Mem.Mat],
    sa_l1: pl.Tile[[PROJ_M_TILE, MAIN_IN // MX_GROUP], pl.FP8E8M0, pl.Mem.Mat],
    sb_l1: pl.Tile[[MAIN_IN // MX_GROUP, PROJ_N_TILE], pl.FP8E8M0, pl.Mem.Mat],
    group: pl.Scalar[pl.INDEX],
) -> pl.Tile:
    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, group % PROJ_L1_BUFFERS)
    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
    scale_offset = group * (PROJ_K_TILE // MX_GROUP)
    a: pl.Tile[
        [PROJ_M_TILE, PROJ_CUBE_K_TILE], pl.FP8E4M3FN,
        pl.MemRef("a_l0", slots=PROJ_L0_BUFFERS)[0], pl.Mem.Left,
    ] = pl.tile.extract(a_l1, 0, 0, [PROJ_M_TILE, PROJ_CUBE_K_TILE], target_memory=pl.Mem.Left)
    b: pl.Tile[
        [PROJ_CUBE_K_TILE, PROJ_N_TILE], pl.FP8E4M3FN,
        pl.MemRef("b_l0", slots=PROJ_L0_BUFFERS)[0], pl.Mem.Right,
    ] = pl.tile.extract(b_l1, 0, 0, [PROJ_CUBE_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.Right)
    sa = pl.tile.extract(
        sa_l1, 0, scale_offset, [PROJ_M_TILE, PROJ_SCALE_K_TILE], target_memory=pl.Mem.LeftScale, scale_for=a,
    )
    sb = pl.tile.extract(
        sb_l1, scale_offset, 0, [PROJ_SCALE_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.RightScale, scale_for=b,
    )
    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
    if group == 0:
        acc = pl.tile.matmul_mx(a, sa, b, sb)
    else:
        acc = pl.tile.matmul_mx_acc(acc, a, sa, b, sb)
    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
    scale_offset = group * (PROJ_K_TILE // MX_GROUP) + PROJ_SCALE_K_TILE
    a: pl.Tile[
        [PROJ_M_TILE, PROJ_CUBE_K_TILE], pl.FP8E4M3FN,
        pl.MemRef("a_l0", slots=PROJ_L0_BUFFERS)[1], pl.Mem.Left,
    ] = pl.tile.extract(a_l1, 0, PROJ_CUBE_K_TILE, [PROJ_M_TILE, PROJ_CUBE_K_TILE], target_memory=pl.Mem.Left)
    b: pl.Tile[
        [PROJ_CUBE_K_TILE, PROJ_N_TILE], pl.FP8E4M3FN,
        pl.MemRef("b_l0", slots=PROJ_L0_BUFFERS)[1], pl.Mem.Right,
    ] = pl.tile.extract(b_l1, PROJ_CUBE_K_TILE, 0, [PROJ_CUBE_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.Right)
    sa = pl.tile.extract(
        sa_l1, 0, scale_offset, [PROJ_M_TILE, PROJ_SCALE_K_TILE], target_memory=pl.Mem.LeftScale, scale_for=a,
    )
    sb = pl.tile.extract(
        sb_l1, scale_offset, 0, [PROJ_SCALE_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.RightScale, scale_for=b,
    )
    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
    acc = pl.tile.matmul_mx_acc(acc, a, sa, b, sb)
    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
    return acc


@pl.jit.inline
def _dspark_publish_quant(
    source: pl.Tile[[QUANT_M_TILE, QUANT_K_TILE], pl.BF16, pl.Mem.Vec],
    payload: pl.Tensor,
    scales: pl.Tensor,
    row: pl.Scalar[pl.INDEX],
    k: pl.Scalar[pl.INDEX],
):
    padded = pl.fillpad(source, pad_value=pl.PadValue.zero)
    full = pl.set_validshape(padded, QUANT_M_TILE, QUANT_K_TILE)
    data, scale = pl.quant_mx(full, group_axis=1, quant_scale_alg="ocp")
    payload = pl.store(data, [row, k], payload)
    packed = pl.reshape(scale, [1, QUANT_M_TILE * (QUANT_K_TILE // MX_GROUP)])
    offset = row * (MAIN_IN // MX_GROUP) + k // MX_GROUP * QUANT_M_TILE
    scales = pl.store(packed, [0, offset], scales)
    return payload, scales


@pl.jit.inline(auto_scope=False)
def dspark_proj(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.FP8E4M3FN, pl.NZ],
    main_proj_scale: pl.Tensor[[MAIN_IN // MX_GROUP, D], pl.FP8E8M0],
    projected: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Quantize and project each row batch in one A5 MIX launch."""
    target_hidden.bind_dynamic(0, T_DYN)
    projected.bind_dynamic(0, T_DYN)
    weight_scale = pl.tensor.view(main_proj_scale, [MAIN_IN // MX_GROUP, D], layout=pl.MX_B_NN)
    workers = pl.system.available_cluster_count()
    for batch_start in pl.range(0, num_tokens, PROJ_ROWS_PER_LAUNCH):
        batch_rows = pl.min(PROJ_ROWS_PER_LAUNCH, num_tokens - batch_start)
        padded_rows = (batch_rows + PROJ_M_TILE - 1) // PROJ_M_TILE * PROJ_M_TILE
        payload = pl.create_tensor([padded_rows, MAIN_IN], dtype=pl.FP8E4M3FN)
        scales = pl.create_tensor([1, PROJ_ROWS_PER_LAUNCH * (MAIN_IN // MX_GROUP)], dtype=pl.FP8E8M0)
        with pl.spmd(workers, name_hint="dspark_projection_mix", sync_start=True):
            with pl.at(level=pl.Level.CORE_GROUP):
                worker = pl.tile.get_block_idx()
                block_workers = pl.tile.get_block_num()
                quant_tasks = padded_rows // QUANT_M_TILE * (MAIN_IN // QUANT_K_TILE)
                for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    first_task = worker * 2 + lane
                    task_stride = block_workers * 2
                    active_tasks = (batch_rows + QUANT_M_TILE - 1) // QUANT_M_TILE * (MAIN_IN // QUANT_K_TILE)
                    pairs = pl.max(0, (active_tasks - first_task + task_stride - 1) // (2 * task_stride))
                    for pair in pl.range(pairs):
                        work0 = first_task + pair * 2 * task_stride
                        work1 = work0 + task_stride
                        row0 = work0 // (MAIN_IN // QUANT_K_TILE) * QUANT_M_TILE
                        row1 = work1 // (MAIN_IN // QUANT_K_TILE) * QUANT_M_TILE
                        k0 = work0 % (MAIN_IN // QUANT_K_TILE) * QUANT_K_TILE
                        k1 = work1 % (MAIN_IN // QUANT_K_TILE) * QUANT_K_TILE
                        rows0 = pl.min(QUANT_M_TILE, batch_rows - row0)
                        rows1 = pl.min(QUANT_M_TILE, batch_rows - row1)
                        source0: pl.Tile[
                            [QUANT_M_TILE, QUANT_K_TILE], pl.BF16, pl.MemRef("quant_input_ub", slots=2)[0], pl.Mem.Vec,
                        ] = pl.load(
                            target_hidden, [batch_start + row0, k0], [QUANT_M_TILE, QUANT_K_TILE],
                            valid_shape=[rows0, QUANT_K_TILE],
                        )
                        source1: pl.Tile[
                            [QUANT_M_TILE, QUANT_K_TILE], pl.BF16, pl.MemRef("quant_input_ub", slots=2)[1], pl.Mem.Vec,
                        ] = pl.load(
                            target_hidden, [batch_start + row1, k1], [QUANT_M_TILE, QUANT_K_TILE],
                            valid_shape=[rows1, QUANT_K_TILE],
                        )
                        payload, scales = _dspark_publish_quant(source0, payload, scales, row0, k0)
                        payload, scales = _dspark_publish_quant(source1, payload, scales, row1, k1)
                    tail_task = first_task + pairs * 2 * task_stride
                    for task in pl.range(tail_task, quant_tasks, task_stride):
                        row = task // (MAIN_IN // QUANT_K_TILE) * QUANT_M_TILE
                        k = task % (MAIN_IN // QUANT_K_TILE) * QUANT_K_TILE
                        rows = pl.max(0, pl.min(QUANT_M_TILE, batch_rows - row))
                        offset = row * (MAIN_IN // MX_GROUP) + k // MX_GROUP * QUANT_M_TILE
                        if rows > 0:
                            # Keep the next input load independent of the previous quantized writeback.
                            source: pl.Tile[
                                [QUANT_M_TILE, QUANT_K_TILE], pl.BF16,
                                pl.MemRef("quant_input_ub", slots=2)[0], pl.Mem.Vec,
                            ] = pl.load(
                                target_hidden,
                                [batch_start + row, k],
                                [QUANT_M_TILE, QUANT_K_TILE],
                                valid_shape=[rows, QUANT_K_TILE],
                            )
                            payload, scales = _dspark_publish_quant(source, payload, scales, row, k)
                        else:
                            data_bytes = pl.tile.full(
                                [QUANT_M_TILE, QUANT_K_TILE], dtype=pl.INT8, value=0
                            )
                            scale_bytes = pl.tile.full(
                                [1, QUANT_M_TILE * (QUANT_K_TILE // MX_GROUP)],
                                dtype=pl.UINT8,
                                value=0,
                            )
                            payload = pl.store(
                                pl.reinterpret_view(data_bytes, pl.FP8E4M3FN), [row, k], payload
                            )
                            scales = pl.store(
                                pl.reinterpret_view(scale_bytes, pl.FP8E8M0), [0, offset], scales
                            )

                # Publish the quantized activations before Cube reads the workspace.
                pl.system.fence()
                pl.system.syncall(core_type=pl.KernelType.MIX)
                activation_scale = pl.tensor.view(
                    scales, [PROJ_ROWS_PER_LAUNCH, MAIN_IN // MX_GROUP], layout=pl.MX_A_ZZ
                )
                column_waves = (D // PROJ_N_TILE + block_workers - 1 - worker) // block_workers
                for column_wave in pl.range(column_waves):
                    column_tile = worker + column_wave * block_workers
                    col = column_tile * PROJ_N_TILE
                    for row in pl.range(0, batch_rows, PROJ_M_TILE):
                        row_base = batch_start + row
                        # Keep the full K dimension of both scale panels resident in L1.
                        sa_l1 = pl.load(
                            activation_scale, [row, 0], [PROJ_M_TILE, MAIN_IN // MX_GROUP],
                            target_memory=pl.Mem.Mat,
                        )
                        sb_l1 = pl.load(
                            weight_scale, [0, col], [MAIN_IN // MX_GROUP, PROJ_N_TILE],
                            target_memory=pl.Mem.Mat,
                        )
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, PROJ_L1_BUFFERS)
                        pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, PROJ_L1_BUFFERS)
                        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                        pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                        a0: pl.Tile[
                            [PROJ_M_TILE, PROJ_K_TILE], pl.FP8E4M3FN,
                            pl.MemRef("a_l1", slots=PROJ_L1_BUFFERS)[0], pl.Mem.Mat,
                        ] = pl.tile.load(payload, [row, 0], [PROJ_M_TILE, PROJ_K_TILE], target_memory=pl.Mem.Mat)
                        b0: pl.Tile[
                            [PROJ_K_TILE, PROJ_N_TILE], pl.FP8E4M3FN,
                            pl.MemRef("b_l1", slots=PROJ_L1_BUFFERS)[0], pl.Mem.Mat,
                        ] = pl.tile.load(main_proj_weight, [0, col], [PROJ_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.Mat)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 0)
                        a1: pl.Tile[
                            [PROJ_M_TILE, PROJ_K_TILE], pl.FP8E4M3FN,
                            pl.MemRef("a_l1", slots=PROJ_L1_BUFFERS)[1], pl.Mem.Mat,
                        ] = pl.tile.load(payload, [row, PROJ_K_TILE], [PROJ_M_TILE, PROJ_K_TILE], target_memory=pl.Mem.Mat)
                        b1: pl.Tile[
                            [PROJ_K_TILE, PROJ_N_TILE], pl.FP8E4M3FN,
                            pl.MemRef("b_l1", slots=PROJ_L1_BUFFERS)[1], pl.Mem.Mat,
                        ] = pl.tile.load(main_proj_weight, [PROJ_K_TILE, col], [PROJ_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.Mat)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 1)
                        a2: pl.Tile[
                            [PROJ_M_TILE, PROJ_K_TILE], pl.FP8E4M3FN,
                            pl.MemRef("a_l1", slots=PROJ_L1_BUFFERS)[2], pl.Mem.Mat,
                        ] = pl.tile.load(payload, [row, 2 * PROJ_K_TILE], [PROJ_M_TILE, PROJ_K_TILE], target_memory=pl.Mem.Mat)
                        b2: pl.Tile[
                            [PROJ_K_TILE, PROJ_N_TILE], pl.FP8E4M3FN,
                            pl.MemRef("b_l1", slots=PROJ_L1_BUFFERS)[2], pl.Mem.Mat,
                        ] = pl.tile.load(main_proj_weight, [2 * PROJ_K_TILE, col], [PROJ_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.Mat)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 2)
                        a3: pl.Tile[
                            [PROJ_M_TILE, PROJ_K_TILE], pl.FP8E4M3FN,
                            pl.MemRef("a_l1", slots=PROJ_L1_BUFFERS)[3], pl.Mem.Mat,
                        ] = pl.tile.load(payload, [row, 3 * PROJ_K_TILE], [PROJ_M_TILE, PROJ_K_TILE], target_memory=pl.Mem.Mat)
                        b3: pl.Tile[
                            [PROJ_K_TILE, PROJ_N_TILE], pl.FP8E4M3FN,
                            pl.MemRef("b_l1", slots=PROJ_L1_BUFFERS)[3], pl.Mem.Mat,
                        ] = pl.tile.load(main_proj_weight, [3 * PROJ_K_TILE, col], [PROJ_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.Mat)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 3)
                        acc = pl.tile.create([PROJ_M_TILE, PROJ_N_TILE], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                        for group in pl.unroll(MAIN_IN // PROJ_K_TILE):
                            acc = _accumulate_dspark_payload(acc, a0, b0, sa_l1, sb_l1, group)
                            a0, b0 = a1, b1
                            a1, b1 = a2, b2
                            a2, b2 = a3, b3
                            next_group = group + PROJ_L1_BUFFERS
                            if next_group < MAIN_IN // PROJ_K_TILE:
                                slot = group % PROJ_L1_BUFFERS
                                pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                                pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                                a3: pl.Tile[
                                    [PROJ_M_TILE, PROJ_K_TILE], pl.FP8E4M3FN,
                                    pl.MemRef("a_l1", slots=PROJ_L1_BUFFERS)[slot], pl.Mem.Mat,
                                ] = pl.tile.load(payload, [row, next_group * PROJ_K_TILE], [PROJ_M_TILE, PROJ_K_TILE], target_memory=pl.Mem.Mat)
                                b3: pl.Tile[
                                    [PROJ_K_TILE, PROJ_N_TILE], pl.FP8E4M3FN,
                                    pl.MemRef("b_l1", slots=PROJ_L1_BUFFERS)[slot], pl.Mem.Mat,
                                ] = pl.tile.load(main_proj_weight, [next_group * PROJ_K_TILE, col], [PROJ_K_TILE, PROJ_N_TILE], target_memory=pl.Mem.Mat)
                                pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                        pl.system.set_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
                        pl.system.wait_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
                        valid = pl.set_validshape(
                            acc, pl.min(PROJ_M_TILE, num_tokens - row_base), PROJ_N_TILE
                        )
                        projected = pl.store(valid, [row_base, col], projected)
                        pl.system.barrier(pipe=pl.PipeType.ALL)
    return projected


@pl.jit.inline(auto_scope=False)
def dspark_metadata(
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    query_rope_cos: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_sin: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    padded_ids: pl.Out[pl.Tensor[[MOE_TOKENS], pl.INT64]],
    padded_slots: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_TOKENS], pl.INT64]],
    padded_indices: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32]],
    rope_cos: pl.Out[pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32]],
    rope_sin: pl.Out[pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32]],
    pre_mix: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32]],
):
    """Build query metadata with 64 row/layer workers and local-page VF gathers."""
    batch = pl.tensor.dim(anchor_ids, 0)
    table_blocks = pl.tensor.dim(block_tables, 2)
    flat_padded_ids = pl.reshape(padded_ids, [1, MOE_TOKENS])
    flat_indices = pl.reshape(padded_indices, [DSPARK_DRAFT_LAYERS * MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH])
    for core in pl.spmd(64, name_hint="dspark_metadata"):
        for work in pl.range(core, MOE_TOKENS * DSPARK_DRAFT_LAYERS, 64):
            layer = work % DSPARK_DRAFT_LAYERS
            row = work // DSPARK_DRAFT_LAYERS
            request = row // DSPARK_QUERY_WIDTH
            first = pl.cast(0, pl.INT32)
            if request < batch:
                first = pl.read(first_query_positions, [request])
            buffer = pl.tile.full([1, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32, value=-1)
            buffer = pl.tile.move(buffer, target_memory=pl.Mem.Vec)
            indices = pl.reshape(buffer, [DSPARK_SWA_INDEX_WIDTH])
            if first > 0 and table_blocks > 0:
                prefix = pl.max(first - DSPARK_WINDOW, 0)
                visible = first - prefix + DSPARK_QUERY_WIDTH
                first_page = prefix // BLOCK_SIZE
                page_count = (first + DSPARK_QUERY_WIDTH - 1) // BLOCK_SIZE - first_page + 1
                pages = pl.create_tile([1, 8], dtype=pl.INT32, target_memory=pl.Mem.Vec)
                for offset in pl.range(page_count):
                    page = (first_page + offset) % table_blocks
                    physical = pl.read(block_tables, [layer, request, page])
                    pl.tile.write(pages, [0, offset], physical)
                page_table = pl.reshape(pages, [8])
                prefix_u = pl.cast(prefix, pl.UINT32)
                visible_u = pl.cast(visible, pl.UINT32)
                first_page_u = pl.cast(first_page, pl.UINT32)
                with pl.reg_scope():
                    r_full = pl.reg.full_mask(lanes=64)
                    r_lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
                    r_low = pl.reg.broadcast(BLOCK_SIZE - 1, lanes=64, dtype=pl.UINT32)
                    r_first_page = pl.reg.broadcast(first_page_u, lanes=64, dtype=pl.UINT32)
                    r_empty = pl.reg.broadcast(-1, lanes=64, dtype=pl.INT32)
                    for segment in pl.hardware_loop(DSPARK_SWA_INDEX_WIDTH // 64):
                        r_column = pl.reg.adds(r_lane, pl.cast(segment * 64, pl.UINT32), r_full)
                        r_position = pl.reg.adds(r_column, prefix_u, r_full)
                        r_page = pl.reg.shift_rights(r_position, 7, r_full)
                        r_page = pl.reg.sub(r_page, r_first_page, r_full)
                        r_valid = pl.reg.compares(r_column, visible_u, r_full, predicate="lt")
                        r_physical = pl.reg.gather(page_table, r_page, r_valid)
                        r_base = pl.reg.muls(r_physical, BLOCK_SIZE, r_full)
                        r_offset_u = pl.reg.bitwise_and(r_position, r_low, r_full)
                        r_offset_i = pl.reg.interpret_cast(r_offset_u, dtype=pl.INT32)
                        r_slot = pl.reg.add(r_base, r_offset_i, r_full)
                        r_result = pl.reg.select(r_valid, r_slot, r_empty)
                        _written = pl.reg.store(indices, segment * 64, r_result)
            slot = pl.cast(-1, pl.INT64)
            if first > 0 and table_blocks > 0:
                position = first + row % DSPARK_QUERY_WIDTH
                page = position // BLOCK_SIZE % table_blocks
                physical = pl.read(block_tables, [layer, request, page])
                slot = pl.cast(physical * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT64)
            slot_tile = pl.tile.create([1, 4], pl.INT64)
            pl.tile.write(slot_tile, [0, 0], slot)
            pl.store(pl.set_validshape(slot_tile, 1, 1), [layer, row], padded_slots)
            index_row = pl.reshape(indices, [1, DSPARK_SWA_INDEX_WIDTH])
            pl.store(index_row, [layer * MOE_TOKENS + row, 0], flat_indices)
            if layer == 0:
                token = pl.cast(0, pl.INT64)
                if first > 0 and table_blocks > 0:
                    token = pl.cast(DSPARK_NOISE_TOKEN_ID, pl.INT64)
                    if row % DSPARK_QUERY_WIDTH == 0:
                        token = pl.read(anchor_ids, [request])
                token_tile = pl.tile.create([1, 4], pl.INT64)
                pl.tile.write(token_tile, [0, 0], token)
                pl.store(pl.set_validshape(token_tile, 1, 1), [0, row], flat_padded_ids)
                cos = pl.full([1, ROPE_DIM // 2], dtype=pl.FP32, value=0.0)
                sin = pl.full([1, ROPE_DIM // 2], dtype=pl.FP32, value=0.0)
                if row < batch * DSPARK_QUERY_WIDTH:
                    cos = query_rope_cos[row : row + 1, :]
                    sin = query_rope_sin[row : row + 1, :]
                rope_cos[row : row + 1, :] = cos
                rope_sin[row : row + 1, :] = sin
                mix = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                pl.tile.write(mix, [0, 0], pl.cast(1.0, pl.FP32))
                pl.store(pl.set_validshape(mix, 1, HC_MULT), [row, 0], pre_mix)
    return padded_ids, padded_slots, padded_indices, rope_cos, rope_sin, pre_mix


@pl.jit.inline(auto_scope=False)
def dspark_drafter(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.FP8E4M3FN, pl.NZ],
    main_proj_scale: pl.Tensor[[MAIN_IN // MX_GROUP, D], pl.FP8E8M0],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    context_slots: pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64],
    context_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    context_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_cos: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_sin: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN // MX_GROUP, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    window_cache: pl.InOut[pl.Tensor[[DSPARK_DRAFT_LAYERS, C.ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    gate_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, W1_DEVICE_ROWS, WEIGHT_PANEL_BYTES], pl.UINT8],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, W2_DEVICE_ROWS, WEIGHT_PANEL_BYTES], pl.UINT8],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, W3_DEVICE_ROWS, WEIGHT_PANEL_BYTES], pl.UINT8],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    draft_x_hc: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    draft_pre_mix: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32]],
    moe_workspace: pld.DistributedTensor,
    rank: pl.Scalar[pl.INT32],
):
    """Compose three draft layers and publish their final HC state and coefficients."""
    target_hidden.bind_dynamic(0, T_DYN)
    context_slots.bind_dynamic(1, T_DYN)
    context_rope_cos.bind_dynamic(0, T_DYN)
    context_rope_sin.bind_dynamic(0, T_DYN)
    query_rope_cos.bind_dynamic(0, DSPARK_QUERY_T_DYN)
    query_rope_sin.bind_dynamic(0, DSPARK_QUERY_T_DYN)
    num_sampled.bind_dynamic(0, B_DYN)
    last_sampled.bind_dynamic(0, B_DYN)
    next_prefill_tokens.bind_dynamic(0, B_DYN)
    anchor_ids.bind_dynamic(0, B_DYN)
    first_query_positions.bind_dynamic(0, B_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, TABLE_DYN)
    window_cache.bind_dynamic(1, C.ORI_BLOCKS_DYN)
    context_tokens = pl.tensor.dim(target_hidden, 0)
    main_x = pl.create_tensor([context_tokens, D], dtype=pl.BF16)
    projected = pl.create_tensor([context_tokens, D], dtype=pl.BF16)
    dspark_proj(target_hidden, main_proj_weight, main_proj_scale, projected, context_tokens)
    rms_norm(projected, main_norm_weight, main_x, context_tokens, width=D, zero_tail=False)

    context_payload, context_scales, context_padded_rows = qkv_quant(main_x, context_tokens, row_tile=80)
    for layer in pl.range(DSPARK_DRAFT_LAYERS):
        layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[layer]
        dspark_context_kv(
            context_payload, context_scales,
            wkv[layer], layer_wkv_scale, kv_norm_weight[layer],
            context_rope_cos, context_rope_sin,
            context_slots[layer], window_cache[layer],
            context_tokens,
        )

    # MOE_TOKENS must cover batch * DSPARK_QUERY_WIDTH query rows.
    padded_ids = pl.create_tensor([MOE_TOKENS], dtype=pl.INT64)
    padded_slots = pl.create_tensor([DSPARK_DRAFT_LAYERS, MOE_TOKENS], dtype=pl.INT64)
    padded_indices = pl.create_tensor([DSPARK_DRAFT_LAYERS, MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], dtype=pl.INT32)
    rope_cos = pl.create_tensor([MOE_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
    rope_sin = pl.create_tensor([MOE_TOKENS, ROPE_DIM // 2], dtype=pl.FP32)
    pre_mix = pl.reshape(draft_pre_mix, [MOE_TOKENS, HC_MULT])
    dspark_metadata(
        anchor_ids, first_query_positions, block_tables, query_rope_cos, query_rope_sin,
        padded_ids, padded_slots, padded_indices, rope_cos, rope_sin, pre_mix,
    )
    x_hc = pl.reshape(draft_x_hc, [MOE_TOKENS, HC_MULT, D])
    pack_x_hc(padded_ids, embedding_weight, x_hc, hidden_tile=512)

    batch = pl.tensor.dim(num_sampled, 0)
    query_tokens = batch * DSPARK_QUERY_WIDTH
    force_topk = pl.create_tensor([MOE_TOKENS, TOPK], dtype=pl.INT32)
    if BALANCED_ROUTING:
        initialize_balanced_routes(force_topk, query_tokens, rank, N_LOCAL_EXPERTS, TOPK)
    for layer in pl.range(DSPARK_DRAFT_LAYERS):
        layer_wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[layer]
        layer_wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[layer]
        layer_wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[layer]
        layer_wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[layer]
        layer_routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w1_scale[layer]
        layer_routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0, pl.MX_B_NN] = routed_w2_scale[layer]
        layer_routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = routed_w3_scale[layer]
        layer_shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w1_scale[layer]
        layer_shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = shared_w2_scale[layer]
        layer_shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = shared_w3_scale[layer]
        x_mixed = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
        attention_pre_mix = pl.create_tensor([MOE_TOKENS, HC_MULT], dtype=pl.FP32)
        attention_hidden = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
        dspark_attn(
            x_hc, pre_mix,
            hc_attn_fn[layer], hc_attn_scale[layer], hc_attn_base[layer], attn_norm_weight[layer],
            wq_a[layer], layer_wq_a_scale, q_norm_weight[layer], wq_b[layer], layer_wq_b_scale,
            wkv[layer], layer_wkv_scale, kv_norm_weight[layer], attn_sink[layer],
            wo_a[layer], wo_a_scale[layer], wo_b[layer], layer_wo_b_scale,
            rope_cos, rope_sin, padded_slots[layer], padded_indices[layer], window_cache[layer],
            attention_pre_mix, attention_hidden,
            query_tokens,
        )

        draft_moe(
            attention_hidden, attention_pre_mix,
            hc_ffn_fn[layer], hc_ffn_scale[layer], hc_ffn_base[layer],
            ffn_norm_weight[layer],
            gate_weight[layer], correction_bias[layer],
            routed_w1[layer], layer_routed_w1_scale,
            routed_w2[layer], layer_routed_w2_scale,
            routed_w3[layer], layer_routed_w3_scale,
            shared_w1[layer], layer_shared_w1_scale,
            shared_w2[layer], layer_shared_w2_scale,
            shared_w3[layer], layer_shared_w3_scale,
            pre_mix, x_mixed, x_hc,
            moe_workspace, query_tokens, force_topk,
        )


@pl.jit(auto_scope=False)
def l2_dspark_drafter(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.FP8E4M3FN, pl.NZ],
    main_proj_scale: pl.Tensor[[MAIN_IN // MX_GROUP, D], pl.FP8E8M0],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    num_sampled: pl.Tensor[[B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[B_DYN], pl.INT64],
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    context_slots: pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64],
    context_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    context_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_cos: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_sin: pl.Tensor[[DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    embedding_weight: pl.Tensor[[VOCAB, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN // MX_GROUP, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    window_cache: pl.InOut[pl.Tensor[[DSPARK_DRAFT_LAYERS, C.ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    hc_ffn_fn: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, D], pl.BF16],
    gate_weight: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, W1_DEVICE_ROWS, WEIGHT_PANEL_BYTES], pl.UINT8],
    routed_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, W2_DEVICE_ROWS, WEIGHT_PANEL_BYTES], pl.UINT8],
    routed_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, W3_DEVICE_ROWS, WEIGHT_PANEL_BYTES], pl.UINT8],
    routed_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    draft_x_hc: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    draft_pre_mix: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32]],
    moe_workspace: pld.DistributedTensor,
    rank: pl.Scalar[pl.INT32],
):
    dspark_drafter(
        target_hidden, main_proj_weight, main_proj_scale, main_norm_weight,
        num_sampled, last_sampled, next_prefill_tokens,
        anchor_ids, first_query_positions,
        block_tables, context_slots,
        context_rope_cos, context_rope_sin,
        query_rope_cos, query_rope_sin,
        embedding_weight,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale,
        q_norm_weight,
        wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale,
        wo_b, wo_b_scale,
        window_cache,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        ffn_norm_weight,
        gate_weight, correction_bias,
        routed_w1, routed_w1_scale,
        routed_w2, routed_w2_scale,
        routed_w3, routed_w3_scale,
        shared_w1, shared_w1_scale,
        shared_w2, shared_w2_scale,
        shared_w3, shared_w3_scale,
        draft_x_hc, draft_pre_mix,
        moe_workspace, rank,
    )


@pl.jit.host
def l3_dspark_drafter(
    target_hidden: pl.Tensor[[C.EP_SIZE, T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[C.EP_SIZE, MAIN_IN, D], pl.FP8E4M3FN, pl.NZ],
    main_proj_scale: pl.Tensor[[C.EP_SIZE, MAIN_IN // MX_GROUP, D], pl.FP8E8M0],
    main_norm_weight: pl.Tensor[[C.EP_SIZE, D], pl.BF16],
    num_sampled: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT32],
    last_sampled: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    next_prefill_tokens: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    anchor_ids: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[C.EP_SIZE, B_DYN], pl.INT32],
    block_tables: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    context_slots: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64],
    context_rope_cos: pl.Tensor[[C.EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    context_rope_sin: pl.Tensor[[C.EP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_cos: pl.Tensor[[C.EP_SIZE, DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    query_rope_sin: pl.Tensor[[C.EP_SIZE, DSPARK_QUERY_T_DYN, ROPE_DIM // 2], pl.FP32],
    embedding_weight: pl.Tensor[[C.EP_SIZE, VOCAB, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    wq_a: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_GROUPS, O_GROUP_IN // MX_GROUP, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    window_cache: pl.InOut[pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, C.ORI_BLOCKS_DYN, BLOCK_SIZE, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    hc_ffn_fn: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D], pl.BF16],
    gate_weight: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, W1_DEVICE_ROWS, WEIGHT_PANEL_BYTES], pl.UINT8],
    routed_w1_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    routed_w2: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, W2_DEVICE_ROWS, WEIGHT_PANEL_BYTES], pl.UINT8],
    routed_w2_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (MOE_INTER // MX_GROUP), D], pl.FP8E8M0],
    routed_w3: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS, W3_DEVICE_ROWS, WEIGHT_PANEL_BYTES], pl.UINT8],
    routed_w3_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, N_LOCAL_EXPERTS * (D // MX_GROUP), MOE_INTER], pl.FP8E8M0],
    shared_w1: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, MOE_INTER // MX_GROUP, D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[C.EP_SIZE, DSPARK_DRAFT_LAYERS, D // MX_GROUP, MOE_INTER], pl.FP8E8M0],
    draft_x_hc: pl.Out[pl.Tensor[[C.EP_SIZE, MOE_TOKENS, HC_MULT, D], pl.FP32]],
    draft_pre_mix: pl.Out[pl.Tensor[[C.EP_SIZE, MOE_TOKENS, HC_MULT], pl.FP32]],
    moe_epoch: pl.Scalar[pl.INT32],
):
    target_hidden.bind_dynamic(1, T_DYN)
    context_slots.bind_dynamic(2, T_DYN)
    context_rope_cos.bind_dynamic(1, T_DYN)
    context_rope_sin.bind_dynamic(1, T_DYN)
    query_rope_cos.bind_dynamic(1, DSPARK_QUERY_T_DYN)
    query_rope_sin.bind_dynamic(1, DSPARK_QUERY_T_DYN)
    num_sampled.bind_dynamic(1, B_DYN)
    last_sampled.bind_dynamic(1, B_DYN)
    next_prefill_tokens.bind_dynamic(1, B_DYN)
    anchor_ids.bind_dynamic(1, B_DYN)
    first_query_positions.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, B_DYN)
    block_tables.bind_dynamic(3, TABLE_DYN)
    window_cache.bind_dynamic(2, C.ORI_BLOCKS_DYN)

    moe_workspace_buf = pld.alloc_window_buffer([MOE_WORKSPACE_BYTES], dtype=pl.UINT8)
    for rank in pl.range(pld.world_size()):
        moe_workspace = pld.window(moe_workspace_buf, [MOE_WORKSPACE_BYTES], dtype=pl.UINT8)
        l2_dspark_drafter(
            target_hidden[rank], main_proj_weight[rank], main_proj_scale[rank], main_norm_weight[rank],
            num_sampled[rank], last_sampled[rank], next_prefill_tokens[rank],
            anchor_ids[rank], first_query_positions[rank],
            block_tables[rank], context_slots[rank],
            context_rope_cos[rank], context_rope_sin[rank],
            query_rope_cos[rank], query_rope_sin[rank],
            embedding_weight[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], wq_a_scale[rank],
            q_norm_weight[rank],
            wq_b[rank], wq_b_scale[rank],
            wkv[rank], wkv_scale[rank], kv_norm_weight[rank],
            attn_sink[rank],
            wo_a[rank], wo_a_scale[rank],
            wo_b[rank], wo_b_scale[rank],
            window_cache[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            ffn_norm_weight[rank],
            gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            draft_x_hc[rank], draft_pre_mix[rank],
            moe_workspace, rank,
            device=rank,
        )


def build_drafter_specs():
    """One request on every DP rank, with nonzero embeddings and two context writes."""
    from golden import ScalarSpec

    dynamic_sizes = {
        "DSPARK_CONTEXT_T_DYN": 6,
        "B_DYN": 1,
        "DSPARK_QUERY_T_DYN": DSPARK_QUERY_WIDTH,
        "TABLE_DYN": 1,
        "ORI_BLOCKS_DYN": 1,
    }
    dtypes = {
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp8e4m3fn": torch.float8_e4m3fn,
        "fp8e8m0": torch.float8_e8m0fnu,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }

    initial_values = {
        "embedding_weight": 0.25,
        "num_sampled": 1, "last_sampled": 7, "anchor_ids": 7,
        "first_query_positions": 2,
        "context_rope_cos": 1, "query_rope_cos": 1,
        "target_hidden": 0.25, "main_norm_weight": 1,
    }

    def initial_value(name, shape, dtype):
        if name == "context_slots":
            value = torch.full(shape, -1, dtype=dtype)
            value[:, :, 0] = 32
            value[:, :, 1] = 33
            return value
        if name == "window_cache":
            value = torch.zeros(shape, dtype=torch.uint8)
            value[..., HEAD_DIM:].view(torch.bfloat16).fill_(1.0)
            return value
        if dtype == torch.float8_e8m0fnu:
            return lambda: torch.full(shape, 127, dtype=torch.uint8).view(dtype)
        return initial_values.get(name)

    specs = []
    for parameter in inspect.signature(l3_dspark_drafter._func).parameters.values():
        annotation = parameter.annotation
        if hasattr(annotation, "shape"):
            shape = [size if isinstance(size, int) else dynamic_sizes[size.name] for size in annotation.shape]
            if parameter.name == "main_proj_weight":
                shape = [shape[0], D // 32, MAIN_IN // 16, 16, 32]
            dtype = dtypes[str(annotation.dtype)]
            init = initial_value(parameter.name, shape, dtype)
            specs.append(forward_routed_weight_spec(parameter.name, shape, dtype, init, PACK_ROUTED_WEIGHTS))
        else:
            specs.append(ScalarSpec(parameter.name, torch.int32, 1, compile_runtime=True))
    return specs


if __name__ == "__main__":
    from golden import run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="V4.1 DSpark drafter validation")
    parser.add_argument("--ep", type=int, default=C.EP_SIZE, choices=C.SUPPORTED_EP_SIZES)
    parser.add_argument("--draft-experts-per-rank", type=int, default=N_LOCAL_EXPERTS)
    parser.add_argument("-p", "--platform", choices=("a5",), default="a5")
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(EP_SIZE)))
    parser.add_argument("--packed-routed-weights", action=argparse.BooleanOptionalAction, default=False, help="pack resident routed FP4 weights before upload")
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(device) for device in args.device.split(",")]
    if len(devices) != EP_SIZE:
        parser.error(f"drafter needs {EP_SIZE} devices")

    result = run(
        a5_default_runtime="host_build_graph",
        fn=l3_dspark_drafter,
        specs=build_drafter_specs(),
        golden_fn=None,
        config={
            "backend_type": args.platform,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            "execution_mode": False,
            "ring_heap": (2 * 1024**3, 2 * 1024**3, 1024**3, 512 * 1024**2),
        },
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
