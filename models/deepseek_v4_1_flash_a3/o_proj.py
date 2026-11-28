# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 inverse RoPE, grouped BF16 output projections and TP reductions."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"


from .gemm import PREFILL_BF16_K_SUBTILE
from .hc import hc_post_collapse, hc_post_collapse_tile, hc_post_token_tile
from .rope import make_pipelined_rope
from .qkv_proj_rope import _rotate_rope

import torch


from .qkv_proj_rope import golden_rope_interleave as rope_interleave


import pypto.language as pl
import pypto.language.distributed as pld

from .config import PREFILL_MAX_TOKENS, FLASH as M, TP_SIZE
from .dsl_types import SP_T_DYN, T_DYN


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim


# tiling
O_A_N_TILE = 128
O_B_N_TILE = 128
DECODE_B1_O_B_N_TILE = 256
DECODE_ORDERED_K_TILE = 64
DECODE_ORDERED_N_TILE = 128
DECODE_B1_O_A_K_TILE = 256
K_TILE = 128
PREFILL_O_M_TILE = 128
PREFILL_O_N_TILE = 256
TP_COLUMNS = 1280
TP_ROW_TILE = 8

rotate_output = make_pipelined_rope(
    T_DYN, LOCAL_H, HEAD_DIM, inverse=True, name_hint="inverse_rope_o", allow_early_resolve=False, inplace=True,
)
rotate_decode_b1 = make_pipelined_rope(
    T_DYN, LOCAL_H, HEAD_DIM, inverse=True, name_hint="inverse_rope_o_b1", allow_early_resolve=False,
)


NZ_WEIGHT_K = pl.dynamic("NZ_WEIGHT_K")
NZ_WEIGHT_N = pl.dynamic("NZ_WEIGHT_N")


def make_gemm_bf16_native(*, k_subtile):
    @pl.jit.inline(auto_scope=False)
    def gemm_bf16_native(
        source: pl.Tensor,
        weight: pl.Tensor[[NZ_WEIGHT_K, NZ_WEIGHT_N], pl.BF16, pl.NZ],
        output: pl.Tensor,
        num_tokens: pl.Scalar[pl.INT32],
        width: pl.constexpr,
        output_width: pl.constexpr,
        n_tile: pl.constexpr,
        k_tile: pl.constexpr,
        workers: pl.constexpr,
        name_hint: pl.constexpr,
        source_col: pl.Scalar[pl.INT32],
        output_col: pl.Scalar[pl.INT32],
        row_tile: pl.constexpr,
        stages: pl.constexpr,
    ):
        """Project BF16 activations against packed Cube weights."""
        col_blocks = output_width // n_tile
        row_blocks = (num_tokens + row_tile - 1) // row_tile
        worker_limit = pl.min(workers, pl.system.available_cluster_count())
        task_workers = pl.max(1, pl.min(row_blocks * col_blocks, worker_limit))
        with pl.spmd(task_workers, name_hint=name_hint):
            worker = pl.tile.get_block_idx()
            if col_blocks == 4:
                inverse = 1
            else:
                inverse = 5
            for panel in pl.range(0, col_blocks):
                col = panel * n_tile
                first_row = (worker + task_workers - (panel * row_blocks) % task_workers) % task_workers
                row_step = task_workers
                if k_subtile > 0 and num_tokens >= 768 and task_workers == 24:
                    # With 24 workers, the 4 or 20 output panels recur every six row tiles.
                    first_row = row_blocks
                    if worker % 4 == panel % 4:
                        first_row = (((worker + 24 - panel) % 24) // 4 * inverse) % 6
                    row_step = 6
                for block in pl.range(first_row, row_blocks, row_step):
                    row = block * row_tile
                    rows = pl.max(0, pl.min(row_tile, num_tokens - row))
                    cube_seed = pl.tile.create([row_tile, n_tile], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, compact=True)
                    cube_acc = pl.set_validshape(cube_seed, rows, n_tile)
                    for k in pl.pipeline(0, width, k_tile, stage=stages):
                        if k_subtile > 0:
                            a = pl.load(source, [row, source_col + k], [row_tile, k_tile],
                                        valid_shape=[rows, k_tile], target_memory=pl.Mem.Mat)
                            b = pl.load(weight, [k, col], [k_tile, n_tile], target_memory=pl.Mem.Mat)
                            for inner in pl.pipeline(0, k_tile, k_subtile, stage=2):
                                af = pl.tile.slice(a, [row_tile, k_subtile], [0, inner])
                                bf = pl.tile.slice(b, [k_subtile, n_tile], [inner, 0])
                                cube_acc = pl.matmul_acc(cube_acc, af, bf, init_cond=(k + inner == 0))
                        else:
                            cube_a = pl.load(source, [row, source_col + k], [row_tile, k_tile], valid_shape=[rows, k_tile])
                            cube_w = pl.load(weight, [k, col], [k_tile, n_tile], target_memory=pl.MemorySpace.Mat)
                            cube_acc = pl.matmul_acc(cube_acc, cube_a, cube_w, init_cond=(k == 0))
                    cube_value = pl.set_validshape(cube_acc, rows, n_tile)
                    output = pl.store(cube_value, [row, output_col + col], output)
        return output

    return gemm_bf16_native


gemm_bf16_native = make_gemm_bf16_native(k_subtile=0)
prefill_bf16_native = make_gemm_bf16_native(k_subtile=PREFILL_BF16_K_SUBTILE)


def make_native_project_output(*, prefill, b1_rotated):
    @pl.jit.inline(auto_scope=False)
    def project_output(
        attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, D], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        """Apply inverse RoPE and grouped BF16 O-A/O-B projections. Prefill consumes attention scratch."""
        tokens = pl.tensor.dim(output, 0)
        if prefill:
            rotated = rotate_output(attended, rope_cos, rope_sin, attended, num_tokens)
        elif b1_rotated:
            rotated = attended
        else:
            rotated = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
            if num_tokens == 1:
                rotated = rotate_decode_b1(attended, rope_cos, rope_sin, rotated, num_tokens)
            else:
                rotated = _rotate_rope(
                    attended, rotated, rope_cos, rope_sin, 0, num_tokens,
                    LOCAL_H, HEAD_DIM, "inverse_rope_o", True, True,
                )
        projected = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
        for group in pl.parallel(LOCAL_O_GROUPS):
            source_col = pl.cast(group * O_GROUP_IN, pl.INT32)
            output_col = pl.cast(group * O_LORA, pl.INT32)
            if prefill and num_tokens >= PREFILL_O_M_TILE:
                prefill_bf16_native(
                    rotated, wo_a[group], projected, num_tokens,
                    O_GROUP_IN, O_LORA, PREFILL_O_N_TILE, K_TILE, 24, "proj_o_a",
                    source_col, output_col, PREFILL_O_M_TILE, stages=2,
                )
            elif not prefill and num_tokens == 1:
                gemm_bf16_native(
                    rotated, wo_a[group], projected, num_tokens,
                    O_GROUP_IN, O_LORA, O_A_N_TILE, DECODE_B1_O_A_K_TILE, 24, "proj_o_a",
                    source_col, output_col, 16, stages=2,
                )
            elif num_tokens >= 64:
                gemm_bf16_native(
                    rotated, wo_a[group], projected, num_tokens,
                    O_GROUP_IN, O_LORA, 256, K_TILE, 24, "proj_o_a",
                    source_col, output_col, 64, stages=1,
                )
            else:
                gemm_bf16_native(
                    rotated, wo_a[group], projected, num_tokens,
                    O_GROUP_IN, O_LORA, O_A_N_TILE, K_TILE, 24, "proj_o_a",
                    source_col, output_col, 16, stages=1,
                )
        if not prefill and num_tokens == 1:
            gemm_bf16_native(
                projected, wo_b, output, num_tokens,
                LOCAL_O_WIDTH, D, DECODE_B1_O_B_N_TILE, K_TILE, 24, "proj_o_b_b1",
                0, 0, 16, stages=2,
            )
        elif prefill and num_tokens >= PREFILL_O_M_TILE:
            prefill_bf16_native(
                projected, wo_b, output, num_tokens,
                LOCAL_O_WIDTH, D, PREFILL_O_N_TILE, K_TILE, 24, "proj_o_b", 0, 0, PREFILL_O_M_TILE, stages=2,
            )
        elif num_tokens >= 64:
            gemm_bf16_native(
                projected, wo_b, output, num_tokens,
                LOCAL_O_WIDTH, D, 256, K_TILE, 24, "proj_o_b", 0, 0, 64, stages=1,
            )
        else:
            gemm_bf16_native(
                projected, wo_b, output, num_tokens,
                LOCAL_O_WIDTH, D, O_B_N_TILE, K_TILE, 24, "proj_o_b",
                0, 0, 16, stages=1,
            )
        return output

    return project_output


prefill_o_proj_nz = make_native_project_output(prefill=True, b1_rotated=False)
_decode_o_proj_nz_rotated = make_native_project_output(prefill=False, b1_rotated=True)
decode_o_proj_nz = make_native_project_output(prefill=False, b1_rotated=False)


@pl.jit.inline(auto_scope=False)
def decode_o_proj_nz_from_attention(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    if num_tokens == 1:
        _decode_o_proj_nz_rotated(attended, wo_a, wo_b, rope_cos, rope_sin, output, num_tokens)
    else:
        decode_o_proj_nz(attended, wo_a, wo_b, rope_cos, rope_sin, output, num_tokens)
    return output


@pl.jit.inline(auto_scope=False)
def _project_ordered_decode_rows(
    source: pl.Tensor,
    weight: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    source_col: pl.Scalar[pl.INT32],
    output_col: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    output_width: pl.constexpr,
    k_tile: pl.constexpr,
    n_tile: pl.constexpr,
    name_hint: pl.constexpr,
):
    """Accumulate each output panel in increasing K order in FP32."""
    panels = output_width // n_tile
    work = num_tokens * panels
    workers = pl.min(work, pl.system.available_cluster_count())
    for worker in pl.spmd(workers, name_hint=name_hint):
        for block in pl.range(worker, work, pl.tile.get_block_num()):
            row = block // panels
            col = (block % panels) * n_tile
            total = pl.tile.full([1, n_tile], dtype=pl.FP32, value=0.0)
            for first in pl.pipeline(0, width, k_tile, stage=2):
                value = pl.cast(pl.load(source, [row, source_col + first], [1, k_tile]), pl.FP32)
                matrix = pl.cast(pl.load(weight, [first, col], [k_tile, n_tile]), pl.FP32)
                column = pl.reshape(value, [k_tile, 1])
                products = pl.row_expand_mul(matrix, column)
                for inner in pl.range(k_tile):
                    product = pl.tile.slice(products, [1, n_tile], [inner, 0])
                    total = pl.add(total, product)
            pl.store(pl.cast(total, pl.BF16, mode="rint"), [row, output_col + col], output)
    return output


@pl.jit.inline(auto_scope=False)
def decode_swa_o_proj_ordered(
    attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Apply the ordered SWA projections independently to each active Decode token."""
    tokens = pl.tensor.dim(attended, 0)
    rotated = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    rotated = rotate_decode_b1(attended, rope_cos, rope_sin, rotated, num_tokens)
    projected = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
    for group in pl.parallel(LOCAL_O_GROUPS):
        _project_ordered_decode_rows(
            rotated, wo_a[group], projected, num_tokens,
            pl.cast(group * O_GROUP_IN, pl.INT32), pl.cast(group * O_LORA, pl.INT32),
            O_GROUP_IN, O_LORA, DECODE_ORDERED_K_TILE, DECODE_ORDERED_N_TILE, "proj_o_a_ordered",
        )
    output = _project_ordered_decode_rows(
        projected, wo_b, output, num_tokens, 0, 0,
        LOCAL_O_WIDTH, D, DECODE_ORDERED_K_TILE, DECODE_ORDERED_N_TILE, "proj_o_b_ordered",
    )
    return output


@pl.jit.inline(auto_scope=False)
def retire_prefill_signals(
    completion_anchor: pl.Tensor,
    arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Clear ready and consumed signals at a serialized entry boundary."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_signal_retire", allow_early_resolve=False):
        anchor = pl.read(completion_anchor, [0, 0, 0])
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                arrived, peer=group_base + tp_rank, offsets=[peer, 0],
                value=0, op=pld.NotifyOp.Set,
            )
        pl.write(completion_anchor, [0, 0, 0], anchor)
    return completion_anchor


@pl.jit.inline(auto_scope=False)
def prefill_sp_output_reduce_scatter(
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Sum BF16 head-TP contributions in FP32 for this rank's token rows."""
    local_tokens = pl.tensor.dim(output, 0)
    output_offset = pl.cast(tp_rank * local_tokens, pl.INDEX)
    return _tp_output_all_reduce(
        partial, output_window, output_arrived, output, group_base, tp_rank,
        num_tokens, attention_epoch, output_offset, False, TP_ROW_TILE,
    )


@pl.jit.inline(auto_scope=False)
def prefill_tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor,
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    owner_rows: pl.constexpr,
):
    return _tp_output_all_reduce(
        output_partial, output_window, output_arrived, output, group_base, tp_rank,
        num_tokens, attention_epoch, pl.cast(0, pl.INDEX), owner_rows, 1 if owner_rows else TP_ROW_TILE,
    )


@pl.jit.inline(auto_scope=False)
def _tp_output_publish(
    output_partial: pl.Tensor,
    output_window: pld.DistributedTensor,
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    owner_rows: pl.constexpr,
    tile_rows: pl.constexpr,
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_reuse", allow_early_resolve=False) as reuse_tid:
        for peer in pl.range(TP_SIZE):
            previous_epoch = (attention_epoch - 1) * 2
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
    publish_rows = num_tokens
    if owner_rows:
        publish_rows = (num_tokens + TP_SIZE - 1 - tp_rank) // TP_SIZE
    column_tiles = D // TP_COLUMNS
    publish_tiles = (publish_rows + tile_rows - 1) // tile_rows * column_tiles
    publish_workers = pl.max(1, pl.min(publish_tiles, pl.system.available_aiv_count()))
    with pl.spmd(publish_workers, name_hint="prefill_tp_publish", deps=[reuse_tid]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, publish_tiles, publish_workers):
            first = tile // column_tiles * tile_rows
            live = pl.min(tile_rows, publish_rows - first)
            row = first
            if owner_rows:
                row = first * TP_SIZE + tp_rank
            col = tile % column_tiles * TP_COLUMNS
            pld.tensor.put(
                dst=output_window, peer=group_base + tp_rank, src=output_partial,
                dst_offsets=[row, col], src_offsets=[row, col], shape=[live, TP_COLUMNS],
                chunk_rows=tile_rows, chunk_cols=TP_COLUMNS,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_ready", deps=[publish_tid]) as ready_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_wait", deps=[ready_tid],
               allow_early_resolve=False) as wait_tid:
        for peer in pl.range(TP_SIZE):
            ready_epoch = attention_epoch * 2 - 1
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=ready_epoch, cmp=pld.WaitCmp.Ge)
    return wait_tid


@pl.jit.inline(auto_scope=False)
def _tp_output_all_reduce(
    output_partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor,
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    output_offset: pl.Scalar[pl.INDEX],
    owner_rows: pl.constexpr,
    tile_rows: pl.constexpr,
):
    """Publish TP partials or owned rows and read the destination token range."""
    wait_tid = _tp_output_publish(
        output_partial, output_window, output_arrived, group_base, tp_rank,
        num_tokens, attention_epoch, owner_rows, tile_rows,
    )
    column_tiles = D // TP_COLUMNS
    output_rows = pl.tensor.dim(output, 0)
    reduce_tiles = (output_rows + tile_rows - 1) // tile_rows * column_tiles
    reduce_workers = pl.max(1, pl.min(reduce_tiles, pl.system.available_aiv_count()))
    with pl.spmd(reduce_workers, name_hint="prefill_tp_reduce", deps=[wait_tid]) as reduce_tid:
        worker = pl.tile.get_block_idx()
        for tile in pl.range(worker, reduce_tiles, reduce_workers):
            local_row = tile // column_tiles * tile_rows
            row = output_offset + local_row
            col = tile % column_tiles * TP_COLUMNS
            acc = pl.tile.full([tile_rows, TP_COLUMNS], dtype=pl.FP32, value=0.0)
            if row < num_tokens:
                if owner_rows:
                    owner = group_base + row % TP_SIZE
                    owned = pld.tile.remote_load(output_window, peer=owner, offsets=[row, col], shape=[1, TP_COLUMNS])
                    acc = pl.add(acc, pl.cast(owned, pl.FP32))
                else:
                    live = pl.min(tile_rows, pl.min(num_tokens - row, output_rows - local_row))
                    if live == tile_rows:
                        for peer in pl.range(TP_SIZE):
                            value = pld.tile.remote_load(
                                output_window, peer=group_base + peer, offsets=[row, col],
                                shape=[tile_rows, TP_COLUMNS],
                            )
                            acc = pl.add(acc, pl.cast(value, pl.FP32))
                    else:
                        for peer in pl.range(TP_SIZE):
                            tail = pld.tile.remote_load(
                                output_window, peer=group_base + peer, offsets=[row, col],
                                shape=[tile_rows, TP_COLUMNS], valid_shape=[live, TP_COLUMNS],
                            )
                            filled = pl.fillpad(tail, pad_value=pl.PadValue.zero)
                            full = pl.set_validshape(filled, tile_rows, TP_COLUMNS)
                            acc = pl.add(acc, pl.cast(full, pl.FP32))
            result = pl.cast(acc, pl.BF16, mode="rint")
            extent = pl.min(tile_rows, output_rows - local_row)
            output = pl.store(pl.set_validshape(result, extent, TP_COLUMNS), [local_row, col], output)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        anchor = pl.read(output, [0, 0])
        pl.write(output, [0, 0], anchor)
    return output


@pl.jit.inline(auto_scope=False)
def decode_tp_output_post_token(
    output_partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor,
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    incoming: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    secondary: pl.Tensor[[T_DYN, D], pl.BF16],
    owner_rows: pl.constexpr,
):
    """Fuse the single-token TP read with the Decode mHC Post kernel."""
    wait_tid = _tp_output_publish(
        output_partial, output_window, output_arrived, group_base, tp_rank,
        num_tokens, attention_epoch, owner_rows, 1,
    )
    tokens = pl.tensor.dim(output_partial, 0)
    residual_flat = pl.reshape(residual, [tokens, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [tokens, HC_DIM])
    reduce_tiles = D // TP_COLUMNS
    reduce_workers = pl.max(1, pl.min(reduce_tiles, pl.system.available_aiv_count()))
    with pl.spmd(reduce_workers, name_hint="decode_tp_reduce_post", deps=[wait_tid]) as reduce_tid:
        reduce_worker = pl.tile.get_block_idx()
        for reduce_tile in pl.range(reduce_worker, reduce_tiles, reduce_workers):
            col = reduce_tile * TP_COLUMNS
            acc = pl.tile.full([1, TP_COLUMNS], dtype=pl.FP32, value=0.0)
            if num_tokens > 0:
                if owner_rows:
                    value = pld.tile.remote_load(output_window, peer=group_base, offsets=[0, col], shape=[1, TP_COLUMNS])
                    acc = pl.add(acc, pl.cast(value, pl.FP32))
                else:
                    for peer in pl.range(TP_SIZE):
                        value = pld.tile.remote_load(
                            output_window, peer=group_base + peer, offsets=[0, col], shape=[1, TP_COLUMNS],
                        )
                        acc = pl.add(acc, pl.cast(value, pl.FP32))
            reduced_tile = pl.cast(acc, pl.BF16, mode="rint")
            reduced_f32 = pl.cast(reduced_tile, pl.FP32)
            if owner_rows:
                pl.store(reduced_tile, [0, col], secondary)
                hc_post_token_tile(
                    reduced_f32, residual_flat, post_mix, residual_mix_flat, output_flat, col, TP_COLUMNS,
                )
            else:
                hc_post_collapse_tile(
                    reduced_f32, residual_flat, post_mix, residual_mix_flat, output_flat,
                    pl.cast(0, pl.INDEX), col, incoming, secondary, TP_COLUMNS,
                )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_release", deps=[reduce_tid]) as release_tid:
        for peer in pl.range(TP_SIZE):
            pld.system.notify(
                output_arrived,
                peer=group_base + peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
            )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_tp_consumed", deps=[release_tid],
               allow_early_resolve=False):
        for peer in pl.range(TP_SIZE):
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=attention_epoch * 2, cmp=pld.WaitCmp.Ge)
        output_anchor = pl.read(output, [0, 0, 0])
        pl.write(output, [0, 0, 0], output_anchor)
        secondary_anchor = pl.read(secondary, [0, 0])
        pl.write(secondary, [0, 0], secondary_anchor)
    return output


@pl.jit.inline(auto_scope=False)
def decode_tp_output_post(
    output_partial: pl.Tensor[[T_DYN, D], pl.BF16],
    output_window: pld.DistributedTensor,
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    incoming: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    collapsed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Fuse one Decode token's local TP reduction and mHC Post."""
    tokens = pl.tensor.dim(output_partial, 0)
    if num_tokens == 1:
        decode_tp_output_post_token(
            output_partial, output_window, output_arrived, group_base, tp_rank, num_tokens, attention_epoch,
            residual, post_mix, residual_mix, incoming, output, collapsed, False,
        )
    else:
        reduced = pl.create_tensor([tokens, D], dtype=pl.BF16)
        prefill_tp_output_all_reduce(
            output_partial, output_window, output_arrived, reduced,
            group_base, tp_rank, num_tokens, attention_epoch, owner_rows=False,
        )
        hc_post_collapse(reduced, residual, post_mix, residual_mix, incoming, output, collapsed)
    return output


# Standalone validation shapes.
VALIDATION_ROWS = 144


@pl.jit
def l2_prefill_o_proj(
    attended: pl.Tensor[[VALIDATION_ROWS, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[VALIDATION_ROWS, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[VALIDATION_ROWS, ROPE_DIM // 2], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.InOut[pl.Tensor[[VALIDATION_ROWS, D], pl.BF16]],
):
    num_tokens = pl.read(counts, [0])
    workers = pl.min(VALIDATION_ROWS, pl.system.available_aiv_count())
    scratch = pl.create_tensor([VALIDATION_ROWS, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    for worker in pl.spmd(workers, name_hint="o_proj_validation_input"):
        for row in pl.range(worker, VALIDATION_ROWS, workers):
            tile = pl.load(attended, [row, 0], [1, LOCAL_H * HEAD_DIM])
            pl.store(tile, [row, 0], scratch)
    prefill_o_proj_nz(scratch, wo_a, wo_b, rope_cos, rope_sin, output, num_tokens)
    return output


@pl.jit
def l2_decode_swa_o_proj(
    attended: pl.Tensor[[VALIDATION_ROWS, LOCAL_H * HEAD_DIM], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16],
    rope_cos: pl.Tensor[[VALIDATION_ROWS, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[VALIDATION_ROWS, ROPE_DIM // 2], pl.FP32],
    counts: pl.Tensor[[1], pl.INT32],
    output: pl.InOut[pl.Tensor[[VALIDATION_ROWS, D], pl.BF16]],
):
    num_tokens = pl.read(counts, [0])
    decode_swa_o_proj_ordered(attended, wo_a, wo_b, rope_cos, rope_sin, output, num_tokens)
    return output


def golden_grouped_output(grouped, wo_a, *, output_dtype=None):
    """Apply grouped BF16 O-A with FP32 accumulation and BF16 rounding."""
    value = torch.einsum("tgk,gkn->tgn", grouped.float(), wo_a.float())
    return value.to(output_dtype or torch.bfloat16)


def _golden_ordered_bf16_mm(left, right):
    """Apply BF16 multiplication and add each K contribution in FP32."""
    shape = (*left.shape[:-1], right.shape[-1])
    output = torch.zeros(shape, dtype=torch.float32, device=left.device)
    left, right = left.float(), right.float()
    for k in range(left.shape[-1]):
        output.add_(left[..., k:k + 1] * right[..., k, :])
    return output


def golden_o_proj(attended, cos, sin, wo_a, wo_b, *, output_dtype=None, sequential_projection=False):
    """Evaluate inverse RoPE and the checkpoint's two BF16 output projections."""
    rope_dim = cos.shape[-1] * 2
    tail = rope_interleave(attended[..., -rope_dim:], cos, sin, inverse=True)
    rotated = torch.cat((attended[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = rotated.flatten(-2).unflatten(-1, (groups, -1))
    if sequential_projection:
        latent = _golden_ordered_bf16_mm(grouped, wo_a).bfloat16()
        result = _golden_ordered_bf16_mm(latent.flatten(-2), wo_b)
    else:
        latent = golden_grouped_output(grouped, wo_a)
        result = latent.flatten(-2).float() @ wo_b.float()
    return result.to(output_dtype or torch.bfloat16)


def golden_l2_o_proj(tensors, *, packed_weights):
    """Project the live rows and preserve the output tail."""
    from .weights import unpack_bf16_nz

    count = int(tensors["counts"][0])
    attended = tensors["attended"][:count].reshape(count, LOCAL_H, HEAD_DIM)
    cos, sin = tensors["rope_cos"][:count], tensors["rope_sin"][:count]
    wo_a, wo_b = tensors["wo_a"], tensors["wo_b"]
    if packed_weights:
        wo_a = unpack_bf16_nz(wo_a)
        wo_b = unpack_bf16_nz(wo_b)
    result = golden_o_proj(attended, cos, sin, wo_a, wo_b, sequential_projection=not packed_weights)
    tensors["output"][:count].copy_(result)


def build_validation_specs(count, *, packed_weights, wo_a, wo_b):
    from golden import TensorSpec

    from .rope_tables import prepare_forward_rope_tables
    from .weights import pack_bf16_nz

    cos, sin = prepare_forward_rope_tables(2 * VALIDATION_ROWS)
    rng = torch.Generator().manual_seed(41)
    attended = torch.randn(VALIDATION_ROWS, LOCAL_H * HEAD_DIM, generator=rng).bfloat16()
    if packed_weights:
        wo_a = pack_bf16_nz(wo_a)
        wo_b = pack_bf16_nz(wo_b)
    return [
        TensorSpec("attended", list(attended.shape), attended.dtype, init_value=attended),
        TensorSpec("wo_a", list(wo_a.shape), wo_a.dtype, init_value=wo_a),
        TensorSpec("wo_b", list(wo_b.shape), wo_b.dtype, init_value=wo_b),
        TensorSpec("rope_cos", [VALIDATION_ROWS, ROPE_DIM // 2], torch.float32, init_value=cos[0, -VALIDATION_ROWS:]),
        TensorSpec("rope_sin", [VALIDATION_ROWS, ROPE_DIM // 2], torch.float32, init_value=sin[0, -VALIDATION_ROWS:]),
        TensorSpec("counts", [1], torch.int32, init_value=count),
        TensorSpec("output", [VALIDATION_ROWS, D], torch.bfloat16, init_value=1),
    ]


if __name__ == "__main__":
    import argparse
    import json
    from functools import partial

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=0, choices=range(M.num_hidden_layers))
    parser.add_argument("--tp-rank", type=int, choices=range(TP_SIZE))
    parser.add_argument("--phase", choices=("prefill", "decode-swa"))
    parser.add_argument("--num-tokens", type=int, choices=range(1, VALIDATION_ROWS + 1))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    checkpoint = None
    if args.checkpoint is not None:
        from .weights import Checkpoint

        checkpoint = Checkpoint(args.checkpoint)
    phase_cases = (("prefill", (17, 129), l2_prefill_o_proj), ("decode-swa", (1, 3), l2_decode_swa_o_proj))
    if args.tp_rank is not None:
        lanes = [args.tp_rank]
    else:
        lanes = range(TP_SIZE) if checkpoint is not None else [0]
    generator = torch.Generator().manual_seed(42)
    reports = []
    passed = True
    for lane in lanes:
        if checkpoint is None:
            wo_a = torch.randn(LOCAL_O_GROUPS, O_GROUP_IN, O_LORA, generator=generator) / O_GROUP_IN**0.5
            wo_b = torch.randn(LOCAL_O_WIDTH, D, generator=generator) / LOCAL_O_WIDTH**0.5
            wo_a, wo_b = wo_a.bfloat16(), wo_b.bfloat16()
        else:
            prefix = f"layers.{args.layer}.attn."
            wo_a = checkpoint.tensor(prefix + "wo_a.weight").reshape(M.o_groups, O_LORA, O_GROUP_IN)
            wo_a = wo_a[lane * LOCAL_O_GROUPS:(lane + 1) * LOCAL_O_GROUPS].transpose(-1, -2).contiguous()
            wo_b = checkpoint.tensor(prefix + "wo_b.weight")
            wo_b = wo_b[:, lane * LOCAL_O_WIDTH:(lane + 1) * LOCAL_O_WIDTH].t().contiguous()
        for phase, counts, kernel in phase_cases:
            if args.phase is not None and phase != args.phase:
                continue
            packed_weights = phase == "prefill"
            for count in counts if args.num_tokens is None else (args.num_tokens,):
                specs = build_validation_specs(count, packed_weights=packed_weights, wo_a=wo_a, wo_b=wo_b)
                print(f"A3_OUTPUT_PROJECTION phase={phase} tokens={count} layer={args.layer} tp_rank={lane}", flush=True)
                result = run(
                    fn=kernel,
                    specs=specs,
                    golden_fn=partial(golden_l2_o_proj, packed_weights=packed_weights),
                    compare_fn={"output": ratio_allclose(atol=0.003, rtol=1.0 / 128, max_error_ratio=0.0)},
                    config={"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device},
                    compile_only=args.compile_only,
                )
                reports.append({"phase": phase, "tokens": count, "layer": args.layer, "tp_rank": lane,
                                "passed": result.passed, "compile_only": args.compile_only, "build": str(result.work_dir)})
                if not result.passed:
                    passed = False
                    if result.error:
                        print(result.error)
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)
