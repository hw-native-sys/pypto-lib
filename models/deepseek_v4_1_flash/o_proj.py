# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 inverse RoPE and grouped output projection stages."""

import pypto.language as pl

from models.deepseek_v4_1_flash.attention_ops import K_TILE, M_TILE, N_TILE, make_mx_projection, make_rope
from models.deepseek_v4_1_flash.config import (
    D,
    HEAD_DIM,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    O_GROUP_IN,
    O_LORA,
    ROPE_DIM,
    T_DYN,
)


_PREFILL_WORKERS = 64


@pl.jit.inline
def _grouped_output_block(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    block: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    t0 = block // (LOCAL_O_WIDTH // N_TILE) * M_TILE
    n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
    group = n0 // O_LORA
    local_n = n0 % O_LORA
    rows = pl.min(M_TILE, num_tokens - t0)
    acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
    for kb in pl.range(O_GROUP_IN // K_TILE):
        k0 = kb * K_TILE
        a = pl.slice(
            x,
            [M_TILE, K_TILE],
            [t0, group * O_GROUP_IN + k0],
            valid_shape=[rows, K_TILE],
        )
        w = pl.reshape(
            weight[group : group + 1, local_n : local_n + N_TILE, k0 : k0 + K_TILE],
            [N_TILE, K_TILE],
        )
        acc = pl.matmul_acc(acc, a, w, b_trans=True, init_cond=(kb == 0))
    value = pl.cast(acc, pl.BF16, mode="rint")
    output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(value, rows, N_TILE)
    return output


@pl.jit.inline
def grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Apply grouped Wo-A with FP32 accumulation and a BF16 boundary."""
    for block in pl.spmd(
        (num_tokens + M_TILE - 1) // M_TILE * (LOCAL_O_WIDTH // N_TILE),
        name_hint="attention_grouped_o_a",
    ):
        _grouped_output_block(x, weight, output, block, num_tokens)
    return output


@pl.jit.inline
def grouped_output_with_deps(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
):
    """Apply grouped Wo-A after an explicit producer dependency."""
    with pl.spmd(
        (num_tokens + M_TILE - 1) // M_TILE * (LOCAL_O_WIDTH // N_TILE),
        name_hint="c1a_grouped_output",
        deps=[ready],
    ) as grouped_tid:
        block = pl.tile.get_block_idx()
        _grouped_output_block(x, weight, output, block, num_tokens)
    return grouped_tid


def _make_o_proj(rotate, grouped, project):
    @pl.jit.inline(auto_scope=False)
    def o_proj(
        attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, D], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(attended, 0)
        unrotated = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        rotate(attended, rope_cos, rope_sin, unrotated, num_tokens)
        latent = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
        grouped(unrotated, wo_a, latent, num_tokens)
        project(latent, wo_b, wo_b_scale, output, num_tokens)
        return output

    return o_proj


def make_o_proj_with_deps(rotate, grouped, project):
    """Compose dependency-aware C1A decode output primitives."""

    @pl.jit.inline(auto_scope=False)
    def o_proj_with_deps(
        attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, D], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        tokens = pl.tensor.dim(attended, 0)
        unrotated = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        rotate_tid = rotate(attended, rope_cos, rope_sin, unrotated, num_tokens, ready)
        latent = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.BF16)
        grouped_tid = grouped(unrotated, wo_a, latent, num_tokens, rotate_tid)
        output_tid = project(latent, wo_b, wo_b_scale, output, num_tokens, grouped_tid)
        return output_tid

    return o_proj_with_deps


_project_ob = make_mx_projection(LOCAL_O_WIDTH, D, pl.FP32, name_hint="attention_o_b")
_rotate_output = make_rope(LOCAL_H, inverse=True, name_hint="attention_o_rope")
_prefill_rotate_output = make_rope(
    LOCAL_H,
    inverse=True,
    max_workers=_PREFILL_WORKERS,
    name_hint="prefill_attention_o_rope",
)

o_proj = _make_o_proj(_rotate_output, grouped_output, _project_ob)
prefill_o_proj = _make_o_proj(_prefill_rotate_output, grouped_output, _project_ob)


__all__ = [
    "grouped_output",
    "grouped_output_with_deps",
    "make_o_proj_with_deps",
    "o_proj",
    "prefill_o_proj",
]
