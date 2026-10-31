# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2 single-layer dependencies extracted from the shared model operators."""

import pypto.language as pl
from .qkv_proj_rope import K_TILE, M_TILE, N_TILE
from .config import FLASH as M, T_DYN, TP_SIZE


LOCAL_H = M.num_attention_heads // TP_SIZE


LOCAL_O_GROUPS = M.o_groups // TP_SIZE


LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank


HEAD_DIM = M.head_dim


O_LORA = M.o_lora_rank


O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups


@pl.jit.inline
def prefill_grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Stage Wo-A values with Vector startup and FP32 Vector stores before narrowing."""
    tokens = pl.tensor.dim(x, 0)
    accumulated = pl.create_tensor([tokens, LOCAL_O_WIDTH], dtype=pl.FP32)
    blocks = (num_tokens + M_TILE - 1) // M_TILE * (LOCAL_O_WIDTH // N_TILE)
    with pl.spmd(blocks, name_hint="prefill_grouped_o_a_cube") as cube_tid:
        block = pl.tile.get_block_idx()
        t0 = block // (LOCAL_O_WIDTH // N_TILE) * M_TILE
        n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
        group = n0 // O_LORA
        local_n = n0 % O_LORA
        rows = pl.min(M_TILE, num_tokens - t0)
        acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
        for kb in pl.range(O_GROUP_IN // K_TILE):
            k0 = kb * K_TILE
            a = pl.slice(x, [M_TILE, K_TILE], [t0, group * O_GROUP_IN + k0], valid_shape=[rows, K_TILE])
            a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
            w = pl.reshape(weight[group : group + 1, local_n : local_n + N_TILE, k0 : k0 + K_TILE], [N_TILE, K_TILE])
            acc = pl.matmul_acc(acc, a, w, b_trans=True, init_cond=(kb == 0))
        accumulated[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.set_validshape(pl.mul(acc, 1.0), rows, N_TILE)
    with pl.spmd(blocks, name_hint="prefill_grouped_o_a_cast", deps=[cube_tid]):
        block = pl.tile.get_block_idx()
        t0 = block // (LOCAL_O_WIDTH // N_TILE) * M_TILE
        n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
        rows = pl.min(M_TILE, num_tokens - t0)
        value = pl.slice(accumulated, [M_TILE, N_TILE], [t0, n0], valid_shape=[rows, N_TILE])
        output[t0 : t0 + M_TILE, n0 : n0 + N_TILE] = pl.cast(value, pl.BF16, mode="rint")
    return output
