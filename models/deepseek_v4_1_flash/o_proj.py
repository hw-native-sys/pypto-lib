# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 TP-local inverse RoPE and output projection, without communication."""

import pypto.language as pl

from models.deepseek_v4_1_flash.config import (
    D, HEAD_DIM, LOCAL_H, LOCAL_O_GROUPS, LOCAL_O_WIDTH, O_GROUP_IN, O_LORA, ROPE_DIM, T_DYN,
)
from models.deepseek_v4_1_flash.qkv_proj_rope import (
    K_TILE, N_TILE, make_projection, make_rope, make_prefill_rope,
)

M_TILE = 16


@pl.jit.inline
def grouped_output(
    x: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    weight: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    output: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    for block in pl.spmd((num_tokens + M_TILE - 1) // M_TILE * (LOCAL_O_WIDTH // N_TILE),
                         name_hint="swa_grouped_output"):
        t0 = block // (LOCAL_O_WIDTH // N_TILE) * M_TILE
        n0 = block % (LOCAL_O_WIDTH // N_TILE) * N_TILE
        group = n0 // O_LORA
        local_n = n0 % O_LORA
        rows = pl.min(M_TILE, num_tokens - t0)
        acc = pl.create_tensor([M_TILE, N_TILE], dtype=pl.FP32)
        for kb in pl.range(O_GROUP_IN // K_TILE):
            k0 = kb * K_TILE
            a = pl.slice(x, [M_TILE, K_TILE], [t0, group * O_GROUP_IN + k0],
                         valid_shape=[rows, K_TILE])
            w = pl.reshape(weight[group:group + 1, local_n:local_n + N_TILE, k0:k0 + K_TILE],
                           [N_TILE, K_TILE])
            acc = pl.matmul_acc(acc, a, w, b_trans=True, init_cond=(kb == 0))
        value = pl.cast(acc, pl.BF16, mode="rint")
        output[t0:t0 + M_TILE, n0:n0 + N_TILE] = pl.set_validshape(value, rows, N_TILE)
    return output


project_ob = make_projection(LOCAL_O_WIDTH, D, pl.FP32)


def make_o_proj(rotate):
    @pl.jit.inline(auto_scope=False)
    def o_proj(
        attended: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        unrotated: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        latent: pl.Tensor[[T_DYN, LOCAL_O_WIDTH], pl.BF16],
        partial: pl.Tensor[[T_DYN, D], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        rotate(attended, cos, sin, unrotated, num_tokens)
        grouped_output(unrotated, wo_a, latent, num_tokens)
        project_ob(latent, wo_b, wo_b_scale, partial, num_tokens)
        return partial

    return o_proj


o_proj = make_o_proj(make_rope(LOCAL_H, inverse=True))
prefill_o_proj = make_o_proj(make_prefill_rope(LOCAL_H, inverse=True))
