# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 attention RMSNorm (dynamic shape): normalizes token-major
activations for both decode and prefill attention paths."""

import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
EPS = M.rms_norm_eps

# tiling
D_TILE = 128
T_TILE = 8
assert D % D_TILE == 0, "D must be divisible by D_TILE"
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0


@pl.jit.inline
def rms_norm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(x, 0)
    # Capture form (not `for ... in pl.spmd`): callers need the producer TaskId to
    # hang a `pl.system.task_dummy` barrier off it and defer non-critical consumers.
    with pl.spmd(t_dim // T_TILE, name_hint="rms_norm", allow_early_resolve=True) as rms_tid:
        tg_idx = pl.tile.get_block_idx()
        tg = tg_idx * T_TILE
        x_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for rms_db in pl.pipeline(D // D_TILE, stage=2):
            rms_d0 = rms_db * D_TILE
            rms_x_chunk = pl.cast(x[tg : tg + T_TILE, rms_d0 : rms_d0 + D_TILE], target_type=pl.FP32)
            x_sq_sum = pl.add(x_sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_chunk, rms_x_chunk)), [1, T_TILE]))
        x_inv_rms = pl.rsqrt(pl.add(pl.mul(x_sq_sum, 1.0 / D), EPS), high_precision=True)
        x_inv_rms_t = pl.reshape(x_inv_rms, [T_TILE, 1])
        for apply_db in pl.pipeline(D // D_TILE, stage=2):
            apply_d0 = apply_db * D_TILE
            apply_x_chunk = pl.cast(x[tg : tg + T_TILE, apply_d0 : apply_d0 + D_TILE], target_type=pl.FP32)
            norm_w_chunk = pl.cast(pl.reshape(norm_w[apply_d0 : apply_d0 + D_TILE], [1, D_TILE]), pl.FP32)
            x_normed_chunk = pl.col_expand_mul(pl.row_expand_mul(apply_x_chunk, x_inv_rms_t), norm_w_chunk)
            x_normed[tg : tg + T_TILE, apply_d0 : apply_d0 + D_TILE] = pl.cast(
                x_normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )

    return rms_tid


def golden_rms_norm(x, norm_w):
    import torch

    x = x.float()
    norm_w = norm_w.float()
    inv = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return (x * inv * norm_w).to(torch.bfloat16)
