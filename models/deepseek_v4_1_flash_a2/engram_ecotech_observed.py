# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Engram INT8 lookup, BF16 projection, original-basis hidden and gate readbacks."""

import pypto.language as pl

from .qkv_proj_rope import make_bf16_projection_staged

from .engram_ecotech import (
    COLS, D, HC, K, N, ROTATION_TILE, WIDTH,
    make_original_basis_gate, make_original_basis_hidden, make_resident_int8_lookup,
)


def make_engram_ecotech_observed(capacity, table_rows, rms_norm_eps=1e-20):
    if capacity < 1 or capacity % 16 or table_rows < 1 or rms_norm_eps <= 0:
        raise ValueError("Use a positive padded capacity, resident row count and source norm epsilon")
    gather = make_resident_int8_lookup(capacity, table_rows)
    project = make_bf16_projection_staged(K, N)
    restore = make_original_basis_hidden(capacity)
    gate = make_original_basis_gate(capacity, rms_norm_eps)

    @pl.jit
    def engram_ecotech_observed(
        staging_ids: pl.InOut[pl.Tensor[[capacity, COLS], pl.INT32]],
        table: pl.InOut[pl.Tensor[[table_rows, WIDTH], pl.INT8]],
        table_scales: pl.InOut[pl.Tensor[[table_rows, 8], pl.FP32]],
        wkv: pl.InOut[pl.Tensor[[K, N], pl.BF16]],
        channel_weight: pl.InOut[pl.Tensor[[HC, D], pl.FP32]],
        hidden: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        rotation_block: pl.InOut[pl.Tensor[[ROTATION_TILE, ROTATION_TILE], pl.FP32]],
        token_mask: pl.InOut[pl.Tensor[[capacity], pl.INT32]],
        lookup: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, N], pl.BF16]],
        original: pl.InOut[pl.Tensor[[capacity, HC, D], pl.FP32]],
        probabilities: pl.InOut[pl.Tensor[[capacity, HC], pl.FP32]],
        output: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        gather(staging_ids, table, table_scales, lookup, active)
        project(lookup, wkv, kv, active)
        restore(hidden, rotation_block, original, active)
        gate(hidden, original, kv, channel_weight, token_mask, probabilities, output, active)
        return (
            staging_ids, table, table_scales, wkv, channel_weight, hidden, rotation_block, token_mask,
            lookup, kv, original, probabilities, output,
        )

    return engram_ecotech_observed
