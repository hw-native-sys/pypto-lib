# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exact prefill input conversion on real BF16/F32 device allocations."""

import pypto.language as pl
import pypto.language.distributed as pld

from .config import FLASH as M

INPUT_ROWS = pl.dynamic("LOWLAT_COMPACT_INPUT_ROWS")
INPUT_RANKS = 8
INPUT_D = M.hidden_size
INPUT_HC = M.hc_mult


@pl.jit(auto_scope=False)
def l2_expand_compact_input(
    embedding: pl.Tensor[[INPUT_ROWS, INPUT_D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[INPUT_ROWS, INPUT_HC, INPUT_D], pl.FP32]],
):
    embedding.bind_dynamic(0, INPUT_ROWS)
    x_hc.bind_dynamic(0, INPUT_ROWS)
    rows = pl.tensor.dim(embedding, 0)
    flat = pl.reshape(x_hc, [rows * INPUT_HC, INPUT_D])
    with pl.spmd(rows, name_hint="prefill_compact_bf16_to_hc"):
        row = pl.tile.get_block_idx()
        source = pl.load(embedding, [row, 0], [1, INPUT_D])
        # BF16 is the upper half of FP32. Preserve every bit, including NaN
        # payloads, instead of relying on a numeric converter's NaN policy.
        halfwords = pl.reinterpret_view(source, pl.INT16)
        # The pinned A3 backend has no S16->S32 cast. Byte-sized integers
        # survive its S16->F16->F32 conversion exactly, then widen to S32.
        high_byte = pl.tile.shrs(halfwords, 8)
        low_byte = pl.tile.ands(halfwords, 255)
        high_fp32 = pl.cast(high_byte, target_type=pl.FP32)
        low_fp32 = pl.cast(low_byte, target_type=pl.FP32)
        high_i32 = pl.cast(high_fp32, target_type=pl.INT32)
        low_i32 = pl.cast(low_fp32, target_type=pl.INT32)
        high_bits = pl.tile.shls(high_i32, 24)
        low_bits = pl.tile.shls(low_i32, 16)
        shifted = pl.add(high_bits, low_bits)
        expanded = pl.reinterpret_view(shifted, pl.FP32)
        for channel in pl.range(INPUT_HC):
            pl.store(expanded, [row * INPUT_HC + channel, 0], flat)
    return x_hc


@pl.jit.host
def l3_expand_compact_input(
    embedding: pl.Tensor[[INPUT_RANKS, INPUT_ROWS, INPUT_D], pl.BF16],
    x_hc: pl.Out[pl.Tensor[[INPUT_RANKS, INPUT_ROWS, INPUT_HC, INPUT_D], pl.FP32]],
):
    for rank in pl.range(pld.world_size()):
        l2_expand_compact_input(embedding[rank], x_hc[rank], device=rank)
    return x_hc
