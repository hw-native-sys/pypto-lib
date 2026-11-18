# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device BF16 embedding lookup and exact FP32 HC input expansion."""

import pypto.language as pl
import pypto.language.distributed as pld

from .config import FLASH as M

# Dynamic shape variables.
LOOKUP_ROWS = pl.dynamic("LOWLAT_LOOKUP_INPUT_ROWS")
LOOKUP_VOCAB = pl.dynamic("LOWLAT_LOOKUP_INPUT_VOCAB")

# Model configuration.
RANKS = 8
HIDDEN = M.hidden_size
HC_MULT = M.hc_mult


@pl.jit(auto_scope=False)
def l2_lookup_prefill_input(
    embedding_weight: pl.Tensor[[LOOKUP_VOCAB, HIDDEN], pl.BF16],
    input_ids: pl.Tensor[[LOOKUP_ROWS], pl.INT64],
    x_hc: pl.Out[pl.Tensor[[LOOKUP_ROWS, HC_MULT, HIDDEN], pl.FP32]],
):
    embedding_weight.bind_dynamic(0, LOOKUP_VOCAB)
    input_ids.bind_dynamic(0, LOOKUP_ROWS)
    x_hc.bind_dynamic(0, LOOKUP_ROWS)
    rows = pl.tensor.dim(input_ids, 0)
    flat_rows = rows * HC_MULT
    flat = pl.reshape(x_hc, [flat_rows, HIDDEN])
    with pl.spmd(rows, name_hint="prefill_device_embedding_to_hc"):
        row = pl.tile.get_block_idx()
        token = pl.read(input_ids, [row])
        token_index = pl.cast(token, pl.INDEX)
        source = pl.load(embedding_weight, [token_index, 0], [1, HIDDEN])
        halfwords = pl.reinterpret_view(source, pl.INT16)
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
        for channel in pl.range(HC_MULT):
            output_row = row * HC_MULT + channel
            pl.store(expanded, [output_row, 0], flat)
    return x_hc


@pl.jit.host
def l3_lookup_prefill_input(
    embedding_weight: pl.Tensor[[RANKS, LOOKUP_VOCAB, HIDDEN], pl.BF16],
    input_ids: pl.Tensor[[RANKS, LOOKUP_ROWS], pl.INT64],
    x_hc: pl.Out[pl.Tensor[[RANKS, LOOKUP_ROWS, HC_MULT, HIDDEN], pl.FP32]],
):
    embedding_weight.bind_dynamic(1, LOOKUP_VOCAB)
    input_ids.bind_dynamic(1, LOOKUP_ROWS)
    x_hc.bind_dynamic(1, LOOKUP_ROWS)
    for rank in pl.range(pld.world_size()):
        l2_lookup_prefill_input(embedding_weight[rank], input_ids[rank], x_hc[rank], device=rank)
    return x_hc
