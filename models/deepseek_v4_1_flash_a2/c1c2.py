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
from .ops import make_quantize


def make_compressed_cache_publish(capacity, blocks, width):
    """Publish BF16 compressed KV rows into PA_BBND pages.

    ``slots`` contains the flattened page row (``page * 128 + offset``), as
    produced by the vLLM A2 scheduler.  A negative slot is padding and leaves
    the cache byte-for-byte untouched.
    """
    rows = blocks * 128

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [rows, width])
        with pl.spmd(active, name_hint="a2_c2_bf16_cache_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                flat[slot : slot + 1, :] = value[token : token + 1, :]
        return cache

    return publish


def make_indexer_cache_publish(capacity, blocks, width):
    """Quantize index keys to INT8/FP16 and publish both cache tensors.

    Ascend's non-A5 indexer path uses one dynamic scale for each key row.  The
    A2 PyPTO representation stores that scale as ``[blocks,128,1,1]`` FP16,
    matching ``indexer_quant_scatter`` in vLLM-Ascend.  Quantization is the
    same symmetric per-token INT8 operation used by the A2 W8A8 projections.
    """
    rows = blocks * 128
    quantize = make_quantize(capacity, width)

    @pl.jit.inline
    def publish(
        value: pl.Tensor[[capacity, width], pl.BF16],
        slots: pl.Tensor[[capacity], pl.INT64],
        cache: pl.Tensor[[blocks, 128, 1, width], pl.INT8],
        scales: pl.Tensor[[blocks, 128, 1, 1], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        payload = pl.create_tensor([capacity, width], dtype=pl.INT8)
        row_scales = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(value, payload, row_scales, active)
        cache_flat = pl.reshape(cache, [rows, width])
        scale_flat = pl.reshape(scales, [rows, 1])
        with pl.spmd(active, name_hint="a2_c1_int8_cache_publish"):
            token = pl.tile.get_block_idx()
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                cache_flat[slot : slot + 1, :] = payload[token : token + 1, :]
        # A scalar FP16 row is only two bytes, while Ascend writes a full
        # cache line.  Serialize these writes so adjacent rows cannot race.
        with pl.spmd(1, name_hint="a2_c1_int8_scale_publish"):
            _worker = pl.tile.get_block_idx()
            for token in pl.range(active):
                slot_i64 = pl.read(slots, [token])
                if slot_i64 >= 0:
                    slot = pl.cast(slot_i64, pl.INDEX)
                    scale = pl.cast(pl.read(row_scales, [token, 0]), pl.FP16)
                    pl.write(scale_flat, [slot, 0], scale)
        return cache, scales

    return publish
