# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Blocked weighted RMSNorm shared by projection kernels."""

import pypto.language as pl

from .activation_quant import VEC_M_TILE


PREFILL_NORM_WORKERS = 48


def make_blocked_rmsnorm(capacity, width, eps, *, dynamic_rows=False):
    """Specialize weighted FP32 RMSNorm with a BF16 output."""
    stream_rows = width == 512 and (dynamic_rows or capacity >= 1024 and capacity % VEC_M_TILE == 0)
    stages = 2 if stream_rows else 1
    read_padded = dynamic_rows or not stream_rows

    @pl.jit.inline
    def norm(
        x: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[capacity, width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        tiles = (active + VEC_M_TILE - 1) // VEC_M_TILE
        workers = pl.min(pl.min(tiles, PREFILL_NORM_WORKERS), pl.system.available_aiv_count())
        for worker in pl.spmd(workers, name_hint="a2_rms_norm"):
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            if stream_rows:
                begin = worker * tiles // workers
                end = (worker + 1) * tiles // workers
                step = 1
            else:
                begin = worker
                end = tiles
                step = workers
            for block in pl.pipeline(begin, end, step, stage=stages):
                t0 = block * VEC_M_TILE
                rows = pl.min(VEC_M_TILE, active - t0)
                source = x[t0 : t0 + VEC_M_TILE, :]
                if read_padded:
                    raw = pl.slice(x, [VEC_M_TILE, width], [t0, 0], valid_shape=[rows, width])
                    source = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), VEC_M_TILE, width)
                value = pl.cast(source, pl.FP32)
                square = pl.mul(value, value)
                mean = pl.mul(pl.row_sum(square), 1.0 / width)
                inverse = pl.rsqrt(pl.add(mean, eps), high_precision=True)
                normalized = pl.row_expand_mul(value, inverse)
                weighted = pl.col_expand_mul(normalized, gamma)
                narrowed = pl.cast(weighted, pl.BF16, mode="rint")
                output[t0 : t0 + VEC_M_TILE, :] = pl.set_validshape(narrowed, rows, width)
        return output

    return norm
