# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure PyPTO dtype boundaries; original own cache publication remains upstream."""

import pypto.language as pl


def make_pack(active, ori_blocks, cmp_blocks):
    @pl.jit
    def pack(query: pl.InOut[pl.Tensor[[active, 64, 512], pl.FP32]],
             window: pl.InOut[pl.Tensor[[ori_blocks, 128, 512], pl.FP32]],
             compressed: pl.InOut[pl.Tensor[[cmp_blocks, 128, 512], pl.FP32]],
             query_bf16: pl.InOut[pl.Tensor[[active, 64, 512], pl.BF16]],
             window_bf16: pl.InOut[pl.Tensor[[ori_blocks, 128, 512], pl.BF16]],
             compressed_bf16: pl.InOut[pl.Tensor[[cmp_blocks, 128, 512], pl.BF16]]):
        q = pl.reshape(query, [active * 64, 512])
        qb = pl.reshape(query_bf16, [active * 64, 512])
        for head in pl.spmd(active * 64, name_hint='ratio1_query_boundary'):
            qb[head:head + 1, :] = pl.cast(q[head:head + 1, :], pl.BF16, mode='rint')
        w = pl.reshape(window, [ori_blocks * 128, 512])
        wb = pl.reshape(window_bf16, [ori_blocks * 128, 512])
        for row in pl.spmd(ori_blocks * 128, name_hint='ratio1_window_boundary'):
            wb[row:row + 1, :] = pl.cast(w[row:row + 1, :], pl.BF16, mode='rint')
        c = pl.reshape(compressed, [cmp_blocks * 128, 512])
        cb = pl.reshape(compressed_bf16, [cmp_blocks * 128, 512])
        for row in pl.spmd(cmp_blocks * 128, name_hint='ratio1_compressed_boundary'):
            cb[row:row + 1, :] = pl.cast(c[row:row + 1, :], pl.BF16, mode='rint')
        return query, window, compressed, query_bf16, window_bf16, compressed_bf16

    return pack


def make_import(active, ori_blocks=2, cmp_blocks=2):
    @pl.jit
    def import_output(query: pl.InOut[pl.Tensor[[active, 64, 512], pl.FP32]],
                      window: pl.InOut[pl.Tensor[[ori_blocks, 128, 512], pl.FP32]],
                      window_indices: pl.InOut[pl.Tensor[[active, 128], pl.INT32]],
                      compressed: pl.InOut[pl.Tensor[[cmp_blocks, 128, 512], pl.FP32]],
                      compressed_indices: pl.InOut[pl.Tensor[[active, 512], pl.INT32]],
                      sink: pl.InOut[pl.Tensor[[64], pl.FP32]],
                      extents: pl.InOut[pl.Tensor[[active, 2], pl.INT32]],
                      output: pl.InOut[pl.Tensor[[active, 64, 512], pl.FP32]],
                      leaf_output: pl.InOut[pl.Tensor[[active, 64, 512], pl.BF16]]):
        src = pl.reshape(leaf_output, [active * 64, 512])
        dst = pl.reshape(output, [active * 64, 512])
        for head in pl.spmd(active * 64, name_hint='ratio1_native_output_import'):
            dst[head:head + 1, :] = pl.cast(src[head:head + 1, :], pl.FP32)
        return query, window, window_indices, compressed, compressed_indices, sink, extents, output, leaf_output

    return import_output
