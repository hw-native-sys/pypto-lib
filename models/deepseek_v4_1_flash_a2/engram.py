# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card BF16 Engram in the unrotated checkpoint basis.

The caller supplies resident table rows and their indices. Host hashing,
large-table offload and cross-rank gathers remain outside this kernel.
"""
import pypto.language as pl

D, HC, COLS, ROW_WIDTH = 5120, 4, 24, 256
K, N = COLS * ROW_WIDTH, (HC + 1) * D
TILE, KT, NT, DT = 16, 128, 256, 512


def make_engram(capacity, table_rows):
    if capacity % TILE or table_rows < 1:
        raise ValueError('Engram needs padded token capacity and a nonempty resident table')

    @pl.jit
    def engram(
        hash_ids: pl.Tensor[[capacity, COLS], pl.INT32],
        table: pl.Tensor[[table_rows, ROW_WIDTH], pl.BF16],
        wkv: pl.Tensor[[K, N], pl.BF16],
        channel_weight: pl.Tensor[[HC, D], pl.FP32],
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        lookup: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, N], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint='a2_engram_lookup'):
            for column in pl.range(COLS):
                row = pl.read(hash_ids, [token, column])
                lookup[token:token + 1, column * ROW_WIDTH:(column + 1) * ROW_WIDTH] = pl.slice(table, [1, ROW_WIDTH], [row, 0])

        blocks = pl.system.available_cluster_count()
        for core in pl.spmd(blocks, name_hint='a2_engram_projection'):
            for task in pl.range(core, ((active + TILE - 1) // TILE) * (N // NT), blocks):
                t0 = (task // (N // NT)) * TILE
                n0 = (task % (N // NT)) * NT
                rows = pl.min(TILE, active - t0)
                a = pl.slice(lookup, [TILE, KT], [t0, 0], valid_shape=[rows, KT])
                b = wkv[:KT, n0:n0 + NT]
                acc = pl.matmul(a, b, out_dtype=pl.FP32)
                for k0 in pl.pipeline(KT, K, KT, stage=2):
                    a = pl.slice(lookup, [TILE, KT], [t0, k0], valid_shape=[rows, KT])
                    b = wkv[k0:k0 + KT, n0:n0 + NT]
                    acc = pl.matmul_acc(acc, a, b)
                # ReplicatedLinear returns BF16 before the FP32 gate arithmetic.
                kv[t0:t0 + TILE, n0:n0 + NT] = pl.set_validshape(pl.cast(acc, pl.BF16, mode='rint'), rows, NT)

        flat_hidden = pl.reshape(hidden, [capacity, HC * D])
        flat_output = pl.reshape(output, [capacity, HC * D])
        mask = pl.reshape(token_mask, [1, capacity])
        for task in pl.spmd(((active + TILE - 1) // TILE) * HC, name_hint='a2_engram_gate'):
            t0, copy = (task // HC) * TILE, task % HC
            rows = pl.min(TILE, active - t0)
            sx = pl.full([1, TILE], dtype=pl.FP32, value=0.)
            sk = pl.full([1, TILE], dtype=pl.FP32, value=0.)
            dot = pl.full([1, TILE], dtype=pl.FP32, value=0.)
            for d0 in pl.range(0, D, DT):
                x = pl.cast(pl.slice(flat_hidden, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                key = pl.cast(pl.slice(kv, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                weight = channel_weight[copy:copy + 1, d0:d0 + DT]
                sx = pl.add(sx, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, TILE]))
                sk = pl.add(sk, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, TILE]))
                dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(pl.col_expand_mul(x, weight), key)), [1, TILE]))
            inv = pl.mul(pl.rsqrt(pl.add(pl.mul(sx, 1. / D), 1e-20)), pl.rsqrt(pl.add(pl.mul(sk, 1. / D), 1e-20)))
            dot = pl.mul(pl.mul(dot, inv), D**-0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1e-6))
            # Signed FP32 bits preserve -0 without A2-unsupported INT32 bitwise AND.
            bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
            negative = pl.maximum(pl.minimum(bits, 0.), -1.)
            signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.), 1.))
            gate = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.))
            eligible = pl.cast(pl.slice(mask, [1, TILE], [0, t0], valid_shape=[1, rows]), pl.FP32)
            gate = pl.reshape(pl.mul(gate, eligible), [TILE, 1])
            for d0 in pl.range(0, D, DT):
                x = pl.cast(pl.slice(flat_hidden, [TILE, DT], [t0, copy * D + d0], valid_shape=[rows, DT]), pl.FP32)
                value = pl.cast(pl.slice(kv, [TILE, DT], [t0, HC * D + d0], valid_shape=[rows, DT]), pl.FP32)
                result = pl.cast(pl.add(x, pl.row_expand_mul(value, gate)), pl.BF16, mode='rint')
                flat_output[t0:t0 + TILE, copy * D + d0:copy * D + d0 + DT] = pl.set_validshape(result, rows, DT)
        return lookup, kv, output

    return engram
