# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resident INT8 Engram lookup, W8A8 projection, and masked residual gate."""

import pypto.language as pl

from .ops import N_TILE, make_dequantize, make_dot, make_pack_weight, make_quantize


# model config
D, HC, COLS, WIDTH = 5120, 4, 24, 256
K, N = COLS * WIDTH, (HC + 1) * D

# tiling
T_TILE, D_TILE = 16, 512


def make_engram_w8a8(capacity, table_rows):
    if capacity % T_TILE or table_rows < 1:
        raise ValueError("Engram requires padded token capacity and a nonempty resident table")
    quantize = make_quantize(capacity, K)
    pack = make_pack_weight(K, N)
    dot = make_dot(capacity, K, N)
    dequantize = make_dequantize(capacity, N, combine_scales=True)
    packed_rows = K * (N // N_TILE)

    @pl.jit
    def engram_w8a8(
        staging_ids: pl.Tensor[[capacity, COLS], pl.INT32],
        table: pl.Tensor[[table_rows, WIDTH], pl.INT8],
        table_scales: pl.Tensor[[table_rows, 8], pl.FP32],
        wkv: pl.Tensor[[K, N], pl.INT8],
        wkv_scale: pl.Tensor[[N], pl.FP32],
        channel_weight: pl.Tensor[[HC, D], pl.FP32],
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        lookup: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, N], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint="engram_int8_lookup"):
            for column in pl.range(COLS):
                row = pl.read(staging_ids, [token, column])
                for group in pl.range(8):
                    lookup_half = pl.cast(pl.slice(table, [1, 32], [row, group * 32]), pl.FP16)
                    lookup_codes = pl.cast(lookup_half, pl.FP32)
                    scale = pl.read(table_scales, [row, group])
                    decoded_fp32 = pl.mul(lookup_codes, scale)
                    decoded = pl.cast(decoded_fp32, pl.BF16, mode="rint")
                    start = column * WIDTH + group * 32
                    lookup[token:token + 1, start:start + 32] = decoded

        payload = pl.create_tensor([capacity, K], dtype=pl.INT8)
        token_scales = pl.create_tensor([capacity, 8], dtype=pl.FP32)
        quantize(lookup, payload, token_scales, active)
        packed = pl.create_tensor([packed_rows, N_TILE], dtype=pl.INT8)
        pack(wkv, packed)
        accumulator = pl.create_tensor([capacity, N], dtype=pl.INT32)
        dot(payload, packed, accumulator, active)
        dequantize(accumulator, wkv_scale, token_scales, kv, active)

        flat_hidden = pl.reshape(hidden, [capacity, HC * D])
        flat_output = pl.reshape(output, [capacity, HC * D])
        mask = pl.reshape(token_mask, [1, capacity])
        for task in pl.spmd(((active + T_TILE - 1) // T_TILE) * HC, name_hint="engram_w8a8_gate"):
            t0, branch = (task // HC) * T_TILE, task % HC
            rows = pl.min(T_TILE, active - t0)
            sx = pl.full([1, T_TILE], dtype=pl.FP32, value=0.)
            sk = pl.full([1, T_TILE], dtype=pl.FP32, value=0.)
            dot_value = pl.full([1, T_TILE], dtype=pl.FP32, value=0.)
            for d0 in pl.range(0, D, D_TILE):
                x_bf16 = pl.slice(flat_hidden, [T_TILE, D_TILE], [t0, branch * D + d0], valid_shape=[rows, D_TILE])
                key_bf16 = pl.slice(kv, [T_TILE, D_TILE], [t0, branch * D + d0], valid_shape=[rows, D_TILE])
                x = pl.cast(x_bf16, pl.FP32)
                key = pl.cast(key_bf16, pl.FP32)
                weight = channel_weight[branch:branch + 1, d0:d0 + D_TILE]
                x_sq = pl.mul(x, x)
                key_sq = pl.mul(key, key)
                sx = pl.add(sx, pl.reshape(pl.row_sum(x_sq), [1, T_TILE]))
                sk = pl.add(sk, pl.reshape(pl.row_sum(key_sq), [1, T_TILE]))
                weighted_x = pl.col_expand_mul(x, weight)
                product = pl.mul(weighted_x, key)
                dot_value = pl.add(dot_value, pl.reshape(pl.row_sum(product), [1, T_TILE]))
            x_mean = pl.add(pl.mul(sx, 1. / D), 1e-20)
            key_mean = pl.add(pl.mul(sk, 1. / D), 1e-20)
            inv_x = pl.rsqrt(x_mean, high_precision=True)
            inv_key = pl.rsqrt(key_mean, high_precision=True)
            inv = pl.mul(inv_x, inv_key)
            dot_value = pl.mul(pl.mul(dot_value, inv), D**-0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot_value), 1e-6))
            bits = pl.cast(pl.reinterpret_view(dot_value, pl.INT32), pl.FP32)
            negative = pl.maximum(pl.minimum(bits, 0.), -1.)
            signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.), 1.))
            gate = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.))
            eligible_i32 = pl.slice(mask, [1, T_TILE], [0, t0], valid_shape=[1, rows])
            gate = pl.reshape(pl.mul(gate, pl.cast(eligible_i32, pl.FP32)), [T_TILE, 1])
            for d0 in pl.range(0, D, D_TILE):
                x_bf16 = pl.slice(flat_hidden, [T_TILE, D_TILE], [t0, branch * D + d0], valid_shape=[rows, D_TILE])
                value_bf16 = pl.slice(kv, [T_TILE, D_TILE], [t0, HC * D + d0], valid_shape=[rows, D_TILE])
                x = pl.cast(x_bf16, pl.FP32)
                value = pl.cast(value_bf16, pl.FP32)
                result_fp32 = pl.add(x, pl.row_expand_mul(value, gate))
                result = pl.cast(result_fp32, pl.BF16, mode="rint")
                start = branch * D + d0
                flat_output[t0:t0 + T_TILE, start:start + D_TILE] = pl.set_validshape(result, rows, D_TILE)
        return lookup, kv, output

    return engram_w8a8
