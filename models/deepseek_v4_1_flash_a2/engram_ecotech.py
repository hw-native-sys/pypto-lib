# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""INT8/F32 Engram lookup, BF16 projection and block32 original-basis gating."""

import pypto.language as pl

from .qkv_proj_rope import make_bf16_projection_staged


# model config
D, HC, COLS, WIDTH = 5120, 4, 24, 256
K, N = COLS * WIDTH, (HC + 1) * D

# tiling
T_TILE, D_TILE, ROTATION_TILE = 16, 512, 32


def make_resident_int8_lookup(capacity, table_rows):
    """Gather resident rows with group32 FP32 scales into a BF16 lookup."""

    @pl.jit.inline
    def gather(
        staging_ids: pl.Tensor[[capacity, COLS], pl.INT32],
        table: pl.Tensor[[table_rows, WIDTH], pl.INT8],
        table_scales: pl.Tensor[[table_rows, 8], pl.FP32],
        lookup: pl.Tensor[[capacity, K], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for token in pl.spmd(active, name_hint="engram_ecotech_i8_lookup"):
            for column in pl.range(COLS):
                row = pl.read(staging_ids, [token, column])
                for group in pl.range(8):
                    decoded = pl.full([1, 32], dtype=pl.BF16, value=0.)
                    if row >= 0:
                        if row < table_rows:
                            codes_half = pl.cast(table[row:row + 1, group * 32:group * 32 + 32], pl.FP16)
                            codes = pl.cast(codes_half, pl.FP32)
                            scale = pl.read(table_scales, [row, group])
                            decoded = pl.cast(pl.mul(codes, scale), pl.BF16, mode="rint")
                    start = column * WIDTH + group * 32
                    lookup[token:token + 1, start:start + 32] = decoded
        return lookup

    return gather


def make_original_basis_hidden(capacity):
    """Restore FP32 hidden blocks with hidden_rotated @ rotation_block.T."""

    @pl.jit.inline
    def restore(
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        rotation_block: pl.Tensor[[ROTATION_TILE, ROTATION_TILE], pl.FP32],
        original: pl.Tensor[[capacity, HC, D], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        flat_hidden = pl.reshape(hidden, [capacity * HC, D])
        flat_original = pl.reshape(original, [capacity * HC, D])
        for row in pl.spmd(active * HC, name_hint="engram_ecotech_original_basis"):
            rotation = rotation_block[:, :]
            for start in pl.range(0, D, ROTATION_TILE):
                values = pl.cast(flat_hidden[row:row + 1, start:start + ROTATION_TILE], pl.FP32)
                products = pl.col_expand_mul(rotation, values)
                restored = pl.reshape(pl.row_sum(products), [1, ROTATION_TILE])
                flat_original[row:row + 1, start:start + ROTATION_TILE] = restored
        return original

    return restore


def make_original_basis_gate(capacity, eps):
    """Compute the original-basis gate and add the already rotated value."""

    @pl.jit.inline
    def gate(
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        original: pl.Tensor[[capacity, HC, D], pl.FP32],
        kv: pl.Tensor[[capacity, N], pl.BF16],
        channel_weight: pl.Tensor[[HC, D], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        probabilities: pl.Tensor[[capacity, HC], pl.FP32],
        output: pl.Tensor[[capacity, HC, D], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat_original = pl.reshape(original, [capacity, HC * D])
        flat_hidden = pl.reshape(hidden, [capacity, HC * D])
        flat_output = pl.reshape(output, [capacity, HC * D])
        mask = pl.reshape(token_mask, [1, capacity])
        for task in pl.spmd(((active + T_TILE - 1) // T_TILE) * HC, name_hint="engram_ecotech_gate"):
            t0, branch = (task // HC) * T_TILE, task % HC
            rows = pl.min(T_TILE, active - t0)
            sx = pl.full([1, T_TILE], dtype=pl.FP32, value=0.)
            sk = pl.full([1, T_TILE], dtype=pl.FP32, value=0.)
            dot = pl.full([1, T_TILE], dtype=pl.FP32, value=0.)
            for d0 in pl.range(0, D, D_TILE):
                x = pl.slice(flat_original, [T_TILE, D_TILE], [t0, branch * D + d0], valid_shape=[rows, D_TILE])
                key_bf16 = pl.slice(kv, [T_TILE, D_TILE], [t0, branch * D + d0], valid_shape=[rows, D_TILE])
                key = pl.cast(key_bf16, pl.FP32)
                weight = channel_weight[branch:branch + 1, d0:d0 + D_TILE]
                sx = pl.add(sx, pl.reshape(pl.row_sum(pl.mul(x, x)), [1, T_TILE]))
                sk = pl.add(sk, pl.reshape(pl.row_sum(pl.mul(key, key)), [1, T_TILE]))
                weighted_x = pl.col_expand_mul(x, weight)
                dot = pl.add(dot, pl.reshape(pl.row_sum(pl.mul(weighted_x, key)), [1, T_TILE]))
            inv_x = pl.rsqrt(pl.add(pl.mul(sx, 1. / D), eps), high_precision=True)
            inv_key = pl.rsqrt(pl.add(pl.mul(sk, 1. / D), eps), high_precision=True)
            inv = pl.mul(inv_x, inv_key)
            dot = pl.mul(pl.mul(dot, inv), D**-0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot), 1e-6))
            bits = pl.cast(pl.reinterpret_view(dot, pl.INT32), pl.FP32)
            negative = pl.maximum(pl.minimum(bits, 0.), -1.)
            signed = pl.mul(magnitude, pl.add(pl.mul(negative, 2.), 1.))
            eligible = pl.cast(pl.slice(mask, [1, T_TILE], [0, t0], valid_shape=[1, rows]), pl.FP32)
            probability = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.))
            probability = pl.reshape(pl.mul(probability, eligible), [T_TILE, 1])
            gate_values = pl.set_validshape(probability, rows, 1)
            probabilities[t0:t0 + T_TILE, branch:branch + 1] = gate_values
            for d0 in pl.range(0, D, D_TILE):
                x_bf16 = pl.slice(flat_hidden, [T_TILE, D_TILE], [t0, branch * D + d0], valid_shape=[rows, D_TILE])
                value_bf16 = pl.slice(kv, [T_TILE, D_TILE], [t0, HC * D + d0], valid_shape=[rows, D_TILE])
                value = pl.cast(value_bf16, pl.FP32)
                result = pl.add(pl.cast(x_bf16, pl.FP32), pl.row_expand_mul(value, probability))
                result = pl.cast(result, pl.BF16, mode="rint")
                start = branch * D + d0
                flat_output[t0:t0 + T_TILE, start:start + D_TILE] = pl.set_validshape(result, rows, D_TILE)
        return probabilities, output

    return gate


def make_engram_ecotech(capacity, table_rows, rms_norm_eps=1e-20):
    """Compose resident lookup, shared BF16 projection and rotated-residual gating."""
    if capacity < 1 or capacity % 16 or table_rows < 1 or rms_norm_eps <= 0:
        raise ValueError("Use a positive padded capacity, resident row count and source norm epsilon")
    gather = make_resident_int8_lookup(capacity, table_rows)
    project = make_bf16_projection_staged(K, N)
    restore = make_original_basis_hidden(capacity)
    gate = make_original_basis_gate(capacity, rms_norm_eps)

    @pl.jit
    def engram_ecotech(
        staging_ids: pl.Tensor[[capacity, COLS], pl.INT32],
        table: pl.Tensor[[table_rows, WIDTH], pl.INT8],
        table_scales: pl.Tensor[[table_rows, 8], pl.FP32],
        wkv: pl.Tensor[[K, N], pl.BF16],
        channel_weight: pl.Tensor[[HC, D], pl.FP32],
        hidden: pl.Tensor[[capacity, HC, D], pl.BF16],
        rotation_block: pl.Tensor[[ROTATION_TILE, ROTATION_TILE], pl.FP32],
        token_mask: pl.Tensor[[capacity], pl.INT32],
        lookup: pl.InOut[pl.Tensor[[capacity, K], pl.BF16]],
        kv: pl.InOut[pl.Tensor[[capacity, N], pl.BF16]],
        original: pl.InOut[pl.Tensor[[capacity, HC, D], pl.FP32]],
        output: pl.InOut[pl.Tensor[[capacity, HC, D], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        gather(staging_ids, table, table_scales, lookup, active)
        project(lookup, wkv, kv, active)
        restore(hidden, rotation_block, original, active)
        probabilities = pl.create_tensor([capacity, HC], dtype=pl.FP32)
        gate(hidden, original, kv, channel_weight, token_mask, probabilities, output, active)
        return lookup, kv, original, output

    return engram_ecotech
