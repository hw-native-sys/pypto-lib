# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FP32 HC coefficients, delayed stream collapse and BF16 residual expansion."""

import pypto.language as pl


# tiling
D_TILE = 256
TOKEN_TILE = 8
CHANNEL_TILE = 8
MIX_TILE = 8


def make_hc_pre_inline(capacity, config):
    """Build four-stream HC coefficient generation and incoming-premix collapse."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    flat_width = hidden * hc
    channels = (hc + 2) * hc
    iterations = config["hc_sinkhorn_iters"]
    eps = config["hc_eps"]
    norm_eps = config["rms_norm_eps"]

    @pl.jit.inline
    def pre(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, flat_width], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        next_pre: pl.Tensor[[capacity, hc], pl.FP32],
        post: pl.Tensor[[capacity, hc], pl.FP32],
        comb: pl.Tensor[[capacity, hc, hc], pl.FP32],
        collapsed: pl.Tensor[[capacity, hidden], pl.BF16],
        collapsed_fp32: pl.Tensor[[capacity, hidden], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(x, [capacity, flat_width])
        inverse = pl.create_tensor([capacity, TOKEN_TILE], dtype=pl.FP32)
        for block in pl.spmd((active + TOKEN_TILE - 1) // TOKEN_TILE, name_hint="a2_hc_rms"):
            token = block * TOKEN_TILE
            rows = pl.min(TOKEN_TILE, active - token)
            square_sum = pl.full([1, TOKEN_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.range(0, flat_width, D_TILE):
                values = pl.slice(flat, [TOKEN_TILE, D_TILE], [token, inner], valid_shape=[rows, D_TILE])
                values = pl.fillpad(values, pad_value=pl.PadValue.zero)
                values_fp32 = pl.cast(values, pl.FP32)
                square = pl.mul(values_fp32, values_fp32)
                square_sum = pl.add(square_sum, pl.reshape(pl.row_sum(square), [1, TOKEN_TILE]))
            mean = pl.mul(square_sum, 1.0 / flat_width)
            argument = pl.add(mean, norm_eps)
            inverse_column = pl.reshape(pl.rsqrt(argument, high_precision=True), [TOKEN_TILE, 1])
            ones = pl.full([TOKEN_TILE, TOKEN_TILE], dtype=pl.FP32, value=1.0)
            inverse_tile = pl.row_expand_mul(ones, inverse_column)
            inverse[token:token + TOKEN_TILE, :] = pl.set_validshape(inverse_tile, rows, TOKEN_TILE)

        projected = pl.create_tensor([capacity, channels], dtype=pl.FP32)
        for block in pl.spmd(active * (channels // CHANNEL_TILE), name_hint="a2_hc_projection"):
            token = block // (channels // CHANNEL_TILE)
            channel = block % (channels // CHANNEL_TILE) * CHANNEL_TILE
            dot_sum = pl.full([1, CHANNEL_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.range(0, flat_width, D_TILE):
                source = pl.cast(flat[token:token + 1, inner:inner + D_TILE], pl.FP32)
                weight = function[channel:channel + CHANNEL_TILE, inner:inner + D_TILE]
                product = pl.col_expand_mul(weight, source)
                dot_sum = pl.add(dot_sum, pl.reshape(pl.row_sum(product), [1, CHANNEL_TILE]))
            inverse_scalar = pl.read(inverse, [token, 0])
            normalized = pl.mul(dot_sum, inverse_scalar)
            projected[token:token + 1, channel:channel + CHANNEL_TILE] = normalized

        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        for token in pl.spmd(active, name_hint="a2_hc_sinkhorn"):
            scale0 = pl.read(scale, [0])
            scale1 = pl.read(scale, [1])
            scale2 = pl.read(scale, [2])
            pre_logits = pl.tile.full([1, MIX_TILE], dtype=pl.FP32, value=0.0)
            post_logits = pl.tile.full([1, MIX_TILE], dtype=pl.FP32, value=0.0)
            for column in pl.unroll(MIX_TILE):
                logical = column % hc
                pre_value = pl.read(projected, [token, logical]) * scale0 + pl.read(base, [logical])
                post_value = pl.read(projected, [token, hc + logical]) * scale1 + pl.read(base, [hc + logical])
                pl.tile.write(pre_logits, [0, column], pre_value)
                pl.tile.write(post_logits, [0, column], post_value)
            pre_neg = pl.neg(pre_logits)
            pre_exp = pl.exp(pre_neg)
            pre_denominator = pl.add(pre_exp, 1.0)
            pre_sigmoid = pl.recip(pre_denominator)
            pre_result = pl.add(pre_sigmoid, eps)
            post_neg = pl.neg(post_logits)
            post_exp = pl.exp(post_neg)
            post_denominator = pl.add(post_exp, 1.0)
            post_sigmoid = pl.recip(post_denominator)
            post_result = pl.mul(post_sigmoid, 2.0)
            pre_valid = pl.set_validshape(pre_result, 1, hc)
            post_valid = pl.set_validshape(post_result, 1, hc)
            pl.store(pre_valid, [token, 0], next_pre)
            pl.store(post_valid, [token, 0], post)
            logits = pl.tile.full([MIX_TILE, MIX_TILE], dtype=pl.FP32, value=0.0)
            for row in pl.unroll(MIX_TILE):
                for column in pl.unroll(MIX_TILE):
                    logical = 2 * hc + (row % hc) * hc + column % hc
                    value = pl.read(projected, [token, logical]) * scale2 + pl.read(base, [logical])
                    pl.tile.write(logits, [row, column], value)
            logits_valid = pl.set_validshape(logits, hc, hc)
            logits_padded = pl.fillpad(logits_valid, pad_value=pl.PadValue.min)
            row_tmp = pl.create_tile([MIX_TILE, MIX_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_max = pl.row_max(logits_padded, row_tmp)
            shifted = pl.row_expand_sub(logits_padded, row_max)
            exponential = pl.exp(shifted)
            row_sum = pl.row_sum(exponential, row_tmp)
            probabilities_softmax = pl.add(pl.row_expand_div(exponential, row_sum), eps)
            probabilities_valid = pl.set_validshape(probabilities_softmax, hc, hc)
            probabilities_padded = pl.fillpad(probabilities_valid, pad_value=pl.PadValue.zero)
            column_sum = pl.col_sum(probabilities_padded)
            column_sum = pl.add(column_sum, eps)
            probabilities = pl.col_expand_div(probabilities_padded, column_sum)
            for _ in pl.range(iterations - 1):
                iter_valid = pl.set_validshape(probabilities, hc, hc)
                iter_padded = pl.fillpad(iter_valid, pad_value=pl.PadValue.zero)
                row_sum = pl.row_sum(iter_padded, row_tmp)
                row_sum = pl.add(row_sum, eps)
                row_normalized = pl.row_expand_div(iter_padded, row_sum)
                row_normalized_valid = pl.set_validshape(row_normalized, hc, hc)
                row_normalized_padded = pl.fillpad(row_normalized_valid, pad_value=pl.PadValue.zero)
                column_sum = pl.col_sum(row_normalized_padded)
                column_sum = pl.add(column_sum, eps)
                probabilities = pl.col_expand_div(row_normalized_padded, column_sum)
            for row in pl.unroll(hc):
                comb_value = pl.slice(probabilities, [1, MIX_TILE], [row, 0], valid_shape=[1, hc])
                pl.store(comb_value, [token, row * hc], comb_flat)

        for block in pl.spmd(active * (hidden // D_TILE), name_hint="a2_hc_collapse"):
            token = block // (hidden // D_TILE)
            offset = block % (hidden // D_TILE) * D_TILE
            collapsed_tile = pl.full([1, D_TILE], dtype=pl.FP32, value=0.0)
            for stream in pl.unroll(hc):
                source_offset = stream * hidden + offset
                source_bf16 = flat[token:token + 1, source_offset:source_offset + D_TILE]
                source_fp32 = pl.cast(source_bf16, pl.FP32)
                coefficient = pl.read(incoming, [token, stream])
                term = pl.mul(source_fp32, coefficient)
                collapsed_tile = pl.add(collapsed_tile, term)
            collapsed_fp32[token:token + 1, offset:offset + D_TILE] = collapsed_tile
            collapsed[token:token + 1, offset:offset + D_TILE] = pl.cast(collapsed_tile, pl.BF16, mode="rint")
        return next_pre, post, comb, collapsed

    return pre


def make_hc_post_inline(capacity, config):
    """Build BF16 residual expansion with FP32 combination arithmetic."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]

    @pl.jit.inline
    def post(
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        residual: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        post_mix: pl.Tensor[[capacity, hc], pl.FP32],
        comb: pl.Tensor[[capacity, hc, hc], pl.FP32],
        output: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(residual, [capacity, hc * hidden])
        output_flat = pl.reshape(output, [capacity, hc * hidden])
        comb_flat = pl.reshape(comb, [capacity, hc * hc])
        for block in pl.spmd(active * hc * (hidden // D_TILE), name_hint="a2_hc_post"):
            token = block // (hc * (hidden // D_TILE))
            stream = block // (hidden // D_TILE) % hc
            offset = block % (hidden // D_TILE) * D_TILE
            value = pl.cast(sublayer[token:token + 1, offset:offset + D_TILE], pl.FP32)
            coefficient = pl.read(post_mix, [token, stream])
            merged = pl.mul(value, coefficient)
            for source_stream in pl.unroll(hc):
                source_offset = source_stream * hidden + offset
                source = flat[token:token + 1, source_offset:source_offset + D_TILE]
                source = pl.cast(source, pl.FP32)
                coefficient = pl.read(comb_flat, [token, source_stream * hc + stream])
                term = pl.mul(source, coefficient)
                merged = pl.add(merged, term)
            destination = stream * hidden + offset
            merged_bf16 = pl.cast(merged, pl.BF16, mode="rint")
            output_flat[token:token + 1, destination:destination + D_TILE] = merged_bf16
        return output

    return post

def make_hc(capacity, config):
    """Build the standalone HC operator diagnostic entry."""
    hidden = config["hidden_size"]
    hc = config["hc_mult"]
    channels = (hc + 2) * hc
    pre = make_hc_pre_inline(capacity, config)
    post_fn = make_hc_post_inline(capacity, config)

    @pl.jit
    def hc_entry(
        x: pl.Tensor[[capacity, hc, hidden], pl.BF16],
        function: pl.Tensor[[channels, hc * hidden], pl.FP32],
        scale: pl.Tensor[[3], pl.FP32],
        base: pl.Tensor[[channels], pl.FP32],
        incoming: pl.Tensor[[capacity, hc], pl.FP32],
        sublayer: pl.Tensor[[capacity, hidden], pl.BF16],
        next_pre: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, hc], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, hc, hc], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        merged: pl.InOut[pl.Tensor[[capacity, hc, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        collapsed_fp32 = pl.create_tensor([capacity, hidden], dtype=pl.FP32)
        pre(
            x, function, scale, base, incoming, next_pre, post, comb,
            collapsed, collapsed_fp32, active,
        )
        post_fn(sublayer, x, post, comb, merged, active)
        return next_pre, post, comb, collapsed, merged

    return hc_entry
