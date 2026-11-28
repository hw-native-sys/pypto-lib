# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-token W8A8 quantization for A3 projections and experts."""

import inspect

import pypto.language as pl


K_TILE = 256
VEC_M_TILE = 8
PREFILL_QUANT_WORKERS = 48
QUANT_REDUCE_TILE = 1024
CHECKPOINT_COLUMNS = pl.dynamic("CHECKPOINT_QUANT_COLUMNS")
CHECKPOINT_ROWS = pl.dynamic("CHECKPOINT_QUANT_ROWS")
_TILE_RSQRT_HAS_HIGH_PRECISION = "high_precision" in inspect.signature(pl.tile.rsqrt).parameters

if _TILE_RSQRT_HAS_HIGH_PRECISION:
    @pl.jit.inline(auto_scope=False)
    def _tile_rsqrt_for_norm(
        value: pl.Tile[[VEC_M_TILE, 1], pl.FP32],
        scratch: pl.Tile[[VEC_M_TILE, 1], pl.FP32],
    ):
        return pl.tile.rsqrt(value, tmp=scratch, high_precision=True)
else:
    @pl.jit.inline(auto_scope=False)
    def _tile_rsqrt_for_norm(
        value: pl.Tile[[VEC_M_TILE, 1], pl.FP32],
        scratch: pl.Tile[[VEC_M_TILE, 1], pl.FP32],
    ):
        return pl.tile.rsqrt(value, tmp=scratch)


@pl.jit.inline(auto_scope=False)
def _checkpoint_codes(
    value: pl.Tile[[CHECKPOINT_ROWS, CHECKPOINT_COLUMNS], pl.FP32],
    denominator: pl.Tile[[CHECKPOINT_ROWS, 1], pl.FP32],
    inverse: pl.Tile[[CHECKPOINT_ROWS, 1], pl.FP32],
):
    """Resolve signed half-way cases against the original BF16 row peak."""
    magnitude = pl.abs(value)
    # Scale both sides of the midpoint comparison by a power of two to keep finite BF16 inputs in range.
    scaled_magnitude = pl.mul(magnitude, 127.0 / 128.0)
    estimate = pl.row_expand_mul(scaled_magnitude, inverse)
    floor_int = pl.cast(pl.minimum(estimate, 127.0), pl.INT32, mode="floor")
    lower = pl.cast(floor_int, pl.FP32)
    midpoint = pl.row_expand_mul(pl.add(lower, 0.5), denominator)
    # Nonnegative FP32 bit patterns are ordered. For an odd floor, include
    # equality at the midpoint so that an exact tie rounds to the even code.
    distance = pl.sub(pl.reinterpret_view(midpoint, pl.INT32), pl.reinterpret_view(scaled_magnitude, pl.INT32))
    odd = pl.sub(floor_int, pl.shls(pl.shrs(floor_int, 1), 1))
    round_up = pl.shrs(pl.sub(distance, odd), 31)
    nearest = pl.cast(pl.sub(floor_int, round_up), pl.FP32)
    signed = pl.tile.select(pl.tile.cmps(value, 0.0, cmp_type=2), pl.neg(nearest), nearest)
    encoded = pl.cast(signed, pl.INT8, mode="rint")
    return encoded


@pl.jit.inline(auto_scope=False)
def _encode_checkpoint_panel(
    source: pl.Tensor, payload: pl.Tensor, row: pl.Scalar[pl.INDEX],
    k0: pl.Scalar[pl.INDEX], rows: pl.Scalar[pl.INDEX],
    denominator: pl.Tile[[8, 1], pl.FP32], inverse: pl.Tile[[8, 1], pl.FP32],
):
    raw = pl.load(source, [row, k0], [8, 256], valid_shape=[rows, 256])
    padded = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 8, 256)
    value = pl.cast(padded, pl.FP32)
    encoded = _checkpoint_codes(value, denominator, inverse)
    pl.store(encoded, [row, k0], payload)


@pl.jit.inline(auto_scope=False)
def _quantize_cached_rows(
    source: pl.Tensor, payload: pl.Tensor, scales: pl.Tensor, active: pl.Scalar[pl.INT32],
    capacity: pl.Scalar[pl.INDEX], width: pl.constexpr, checkpoint: pl.constexpr,
):
    """Keep complete input rows in UB for the peak and midpoint passes."""
    blocks = capacity // VEC_M_TILE
    if not checkpoint:
        blocks = (active + VEC_M_TILE - 1) // VEC_M_TILE
    workers = pl.min(pl.min(blocks, PREFILL_QUANT_WORKERS), pl.system.available_aiv_count())
    for worker in pl.spmd(workers, name_hint="prefill_token_quantize_whole_row"):
        start = worker * blocks // workers
        end = (worker + 1) * blocks // workers
        for block in pl.pipeline(start, end, stage=2):
            row = block * VEC_M_TILE
            rows = pl.max(0, pl.min(VEC_M_TILE, active - row))
            raw = pl.load(source, [row, 0], [VEC_M_TILE, width], valid_shape=[rows, width])
            cached = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), VEC_M_TILE, width)
            fp = pl.cast(cached, pl.FP32)
            scratch = pl.create_tile([VEC_M_TILE, width], dtype=pl.FP32)
            maximum = pl.row_max(pl.abs(fp), tmp_tile=scratch)
            if checkpoint:
                factor = pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30)
                pl.store(factor, [row, 0], scales)
                denominator = pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0)
                inverse = pl.recip(denominator)
                for k0 in pl.range(0, width, 256):
                    part = pl.tile.slice(fp, [VEC_M_TILE, 256], [0, k0])
                    encoded = _checkpoint_codes(part, denominator, inverse)
                    pl.store(encoded, [row, k0], payload)
            else:
                row_peak = pl.reshape(maximum, [1, VEC_M_TILE])
                numerator = pl.tile.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
                inverse = pl.reshape(pl.div(numerator, pl.maximum(row_peak, 1e-30)), [VEC_M_TILE, 1])
                scale = pl.mul(maximum, 1.0 / 127.0)
                pl.store(pl.set_validshape(scale, rows, 1), [row, 0], scales)
                rounded = pl.cast(pl.row_expand_mul(fp, inverse), pl.INT32, mode="rint")
                codes = pl.cast(rounded, pl.INT8, mode="trunc")
                pl.store(pl.set_validshape(codes, rows, width), [row, 0], payload)
    return payload, scales


@pl.jit.inline(auto_scope=False)
def _prefill_quant_params(maximum: pl.Tile[[VEC_M_TILE, 1], pl.FP32]):
    factor = pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30)
    divisor = pl.reshape(pl.maximum(maximum, 127e-30), [1, VEC_M_TILE])
    numerator = pl.tile.full([1, VEC_M_TILE], dtype=pl.FP32, value=127.0)
    multiplier = pl.div(numerator, divisor)
    # Preserve exact powers of two. Round other multipliers up for the peak/2 tie.
    bits = pl.reinterpret_view(multiplier, pl.INT32)
    exponent = pl.shls(pl.shrs(bits, 23), 23)
    bias = pl.minimum(pl.sub(bits, exponent), 1)
    biased = pl.reinterpret_view(pl.add(bits, bias), pl.FP32)
    return factor, pl.reshape(biased, [VEC_M_TILE, 1])


@pl.jit.inline(auto_scope=False)
def _quantize_prefill_rows(
    source: pl.Tensor,
    payload: pl.Tensor,
    scales: pl.Tensor,
    active: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
    width: pl.constexpr,
    panel_width: pl.constexpr,
):
    """Read BF16 activations once and encode from the cached row tile."""
    blocks = padded_rows // VEC_M_TILE
    workers = pl.min(pl.min(blocks, PREFILL_QUANT_WORKERS), pl.system.available_aiv_count())
    for worker in pl.spmd(workers, name_hint="prefill_activation_quant"):
        begin = worker * blocks // workers
        end = (worker + 1) * blocks // workers
        for block in pl.range(begin, end):
            row = block * VEC_M_TILE
            valid = pl.max(0, pl.min(VEC_M_TILE, active - row))
            raw = pl.load(source, [row, 0], [VEC_M_TILE, width], valid_shape=[valid, width])
            padded = pl.fillpad(raw, pad_value=pl.PadValue.zero)
            cached = pl.set_validshape(padded, VEC_M_TILE, width)
            peak_bits = pl.tile.full([VEC_M_TILE, panel_width], dtype=pl.INT16, value=0)
            for col in pl.range(0, width, panel_width):
                panel = pl.tile.slice(cached, [VEC_M_TILE, panel_width], [0, col])
                panel_bits = pl.reinterpret_view(panel, pl.INT16)
                magnitude_bits = pl.ands(panel_bits, 32767)
                peak_bits = pl.maximum(peak_bits, magnitude_bits)
            peak_bf16 = pl.reinterpret_view(peak_bits, pl.BF16)
            peak = pl.cast(peak_bf16, pl.FP32)
            scratch = pl.create_tile([VEC_M_TILE, panel_width], dtype=pl.FP32)
            maximum = pl.row_max(peak, tmp_tile=scratch)
            factor, inverse = _prefill_quant_params(maximum)
            pl.store(factor, [row, 0], scales)
            for col in pl.range(0, width, panel_width):
                panel = pl.tile.slice(cached, [VEC_M_TILE, panel_width], [0, col])
                value = pl.cast(panel, pl.FP32)
                normalized = pl.row_expand_mul(value, inverse)
                # Round to integer before converting the exact codes through FP16.
                rounded = pl.cast(normalized, pl.INT16, mode="rint")
                exact_fp16 = pl.cast(rounded, pl.FP16, mode="rint")
                codes = pl.cast(exact_fp16, pl.INT8, mode="trunc")
                pl.store(codes, [row, col], payload)
    return payload, scales


@pl.jit.inline(auto_scope=False)
def normalize_quantize_prefill_projection(
    source: pl.Tensor, weight: pl.Tensor, normalized: pl.Tensor,
    payload: pl.Tensor, scales: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], padded_rows: pl.Scalar[pl.INDEX],
    width: pl.constexpr, eps: pl.constexpr,
):
    """Normalize Q-latent rows and quantize the rounded BF16 values in the same tile."""
    blocks = padded_rows // VEC_M_TILE
    workers = pl.min(pl.min(blocks, PREFILL_QUANT_WORKERS), pl.system.available_aiv_count())
    for worker in pl.spmd(workers, name_hint="prefill_q_norm_quant"):
        gamma = pl.reshape(pl.cast(pl.load(weight, [0], [width]), pl.FP32), [1, width])
        begin = worker * blocks // workers
        end = (worker + 1) * blocks // workers
        for block in pl.range(begin, end):
            row = block * VEC_M_TILE
            rows = pl.max(0, pl.min(VEC_M_TILE, num_tokens - row))
            raw = pl.load(source, [row, 0], [VEC_M_TILE, width], valid_shape=[rows, width])
            padded = pl.fillpad(raw, pad_value=pl.PadValue.zero)
            value = pl.cast(pl.set_validshape(padded, VEC_M_TILE, width), pl.FP32)
            square = pl.mul(value, value)
            scratch = pl.create_tile([VEC_M_TILE, width], dtype=pl.FP32)
            total = pl.reshape(pl.row_sum(square, tmp_tile=scratch), [1, VEC_M_TILE])
            mean = pl.add(pl.mul(total, 1.0 / width), eps)
            rsqrt_scratch = pl.create_tile([1, VEC_M_TILE], dtype=pl.FP32)
            inv = pl.tile.rsqrt(mean, rsqrt_scratch, high_precision=True)
            scaled = pl.row_expand_mul(value, pl.reshape(inv, [VEC_M_TILE, 1]))
            weighted = pl.col_expand_mul(scaled, gamma)
            rounded = pl.cast(weighted, pl.BF16, mode="rint")
            pl.store(pl.set_validshape(rounded, rows, width), [row, 0], normalized)
            activation = pl.cast(rounded, pl.FP32)
            peak = pl.row_max(pl.abs(activation), tmp_tile=scratch)
            scale, inverse = _prefill_quant_params(peak)
            pl.store(scale, [row, 0], scales)
            quant = pl.row_expand_mul(activation, inverse)
            encoded = pl.cast(quant, pl.INT32, mode="rint")
            pl.store(pl.cast(encoded, pl.INT8, mode="trunc"), [row, 0], payload)
    return normalized, payload, scales


DECODE_QUANT_ENCODE_TILE = 256
DECODE_QUANT_REDUCE_TILE = 1024
DECODE_HIDDEN_WIDTH = pl.dynamic("DECODE_HIDDEN_WIDTH")
@pl.jit.inline(auto_scope=False)
def _quantize_decode_active_row(
    source: pl.Tensor,
    payload: pl.Tensor,
    scale: pl.Tensor,
    width: pl.constexpr,
):
    """Quantize only the live Decode row while preserving padded-row outputs."""
    encode_tiles = width // DECODE_QUANT_ENCODE_TILE
    encode_workers = pl.max(1, pl.min(encode_tiles, pl.system.available_aiv_count()))
    for encode_worker in pl.spmd(encode_workers, name_hint="quant_decode_active_row"):
        for panel in pl.range(encode_worker, encode_tiles, encode_workers):
            col = panel * DECODE_QUANT_ENCODE_TILE
            whole = pl.reshape(pl.load(source, [0, 0], [1, width]), [8, width // 8])
            fp32 = pl.cast(whole, pl.FP32)
            scratch = pl.create_tile([8, width // 8], dtype=pl.FP32)
            slice_peaks = pl.row_max(pl.abs(fp32), tmp_tile=scratch)
            zeros = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
            maximum = pl.row_max(
                pl.col_expand_add(zeros, pl.reshape(slice_peaks, [1, 8])),
                tmp_tile=zeros,
            )
            if panel == 0:
                active_scale = pl.reshape(pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30), [1, 8])
                padding_scale = pl.tile.full([1, 8], dtype=pl.FP32, value=1e-30)
                lanes = pl.cast(pl.tile.arange(0, [1, 8], dtype=pl.INT32), pl.FP32)
                row_scales = pl.tile.select(pl.tile.cmps(lanes, 0.0, cmp_type=0), active_scale, padding_scale)
                pl.store(pl.reshape(row_scales, [8, 1]), [0, 0], scale)
                pl.store(pl.reshape(padding_scale, [8, 1]), [8, 0], scale)
            denominator = pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0)
            inverse = pl.recip(denominator)
            value = pl.cast(
                pl.reshape(pl.load(source, [0, col], [1, DECODE_QUANT_ENCODE_TILE]), [8, DECODE_QUANT_ENCODE_TILE // 8]),
                pl.FP32,
            )
            codes = _checkpoint_codes(value, denominator, inverse)
            pl.store(pl.reshape(codes, [1, DECODE_QUANT_ENCODE_TILE]), [0, col], payload)
            padding = pl.cast(
                pl.tile.full([8, DECODE_QUANT_ENCODE_TILE], dtype=pl.FP32, value=0.0),
                pl.INT8,
                mode="rint",
            )
            pl.store(pl.set_validshape(padding, 7, DECODE_QUANT_ENCODE_TILE), [1, col], payload)
            pl.store(padding, [8, col], payload)
    return payload, scale



@pl.jit.inline(auto_scope=False)
def _quantize_decode_wide_row(
    source: pl.Tensor,
    payload: pl.Tensor,
    scale: pl.Tensor,
    width: pl.constexpr,
):
    """Encode two chunks of the active row and clear padding in one Vector task."""
    wide_tiles = 2 + width // DECODE_QUANT_REDUCE_TILE
    wide_workers = pl.max(1, pl.min(wide_tiles, pl.system.available_aiv_count()))
    for wide_worker in pl.spmd(wide_workers, name_hint="quant_decode_wide_row"):
        for worker in pl.range(wide_worker, wide_tiles, wide_workers):
            if worker < 2:
                raw = pl.load(source, [0, 0], [1, width])
                grouped = pl.reshape(raw, [8, width // 8])
                value = pl.cast(grouped, pl.FP32)
                absolute = pl.abs(value)
                scratch = pl.create_tile([8, width // 8], dtype=pl.FP32)
                peaks = pl.row_max(absolute, tmp_tile=scratch)
                zero = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
                peak_tile = pl.col_expand_add(zero, pl.reshape(peaks, [1, 8]))
                maximum = pl.row_max(peak_tile, tmp_tile=zero)
                if worker == 0:
                    active = pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30)
                    inactive = pl.tile.full([1, 8], dtype=pl.FP32, value=1e-30)
                    lanes = pl.cast(pl.tile.arange(0, [1, 8], dtype=pl.INT32), pl.FP32)
                    active_lane = pl.tile.cmps(lanes, 0.0, cmp_type=0)
                    row_scales = pl.tile.select(active_lane, pl.reshape(active, [1, 8]), inactive)
                    pl.store(pl.reshape(row_scales, [8, 1]), [0, 0], scale)
                    pl.store(pl.reshape(inactive, [8, 1]), [8, 0], scale)

                denominator = pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0)
                inverse = pl.recip(denominator)
                column = worker * (width // 2)
                panel = pl.load(source, [0, column], [1, width // 2])
                panel_rows = pl.reshape(panel, [8, width // 16])
                activation = pl.cast(panel_rows, pl.FP32)
                encoded = _checkpoint_codes(activation, denominator, inverse)
                pl.store(pl.reshape(encoded, [1, width // 2]), [0, column], payload)
            else:
                column = (worker - 2) * DECODE_QUANT_REDUCE_TILE
                zeros = pl.tile.full([8, DECODE_QUANT_REDUCE_TILE], dtype=pl.FP32, value=0.0)
                padding = pl.cast(zeros, pl.INT8, mode="rint")
                leading = pl.set_validshape(padding, 7, DECODE_QUANT_REDUCE_TILE)
                pl.store(leading, [1, column], payload)
                pl.store(padding, [8, column], payload)
    return payload, scale


@pl.jit.inline(auto_scope=False)
def _wide_decode_peak(row: pl.Tile[[1, DECODE_HIDDEN_WIDTH], pl.BF16], width: pl.constexpr):
    value = pl.cast(pl.reshape(row, [8, width // 8]), pl.FP32)
    scratch = pl.create_tile([8, width // 8], dtype=pl.FP32)
    peaks = pl.reshape(pl.row_max(pl.abs(value), tmp_tile=scratch), [1, 8])
    zero = pl.tile.full([8, 8], dtype=pl.FP32, value=0.0)
    return pl.row_max(pl.col_expand_add(zero, peaks), tmp_tile=zero)


@pl.jit.inline(auto_scope=False)
def _quantize_decode_wide_pair(
    source: pl.Tensor, payload: pl.Tensor, scale: pl.Tensor,
    width: pl.constexpr,
):
    """Quantize two live rows and clear the rest of the 16-row slab."""
    pair_quant_tiles = 4 + width // DECODE_QUANT_ENCODE_TILE
    pair_quant_workers = pl.max(1, pl.min(pair_quant_tiles, pl.system.available_aiv_count()))
    for pair_quant_worker in pl.spmd(pair_quant_workers, name_hint="quant_fused_small_batch"):
        for worker in pl.range(pair_quant_worker, pair_quant_tiles, pair_quant_workers):
            if worker < 4:
                row = worker // 2
                col = (worker % 2) * (width // 2)
                full = pl.load(source, [row, 0], [1, width])
                maximum = _wide_decode_peak(full, width)
                if worker == 0:
                    inactive = pl.tile.full([1, 8], dtype=pl.FP32, value=1e-30)
                    lanes = pl.cast(pl.tile.arange(0, [1, 8], dtype=pl.INT32), pl.FP32)
                    peak0 = pl.reshape(pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30), [1, 8])
                    row_scales = pl.tile.select(pl.tile.cmps(lanes, 0.0, cmp_type=0), peak0, inactive)
                    second = pl.load(source, [1, 0], [1, width])
                    other = _wide_decode_peak(second, width)
                    peak1 = pl.reshape(pl.maximum(pl.mul(other, 1.0 / 127.0), 1e-30), [1, 8])
                    row_scales = pl.tile.select(pl.tile.cmps(lanes, 1.0, cmp_type=0), peak1, row_scales)
                    pl.store(pl.reshape(row_scales, [8, 1]), [0, 0], scale)
                    pl.store(pl.reshape(inactive, [8, 1]), [8, 0], scale)
                denominator = pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0)
                inverse = pl.recip(denominator)
                panel = pl.load(source, [row, col], [1, width // 2])
                value = pl.cast(pl.reshape(panel, [8, width // 16]), pl.FP32)
                encoded = _checkpoint_codes(value, denominator, inverse)
                pl.store(pl.reshape(encoded, [1, (width // 2)]), [row, col], payload)
            else:
                col = (worker - 4) * DECODE_QUANT_ENCODE_TILE
                zeros = pl.cast(pl.tile.full([8, DECODE_QUANT_ENCODE_TILE], dtype=pl.FP32, value=0.0),
                                pl.INT8, mode="rint")
                pl.store(pl.set_validshape(zeros, 6, DECODE_QUANT_ENCODE_TILE), [2, col], payload)
                pl.store(zeros, [8, col], payload)
    return payload, scale



@pl.jit.inline(auto_scope=False)
def quantize_projection_activation(
    source: pl.Tensor, num_tokens: pl.Scalar[pl.INT32], width: pl.constexpr, dense_query: pl.constexpr,
    prefill: pl.constexpr,
):
    """Quantize the resident A3 checkpoint's padded rows with ties-to-even rounding."""
    capacity = pl.tensor.dim(source, 0)
    padded_rows = (capacity + 15) // 16 * 16
    payload = pl.create_tensor([padded_rows, width], dtype=pl.INT8)
    scale = pl.create_tensor([padded_rows, 1], dtype=pl.FP32)
    _quantize_projection_activation_into(source, payload, scale, num_tokens, padded_rows, width, dense_query, prefill)
    return payload, scale


@pl.jit.inline(auto_scope=False)
def _quantize_projection_activation_into(
    source: pl.Tensor, payload: pl.Tensor, scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], padded_rows: pl.Scalar[pl.INT32],
    width: pl.constexpr, dense_query: pl.constexpr, prefill: pl.constexpr,
):
    if not prefill and num_tokens == 1 and padded_rows == 16:
        if width == 5120:
            _quantize_decode_wide_row(source, payload, scale, width)
        else:
            _quantize_decode_active_row(source, payload, scale, width)
    elif not prefill and width == 5120 and num_tokens == 2 and padded_rows == 16:
        _quantize_decode_wide_pair(source, payload, scale, width)
    elif prefill:
        if width == 2304:
            _quantize_prefill_rows(source, payload, scale, num_tokens, padded_rows, width, width)
        else:
            _quantize_prefill_rows(source, payload, scale, num_tokens, padded_rows, width, 1280)
    elif padded_rows <= 32:
        panels = (width + QUANT_REDUCE_TILE - 1) // QUANT_REDUCE_TILE
        partials = pl.create_tensor([padded_rows // 8 * panels, 8], dtype=pl.FP32)
        absmax_tiles = padded_rows // 8 * panels
        absmax_workers = pl.max(1, pl.min(absmax_tiles, pl.system.available_aiv_count()))
        with pl.spmd(absmax_workers, name_hint="quant_absmax") as partials_ready:
            absmax_worker = pl.tile.get_block_idx()
            for shard in pl.range(absmax_worker, absmax_tiles, absmax_workers):
                shard_row = shard // panels * 8
                shard_col = shard % panels * QUANT_REDUCE_TILE
                shard_rows = pl.max(0, pl.min(8, num_tokens - shard_row))
                shard_cols = pl.min(QUANT_REDUCE_TILE, width - shard_col)
                raw = pl.tile.load(source, [shard_row, shard_col], [8, QUANT_REDUCE_TILE], valid_shape=[shard_rows, shard_cols])
                padded = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), 8, QUANT_REDUCE_TILE)
                absolute = pl.abs(pl.cast(padded, pl.FP32))
                scratch = pl.create_tile([8, QUANT_REDUCE_TILE], dtype=pl.FP32)
                shard_max = pl.reshape(pl.row_max(absolute, tmp_tile=scratch), [1, 8])
                pl.tile.store(shard_max, [shard, 0], partials)
        tiles = padded_rows // 8 * (width // 256)
        workers = pl.min(tiles, pl.system.available_aiv_count())
        with pl.spmd(workers, name_hint="quant_encode", deps=[partials_ready]):
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, tiles, workers):
                group = block // (width // 256)
                row = group * 8
                k0 = block % (width // 256) * 256
                rows = pl.max(0, pl.min(8, num_tokens - row))
                maximum = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                for panel in pl.range(panels):
                    local = pl.load(partials, [group * panels + panel, 0], [1, 8])
                    maximum = pl.maximum(maximum, local)
                if k0 == 0:
                    factor = pl.reshape(pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30), [8, 1])
                    pl.store(factor, [row, 0], scale)
                denominator = pl.reshape(pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0), [8, 1])
                inverse = pl.recip(denominator)
                _encode_checkpoint_panel(source, payload, row, k0, rows, denominator, inverse)
    elif dense_query and width == 1280:
        _quantize_cached_rows(source, payload, scale, num_tokens, padded_rows, width, True)
    else:
        workers = pl.max(1, pl.min(pl.system.available_aiv_count(), padded_rows // 8))
        for worker in pl.spmd(workers, name_hint="dynamic_int8_quant"):
            for row in pl.range(worker * 8, padded_rows, workers * 8):
                rows = pl.max(0, pl.min(8, num_tokens - row))
                maximum = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                for k0 in pl.range(0, width, QUANT_REDUCE_TILE):
                    cols = pl.min(QUANT_REDUCE_TILE, width - k0)
                    reduce_raw = pl.load(source, [row, k0], [8, QUANT_REDUCE_TILE], valid_shape=[rows, cols])
                    reduce_padded = pl.set_validshape(pl.fillpad(reduce_raw, pad_value=pl.PadValue.zero), 8, QUANT_REDUCE_TILE)
                    reduce_value = pl.cast(reduce_padded, pl.FP32)
                    tmp = pl.create_tile([8, QUANT_REDUCE_TILE], dtype=pl.FP32)
                    local_maximum = pl.reshape(pl.row_max(pl.abs(reduce_value), tmp_tile=tmp), [1, 8])
                    maximum = pl.maximum(maximum, local_maximum)
                factor = pl.reshape(pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30), [8, 1])
                scale = pl.store(factor, [row, 0], scale)
                denominator = pl.reshape(pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0), [8, 1])
                inverse = pl.recip(denominator)
                for k0 in pl.range(0, width, 256):
                    _encode_checkpoint_panel(source, payload, row, k0, rows, denominator, inverse)
    return payload, scale


@pl.jit.inline(auto_scope=False)
def _normalize_quantize_projection_row(
    source: pl.Tensor,
    norm_weight: pl.Tensor,
    normalized: pl.Tensor,
    payload: pl.Tensor,
    scale: pl.Tensor,
    blocks: pl.Scalar[pl.INDEX],
    workers: pl.Scalar[pl.INDEX],
    width: pl.constexpr,
    eps: pl.constexpr,
):
    """Normalize one Decode row, encode its full width and initialize the padding."""
    weight = pl.tensor.reshape(norm_weight, [1, width])
    quant_workers = pl.max(1, pl.min(pl.min(blocks, workers), pl.system.available_aiv_count()))
    for worker in pl.spmd(quant_workers, name_hint="decode_q_norm_quant_b1"):
        for block in pl.range(worker, blocks, quant_workers):
            row = block * VEC_M_TILE
            zero = pl.tile.full([VEC_M_TILE, width], dtype=pl.FP32, value=0.0)
            padding = pl.cast(zero, pl.INT8, mode="rint")
            if block == 0:
                raw = pl.load(source, [0, 0], [VEC_M_TILE, width], valid_shape=[1, width])
                padded = pl.fillpad(raw, pad_value=pl.PadValue.zero)
                valid = pl.set_validshape(padded, VEC_M_TILE, width)
                value = pl.cast(valid, pl.FP32)
                squares = pl.mul(value, value)
                scratch = pl.create_tile([VEC_M_TILE, width], dtype=pl.FP32)
                square_sum = pl.row_sum(squares, tmp_tile=scratch)
                mean = pl.mul(square_sum, 1.0 / width)
                argument = pl.add(mean, eps)
                rsqrt_scratch = pl.create_tile([VEC_M_TILE, 1], dtype=pl.FP32)
                inverse_rms = _tile_rsqrt_for_norm(argument, rsqrt_scratch)
                scaled = pl.row_expand_mul(value, inverse_rms)
                gamma = pl.cast(pl.load(weight, [0, 0], [1, width]), pl.FP32)
                weighted = pl.col_expand_mul(scaled, gamma)
                rounded = pl.cast(weighted, pl.BF16, mode="rint")
                pl.store(pl.set_validshape(rounded, 1, width), [0, 0], normalized)

                quantized = pl.cast(rounded, pl.FP32)
                absolute = pl.abs(quantized)
                maximum = pl.row_max(absolute, tmp_tile=scratch)
                factor = pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30)
                pl.store(factor, [0, 0], scale)

                denominator = pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0)
                first = pl.tile.read(denominator, [0, 0])
                lanes = pl.tile.full([VEC_M_TILE, VEC_M_TILE], dtype=pl.FP32, value=0.0)
                repeated = pl.add(lanes, first)
                broadcast = pl.row_max(repeated, tmp_tile=lanes)
                inverse = pl.recip(broadcast)
                live = pl.tile.slice(quantized, [1, width], [0, 0])
                chunks = pl.reshape(live, [VEC_M_TILE, width // VEC_M_TILE])
                encoded = _checkpoint_codes(chunks, broadcast, inverse)
                pl.store(pl.reshape(encoded, [1, width]), [0, 0], payload)
                pl.store(pl.set_validshape(padding, VEC_M_TILE - 1, width), [1, 0], payload)
            else:
                padding_scale = pl.tile.full([1, VEC_M_TILE], dtype=pl.FP32, value=1e-30)
                pl.store(pl.reshape(padding_scale, [VEC_M_TILE, 1]), [row, 0], scale)
                pl.store(padding, [row, 0], payload)
    return normalized, payload, scale


@pl.jit.inline(auto_scope=False)
def normalize_quantize_projection_activation(
    source: pl.Tensor,
    norm_weight: pl.Tensor,
    normalized: pl.Tensor,
    payload: pl.Tensor,
    scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.constexpr,
    eps: pl.constexpr,
):
    """Apply BF16 RMSNorm and checkpoint INT8 quantization to projection rows."""
    capacity = pl.tensor.dim(source, 0)
    blocks = ((capacity + 15) // 16 * 16) // VEC_M_TILE
    workers = pl.min(blocks, pl.system.available_cluster_count())
    if width == 1280 and num_tokens == 1:
        _normalize_quantize_projection_row(
            source, norm_weight, normalized, payload, scale, blocks, workers, width, eps,
        )
    else:
        weight_row = pl.tensor.view(norm_weight, [1, width])
        for worker in pl.spmd(workers, name_hint="decode_norm_quant"):
            for block in pl.range(worker, blocks, workers):
                row = block * VEC_M_TILE
                rows = pl.max(0, pl.min(VEC_M_TILE, num_tokens - row))
                raw = pl.load(source, [row, 0], [VEC_M_TILE, width], valid_shape=[rows, width])
                padded = pl.set_validshape(pl.fillpad(raw, pad_value=pl.PadValue.zero), VEC_M_TILE, width)
                value = pl.cast(padded, pl.FP32)
                square = pl.mul(value, value)
                norm_scratch = pl.create_tile([VEC_M_TILE, width], dtype=pl.FP32)
                row_sum = pl.row_sum(square, tmp_tile=norm_scratch)
                mean = pl.mul(row_sum, 1.0 / width)
                rms_argument = pl.add(mean, eps)
                rsqrt_scratch = pl.create_tile([VEC_M_TILE, 1], dtype=pl.FP32)
                inverse_rms = _tile_rsqrt_for_norm(rms_argument, rsqrt_scratch)
                normalized_fp32 = pl.row_expand_mul(value, inverse_rms)
                gamma = pl.cast(pl.load(weight_row, [0, 0], [1, width]), pl.FP32)
                weighted = pl.col_expand_mul(normalized_fp32, gamma)
                normalized_bf16 = pl.cast(weighted, pl.BF16, mode="rint")
                normalized = pl.store(
                    pl.set_validshape(normalized_bf16, rows, width),
                    [row, 0],
                    normalized,
                )
                quantized_fp32 = pl.cast(normalized_bf16, pl.FP32)
                scratch = pl.create_tile([VEC_M_TILE, width], dtype=pl.FP32)
                maximum = pl.row_max(pl.abs(quantized_fp32), tmp_tile=scratch)
                factor = pl.reshape(pl.maximum(pl.mul(maximum, 1.0 / 127.0), 1e-30), [VEC_M_TILE, 1])
                pl.store(factor, [row, 0], scale)
                denominator = pl.reshape(pl.mul(pl.maximum(maximum, 127e-30), 1.0 / 128.0), [VEC_M_TILE, 1])
                inverse = pl.recip(denominator)
                for k0 in pl.range(0, width, K_TILE):
                    part = pl.tile.slice(quantized_fp32, [VEC_M_TILE, K_TILE], [0, k0])
                    encoded = _checkpoint_codes(part, denominator, inverse)
                    pl.store(encoded, [row, k0], payload)
    return normalized, payload, scale


@pl.jit.inline(auto_scope=False)
def quantize_activation(source: pl.Tensor, num_tokens: pl.Scalar[pl.INT32], width: pl.constexpr):
    """Quantize A3 checkpoint activations using the matching width and row schedule."""
    return quantize_projection_activation(source, num_tokens, width, True, False)
