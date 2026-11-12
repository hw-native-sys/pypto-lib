# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""RMS normalization, rotary embedding and grouped cache publication."""

import pypto.language as pl

from config import CMP_BLOCKS_DYN, COMPRESSED_CACHE_BYTES, T_DYN

HEAD_DIM = 512


@pl.jit.inline(auto_scope=False)
def _process_batch(
    source: pl.Tile[[4096], pl.BF16, pl.Mem.Vec],
    weight: pl.Tile[[512], pl.BF16, pl.Mem.Vec],
    cosine: pl.Tile[[512], pl.FP32, pl.Mem.Vec],
    sine: pl.Tile[[512], pl.FP32, pl.Mem.Vec],
    rows: pl.Scalar[pl.INDEX],
):
    normalized_rows = pl.tile.create([4096], dtype=pl.BF16, target_memory=pl.Mem.Vec)
    packed = pl.tile.create([4096], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
    scale_values = pl.tile.create([4096], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    factors = pl.tile.create([512], dtype=pl.BF16, target_memory=pl.Mem.Vec)
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=64)
        lower = pl.reg.active_mask(32, lanes=64)
        lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
        for row in pl.hardware_loop(rows):
            summed = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
            for segment in pl.hardware_loop(8):
                value = pl.reg.convert(
                    pl.reg.load(source, row * HEAD_DIM + segment * 64, lanes=64), dtype=pl.FP32
                )
                squared = pl.reg.mul(value, value, full)
                summed = pl.reg.add(summed, squared, full)
            sum_one = pl.reg.reduce_sum(summed, full, reassoc=True)
            total = pl.reg.broadcast(sum_one, lanes=64, dtype=pl.FP32)
            mean = pl.reg.muls(total, 1.0 / HEAD_DIM, full)
            rms = pl.reg.sqrt(pl.reg.adds(mean, 1e-06, full), full)
            inverse = pl.reg.div(pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32), rms, full)
            for segment in pl.hardware_loop(7):
                value = pl.reg.convert(pl.reg.load(source, row * 512 + segment * 64, lanes=64), dtype=pl.FP32)
                gain = pl.reg.convert(pl.reg.load(weight, segment * 64, lanes=64), dtype=pl.FP32)
                value = pl.reg.mul(value, inverse, full)
                value = pl.reg.mul(value, gain, full)
                bf = pl.reg.convert(value, dtype=pl.BF16, rounding="R", saturate=False)
                _latent = pl.reg.store(normalized_rows, row * 512 + segment * 64, bf)
                value = pl.reg.convert(bf, dtype=pl.FP32)
                magnitude = pl.reg.abs(value, full)
                maximum = magnitude
                for stage in pl.unroll(4):
                    partner = pl.reg.bitwise_xor(
                        lane,
                        pl.reg.broadcast(pl.cast(1 << stage, pl.UINT32), lanes=64, dtype=pl.UINT32),
                        full,
                    )
                    maximum = pl.reg.maximum(maximum, pl.reg.gather_lanes(maximum, partner), full)
                scale = pl.reg.maximums(pl.reg.muls(maximum, 1.0 / 6.0, full), 0.001953125, full)
                bf_scale = pl.reg.convert(scale, dtype=pl.BF16, rounding="R", saturate=False)
                scale = pl.reg.convert(bf_scale, dtype=pl.FP32)
                scale_record = scale
                _scales = pl.reg.store(scale_values, row * HEAD_DIM + segment * 64, scale_record)
                normalized = pl.reg.div(value, scale, full)
                magnitude = pl.reg.abs(normalized, full)
                codes = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
                selected = pl.reg.compares(magnitude, 0.25, full, predicate="gt")
                codes = pl.reg.select(selected, pl.reg.broadcast(1, lanes=64, dtype=pl.INT32), codes)
                selected = pl.reg.compares(magnitude, 0.75, full, predicate="ge")
                codes = pl.reg.select(selected, pl.reg.broadcast(2, lanes=64, dtype=pl.INT32), codes)
                selected = pl.reg.compares(magnitude, 1.25, full, predicate="gt")
                codes = pl.reg.select(selected, pl.reg.broadcast(3, lanes=64, dtype=pl.INT32), codes)
                selected = pl.reg.compares(magnitude, 1.75, full, predicate="ge")
                codes = pl.reg.select(selected, pl.reg.broadcast(4, lanes=64, dtype=pl.INT32), codes)
                selected = pl.reg.compares(magnitude, 2.5, full, predicate="gt")
                codes = pl.reg.select(selected, pl.reg.broadcast(5, lanes=64, dtype=pl.INT32), codes)
                selected = pl.reg.compares(magnitude, 3.5, full, predicate="ge")
                codes = pl.reg.select(selected, pl.reg.broadcast(6, lanes=64, dtype=pl.INT32), codes)
                selected = pl.reg.compares(magnitude, 5.0, full, predicate="gt")
                codes = pl.reg.select(selected, pl.reg.broadcast(7, lanes=64, dtype=pl.INT32), codes)
                sign = pl.reg.interpret_cast(normalized, dtype=pl.INT32)
                sign = pl.reg.shift_rights(sign, 28, full)
                sign = pl.reg.bitwise_and(sign, pl.reg.broadcast(8, lanes=64, dtype=pl.INT32), full)
                codes = pl.reg.bitwise_or(codes, sign, full)
                even_code, odd_code = pl.reg.dintlv(codes, codes, full)
                odd_code = pl.reg.shift_lefts(odd_code, 4, lower)
                byte_code = pl.reg.bitwise_or(even_code, odd_code, lower)
                signed_bytes = pl.reg.convert(byte_code, dtype=pl.INT8, saturate=False)
                bytes8 = pl.reg.interpret_cast(signed_bytes, dtype=pl.UINT8)
                _payload = pl.reg.store(packed, row * HEAD_DIM + segment * 32, bytes8, lower)
            value = pl.reg.convert(pl.reg.load(source, row * 512 + 7 * 64, lanes=64), dtype=pl.FP32)
            gain = pl.reg.convert(pl.reg.load(weight, 7 * 64, lanes=64), dtype=pl.FP32)
            value = pl.reg.mul(value, inverse, full)
            value = pl.reg.mul(value, gain, full)
            bf = pl.reg.convert(value, dtype=pl.BF16, rounding="R", saturate=False)
            _latent = pl.reg.store(normalized_rows, row * 512 + 7 * 64, bf)
            value = pl.reg.convert(bf, dtype=pl.FP32)
            even, odd = pl.reg.dintlv(value, value, full)
            cosv = pl.reg.load(cosine, row * 64, lanes=64)
            sinv = pl.reg.load(sine, row * 64, lanes=64)
            a = pl.reg.mul(even, cosv, lower)
            b = pl.reg.mul(odd, sinv, lower)
            real = pl.reg.sub(a, b, lower)
            a = pl.reg.mul(even, sinv, lower)
            b = pl.reg.mul(odd, cosv, lower)
            imag = pl.reg.add(a, b, lower)
            interleaved, unused = pl.reg.intlv(real, imag, full)
            bf = pl.reg.convert(interleaved, dtype=pl.BF16, rounding="R", saturate=False)
            value = pl.reg.convert(bf, dtype=pl.FP32)
            magnitude = pl.reg.abs(value, full)
            maximum = magnitude
            for stage in pl.unroll(4):
                partner = pl.reg.bitwise_xor(
                    lane, pl.reg.broadcast(pl.cast(1 << stage, pl.UINT32), lanes=64, dtype=pl.UINT32), full
                )
                maximum = pl.reg.maximum(maximum, pl.reg.gather_lanes(maximum, partner), full)
            scale = pl.reg.maximums(pl.reg.muls(maximum, 1.0 / 6.0, full), 0.001953125, full)
            bf_scale = pl.reg.convert(scale, dtype=pl.BF16, rounding="R", saturate=False)
            scale = pl.reg.convert(bf_scale, dtype=pl.FP32)
            scale_record = scale
            _scales = pl.reg.store(scale_values, row * HEAD_DIM + 7 * 64, scale_record)
            normalized = pl.reg.div(value, scale, full)
            magnitude = pl.reg.abs(normalized, full)
            codes = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
            selected = pl.reg.compares(magnitude, 0.25, full, predicate="gt")
            codes = pl.reg.select(selected, pl.reg.broadcast(1, lanes=64, dtype=pl.INT32), codes)
            selected = pl.reg.compares(magnitude, 0.75, full, predicate="ge")
            codes = pl.reg.select(selected, pl.reg.broadcast(2, lanes=64, dtype=pl.INT32), codes)
            selected = pl.reg.compares(magnitude, 1.25, full, predicate="gt")
            codes = pl.reg.select(selected, pl.reg.broadcast(3, lanes=64, dtype=pl.INT32), codes)
            selected = pl.reg.compares(magnitude, 1.75, full, predicate="ge")
            codes = pl.reg.select(selected, pl.reg.broadcast(4, lanes=64, dtype=pl.INT32), codes)
            selected = pl.reg.compares(magnitude, 2.5, full, predicate="gt")
            codes = pl.reg.select(selected, pl.reg.broadcast(5, lanes=64, dtype=pl.INT32), codes)
            selected = pl.reg.compares(magnitude, 3.5, full, predicate="ge")
            codes = pl.reg.select(selected, pl.reg.broadcast(6, lanes=64, dtype=pl.INT32), codes)
            selected = pl.reg.compares(magnitude, 5.0, full, predicate="gt")
            codes = pl.reg.select(selected, pl.reg.broadcast(7, lanes=64, dtype=pl.INT32), codes)
            sign = pl.reg.interpret_cast(normalized, dtype=pl.INT32)
            sign = pl.reg.shift_rights(sign, 28, full)
            sign = pl.reg.bitwise_and(sign, pl.reg.broadcast(8, lanes=64, dtype=pl.INT32), full)
            codes = pl.reg.bitwise_or(codes, sign, full)
            even_code, odd_code = pl.reg.dintlv(codes, codes, full)
            odd_code = pl.reg.shift_lefts(odd_code, 4, lower)
            byte_code = pl.reg.bitwise_or(even_code, odd_code, lower)
            signed_bytes = pl.reg.convert(byte_code, dtype=pl.INT8, saturate=False)
            bytes8 = pl.reg.interpret_cast(signed_bytes, dtype=pl.UINT8)
            _payload = pl.reg.store(packed, row * HEAD_DIM + 7 * 32, bytes8, lower)
            indices = pl.reg.shift_lefts(lane, 4, full)
            active = pl.reg.active_mask(HEAD_DIM // 16, lanes=64)
            scales = pl.reg.gather(
                scale_values, pl.reg.adds(indices, pl.cast(row * HEAD_DIM, pl.UINT32), full), active
            )
            bf_scales = pl.reg.convert(scales, dtype=pl.BF16, rounding="R", saturate=False)
            _bf_written = pl.reg.store(factors, row * 64, bf_scales, active)
    return (normalized_rows, packed, factors)


@pl.jit.inline(auto_scope=False)
def _flush(
    inputs: pl.Tile[[8, HEAD_DIM], pl.BF16, pl.Mem.Vec],
    weight: pl.Tile[[HEAD_DIM], pl.BF16, pl.Mem.Vec],
    cosv: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec],
    sinv: pl.Tile[[8, 64], pl.FP32, pl.Mem.Vec],
    targets: pl.Tile[[8], pl.INT64, pl.Mem.Vec],
    tokens: pl.Tile[[8], pl.INT32, pl.Mem.Vec],
    latent: pl.Tensor,
    cache: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    source: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
):
    for row in pl.range(rows):
        load_token = pl.cast(pl.tile.read(tokens, [row]), pl.INDEX)
        inputs = pl.tile.gather_row(inputs, source, [row, 0], [load_token, 0], [1, HEAD_DIM])
        cosv = pl.tile.gather_row(cosv, cosine, [row, 0], [load_token, 0], [1, 32])
        sinv = pl.tile.gather_row(sinv, sine, [row, 0], [load_token, 0], [1, 32])
    flat_inputs = pl.reshape(inputs, [4096])
    norm, payload, bf_factors = _process_batch(
        flat_inputs, weight, pl.reshape(cosv, [512]), pl.reshape(sinv, [512]), rows
    )
    norm_rows = pl.reshape(norm, [8, HEAD_DIM])
    for row in pl.range(rows):
        token = pl.tile.read(tokens, [row])
        value = pl.tile.slice(norm_rows, [1, HEAD_DIM], [row, 0])
        pl.store(value, [token, 0], latent)
    payload_rows = pl.reshape(payload, [8, HEAD_DIM])
    bf_factor_rows = pl.reshape(bf_factors, [8, 64])
    for row in pl.range(rows):
        target = pl.tile.read(targets, [row])
        if target >= 0:
            value_fp4 = pl.tile.slice(payload_rows, [1, HEAD_DIM // 2], [row, 0])
            pl.store(value_fp4, [target, 0], cache)
            factors_cmp = pl.tile.slice(bf_factor_rows, [1, 32], [row, 0])
            scale_bytes = pl.reinterpret_view(factors_cmp, pl.UINT8)
            pl.store(scale_bytes, [target, 256], cache)
    return cache


@pl.jit.inline(auto_scope=False)
def publish_body(
    source: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.Tensor,
    latent: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    worker = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    gamma = pl.reshape(pl.load(weight, [0], [HEAD_DIM], target_memory=pl.Mem.Vec), [HEAD_DIM])
    inputs = pl.tile.create([8, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
    cosv = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    sinv = pl.tile.create([8, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    targets = pl.tile.create([8], dtype=pl.INT64, target_memory=pl.Mem.Vec)
    tokens = pl.tile.create([8], dtype=pl.INT32, target_memory=pl.Mem.Vec)
    pending = 0
    for base in pl.range(0, num_tokens, 32):
        count = pl.min(32, num_tokens - base)
        slot_tile = pl.load(slots, [base], [32], valid_shape=[count], target_memory=pl.Mem.Vec)
        for lane in pl.range(count):
            token = base + lane
            slot = pl.tile.read(slot_tile, [lane])
            owner = pl.cast(slot // 128, pl.INDEX) % workers
            if slot < 0:
                owner = token // 8 % workers
            if owner == worker:
                _target = pl.tile.write(targets, [pending], slot)
                _token = pl.tile.write(tokens, [pending], pl.cast(token, pl.INT32))
                pending = pending + 1
                if pending == 8:
                    cache = _flush(
                        inputs,
                        gamma,
                        cosv,
                        sinv,
                        targets,
                        tokens,
                        latent,
                        cache,
                        pending,
                        source,
                        cosine,
                        sine,
                    )
                    pending = 0
    if pending > 0:
        cache = _flush(
            inputs, gamma, cosv, sinv, targets, tokens, latent, cache, pending, source, cosine, sine
        )
    return cache


@pl.jit.incore
def _c1a_compress_python(
    source: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    norm_weight: pl.Tensor,
    latent: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Normalize, retain unrotated latent, and publish nearest-even E2M1 rows."""
    _published = publish_body(source, norm_weight, cosine, sine, slots, cache, latent, num_tokens)
    return cache


@pl.jit.inline(auto_scope=False)
def c1a_compressor_publish_python(
    source: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    source_ready: pl.Scalar[pl.TASK_ID],
):
    """Normalize and publish MXFP4 rows while retaining the unrotated latent."""
    active_tokens = pl.cast(num_tokens, pl.INT32)
    workers = pl.max(pl.min(active_tokens, 8), 1)
    flat_cache = pl.reshape(cache, [pl.tensor.dim(cache, 0) * 128, COMPRESSED_CACHE_BYTES])
    with pl.spmd(workers, name_hint="c1a_compress_python", deps=[cache_ready, source_ready]) as published:
        _published = _c1a_compress_python(source, slots, flat_cache, cosine, sine, norm_weight, latent, active_tokens)
    return published
