# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Window cache publication with BF16 RMSNorm and RoPE followed by group-32 FP8."""

import pypto.language as pl

# model config
HEAD_DIM = 512
PAGE_SIZE = 128

# tiling
ROW_TILE = 8
SLOT_TILE = 128


@pl.jit.inline(auto_scope=False)
def _postprocess(
    raw: pl.Tile[[ROW_TILE * HEAD_DIM], pl.BF16, pl.Mem.Vec],
    weight: pl.Tile[[HEAD_DIM], pl.BF16, pl.Mem.Vec],
    cosine: pl.Tile[[ROW_TILE * 32], pl.FP32, pl.Mem.Vec],
    sine: pl.Tile[[ROW_TILE * 32], pl.FP32, pl.Mem.Vec],
    rows: pl.Scalar[pl.INDEX],
):
    packed = pl.tile.create([ROW_TILE * HEAD_DIM], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
    factors = pl.tile.create([ROW_TILE * 8], dtype=pl.UINT32, target_memory=pl.Mem.Vec)
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=64)
        lower = pl.reg.active_mask(32, lanes=64)
        upper = pl.reg.mask_not(lower)
        first_index = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
        last_lane = pl.reg.broadcast(63, lanes=64, dtype=pl.UINT32)
        one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
        scale_words = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
        for row in pl.hardware_loop(rows):
            summed = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
            for segment in pl.hardware_loop(HEAD_DIM // 64):
                source_bf = pl.reg.load(raw, row * HEAD_DIM + segment * 64, lanes=64)
                source_f32 = pl.reg.convert(source_bf, dtype=pl.FP32)
                summed = pl.reg.add(summed, pl.reg.mul(source_f32, source_f32, full), full)
            total = pl.reg.reduce_sum(summed, full, reassoc=True)
            total = pl.reg.broadcast(total, lanes=64, dtype=pl.FP32)
            mean = pl.reg.muls(total, 1.0 / HEAD_DIM, full)
            rms = pl.reg.sqrt(pl.reg.adds(mean, 1.0e-6, full), full)
            inverse = pl.reg.div(one, rms, full)
            cos = pl.reg.load(cosine, row * 32, lanes=64)
            sin = pl.reg.load(sine, row * 32, lanes=64)
            for segment in pl.hardware_loop(HEAD_DIM // 64 - 1):
                offset = row * HEAD_DIM + segment * 64
                value = pl.reg.convert(pl.reg.load(raw, offset, lanes=64), dtype=pl.FP32)
                gain = pl.reg.convert(pl.reg.load(weight, segment * 64, lanes=64), dtype=pl.FP32)
                normalized = pl.reg.mul(pl.reg.mul(value, inverse, full), gain, full)
                bf = pl.reg.convert(normalized, dtype=pl.BF16, rounding="R", saturate=False)
                value = pl.reg.convert(bf, dtype=pl.FP32)
                magnitude = pl.reg.abs(value, full)
                lo = pl.reg.broadcast(pl.reg.reduce_max(magnitude, lower), lanes=64, dtype=pl.FP32)
                hi = pl.reg.broadcast(pl.reg.reduce_max(magnitude, upper), lanes=64, dtype=pl.FP32)
                maximum = pl.reg.select(lower, lo, hi)
                scale = pl.reg.muls(pl.reg.maximums(maximum, 1.0e-4, full), 1.0 / 448.0, full)
                scale_exponent_bits = pl.reg.interpret_cast(scale, dtype=pl.INT32)
                rounded_bits = pl.reg.adds(scale_exponent_bits, 8388607, full)
                exponent = pl.reg.shift_rights(rounded_bits, 23, full)
                power_bits = pl.reg.shift_lefts(exponent, 23, full)
                scale = pl.reg.interpret_cast(power_bits, dtype=pl.FP32)
                quantized = pl.reg.div(value, scale, full)
                fp8 = pl.reg.convert(quantized, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
                bytes8 = pl.reg.interpret_cast(fp8, dtype=pl.UINT8)
                _packed = pl.reg.store(packed, offset, bytes8)
                # Pack two BF16 scale codes per segment.
                scale_bits = pl.reg.interpret_cast(scale, dtype=pl.UINT32)
                high_bits = pl.reg.gather_lanes(scale_bits, last_lane)
                low_bits = pl.reg.shift_rights(scale_bits, 16, full)
                codes = pl.reg.bitwise_or(low_bits, high_bits, full)
                code_index = row * 8 + segment
                before = pl.reg.active_mask(code_index, lanes=64)
                through = pl.reg.active_mask(code_index + 1, lanes=64)
                code_lane = pl.reg.mask_xor(before, through)
                code_word = pl.reg.gather_lanes(codes, first_index)
                scale_words = pl.reg.select(code_lane, code_word, scale_words)
            offset = row * HEAD_DIM + (HEAD_DIM // 64 - 1) * 64
            value = pl.reg.convert(pl.reg.load(raw, offset, lanes=64), dtype=pl.FP32)
            gain = pl.reg.convert(pl.reg.load(weight, (HEAD_DIM // 64 - 1) * 64, lanes=64), dtype=pl.FP32)
            normalized = pl.reg.mul(pl.reg.mul(value, inverse, full), gain, full)
            bf = pl.reg.convert(normalized, dtype=pl.BF16, rounding="R", saturate=False)
            value = pl.reg.convert(bf, dtype=pl.FP32)
            even, odd = pl.reg.dintlv(value, value, full)
            real = pl.reg.sub(pl.reg.mul(even, cos, lower), pl.reg.mul(odd, sin, lower), lower)
            imag = pl.reg.add(pl.reg.mul(even, sin, lower), pl.reg.mul(odd, cos, lower), lower)
            rotated, _unused = pl.reg.intlv(real, imag, full)
            bf = pl.reg.convert(rotated, dtype=pl.BF16, rounding="R", saturate=False)
            value = pl.reg.convert(bf, dtype=pl.FP32)
            magnitude = pl.reg.abs(value, full)
            lo = pl.reg.broadcast(pl.reg.reduce_max(magnitude, lower), lanes=64, dtype=pl.FP32)
            hi = pl.reg.broadcast(pl.reg.reduce_max(magnitude, upper), lanes=64, dtype=pl.FP32)
            maximum = pl.reg.select(lower, lo, hi)
            scale = pl.reg.muls(pl.reg.maximums(maximum, 1.0e-4, full), 1.0 / 448.0, full)
            scale_exponent_bits = pl.reg.interpret_cast(scale, dtype=pl.INT32)
            rounded_bits = pl.reg.adds(scale_exponent_bits, 8388607, full)
            exponent = pl.reg.shift_rights(rounded_bits, 23, full)
            power_bits = pl.reg.shift_lefts(exponent, 23, full)
            scale = pl.reg.interpret_cast(power_bits, dtype=pl.FP32)
            quantized = pl.reg.div(value, scale, full)
            fp8 = pl.reg.convert(quantized, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
            bytes8 = pl.reg.interpret_cast(fp8, dtype=pl.UINT8)
            _packed = pl.reg.store(packed, offset, bytes8)
            # Pack two BF16 scale codes per segment.
            scale_bits = pl.reg.interpret_cast(scale, dtype=pl.UINT32)
            high_bits = pl.reg.gather_lanes(scale_bits, last_lane)
            low_bits = pl.reg.shift_rights(scale_bits, 16, full)
            codes = pl.reg.bitwise_or(low_bits, high_bits, full)
            code_index = row * 8 + (HEAD_DIM // 64 - 1)
            before = pl.reg.active_mask(code_index, lanes=64)
            through = pl.reg.active_mask(code_index + 1, lanes=64)
            code_lane = pl.reg.mask_xor(before, through)
            code_word = pl.reg.gather_lanes(codes, first_index)
            scale_words = pl.reg.select(code_lane, code_word, scale_words)
        _factors = pl.reg.store(factors, 0, scale_words)
    return packed, factors


@pl.jit.inline(auto_scope=False)
def _flush(
    inputs: pl.Tile[[ROW_TILE, HEAD_DIM], pl.BF16, pl.Mem.Vec],
    weight: pl.Tile[[HEAD_DIM], pl.BF16, pl.Mem.Vec],
    cosine: pl.Tile[[ROW_TILE, 32], pl.FP32, pl.Mem.Vec],
    sine: pl.Tile[[ROW_TILE, 32], pl.FP32, pl.Mem.Vec],
    targets: pl.Tile[[ROW_TILE], pl.INT64, pl.Mem.Vec],
    rows: pl.Scalar[pl.INDEX],
    cache: pl.Tensor,
):
    raw = pl.reshape(inputs, [ROW_TILE * HEAD_DIM])
    cos = pl.reshape(cosine, [ROW_TILE * 32])
    sin = pl.reshape(sine, [ROW_TILE * 32])
    payload, factors = _postprocess(raw, weight, cos, sin, rows)
    payload_rows = pl.reshape(payload, [ROW_TILE, HEAD_DIM])
    factor_bytes = pl.reinterpret_view(factors, pl.UINT8)
    factor_rows = pl.reshape(factor_bytes, [ROW_TILE, HEAD_DIM // 16])
    for (row,) in pl.while_(init_values=(pl.cast(0, pl.INDEX),)):
        pl.cond(row < rows)
        slot = pl.tile.read(targets, [row])
        count = 1
        scanning = True
        while row + count < rows and scanning:
            next_slot = pl.tile.read(targets, [row + count])
            if next_slot == slot + count:
                count = count + 1
            else:
                scanning = False
        value = pl.tile.slice(payload_rows, [ROW_TILE, HEAD_DIM], [row, 0], valid_shape=[count, HEAD_DIM])
        factor = pl.tile.slice(factor_rows, [ROW_TILE, HEAD_DIM // 16], [row, 0], valid_shape=[count, HEAD_DIM // 16])
        _payload = pl.store(value, [slot, 0], cache)
        _scales = pl.store(factor, [slot, HEAD_DIM], cache)
        next_row = row + count
        final_row = pl.yield_(next_row)
    return cache


@pl.jit.incore
def norm_rope_publish(
    source: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Normalize, rotate, quantize, and publish rows owned by this cache page worker."""
    worker = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    gamma = pl.load(weight, [0], [HEAD_DIM], target_memory=pl.Mem.Vec)
    inputs = pl.tile.create([ROW_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Vec)
    cosv = pl.tile.create([ROW_TILE, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    sinv = pl.tile.create([ROW_TILE, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    targets = pl.tile.create([ROW_TILE], dtype=pl.INT64, target_memory=pl.Mem.Vec)
    pending = 0
    for base in pl.range(0, num_tokens, SLOT_TILE):
        count = pl.min(SLOT_TILE, num_tokens - base)
        slot_tile = pl.load(slots, [base], [SLOT_TILE], valid_shape=[count], target_memory=pl.Mem.Vec)
        lane = 0
        while lane < count:
            slot = pl.tile.read(slot_tile, [lane])
            accepted = False
            if slot >= 0:
                unsigned_slot = pl.cast(slot, pl.UINT64)
                owner = (unsigned_slot // PAGE_SIZE) % pl.cast(workers, pl.UINT64)
                accepted = owner == pl.cast(worker, pl.UINT64)
            if accepted:
                run = 0
                scanning = True
                while lane + run < count and pending + run < ROW_TILE and scanning:
                    target = pl.tile.read(slot_tile, [lane + run])
                    owned = False
                    if target >= 0:
                        unsigned_target = pl.cast(target, pl.UINT64)
                        target_owner = (unsigned_target // PAGE_SIZE) % pl.cast(workers, pl.UINT64)
                        owned = target_owner == pl.cast(worker, pl.UINT64)
                    if owned:
                        _target = pl.tile.write(targets, [pending + run], target)
                        run = run + 1
                    else:
                        scanning = False
                token = base + lane
                inputs = pl.tile.gather_row(
                    inputs, source, [pending, 0], [token, 0], [ROW_TILE, HEAD_DIM], valid_shape=[run, HEAD_DIM]
                )
                cosv = pl.tile.gather_row(
                    cosv, cosine, [pending, 0], [token, 0], [ROW_TILE, 32], valid_shape=[run, 32]
                )
                sinv = pl.tile.gather_row(
                    sinv, sine, [pending, 0], [token, 0], [ROW_TILE, 32], valid_shape=[run, 32]
                )
                lane = lane + run
                pending = pending + run
                if pending == ROW_TILE:
                    cache = _flush(inputs, gamma, cosv, sinv, targets, ROW_TILE, cache)
                    pending = 0
            else:
                lane = lane + 1
    if pending > 0:
        cache = _flush(inputs, gamma, cosv, sinv, targets, pending, cache)
    return cache
