# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pool C2A pairs, normalize latent rows, and publish packed MXFP4 cache."""

import pypto.language as pl


@pl.jit.inline(auto_scope=False)
def _pool_normalize(
    ck: pl.Tile[[512], pl.FP32, pl.Mem.Vec],
    cs: pl.Tile[[512], pl.FP32, pl.Mem.Vec],
    pk: pl.Tile[[512], pl.FP32, pl.Mem.Vec],
    ps: pl.Tile[[512], pl.FP32, pl.Mem.Vec],
    weight: pl.Tile[[512], pl.BF16, pl.Mem.Vec],
):
    rounded = pl.create_tile([512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    output = pl.create_tile([512], dtype=pl.BF16, target_memory=pl.Mem.Vec)
    rotated = pl.create_tile([512], dtype=pl.BF16, target_memory=pl.Mem.Vec)
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=64)
        total = pl.reg.broadcast(0.0, lanes=64, dtype=pl.FP32)
        for chunk in pl.hardware_loop(8):
            a = pl.reg.load(ps, chunk * 64, lanes=64)
            b = pl.reg.load(cs, chunk * 64, lanes=64)
            high = pl.reg.maximum(a, b, full)
            previous_delta = pl.reg.sub(a, high, full)
            a = pl.reg.exp(previous_delta, full)
            current_delta = pl.reg.sub(b, high, full)
            b = pl.reg.exp(current_delta, full)
            denom = pl.reg.add(a, b, full)
            previous_value = pl.reg.load(pk, chunk * 64, lanes=64)
            previous_weight = pl.reg.div(a, denom, full)
            av = pl.reg.mul(previous_value, previous_weight, full)
            current_value = pl.reg.load(ck, chunk * 64, lanes=64)
            current_weight = pl.reg.div(b, denom, full)
            bv = pl.reg.mul(current_value, current_weight, full)
            pooled = pl.reg.add(av, bv, full)
            bf = pl.reg.convert(pooled, dtype=pl.BF16, rounding="R", saturate=False)
            value = pl.reg.convert(bf, dtype=pl.FP32)
            _rounded_written = pl.reg.store(rounded, chunk * 64, value)
            square = pl.reg.mul(value, value, full)
            total = pl.reg.add(total, square, full)
        reduced = pl.reg.reduce_sum(total, full, reassoc=True)
        reduced = pl.reg.broadcast(reduced, lanes=64, dtype=pl.FP32)
        mean_square = pl.reg.muls(reduced, 1.0 / 512.0, full)
        variance = pl.reg.adds(mean_square, 1e-06, full)
        rms = pl.reg.sqrt(variance, full)
        one = pl.reg.broadcast(1.0, lanes=64, dtype=pl.FP32)
        inv = pl.reg.div(one, rms, full)
        for chunk in pl.hardware_loop(8):
            value = pl.reg.load(rounded, chunk * 64, lanes=64)
            gamma_bf = pl.reg.load(weight, chunk * 64, lanes=64)
            gamma = pl.reg.convert(gamma_bf, dtype=pl.FP32)
            scaled_value = pl.reg.mul(value, inv, full)
            norm = pl.reg.mul(scaled_value, gamma, full)
            bf = pl.reg.convert(norm, dtype=pl.BF16, rounding="R", saturate=False)
            _output_written = pl.reg.store(output, chunk * 64, bf)
            # Keep the unrotated latent live until its GM store completes.
            _rotated_written = pl.reg.store(rotated, chunk * 64, bf)
    return (output, rotated)


@pl.jit.inline(auto_scope=False)
def _rotate_quantize(
    raw: pl.Tile[[512], pl.BF16, pl.Mem.Vec],
    cosine: pl.Tile[[32], pl.FP32, pl.Mem.Vec],
    sine: pl.Tile[[32], pl.FP32, pl.Mem.Vec],
):
    packed = pl.create_tile([256], dtype=pl.UINT8, target_memory=pl.Mem.Vec)
    factors = pl.create_tile([64], dtype=pl.BF16, target_memory=pl.Mem.Vec)
    scale_storage = pl.create_tile([512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
    with pl.reg_scope():
        full = pl.reg.full_mask(lanes=64)
        lower = pl.reg.active_mask(32, lanes=64)
        lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
        rope_bf = pl.reg.load(raw, 448, lanes=64)
        value = pl.reg.convert(rope_bf, dtype=pl.FP32)
        even, odd = pl.reg.dintlv(value, value, full)
        cosv = pl.reg.gather(cosine, lane, lower)
        sinv = pl.reg.gather(sine, lane, lower)
        even_cos = pl.reg.mul(even, cosv, lower)
        odd_sin = pl.reg.mul(odd, sinv, lower)
        real = pl.reg.sub(even_cos, odd_sin, lower)
        even_sin = pl.reg.mul(even, sinv, lower)
        odd_cos = pl.reg.mul(odd, cosv, lower)
        imag = pl.reg.add(even_sin, odd_cos, lower)
        interleaved, unused = pl.reg.intlv(real, imag, full)
        bf = pl.reg.convert(interleaved, dtype=pl.BF16, rounding="R", saturate=False)
        _raw_written = pl.reg.store(raw, 448, bf)
        for chunk in pl.hardware_loop(8):
            bf_values = pl.reg.load(raw, chunk * 64, lanes=64)
            values = pl.reg.convert(bf_values, dtype=pl.FP32)
            # Unsigned BF16 magnitudes reduce in groups of sixteen.
            magnitude_bits = pl.reg.interpret_cast(bf_values, dtype=pl.UINT16)
            mask16 = pl.reg.full_mask(lanes=64)
            sign_mask = pl.reg.broadcast(32767, lanes=64, dtype=pl.UINT16)
            magnitude_bits = pl.reg.bitwise_and(magnitude_bits, sign_mask, mask16)
            group_maximum = pl.reg.reduce_max(magnitude_bits, mask16, group_size=16)
            repeated_maximum = pl.reg.group_broadcast(group_maximum, lanes=64)
            bf_maximum = pl.reg.interpret_cast(repeated_maximum, dtype=pl.BF16)
            maximum = pl.reg.convert(bf_maximum, dtype=pl.FP32)
            raw_scale = pl.reg.muls(maximum, 1.0 / 6.0, full)
            scale = pl.reg.maximums(raw_scale, 0.001953125, full)
            bf_scale = pl.reg.convert(scale, dtype=pl.BF16, rounding="R", saturate=False)
            rounded_scale = pl.reg.convert(bf_scale, dtype=pl.FP32)
            _scale_written = pl.reg.store(scale_storage, chunk * 64, rounded_scale)
            normalized = pl.reg.div(values, rounded_scale, full)
            magnitude = pl.reg.abs(normalized, full)
            codes = pl.reg.broadcast(0, lanes=64, dtype=pl.INT32)
            # Strict E2M1 midpoint comparisons (RoundEven=false).
            selected = pl.reg.compares(magnitude, 0.25, full, predicate="gt")
            code_one = pl.reg.broadcast(1, lanes=64, dtype=pl.INT32)
            codes = pl.reg.select(selected, code_one, codes)
            selected = pl.reg.compares(magnitude, 0.75, full, predicate="gt")
            code_two = pl.reg.broadcast(2, lanes=64, dtype=pl.INT32)
            codes = pl.reg.select(selected, code_two, codes)
            selected = pl.reg.compares(magnitude, 1.25, full, predicate="gt")
            code_three = pl.reg.broadcast(3, lanes=64, dtype=pl.INT32)
            codes = pl.reg.select(selected, code_three, codes)
            selected = pl.reg.compares(magnitude, 1.75, full, predicate="gt")
            code_four = pl.reg.broadcast(4, lanes=64, dtype=pl.INT32)
            codes = pl.reg.select(selected, code_four, codes)
            selected = pl.reg.compares(magnitude, 2.5, full, predicate="gt")
            code_five = pl.reg.broadcast(5, lanes=64, dtype=pl.INT32)
            codes = pl.reg.select(selected, code_five, codes)
            selected = pl.reg.compares(magnitude, 3.5, full, predicate="gt")
            code_six = pl.reg.broadcast(6, lanes=64, dtype=pl.INT32)
            codes = pl.reg.select(selected, code_six, codes)
            selected = pl.reg.compares(magnitude, 5.0, full, predicate="gt")
            code_seven = pl.reg.broadcast(7, lanes=64, dtype=pl.INT32)
            codes = pl.reg.select(selected, code_seven, codes)
            bits = pl.reg.interpret_cast(normalized, dtype=pl.INT32)
            sign_shifted = pl.reg.shift_rights(bits, 28, full)
            sign_mask32 = pl.reg.broadcast(8, lanes=64, dtype=pl.INT32)
            sign = pl.reg.bitwise_and(sign_shifted, sign_mask32, full)
            codes = pl.reg.bitwise_or(codes, sign, full)
            even_code, odd_code = pl.reg.dintlv(codes, codes, full)
            high_nibble = pl.reg.shift_lefts(odd_code, 4, lower)
            byte_code = pl.reg.bitwise_or(even_code, high_nibble, lower)
            bytes8 = pl.reg.convert(byte_code, dtype=pl.INT8, saturate=False)
            bytes8 = pl.reg.interpret_cast(bytes8, dtype=pl.UINT8)
            _packed_written = pl.reg.store(packed, chunk * 32, bytes8, lower)
        group_indices = pl.reg.shift_lefts(lane, 4, full)
        group_scale = pl.reg.gather(scale_storage, group_indices, lower)
        group_bf = pl.reg.convert(group_scale, dtype=pl.BF16, rounding="R", saturate=False)
        _factors_written = pl.reg.store(factors, 0, group_bf, lower)
    return (packed, factors)


@pl.jit.incore
def compressor_pool_body(
    kv_proj: pl.Tensor,
    score_proj: pl.Tensor,
    state_cache: pl.InOut[pl.Tensor],
    position_ids: pl.Tensor,
    norm_weight: pl.Tensor,
    latent: pl.InOut[pl.Tensor],
    query_start_loc: pl.Tensor,
    token_to_req_indices: pl.Tensor,
    state_block_table: pl.Tensor,
    slots: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    num_state_blocks: pl.Scalar[pl.INT32],
):
    worker = pl.tile.get_block_idx()
    workers = pl.tile.get_block_num()
    requests = pl.tensor.dim(state_block_table, 0)
    state_flat = state_cache
    cache_flat = cache
    weight_row = pl.load(norm_weight, [0], [512], target_memory=pl.Mem.Vec)
    gamma = pl.reshape(weight_row, [512])
    for phase in pl.range(2):
        for request in pl.range(requests + 2):
            block = -1
            start = 0
            end = pl.cast(num_tokens, pl.INDEX)
            if request < requests:
                block_id = pl.tensor.read(state_block_table, [request, 0])
                block = pl.cast(block_id, pl.INDEX)
                request_start = pl.tensor.read(query_start_loc, [request])
                start = pl.cast(request_start, pl.INDEX)
                request_end = pl.tensor.read(query_start_loc, [request + 1])
                end = pl.cast(request_end, pl.INDEX)
            elif request == requests:
                prefix_end = pl.tensor.read(query_start_loc, [0])
                end = pl.cast(prefix_end, pl.INDEX)
            else:
                suffix_start = pl.tensor.read(query_start_loc, [requests])
                start = pl.cast(suffix_start, pl.INDEX)
            valid_block = block >= 0 and block < num_state_blocks
            owner = request % workers
            if valid_block:
                owner = block % workers
            start = pl.max(start, 0)
            end = pl.min(end, num_tokens)
            if owner == worker:
                begin = start
                if phase == 1:
                    begin = pl.max(start, end - 4)
                for token in pl.range(begin, end):
                    position = pl.tensor.read(position_ids, [token])
                    req = pl.tensor.read(token_to_req_indices, [token])
                    valid = valid_block and req == request and (position >= 0)
                    if phase == 0:
                        if valid and position % 2 == 1:
                            current_kv_row = pl.load(kv_proj, [token, 0], [1, 512], target_memory=pl.Mem.Vec)
                            ck = pl.reshape(current_kv_row, [512])
                            current_score_row = pl.load(score_proj, [token, 0], [1, 512], target_memory=pl.Mem.Vec)
                            cs = pl.reshape(current_score_row, [512])
                            pk = pl.create_tile([512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            ps = pl.create_tile([512], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                            if token > start:
                                previous_kv_row = pl.load(kv_proj, [token - 1, 0], [1, 512], target_memory=pl.Mem.Vec)
                                pk = pl.reshape(previous_kv_row, [512])
                                previous_score_row = pl.load(score_proj, [token - 1, 0], [1, 512], target_memory=pl.Mem.Vec)
                                ps = pl.reshape(previous_score_row, [512])
                            else:
                                history = block * 4 + (position - 1) % 4
                                history_kv_row = pl.load(state_flat, [history, 0], [1, 512], target_memory=pl.Mem.Vec)
                                pk = pl.reshape(history_kv_row, [512])
                                history_score_row = pl.load(state_flat, [history, 512], [1, 512], target_memory=pl.Mem.Vec)
                                ps = pl.reshape(history_score_row, [512])
                            pooled, rotated = _pool_normalize(ck, cs, pk, ps, gamma)
                            latent_row = pl.reshape(pooled, [1, 512])
                            pl.store(latent_row, [token, 0], latent)
                            target = pl.tensor.read(slots, [token])
                            for future in pl.range(token + 1, num_tokens):
                                future_slot = pl.tensor.read(slots, [future])
                                same_slot = future_slot == target
                                if same_slot and target >= 0:
                                    future_req = pl.tensor.read(token_to_req_indices, [future])
                                    future_pos = pl.tensor.read(position_ids, [future])
                                    if (
                                        future_req >= 0
                                        and future_req < requests
                                        and (future_pos >= 0)
                                        and (future_pos % 2 == 1)
                                    ):
                                        future_block = pl.tensor.read(state_block_table, [future_req, 0])
                                        future_start = pl.tensor.read(query_start_loc, [future_req])
                                        future_end = pl.tensor.read(query_start_loc, [future_req + 1])
                                        if (
                                            future_block >= 0
                                            and future_block < num_state_blocks
                                            and (future >= future_start)
                                            and (future < future_end)
                                        ):
                                            target = pl.cast(-1, pl.INT64)
                            if target >= 0:
                                cosine_row = pl.load(cosine, [token, 0], [1, 32], target_memory=pl.Mem.Vec)
                                cosv = pl.reshape(cosine_row, [32])
                                sine_row = pl.load(sine, [token, 0], [1, 32], target_memory=pl.Mem.Vec)
                                sinv = pl.reshape(sine_row, [32])
                                payload, factors = _rotate_quantize(rotated, cosv, sinv)
                                payload_row = pl.reshape(payload, [1, 256])
                                pl.store(payload_row, [target, 0], cache_flat)
                                factor_tile = pl.reshape(factors, [1, 64])
                                factor_row = pl.tile.slice(factor_tile, [1, 32], [0, 0])
                                scale_bytes = pl.reinterpret_view(factor_row, pl.UINT8)
                                pl.store(scale_bytes, [target, 256], cache_flat)
                        else:
                            zero = pl.tile.full([1, 512], dtype=pl.BF16, value=0.0)
                            pl.store(zero, [token, 0], latent)
                    elif valid:
                        row = block * 4 + position % 4
                        kv = pl.load(kv_proj, [token, 0], [1, 512], target_memory=pl.Mem.Vec)
                        score = pl.load(score_proj, [token, 0], [1, 512], target_memory=pl.Mem.Vec)
                        pl.store(kv, [row, 0], state_flat)
                        pl.store(score, [row, 512], state_flat)
