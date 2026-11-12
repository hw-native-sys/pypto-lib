# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compression projections, rotary embedding, cache publication, and ratio-2 ring state."""

from pathlib import Path

import pypto.language as pl

from pypto.runtime import pto_isa_include_dir
import torch

from config import (
    FLASH as M, STATE_CAPACITY, B_DYN, Q_START_DYN, STATE_BLOCKS_DYN, T_DYN, CMP_BLOCKS_DYN,
    COMPRESSED_CACHE_BYTES,
)
from quant import unpack_bf16_nz
from rmsnorm import golden_rms_norm


# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
EPS = M.rms_norm_eps
STATE_WIDTH = 2 * M.head_dim

# tiling
C1A_PROJ_M_TILE = 48
C1A_PROJ_N_TILE = 64
C1A_PROJ_K_TILE = 512
C2A_PROJ_M_TILE = 80
C2A_PROJ_K_TILE = 256


@pl.jit.inline(auto_scope=False)
def c1a_compressor_project(
    source: pl.Tensor,
    weight: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    ready: pl.Scalar[pl.TASK_ID],
    m_tile: pl.constexpr,
    n_tile: pl.constexpr,
):
    """Project NZ weights with full-height Cube tiles and store active rows."""
    blocks = (num_tokens + m_tile - 1) // m_tile * (HEAD_DIM // n_tile)
    workers = pl.max(pl.min(blocks, 8), 1)
    with pl.spmd(workers, name_hint="proj_cmp", deps=[ready]) as completed:
        with pl.at(level=pl.Level.CORE_GROUP):
            worker = pl.tile.get_block_idx()
            col = worker % (HEAD_DIM // n_tile) * n_tile
            row_workers = pl.max(workers // (HEAD_DIM // n_tile), 1)
            row_blocks = (num_tokens + m_tile - 1) // m_tile
            for row_block in pl.range(worker // (HEAD_DIM // n_tile), row_blocks, row_workers):
                row = row_block * m_tile
                rows = pl.max(pl.min(m_tile, num_tokens - row), 0)
                a = pl.load(source, [row, 0], [m_tile, C1A_PROJ_K_TILE], valid_shape=[rows, C1A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
                b = pl.load(weight, [0, col], [C1A_PROJ_K_TILE, n_tile], target_memory=pl.Mem.Mat)
                full_a = pl.set_validshape(a, m_tile, C1A_PROJ_K_TILE)
                acc = pl.matmul(full_a, b)
                for k in pl.pipeline(C1A_PROJ_K_TILE, D, C1A_PROJ_K_TILE, stage=2):
                    a = pl.load(source, [row, k], [m_tile, C1A_PROJ_K_TILE], valid_shape=[rows, C1A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
                    b = pl.load(weight, [k, col], [C1A_PROJ_K_TILE, n_tile], target_memory=pl.Mem.Mat)
                    full_a = pl.set_validshape(a, m_tile, C1A_PROJ_K_TILE)
                    acc = pl.matmul_acc(acc, full_a, b)
                value = pl.set_validshape(acc, rows, n_tile)
                output = pl.store(value, [row, col], output)
    return output, completed


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "compressor_publish.cpp",
               include_dirs=(pto_isa_include_dir(), str(Path(__file__).parent / "kernels")))
def _c1a_compress(
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
    ...


@pl.jit.inline(auto_scope=False)
def c1a_compressor_publish(
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
    with pl.spmd(workers, name_hint="c1a_compress", deps=[cache_ready, source_ready]) as published:
        cache = _c1a_compress(source, slots, cache, cosine, sine, norm_weight, latent, active_tokens)
    return published


@pl.jit.inline(auto_scope=False)
def _c1a_publish_process_batch(
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
def _c1a_publish_flush(
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
    norm, payload, bf_factors = _c1a_publish_process_batch(
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
def _c1a_publish_body(
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
                    cache = _c1a_publish_flush(
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
        cache = _c1a_publish_flush(
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
    _published = _c1a_publish_body(source, norm_weight, cosine, sine, slots, cache, latent, num_tokens)
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


@pl.jit.inline
def _c2a_compressor_project_tile(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    output: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    row: pl.Scalar[pl.INDEX],
    col: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
    n_tile: pl.constexpr,
):
    a = pl.load(x, [row, 0], [C2A_PROJ_M_TILE, C2A_PROJ_K_TILE], valid_shape=[rows, C2A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
    b = pl.load(weight, [0, col], [C2A_PROJ_K_TILE, n_tile], target_memory=pl.Mem.Mat)
    a_full = pl.set_validshape(a, C2A_PROJ_M_TILE, C2A_PROJ_K_TILE)
    acc = pl.matmul(a_full, b)
    for k in pl.pipeline(C2A_PROJ_K_TILE, D, C2A_PROJ_K_TILE, stage=2):
        a = pl.load(x, [row, k], [C2A_PROJ_M_TILE, C2A_PROJ_K_TILE], valid_shape=[rows, C2A_PROJ_K_TILE], target_memory=pl.Mem.Mat)
        b = pl.load(weight, [k, col], [C2A_PROJ_K_TILE, n_tile], target_memory=pl.Mem.Mat)
        a_full = pl.set_validshape(a, C2A_PROJ_M_TILE, C2A_PROJ_K_TILE)
        acc = pl.matmul_acc(acc, a_full, b)
    value = pl.set_validshape(acc, rows, n_tile)
    output = pl.store(value, [row, col], output)
    return output


@pl.jit.inline
def c2a_compressor_project(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    kv_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_out: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    n_tile: pl.constexpr,
):
    """Project KV and gate with NZ BF16 weights and FP32 Cube output."""
    blocks = (num_tokens + C2A_PROJ_M_TILE - 1) // C2A_PROJ_M_TILE * (2 * HEAD_DIM // n_tile)
    workers = pl.max(pl.min(blocks, 16), 1)
    with pl.spmd(workers, name_hint="proj_cmp"):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_cmp"):
            worker = pl.tile.get_block_idx()
            col = worker % (HEAD_DIM // n_tile) * n_tile
            row_blocks = (num_tokens + C2A_PROJ_M_TILE - 1) // C2A_PROJ_M_TILE
            row_workers = pl.max(workers // (2 * HEAD_DIM // n_tile), 1)
            for row_block in pl.range(worker // (2 * HEAD_DIM // n_tile), row_blocks, row_workers):
                row = row_block * C2A_PROJ_M_TILE
                rows = pl.min(C2A_PROJ_M_TILE, num_tokens - row)
                if worker % (2 * HEAD_DIM // n_tile) < HEAD_DIM // n_tile:
                    kv_out = _c2a_compressor_project_tile(x, wkv, kv_out, row, col, rows, n_tile)
                else:
                    score_out = _c2a_compressor_project_tile(x, wgate, score_out, row, col, rows, n_tile)
    return kv_out, score_out


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "compressor_pool.cpp",
               include_dirs=(pto_isa_include_dir(), str(Path(__file__).parent / "kernels")))
def _c2a_compress(
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
) -> pl.Tensor:
    """Pool, normalize, publish compressed rows, and update request ring state."""
    ...


@pl.jit.inline(auto_scope=False)
def c2a_compressor_publish(
    kv_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    score_proj: pl.Tensor[[T_DYN, HEAD_DIM], pl.FP32],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    state_block_table: pl.Tensor[[B_DYN, 1], pl.INT32],
    state_cache: pl.Tensor[[STATE_BLOCKS_DYN, STATE_CAPACITY, STATE_WIDTH], pl.FP32],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    state_ready: pl.Scalar[pl.TASK_ID],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
):
    """Pool and publish rows with state reads and writes owned by physical block."""
    active_tokens = pl.cast(num_tokens, pl.INT32)
    blocks = pl.cast(pl.tensor.dim(state_cache, 0), pl.INT32)
    requests = pl.tensor.dim(state_block_table, 0)
    tokens = pl.tensor.dim(kv_proj, 0)
    # Each request can add both a leading and a trailing partial four-position span.
    groups = (tokens + 3) // 4 + 2 * requests
    mapped_starts = pl.create_tensor([groups + 1], dtype=pl.INT32)
    mapped_requests = pl.create_tensor([tokens], dtype=pl.INT32)
    mapped_blocks = pl.create_tensor([groups, 1], dtype=pl.INT32)
    native_blocks = pl.tensor.dim(state_cache, 0) * (STATE_CAPACITY // 4)
    native_state = pl.reshape(state_cache, [native_blocks, 4, STATE_WIDTH])
    with pl.spmd(1, name_hint="c2a_four_slot_requests", deps=[state_ready]) as mapped:
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, tokens):
            pl.write(mapped_requests, [row], pl.cast(-1, pl.INT32))
        for group in pl.range(groups):
            pl.write(mapped_starts, [group], active_tokens)
            pl.write(mapped_blocks, [group, 0], pl.cast(-1, pl.INT32))
        pl.write(mapped_starts, [groups], active_tokens)
        count = pl.cast(0, pl.INT32)
        for request in pl.range(requests):
            start = pl.max(pl.min(pl.read(query_start_loc, [request]), active_tokens), 0)
            end = pl.max(pl.min(pl.read(query_start_loc, [request + 1]), active_tokens), start)
            if start < end:
                first_position = pl.max(pl.read(position_ids, [start]), 0)
                offset = first_position % 4
                chunks = (end - start + offset + 3) // 4
                block = pl.read(state_block_table, [request, 0])
                for chunk in pl.range(chunks):
                    begin = start + pl.max(chunk * 4 - offset, 0)
                    limit = pl.min(start + (chunk + 1) * 4 - offset, end)
                    position = first_position + begin - start
                    native_block = pl.cast(-1, pl.INT32)
                    if block >= 0 and block < blocks:
                        native_block = pl.cast(block * (STATE_CAPACITY // 4) + position // 4 % (STATE_CAPACITY // 4), pl.INT32)
                    pl.write(mapped_starts, [count], pl.cast(begin, pl.INT32))
                    pl.write(mapped_starts, [count + 1], pl.cast(limit, pl.INT32))
                    pl.write(mapped_blocks, [count, 0], native_block)
                    for token in pl.range(begin, limit):
                        if pl.read(token_to_req_indices, [token]) == request:
                            pl.write(mapped_requests, [token], count)
                    count = pl.cast(count + 1, pl.INT32)
    # Request-based decode worker count.
    workers = pl.max(pl.min(pl.tensor.dim(state_block_table, 0), 16), 1)
    native_count = pl.cast(native_blocks, pl.INT32)
    with pl.spmd(workers, name_hint="c2a_compress", deps=[mapped]) as completed:
        _c2a_compress(
            kv_proj, score_proj, native_state, position_ids, norm_weight, latent,
            mapped_starts, mapped_requests, mapped_blocks,
            slots, cosine, sine, cache, active_tokens, native_count,
        )
    return completed


@pl.jit
def c2a_state_feedback_test(
    kv: pl.Tensor[[8, HEAD_DIM], pl.FP32],
    score: pl.Tensor[[8, HEAD_DIM], pl.FP32],
    positions: pl.Tensor[[8], pl.INT32],
    request_ids: pl.Tensor[[8], pl.INT32],
    query_starts: pl.Tensor[[16], pl.INT32],
    state_blocks: pl.Tensor[[8], pl.INT32],
    slots: pl.Tensor[[8], pl.INT64],
    norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    cosine: pl.Tensor[[8, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[8, ROPE_DIM // 2], pl.FP32],
    state_cache: pl.InOut[pl.Tensor[[3, STATE_CAPACITY, STATE_WIDTH], pl.FP32]],
    latent: pl.InOut[pl.Tensor[[8, HEAD_DIM], pl.BF16]],
):
    """Validate the ring retained after a six-row speculative verification window."""
    cache = pl.create_tensor([1, 128, 1, COMPRESSED_CACHE_BYTES], dtype=pl.UINT8)
    state_ready = pl.system.task_dummy(deps=[])
    state_table = pl.reshape(state_blocks, [8, 1])
    c2a_compressor_publish(
        kv, score, query_starts, positions, request_ids, state_table,
        state_cache, norm_weight, latent, pl.cast(6, pl.INT32), state_ready,
        cosine, sine, slots, cache,
    )
    return state_cache, latent


def golden_c1a_compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute C1A projection with K-tile FP32 accumulation followed by RMSNorm."""
    weight = unpack_bf16_nz(wkv).float()
    source = x.float()
    projected = torch.zeros((*x.shape[:-1], weight.shape[-1]), dtype=torch.float32, device=x.device)
    for begin in range(0, x.shape[-1], C1A_PROJ_K_TILE):
        end = begin + C1A_PROJ_K_TILE
        projected.add_(torch.matmul(source[..., begin:end], weight[begin:end]))
    projected = projected.to(x.dtype)
    return golden_rms_norm(projected, norm_weight)


def golden_c2a_compressor(
    x: torch.Tensor,
    query_start_loc: torch.Tensor,
    position_ids: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    state_block_table: torch.Tensor,
    state_cache: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute C2A compression with request-owned FP32 ring state.

    Query boundaries count valid packed tokens; any tensor tail is padding.
    Negative/out-of-range request or block indices suppress state access and
    publication. Valid chunks contain consecutive absolute positions.
    """
    kv = torch.matmul(x.float(), unpack_bf16_nz(wkv).float())
    score = torch.matmul(x.float(), unpack_bf16_nz(wgate).float())
    latent = torch.zeros_like(kv, dtype=x.dtype)
    publish = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    blocks, capacity, width = state_cache.shape
    if capacity < 1 or width != 2 * kv.shape[-1] or state_cache.dtype != torch.float32:
        raise ValueError("state_cache must be FP32 [blocks, positive capacity, 2 * head_dim]")
    head_dim = kv.shape[-1]
    for request in range(state_block_table.shape[0]):
        start = int(query_start_loc[request])
        end = min(int(query_start_loc[request + 1]), x.shape[0])
        block = int(state_block_table[request, 0])
        if start >= end or not 0 <= block < blocks:
            continue
        for token in range(start, end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) != request or position < 0:
                continue
            if position % 2:
                if token > start:
                    previous_kv, previous_score = kv[token - 1], score[token - 1]
                else:
                    previous = state_cache[block, (position - 1) % capacity]
                    previous_kv, previous_score = previous[:head_dim], previous[head_dim:]
                pair_kv = torch.stack((previous_kv, kv[token]))
                pair_score = torch.stack((previous_score, score[token]))
                pooled = (pair_kv * pair_score.softmax(dim=0)).sum(dim=0)
                latent[token] = golden_rms_norm(pooled.to(x.dtype), norm_weight)
                publish[token] = True
        for token in range(max(start, end - capacity), end):
            position = int(position_ids[token])
            if int(token_to_req_indices[token]) == request and position >= 0:
                state_cache[block, position % capacity, :head_dim] = kv[token]
                state_cache[block, position % capacity, head_dim:] = score[token]
    return latent, publish


if __name__ == "__main__":
    import argparse
    import functools

    from golden import TensorSpec, ratio_allclose, run
    from quant import pack_bf16_nz

    parser = argparse.ArgumentParser(description="C2A state feedback after speculative prefix acceptance")
    parser.add_argument("-p", "--platform", choices=("a5", "a5sim"), default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--start-position", type=int, default=12)
    parser.add_argument("--accepted", type=int, choices=range(1, 7), default=1)
    args = parser.parse_args()
    if args.start_position < 1:
        parser.error("--start-position must be positive")

    generator = torch.Generator().manual_seed(42)
    kv = (torch.randint(0, 2, (8, HEAD_DIM), generator=generator) * 2 - 1).float()
    positions = torch.full((8,), -1, dtype=torch.int32)
    positions[:6] = torch.arange(args.start_position, args.start_position + 6)
    request_ids = torch.tensor([0] * 6 + [-1] * 2, dtype=torch.int32)
    query_starts = torch.tensor([0] + [6] * 15, dtype=torch.int32)
    state_blocks = torch.tensor([1] + [-1] * 7, dtype=torch.int32)
    state = torch.full((3, STATE_CAPACITY, STATE_WIDTH), 0.25)
    state[1].zero_()
    state[2].fill_(0.75)
    state[1, (args.start_position - 1) % STATE_CAPACITY, :HEAD_DIM] = 1.0
    weight = pack_bf16_nz(torch.eye(HEAD_DIM, dtype=torch.bfloat16))
    gate = torch.zeros_like(weight)

    def golden(values):
        values["latent"].copy_(golden_c2a_compressor(
            values["kv"].bfloat16(), values["query_starts"], values["positions"],
            values["request_ids"], values["state_blocks"].unsqueeze(1), values["state_cache"],
            weight, gate, values["norm_weight"],
        )[0])

    def compare_feedback(actual, expected, *, compare, **kwargs):
        ok, detail = compare(actual, expected, **kwargs)
        next_position = args.start_position + args.accepted
        if not ok or next_position % 2 == 0:
            return ok, detail
        # An odd first row in the next window reads its accepted predecessor from the ring.
        inputs = kwargs["inputs"]
        block = int(inputs["state_blocks"][0])
        predecessor = next_position - 1
        retained = actual[block, predecessor % actual.shape[1]]
        required = torch.cat((inputs["kv"][args.accepted - 1], inputs["score"][args.accepted - 1]))
        if torch.equal(retained, required):
            return True, ""
        return False, (
            f"    accepted predecessor position {predecessor} was overwritten before the next window; "
            f"{int((retained != required).sum())}/{required.numel()} ring values differ"
        )

    specs = [
        TensorSpec("kv", [8, HEAD_DIM], torch.float32, init_value=kv),
        TensorSpec("score", [8, HEAD_DIM], torch.float32, init_value=0.0),
        TensorSpec("positions", [8], torch.int32, init_value=positions),
        TensorSpec("request_ids", [8], torch.int32, init_value=request_ids),
        TensorSpec("query_starts", [16], torch.int32, init_value=query_starts),
        TensorSpec("state_blocks", [8], torch.int32, init_value=state_blocks),
        TensorSpec("slots", [8], torch.int64, init_value=-1),
        TensorSpec("norm_weight", [HEAD_DIM], torch.bfloat16, init_value=1.0),
        TensorSpec("cosine", [8, ROPE_DIM // 2], torch.float32, init_value=1.0),
        TensorSpec("sine", [8, ROPE_DIM // 2], torch.float32, init_value=0.0),
        TensorSpec("state_cache", [3, STATE_CAPACITY, STATE_WIDTH], torch.float32, init_value=state),
        TensorSpec("latent", [8, HEAD_DIM], torch.bfloat16, init_value=0.0),
    ]
    result = run(
        fn=c2a_state_feedback_test, specs=specs, golden_fn=golden,
        compare_fn={
            "state_cache": functools.partial(
                compare_feedback, compare=ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            ),
            "latent": ratio_allclose(atol=1e-3, rtol=1e-3, valid_rows=6, zero_tail=True),
        },
        config={"backend_type": "a5", "device_id": args.device,
                "execution_mode": args.execution_mode or args.platform == "a5sim"},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
