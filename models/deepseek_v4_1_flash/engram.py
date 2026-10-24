# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Engram hashing, resident sharded lookup, projection and gated residual injection."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from config import EP_SIZE, FLASH as M, T_DYN
from quant import mx_projection


_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--engram-shards", type=int, default=EP_SIZE)
_parser.add_argument("--lookup-capacity", type=int)
_parser.add_argument("--capacity", type=int, default=128)
_parser.add_argument("--engram-compact", action="store_true")
_options = _parser.parse_known_args()[0]
WORLD = EP_SIZE
SHARDS = _options.engram_shards
CAPACITY = _options.capacity if _options.lookup_capacity is None else _options.lookup_capacity
COMPACT = _options.engram_compact
if not 1 <= SHARDS <= WORLD or CAPACITY < 1:
    raise ValueError("Engram requires 1 <= shards <= DP ranks and positive lookup capacity")


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
ENGRAM_LAYERS = len(M.engram_layer_ids)
N_HASH_COLS = (M.engram_max_ngram_size - 1) * M.engram_n_heads
HEAD_DIM = M.engram_head_dim
PAD_TOKEN_ID = M.engram_pad_token_id
HEADS_PER_NGRAM = M.engram_n_heads
ENGRAM_K = N_HASH_COLS * HEAD_DIM
KV_OUT = (HC_MULT + 1) * D
CLAMP = 1e-6
NORM_EPS = M.rms_norm_eps

# tiling
N_TILE = 256
K_TILE = 256
T_TILE = 16
D_TILE = 512


@pl.jit.inline(auto_scope=False)
def engram_gate(
    kv: pl.Tensor,
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    token_mask: pl.Tensor,
    x: pl.Tensor,
    out: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Inject BF16-rounded gated values into active FP32 HC rows, preserving masked rows."""
    tokens = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [tokens, HC_MULT * D])
    out_flat = pl.reshape(out, [tokens, HC_MULT * D])
    mask_rows = pl.reshape(token_mask, [1, tokens])
    for block in pl.spmd((tokens + T_TILE - 1) // T_TILE * HC_MULT, name_hint="engram_gate"):
        t0 = block // HC_MULT * T_TILE
        lane = block % HC_MULT
        valid_rows = pl.min(T_TILE, tokens - t0)
        active_rows = pl.min(T_TILE, pl.max(0, num_tokens - t0))
        gate = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        keep_zero = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        keep = pl.fillpad(keep_zero, pad_value=pl.PadValue.zero)
        if t0 < num_tokens:
            sq_x = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            sq_key = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            dot = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            reduce_tmp = pl.create_tile([T_TILE, D_TILE], dtype=pl.FP32)
            for d0 in pl.range(0, D, D_TILE):
                source = pl.load(x_flat, [t0, lane * D + d0], [T_TILE, D_TILE], valid_shape=[active_rows, D_TILE])
                source = pl.fillpad(source, pad_value=pl.PadValue.zero)
                source = pl.set_validshape(source, T_TILE, D_TILE)
                key = pl.load(kv, [t0, lane * D + d0], [T_TILE, D_TILE], valid_shape=[active_rows, D_TILE])
                key = pl.fillpad(key, pad_value=pl.PadValue.zero)
                key = pl.cast(pl.set_validshape(key, T_TILE, D_TILE), pl.FP32)
                scale = pl.load(weight, [lane, d0], [1, D_TILE])
                source_sq = pl.mul(source, source)
                key_sq = pl.mul(key, key)
                source_sum = pl.reshape(pl.row_sum(source_sq, tmp_tile=reduce_tmp), [1, T_TILE])
                key_sum = pl.reshape(pl.row_sum(key_sq, tmp_tile=reduce_tmp), [1, T_TILE])
                sq_x = pl.add(sq_x, source_sum)
                sq_key = pl.add(sq_key, key_sum)
                weighted_source = pl.col_expand_mul(source, scale)
                product = pl.mul(weighted_source, key)
                dot_sum = pl.reshape(pl.row_sum(product, tmp_tile=reduce_tmp), [1, T_TILE])
                dot = pl.add(dot, dot_sum)
            source_rms = pl.rsqrt(pl.add(pl.mul(sq_x, 1.0 / D), NORM_EPS))
            key_rms = pl.rsqrt(pl.add(pl.mul(sq_key, 1.0 / D), NORM_EPS))
            dot = pl.mul(pl.mul(pl.mul(dot, source_rms), key_rms), D**-0.5)
            magnitude = pl.sqrt(pl.maximum(pl.abs(dot), CLAMP))
            negative = pl.cmps(dot, 0.0, cmp_type=2)
            select_tmp = pl.create_tile([1, 16], dtype=pl.UINT32)
            signed = pl.sel(negative, pl.neg(magnitude), magnitude, select_tmp)
            gate = pl.recip(pl.add(pl.exp(pl.neg(signed)), 1.0))
            mask = pl.load(mask_rows, [0, t0], [1, T_TILE], valid_shape=[1, active_rows])
            mask = pl.set_validshape(pl.fillpad(mask, pad_value=pl.PadValue.zero), 1, T_TILE)
            keep = pl.mul(pl.cast(mask, pl.FP32), 1.0)
        gate_column = pl.reshape(gate, [T_TILE, 1])
        keep_column = pl.reshape(keep, [T_TILE, 1])
        for d0 in pl.range(0, D, D_TILE):
            residual_chunk = pl.load(x_flat, [t0, lane * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE])
            result = pl.mul(residual_chunk, 1.0)
            if t0 < num_tokens:
                value = pl.load(kv, [t0, HC_MULT * D + d0], [T_TILE, D_TILE], valid_shape=[active_rows, D_TILE])
                value = pl.fillpad(value, pad_value=pl.PadValue.zero)
                value = pl.cast(pl.set_validshape(value, T_TILE, D_TILE), pl.FP32)
                updated = pl.add(residual_chunk, pl.row_expand_mul(value, gate_column))
                rounded = pl.cast(pl.cast(updated, pl.BF16, mode="rint"), pl.FP32)
                ones = pl.tile.full([T_TILE, D_TILE], dtype=pl.FP32, value=1.0)
                expanded_mask = pl.row_expand_mul(ones, keep_column)
                predicate = pl.cmps(expanded_mask, 0.0, cmp_type=1)
                result_tmp = pl.create_tile([1, 16], dtype=pl.UINT32)
                result = pl.sel(predicate, rounded, residual_chunk, result_tmp)
            result = pl.set_validshape(result, valid_rows, D_TILE)
            out_flat = pl.store(result, [t0, lane * D + d0], out_flat)
    return out


@pl.jit.inline
def engram(
    rows: pl.Tensor,
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    token_mask: pl.Tensor,
    x: pl.Tensor,
    out: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project BF16 lookup rows with native MXFP8 weights and inject the HC residual."""
    tokens = pl.tensor.dim(x, 0)
    kv = pl.create_tensor([tokens, KV_OUT], dtype=pl.BF16)
    mx_projection(
        rows, wkv_weight, wkv_scale, kv, num_tokens,
        ENGRAM_K, KV_OUT, "engram_wkv", "engram_quant", N_TILE, K_TILE,
        fp32_output=False, floor_scale=False, group_count=1,
    )
    engram_gate(kv, weight, token_mask, x, out, num_tokens)
    return out


def golden_engram(rows, wkv_weight, wkv_scale, weight, token_mask, x, num_tokens):
    """Compute native MXFP8 projection, normalized gates and BF16 residual rounding."""
    from quant import decode_e8m0_codes_v41, host_quant_mxfp8_v41, matmul_mx_golden_v41

    result = x.clone()
    if not num_tokens:
        return result
    payload, codes = host_quant_mxfp8_v41(rows[:num_tokens], return_e8m0=True)
    weight_codes = decode_e8m0_codes_v41(wkv_scale, side="b")
    kv = matmul_mx_golden_v41(payload, codes, wkv_weight, weight_codes).to(torch.bfloat16).float()
    key, value = kv.split([HC_MULT * D, D], dim=-1)
    key = key.reshape(num_tokens, HC_MULT, D)
    hidden = x[:num_tokens].float()
    hidden_rms = torch.rsqrt(hidden.square().mean(-1) + NORM_EPS)
    key_rms = torch.rsqrt(key.square().mean(-1) + NORM_EPS)
    dot = (hidden * weight.float() * key).sum(-1) * hidden_rms * key_rms * D**-0.5
    signed = torch.copysign(dot.abs().clamp_min(CLAMP).sqrt(), dot)
    gate = torch.sigmoid(signed)
    updated = (hidden + gate.unsqueeze(-1) * value.unsqueeze(1)).to(torch.bfloat16).float()
    active = token_mask[:num_tokens].bool().reshape(num_tokens, 1, 1)
    result[:num_tokens] = torch.where(active, updated, hidden)
    return result


@pl.jit
def l2_engram(
    rows: pl.Tensor[[T_DYN, ENGRAM_K], pl.BF16],
    wkv_weight: pl.Tensor[[ENGRAM_K, KV_OUT], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[ENGRAM_K // 32, KV_OUT], pl.FP8E8M0],
    weight: pl.Tensor[[HC_MULT, D], pl.FP32],
    token_mask: pl.Tensor[[T_DYN], pl.INT32],
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Validate the Engram computation shared by prefill and decode."""
    rows.bind_dynamic(0, T_DYN)
    token_mask.bind_dynamic(0, T_DYN)
    x.bind_dynamic(0, T_DYN)
    out.bind_dynamic(0, T_DYN)
    engram(rows, wkv_weight, wkv_scale, weight, token_mask, x, out, num_tokens)
    return out


def build_engram_tensor_specs(tokens=17, num_tokens=13):
    """Build deterministic native-weight inputs with masked and inactive HC rows."""
    from golden import ScalarSpec, TensorSpec
    from quant import gen_mxfp8_weight_kn_v41

    generator = torch.Generator().manual_seed(17)
    rows = torch.randn(tokens, ENGRAM_K, generator=generator).to(torch.bfloat16)
    weight, scale = gen_mxfp8_weight_kn_v41(KV_OUT, ENGRAM_K, 1.0 / math.sqrt(ENGRAM_K), seed=19)
    gate_weight = torch.randn(HC_MULT, D, generator=generator) * 0.1 + 1.0
    hidden = torch.randn(tokens, HC_MULT, D, generator=generator)
    hidden[0] = 0
    mask = torch.ones(tokens, dtype=torch.int32)
    mask[1::3] = 0
    return [
        TensorSpec("rows", list(rows.shape), rows.dtype, init_value=rows),
        TensorSpec("wkv_weight", list(weight.shape), weight.dtype, init_value=weight),
        TensorSpec("wkv_scale", list(scale.shape), scale.dtype, init_value=scale),
        TensorSpec("weight", list(gate_weight.shape), gate_weight.dtype, init_value=gate_weight),
        TensorSpec("token_mask", [tokens], mask.dtype, init_value=mask),
        TensorSpec("x", list(hidden.shape), hidden.dtype, init_value=hidden),
        TensorSpec("out", list(hidden.shape), hidden.dtype),
        ScalarSpec("num_tokens", torch.int32, num_tokens, compile_runtime=True),
    ]


def golden_engram_case(tensors):
    """Fill expected active, masked and inactive residual rows."""
    tensors["out"][:] = golden_engram(
        tensors["rows"], tensors["wkv_weight"], tensors["wkv_scale"], tensors["weight"],
        tensors["token_mask"], tensors["x"], int(tensors["num_tokens"]),
    )


def compare_engram(actual, expected, *, inputs, **kwargs):
    """Check the numerical output and exact preservation of masked or inactive rows."""
    from golden import ratio_allclose

    count = int(inputs["num_tokens"])
    preserved = (torch.arange(actual.shape[0]) >= count) | (inputs["token_mask"] == 0)
    if not torch.equal(actual[preserved], inputs["x"][preserved]):
        return False, "masked and inactive Engram rows must preserve the FP32 input exactly"
    check = ratio_allclose(atol=1e-3, rtol=1e-2)
    return check(actual, expected, inputs=inputs, **kwargs)


# Dynamic shape variables.
TABLE_ROWS = pl.dynamic("ENGRAM_TABLE_ROWS")
MAP_ROWS = pl.dynamic("ENGRAM_MAP_ROWS")
OUTPUT_ROWS = pl.dynamic("ENGRAM_OUTPUT_ROWS")

# model config
LAYERS = len(M.engram_layer_ids)
HEADS = (M.engram_max_ngram_size - 1) * M.engram_n_heads
WIDTH = M.engram_head_dim
LAYER0_ROWS, LAYER1_ROWS = M.engram_num_embeddings
SLOTS = LAYERS * CAPACITY * HEADS

# tiling
REQUEST_TILE = 16
SIGNAL_TILE = 16


@pl.jit.inline(auto_scope=False)
def engram_hash_context(
    context: pl.Tensor,
    token_map: pl.Tensor[[M.vocab_size], pl.INT32],
    multipliers: pl.Tensor[[LAYERS, 4], pl.INT64],
    primes: pl.Tensor[[LAYERS, HEADS], pl.INT32],
    offsets: pl.Tensor[[LAYERS, HEADS], pl.INT32],
    hash_ids: pl.Tensor[[LAYERS, CAPACITY, HEADS], pl.INT32],
    count: pl.Scalar[pl.INT32],
):
    """Compress raw token contexts and hash both layers on the requesting NPU."""
    hashes = pl.reshape(hash_ids, [1, SLOTS])
    context_capacity = pl.tensor.dim(context, 0)
    for block in pl.spmd(SLOTS // REQUEST_TILE, name_hint="engram_hash"):
        first = block * REQUEST_TILE
        result = pl.tile.full([1, REQUEST_TILE], dtype=pl.INT32, value=-1)
        for lane in pl.range(REQUEST_TILE):
            slot = first + lane
            layer = slot // (CAPACITY * HEADS)
            token = slot // HEADS % CAPACITY
            head = slot % HEADS
            if token < count and token < context_capacity:
                pad = pl.read(token_map, [PAD_TOKEN_ID])
                zero: pl.Scalar[pl.INT64] = 0
                for shift, (rolling, blocked) in pl.range(head // HEADS_PER_NGRAM + 2, init_values=(zero, False)):
                    raw = pl.read(context, [token, shift])
                    next_blocked = blocked or raw < 0
                    compressed = pad
                    if not next_blocked:
                        compressed = pl.read(token_map, [raw])
                    multiplier = pl.read(multipliers, [layer, shift])
                    mixed = pl.cast(compressed, pl.INT64) * multiplier
                    next_rolling = rolling ^ mixed
                    rolling, blocked = pl.yield_(next_rolling, next_blocked)
                prime = pl.read(primes, [layer, head])
                offset = pl.read(offsets, [layer, head])
                original = rolling % pl.cast(prime, pl.INT64) + pl.cast(offset, pl.INT64)
                pl.tile.write(result, [0, lane], pl.cast(original, pl.INT32))
        hashes = pl.store(result, [0, first], hashes)


@pl.jit.inline(auto_scope=False)
def prepare_engram(
    context: pl.Tensor[[OUTPUT_ROWS, 4], pl.INT32],
    token_map: pl.Tensor[[M.vocab_size], pl.INT32],
    multipliers: pl.Tensor[[LAYERS, 4], pl.INT64],
    primes: pl.Tensor[[LAYERS, HEADS], pl.INT32],
    offsets: pl.Tensor[[LAYERS, HEADS], pl.INT32],
    hash_ids: pl.Tensor[[LAYERS, CAPACITY, HEADS], pl.INT32],
    token_mask: pl.Tensor[[OUTPUT_ROWS], pl.INT32],
    table: pl.Tensor[[LAYERS, TABLE_ROWS, WIDTH], pl.FP8E4M3FN],
    scale: pl.Tensor[[LAYERS, TABLE_ROWS, WIDTH // 32], pl.FP8E8M0],
    row_ids: pl.Tensor[[LAYERS, MAP_ROWS], pl.INT32],
    rows: pl.Out[pl.Tensor[[LAYERS, OUTPUT_ROWS, HEADS * WIDTH], pl.BF16]],
    missing: pl.Out[pl.Tensor[[SLOTS, SIGNAL_TILE], pl.INT32]],
    counts: pl.Tensor[[WORLD], pl.INT32],
    requests: pld.DistributedTensor[[WORLD * SLOTS, SIGNAL_TILE], pl.INT32],
    request_done: pld.DistributedTensor[[WORLD, SIGNAL_TILE], pl.INT32],
    responses: pld.DistributedTensor[[SLOTS, WIDTH], pl.BF16],
    response_status: pld.DistributedTensor[[SLOTS, SIGNAL_TILE], pl.INT32],
    response_done: pld.DistributedTensor[[SHARDS, SIGNAL_TILE], pl.INT32],
    readers_done: pld.DistributedTensor[[WORLD, SIGNAL_TILE], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    compute_hash: pl.constexpr,
):
    """Route independent DP requests to striped table owners and restore layer/token/head order."""
    output_capacity = pl.tensor.dim(rows, 1)
    count = pl.read(counts, [rank])
    if compute_hash:
        engram_hash_context(context, token_map, multipliers, primes, offsets, hash_ids, count)
    packed = pl.create_tensor([SHARDS * SLOTS, SIGNAL_TILE], dtype=pl.INT32)
    with pl.spmd(SHARDS * ((SLOTS + REQUEST_TILE - 1) // REQUEST_TILE), name_hint="engram_request") as push:
        block = pl.tile.get_block_idx()
        owner = block // ((SLOTS + REQUEST_TILE - 1) // REQUEST_TILE)
        first = block % ((SLOTS + REQUEST_TILE - 1) // REQUEST_TILE) * REQUEST_TILE
        valid = pl.min(REQUEST_TILE, SLOTS - first)
        record = pl.tile.full([REQUEST_TILE, SIGNAL_TILE], dtype=pl.INT32, value=-1)
        for lane in pl.range(REQUEST_TILE):
            slot = first + lane
            if slot < SLOTS:
                layer = slot // (CAPACITY * HEADS)
                token = slot // HEADS % CAPACITY
                head = slot % HEADS
                if token < count:
                    keep = pl.read(token_mask, [token])
                    if keep != 0:
                        original = pl.read(hash_ids, [layer, token, head])
                        if original >= 0 and original % SHARDS == owner:
                            pl.tile.write(record, [lane, 0], original)
        packed = pl.store(pl.set_validshape(record, valid, SIGNAL_TILE), [owner * SLOTS + first, 0], packed)
        pld.tensor.put(requests, peer=owner, src=packed, dst_offsets=[rank * SLOTS + first, 0], src_offsets=[owner * SLOTS + first, 0], shape=[REQUEST_TILE, SIGNAL_TILE])
        pld.system.notify(request_done, peer=owner, offsets=[rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="engram_request_wait") as wait_requests:
        if rank < SHARDS:
            for source in pl.range(WORLD):
                pld.system.defer_wait(request_done, offsets=[source, 0], expected=(SLOTS + REQUEST_TILE - 1) // REQUEST_TILE, cmp=pld.WaitCmp.Ge)
    replies = pl.create_tensor([WORLD * SLOTS, WIDTH], dtype=pl.BF16)
    statuses = pl.create_tensor([WORLD * SLOTS, SIGNAL_TILE], dtype=pl.INT32)
    table_flat = pl.reshape(table, [LAYERS * pl.tensor.dim(table, 1), WIDTH])
    scale_flat = pl.reshape(scale, [LAYERS * pl.tensor.dim(table, 1), WIDTH // 32])
    with pl.spmd(WORLD * ((SLOTS + REQUEST_TILE - 1) // REQUEST_TILE), name_hint="engram_owner_lookup", deps=[push, wait_requests]) as lookup:
        block = pl.tile.get_block_idx()
        source = block // ((SLOTS + REQUEST_TILE - 1) // REQUEST_TILE)
        first = block % ((SLOTS + REQUEST_TILE - 1) // REQUEST_TILE) * REQUEST_TILE
        if rank < SHARDS:
            for lane in pl.range(REQUEST_TILE):
                slot = first + lane
                if slot < SLOTS:
                    original = pl.read(requests, [source * SLOTS + slot, 0])
                    if original >= 0:
                        layer = slot // (CAPACITY * HEADS)
                        local = original // SHARDS
                        found = original < LAYER0_ROWS
                        if layer == 1:
                            found = original < LAYER1_ROWS
                        if COMPACT:
                            for (low, high) in pl.while_(init_values=(0, pl.tensor.dim(row_ids, 1))):
                                pl.cond(low < high)
                                middle = (low + high) // 2
                                candidate = pl.read(row_ids, [layer, middle])
                                if candidate < original:
                                    next_low = middle + 1
                                    next_high = high
                                else:
                                    next_low = low
                                    next_high = middle
                                low, high = pl.yield_(next_low, next_high)
                            local = low
                            found = False
                            if low < pl.tensor.dim(row_ids, 1):
                                found = pl.read(row_ids, [layer, low]) == original
                        if local >= pl.tensor.dim(table, 1):
                            found = False
                        value = pl.tile.full([1, WIDTH], dtype=pl.BF16, value=0.0)
                        status = pl.tile.full([1, SIGNAL_TILE], dtype=pl.INT32, value=0)
                        if found:
                            payload = pl.load(table_flat, [layer * pl.tensor.dim(table, 1) + local, 0], [1, WIDTH])
                            payload = pl.cast(pl.reshape(payload, [WIDTH // 32, 32]), pl.FP32)
                            exponent = pl.load(scale_flat, [layer * pl.tensor.dim(table, 1) + local, 0], [1, 32], valid_shape=[1, WIDTH // 32])
                            exponent = pl.cast(pl.cast(pl.reinterpret_view(exponent, pl.UINT8), pl.FP16), pl.INT32)
                            exponent = pl.reshape(exponent, [1, 32])
                            bits = pl.shls(pl.maximum(exponent, 1), 23)
                            decoded = pl.reinterpret_view(bits, pl.FP32)
                            multiplier = pl.minimum(pl.add(pl.cast(exponent, pl.FP32), 1.0), 2.0)
                            decoded = pl.mul(decoded, pl.mul(multiplier, 0.5))
                            decoded = pl.reshape(pl.tile.slice(decoded, [1, WIDTH // 32], [0, 0]), [WIDTH // 32, 1])
                            value = pl.cast(pl.reshape(pl.row_expand_mul(payload, decoded), [1, WIDTH]), pl.BF16, mode="rint")
                        else:
                            pl.tile.write(status, [0, 0], pl.cast(1, pl.INT32))
                        replies = pl.store(value, [source * SLOTS + slot, 0], replies)
                        statuses = pl.store(status, [source * SLOTS + slot, 0], statuses)
                        pld.tensor.put(responses, peer=source, src=replies, dst_offsets=[slot, 0], src_offsets=[source * SLOTS + slot, 0], shape=[1, WIDTH])
                        pld.tensor.put(response_status, peer=source, src=statuses, dst_offsets=[slot, 0], src_offsets=[source * SLOTS + slot, 0], shape=[1, SIGNAL_TILE])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="engram_response_notify", deps=[lookup]) as notify_responses:
        if rank < SHARDS:
            for source in pl.range(WORLD):
                pld.system.notify(response_done, peer=source, offsets=[rank, 0], value=1, op=pld.NotifyOp.Set)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="engram_response_wait") as wait_responses:
        for owner in pl.range(SHARDS):
            pld.system.defer_wait(response_done, offsets=[owner, 0], expected=1, cmp=pld.WaitCmp.Ge)
    output = pl.reshape(rows, [LAYERS * output_capacity * HEADS, WIDTH])
    with pl.spmd((SLOTS + REQUEST_TILE - 1) // REQUEST_TILE, name_hint="engram_restore", deps=[notify_responses, wait_responses]) as restore:
        first = pl.tile.get_block_idx() * REQUEST_TILE
        for lane in pl.range(REQUEST_TILE):
            slot = first + lane
            if slot < SLOTS:
                token = slot // HEADS % CAPACITY
                restored_value = pl.tile.full([1, WIDTH], dtype=pl.BF16, value=0.0)
                restored_status = pl.tile.full([1, SIGNAL_TILE], dtype=pl.INT32, value=0)
                if token < count:
                    keep = pl.read(token_mask, [token])
                    if keep != 0:
                        requested_id = pl.read(hash_ids, [slot // (CAPACITY * HEADS), token, slot % HEADS])
                        if requested_id >= 0:
                            restored_value = pl.load(responses, [slot, 0], [1, WIDTH])
                            restored_status = pl.load(response_status, [slot, 0], [1, SIGNAL_TILE])
                        else:
                            pl.tile.write(restored_status, [0, 0], pl.cast(1, pl.INT32))
                if token < output_capacity:
                    output_slot = slot // (CAPACITY * HEADS) * output_capacity * HEADS + token * HEADS + slot % HEADS
                    output = pl.store(restored_value, [output_slot, 0], output)
                missing = pl.store(restored_status, [slot, 0], missing)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="engram_readers_notify", deps=[restore]) as notify_readers:
        for peer in pl.range(WORLD):
            pld.system.notify(readers_done, peer=peer, offsets=[rank, 0], value=1, op=pld.NotifyOp.Set)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="engram_readers_wait") as wait_readers:
        for peer in pl.range(WORLD):
            pld.system.defer_wait(readers_done, offsets=[peer, 0], expected=1, cmp=pld.WaitCmp.Ge)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="engram_retire", deps=[notify_readers, wait_readers]):
        anchor = pl.read(missing, [0, 0])
        for peer in pl.range(WORLD):
            pld.system.notify(request_done, peer=rank, offsets=[peer, 0], value=0, op=pld.NotifyOp.Set)
            pld.system.notify(readers_done, peer=rank, offsets=[peer, 0], value=0, op=pld.NotifyOp.Set)
        for owner in pl.range(SHARDS):
            pld.system.notify(response_done, peer=rank, offsets=[owner, 0], value=0, op=pld.NotifyOp.Set)
        pl.write(missing, [0, 0], anchor)
    return rows, missing


@pl.jit(auto_scope=False)
def l2_engram_lookup(
    context: pl.Tensor[[OUTPUT_ROWS, 4], pl.INT32],
    token_map: pl.Tensor[[M.vocab_size], pl.INT32],
    multipliers: pl.Tensor[[LAYERS, 4], pl.INT64],
    primes: pl.Tensor[[LAYERS, HEADS], pl.INT32],
    offsets: pl.Tensor[[LAYERS, HEADS], pl.INT32],
    hash_ids: pl.Tensor[[LAYERS, CAPACITY, HEADS], pl.INT32],
    token_mask: pl.Tensor[[OUTPUT_ROWS], pl.INT32],
    table: pl.Tensor[[LAYERS, TABLE_ROWS, WIDTH], pl.FP8E4M3FN],
    scale: pl.Tensor[[LAYERS, TABLE_ROWS, WIDTH // 32], pl.FP8E8M0],
    row_ids: pl.Tensor[[LAYERS, MAP_ROWS], pl.INT32],
    rows: pl.Out[pl.Tensor[[LAYERS, OUTPUT_ROWS, HEADS * WIDTH], pl.BF16]],
    missing: pl.Out[pl.Tensor[[SLOTS, SIGNAL_TILE], pl.INT32]],
    counts: pl.Tensor[[WORLD], pl.INT32],
    requests: pld.DistributedTensor[[WORLD * SLOTS, SIGNAL_TILE], pl.INT32],
    request_done: pld.DistributedTensor[[WORLD, SIGNAL_TILE], pl.INT32],
    responses: pld.DistributedTensor[[SLOTS, WIDTH], pl.BF16],
    response_status: pld.DistributedTensor[[SLOTS, SIGNAL_TILE], pl.INT32],
    response_done: pld.DistributedTensor[[SHARDS, SIGNAL_TILE], pl.INT32],
    readers_done: pld.DistributedTensor[[WORLD, SIGNAL_TILE], pl.INT32],
    rank: pl.Scalar[pl.INT32],
):
    """Run the shared lookup protocol as an independent per-rank validation entry."""
    context.bind_dynamic(0, OUTPUT_ROWS)
    token_mask.bind_dynamic(0, OUTPUT_ROWS)
    table.bind_dynamic(1, TABLE_ROWS)
    scale.bind_dynamic(1, TABLE_ROWS)
    row_ids.bind_dynamic(1, MAP_ROWS)
    rows.bind_dynamic(1, OUTPUT_ROWS)
    prepare_engram(
        context, token_map, multipliers, primes, offsets,
        hash_ids, token_mask, table, scale, row_ids, rows, missing, counts,
        requests, request_done, responses, response_status, response_done, readers_done, rank, False,
    )
    return rows, missing


@pl.jit.host
def l3_engram_lookup(
    context: pl.Tensor[[WORLD, OUTPUT_ROWS, 4], pl.INT32],
    token_map: pl.Tensor[[WORLD, M.vocab_size], pl.INT32],
    multipliers: pl.Tensor[[WORLD, LAYERS, 4], pl.INT64],
    primes: pl.Tensor[[WORLD, LAYERS, HEADS], pl.INT32],
    offsets: pl.Tensor[[WORLD, LAYERS, HEADS], pl.INT32],
    hash_ids: pl.Tensor[[WORLD, LAYERS, CAPACITY, HEADS], pl.INT32],
    token_mask: pl.Tensor[[WORLD, OUTPUT_ROWS], pl.INT32],
    table: pl.Tensor[[WORLD, LAYERS, TABLE_ROWS, WIDTH], pl.FP8E4M3FN],
    scale: pl.Tensor[[WORLD, LAYERS, TABLE_ROWS, WIDTH // 32], pl.FP8E8M0],
    row_ids: pl.Tensor[[WORLD, LAYERS, MAP_ROWS], pl.INT32],
    rows: pl.Out[pl.Tensor[[WORLD, LAYERS, OUTPUT_ROWS, HEADS * WIDTH], pl.BF16]],
    missing: pl.Out[pl.Tensor[[WORLD, SLOTS, SIGNAL_TILE], pl.INT32]],
    counts: pl.Tensor[[WORLD], pl.INT32],
):
    """Prepare both Engram layers before a backbone call; serialized calls reuse communication windows."""
    rows.bind_dynamic(2, OUTPUT_ROWS)
    table.bind_dynamic(2, TABLE_ROWS)
    scale.bind_dynamic(2, TABLE_ROWS)
    row_ids.bind_dynamic(2, MAP_ROWS)
    request_buffer = pld.alloc_window_buffer([WORLD * SLOTS, SIGNAL_TILE], dtype=pl.INT32)
    request_signal = pld.alloc_window_buffer([WORLD, SIGNAL_TILE], dtype=pl.INT32)
    response_buffer = pld.alloc_window_buffer([SLOTS, WIDTH], dtype=pl.BF16)
    status_buffer = pld.alloc_window_buffer([SLOTS, SIGNAL_TILE], dtype=pl.INT32)
    response_signal = pld.alloc_window_buffer([SHARDS, SIGNAL_TILE], dtype=pl.INT32)
    readers_signal = pld.alloc_window_buffer([WORLD, SIGNAL_TILE], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        requests = pld.window(request_buffer, [WORLD * SLOTS, SIGNAL_TILE], dtype=pl.INT32)
        request_done = pld.window(request_signal, [WORLD, SIGNAL_TILE], dtype=pl.INT32)
        responses = pld.window(response_buffer, [SLOTS, WIDTH], dtype=pl.BF16)
        response_status = pld.window(status_buffer, [SLOTS, SIGNAL_TILE], dtype=pl.INT32)
        response_done = pld.window(response_signal, [SHARDS, SIGNAL_TILE], dtype=pl.INT32)
        readers_done = pld.window(readers_signal, [WORLD, SIGNAL_TILE], dtype=pl.INT32)
        l2_engram_lookup(
            context[rank], token_map[rank], multipliers[rank], primes[rank], offsets[rank],
            hash_ids[rank], token_mask[rank], table[rank], scale[rank], row_ids[rank], rows[rank], missing[rank], counts,
            requests, request_done, responses, response_status, response_done, readers_done, rank, device=rank,
        )


def write_engram_contract(output_dir):
    """Record the compiled rank, shard, request-capacity and table-mode specialization."""
    contract = {"world": WORLD, "shards": SHARDS, "capacity": CAPACITY, "compact": COMPACT}
    Path(output_dir, "engram_lookup_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


def check_engram_contract(compiled, shards, compact):
    """Reject table layouts that disagree with the compiled forward specialization."""
    expected = {"world": WORLD, "shards": shards, "capacity": CAPACITY, "compact": compact}
    path = compiled.output_dir / "engram_lookup_contract.json"
    if not path.is_file() or json.loads(path.read_text()) != expected:
        raise ValueError("compiled Engram ranks, shards, capacity or compact-table mode do not match the resident tables")


def golden_engram_lookup(values, masks, counts):
    """Compute exact CPU hashes and BF16 rows for native resident-table validation."""
    context = values["engram_context"]
    ranks, output_capacity = context.shape[:2]
    hashes = torch.full((ranks, LAYERS, CAPACITY, HEADS), -1, dtype=torch.int32)
    rows = torch.zeros(ranks, LAYERS, output_capacity, HEADS, WIDTH, dtype=torch.bfloat16)
    for rank in range(ranks):
        count = int(counts[rank])
        raw = context[rank, :count].long()
        compressed = values["engram_token_map"][rank][raw.clamp_min(0)].long()
        pad = int(values["engram_token_map"][rank, PAD_TOKEN_ID])
        blocked = (raw < 0).cumsum(-1) != 0
        compressed[blocked] = pad
        for layer in range(LAYERS):
            rolling = torch.zeros(count, dtype=torch.int64)
            for shift in range(4):
                rolling ^= compressed[:, shift] * values["engram_multipliers"][rank, layer, shift]
                if shift:
                    first, last = (shift - 1) * HEADS_PER_NGRAM, shift * HEADS_PER_NGRAM
                    prime = values["engram_primes"][rank, layer, first:last].long()
                    offset = values["engram_offsets"][rank, layer, first:last].long()
                    hashes[rank, layer, :count, first:last] = (rolling[:, None] % prime + offset).int()
            ids = hashes[rank, layer, :count].long()
            for owner in range(SHARDS):
                selected = (ids % SHARDS == owner) & masks[rank, :count, None].bool()
                requested = ids[selected]
                if COMPACT:
                    known = values["engram_row_ids"][owner, layer].long()
                    local = torch.searchsorted(known, requested)
                    if bool((local >= known.numel()).any()) or not torch.equal(known[local], requested):
                        raise ValueError("validation oracle encountered a missing compact original ID")
                else:
                    local = requested // SHARDS
                payload = values["engram_table"][owner, layer].view(torch.uint8)[local].view(torch.float8_e4m3fn)
                scale = values["engram_scale"][owner, layer].view(torch.uint8)[local].view(torch.float8_e8m0fnu)
                decoded = (payload.float().reshape(-1, 8, 32) * scale.float().unsqueeze(-1)).reshape(-1, WIDTH).bfloat16()
                rows[rank, layer, :count][selected] = decoded
    return hashes, rows.flatten(-2)


def validate_forward_engram(runtime, handles, values, masks, counts):
    """Check device hashes, native-table rows and missing status after a forward."""
    expected_hashes, expected_rows = golden_engram_lookup(values, masks, counts)
    for name, expected in (("engram_hash_ids", expected_hashes), ("engram_rows", expected_rows)):
        actual = torch.empty_like(expected)
        runtime.copy_stacked_from(handles[name], actual)
        if not torch.equal(actual, expected):
            locations = (actual != expected).nonzero()[:8].tolist()
            raise AssertionError(f"forward {name} differs from host reference at {locations}")
    errors = torch.empty((len(counts), SLOTS, SIGNAL_TILE), dtype=torch.int32)
    runtime.copy_stacked_from(handles["engram_missing"], errors)
    if bool(errors.any()):
        raise ValueError("forward Engram lookup reported missing original row IDs")
    print("FWD_ENGRAM_HASH_ROWS_EXACT_PASS", list(map(int, counts)), flush=True)


def build_engram_fixture(ranks, capacity, count, sequence=None):
    """Build bounded native tables and explicit small hash buckets for forward CI."""
    generator = torch.Generator().manual_seed(122)
    table_rows = (HEADS * 17 + SHARDS - 1) // SHARDS
    table = (torch.randn(ranks, LAYERS, table_rows, WIDTH, generator=generator) * 0.125).to(torch.float8_e4m3fn)
    codes = torch.randint(124, 129, (ranks, LAYERS, table_rows, 8), generator=generator, dtype=torch.uint8)
    context = torch.full((ranks, capacity, 4), -1, dtype=torch.int32)
    for rank in range(ranks):
        for token in range(count):
            position = token if sequence is None else token % sequence
            for shift in range(4):
                if position >= shift:
                    context[rank, token, shift] = 11 + rank * 127 + token - shift
    mapping = torch.full((ranks, LAYERS, table_rows), torch.iinfo(torch.int32).max, dtype=torch.int32)
    for rank in range(min(SHARDS, ranks)):
        ids = torch.arange(rank, HEADS * 17, SHARDS, dtype=torch.int32)
        mapping[rank, :, :ids.numel()] = ids
    return {
        "engram_context": context,
        "engram_mask": (torch.arange(capacity) < count).to(torch.int32).repeat(ranks, 1),
        "engram_token_map": torch.arange(M.vocab_size, dtype=torch.int32).repeat(ranks, 1),
        "engram_multipliers": torch.tensor([[1000000007, 1000000009, 1000000033, 1000000087], [1000000093, 1000000097, 1000000103, 1000000123]], dtype=torch.int64).repeat(ranks, 1, 1),
        "engram_primes": torch.full((ranks, LAYERS, HEADS), 17, dtype=torch.int32),
        "engram_offsets": (torch.arange(HEADS, dtype=torch.int32) * 17).reshape(1, 1, HEADS).repeat(ranks, LAYERS, 1),
        "engram_table": table, "engram_scale": codes.view(torch.float8_e8m0fnu),
        "engram_row_ids": mapping,
        "engram_hash_ids": torch.zeros(ranks, LAYERS, CAPACITY, HEADS, dtype=torch.int32),
        "engram_missing": torch.zeros(ranks, SLOTS, SIGNAL_TILE, dtype=torch.int32),
        "engram_counts": torch.full((ranks,), count, dtype=torch.int32),
    }


def lookup_fixture(round_index, table_rows=64):
    """Build different DP requests, all-empty calls and compact misses against native FP8 shards."""
    generator = torch.Generator().manual_seed(100)
    table = (torch.randn(WORLD, LAYERS, table_rows, WIDTH, generator=generator) * 2).to(torch.float8_e4m3fn)
    scale = torch.randint(120, 130, (WORLD, LAYERS, table_rows, WIDTH // 32), generator=generator)
    scale = scale.to(torch.uint8).view(torch.float8_e8m0fnu)
    maps = torch.empty(WORLD, LAYERS, table_rows, dtype=torch.int32)
    for rank in range(WORLD):
        for layer in range(LAYERS):
            maps[rank, layer] = torch.arange(table_rows) * SHARDS + rank + layer * SHARDS * 1000
    hashes = torch.zeros(WORLD, LAYERS, CAPACITY, HEADS, dtype=torch.int32)
    counts = torch.tensor([max(0, CAPACITY - rank % 3) for rank in range(WORLD)], dtype=torch.int32)
    counts[round_index % WORLD] = 0
    if round_index == 2:
        counts.zero_()
    mask = torch.ones(WORLD, CAPACITY, dtype=torch.int32)
    if CAPACITY > 1:
        mask[:, 1] = 0
    expected = torch.zeros(WORLD, LAYERS, CAPACITY, HEADS, WIDTH, dtype=torch.bfloat16)
    errors = torch.zeros(WORLD, SLOTS, SIGNAL_TILE, dtype=torch.int32)
    for rank in range(WORLD):
        for layer in range(LAYERS):
            for token in range(CAPACITY):
                for head in range(HEADS):
                    owner = (rank + head + round_index) % SHARDS
                    local = (token * HEADS + head + round_index * 7) % table_rows
                    original = int(maps[owner, layer, local]) if COMPACT else local * SHARDS + owner
                    hashes[rank, layer, token, head] = original
                    if token < counts[rank] and mask[rank, token]:
                        payload = table[owner, layer, local].float().reshape(-1, 32)
                        decoded = payload * scale[owner, layer, local].float().unsqueeze(-1)
                        expected[rank, layer, token, head] = decoded.reshape(WIDTH).bfloat16()
    if round_index == 3:
        hashes[0, 0, 0, 0] = -1
        expected[0, 0, 0, 0] = 0
        errors[0, 0, 0] = 1
    if COMPACT and round_index == 4:
        target = (round_index + 1) % WORLD
        hashes[target, 0, 0, 0] = 300000000
        expected[target, 0, 0, 0] = 0
        errors[target, 0, 0] = 1
    return {
        "context": torch.zeros(WORLD, CAPACITY, 4, dtype=torch.int32),
        "token_map": torch.arange(M.vocab_size, dtype=torch.int32).repeat(WORLD, 1),
        "multipliers": torch.ones(WORLD, LAYERS, 4, dtype=torch.int64),
        "primes": torch.full((WORLD, LAYERS, HEADS), 17, dtype=torch.int32),
        "offsets": torch.zeros(WORLD, LAYERS, HEADS, dtype=torch.int32),
        "hash_ids": hashes, "token_mask": mask, "table": table, "scale": scale,
        "row_ids": maps if COMPACT else torch.zeros(WORLD, LAYERS, 1, dtype=torch.int32),
        "rows": expected.flatten(-2), "missing": errors, "counts": counts,
    }


def validate_lookup(devices, compile_only=False):
    """Compare every returned element across serialized calls on retained communication buffers."""
    from golden import TensorSpec, normalize_config, run
    from pypto.ir import DistributedConfig
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    from utils import upload_forward_inputs

    values = lookup_fixture(0)
    specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
    config = dict(backend_type="a5", execution_mode=False, distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    result = run(fn=l3_engram_lookup, specs=specs, config=config, compile_only=True)
    if not result.passed:
        raise RuntimeError(result.error)
    write_engram_contract(result.work_dir)
    if compile_only:
        return
    cfg = normalize_config(config)
    compiled = DistributedCompiledProgram.from_dir(result.work_dir, backend_type=cfg.backend_type,
        execution_mode=cfg.execution_mode, distributed_config=cfg.distributed_config)
    counts = values["counts"].share_memory_()
    buffer = torch.empty(1024 * 1024, dtype=torch.uint8).share_memory_()
    outputs = {name: torch.empty_like(values[name]).share_memory_() for name in ("rows", "missing")}
    with compiled.prepare(cfg, persistent=True, reset_persistent_windows=False,
            inherited_host_tensors=[counts, buffer, *outputs.values()]) as runtime:
        parameters = compiled._get_metadata()[0]
        arguments = upload_forward_inputs(runtime, specs, parameters, WORLD, {"counts": counts}, buffer)
        handles = dict(zip(values, arguments, strict=True))
        for iteration in range(6):
            current = lookup_fixture(iteration)
            counts.copy_(current["counts"])
            for name in ("hash_ids", "token_mask"):
                for rank in range(WORLD):
                    value = current[name][rank].contiguous()
                    buffer[:value.numel() * value.element_size()].copy_(value.view(torch.uint8).reshape(-1))
                    runtime.copy_to(handles[name].shards[rank].data_ptr, buffer.data_ptr(), value.numel() * value.element_size(), worker_id=rank)
            runtime(*arguments, config=cfg)
            for name, output in outputs.items():
                runtime.copy_stacked_from(handles[name], output)
                if not torch.equal(output, current[name]):
                    delta = (output != current[name]).nonzero()
                    raise AssertionError(f"Engram lookup {name} differs at round {iteration}: {delta[:8].tolist()}")
            print(f"ENGRAM_LOOKUP_EXACT_PASS round={iteration} counts={counts.tolist()}", flush=True)
    print("ENGRAM_LOOKUP_VALIDATION_PASS: native FP8/E8M0, independent ranks, empty owners, masks, reuse and miss status", flush=True)


def engram_weight_layouts():
    """Describe the native projection weights and folded FP32 gate storage."""
    return {
        "engram_wkv": ((ENGRAM_LAYERS, ENGRAM_K, KV_OUT), torch.float8_e4m3fn),
        "engram_wkv_scale": ((ENGRAM_LAYERS, ENGRAM_K // 32, KV_OUT), torch.float8_e8m0fnu),
        "engram_gate_weight": ((ENGRAM_LAYERS, HC_MULT, D), torch.float32),
    }


def load_engram_gate_weight(checkpoint, prefix):
    """Load checkpoint Q/K gate weights and fold their FP32 product."""
    q_weight = checkpoint.tensor(prefix + ".q_weight")
    k_weight = checkpoint.tensor(prefix + ".k_weight")
    if tuple(q_weight.shape) != (HC_MULT, D) or tuple(k_weight.shape) != (HC_MULT, D):
        raise ValueError(f"invalid Engram gate weight shape: {prefix}")
    if any(value.dtype not in (torch.bfloat16, torch.float32) for value in (q_weight, k_weight)):
        raise ValueError(f"unsupported Engram gate weight dtype: {prefix}")
    return q_weight.float() * k_weight.float()


class EngramLookup:
    """Pack caller-owned token histories, build hash constants and provide CPU lookup references."""

    def __init__(self, checkpoint, tokenizer):
        from tokenizers import Regex, normalizers

        self.checkpoint = checkpoint
        sentinel = "\ue000"
        normalizer = normalizers.Sequence([
            normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ])
        keys = {}
        token_map = []
        for token in range(tokenizer.get_vocab_size(with_added_tokens=True)):
            text = tokenizer.decode([token], skip_special_tokens=False)
            key = tokenizer.id_to_token(token) if "\ufffd" in text else normalizer.normalize_str(text) or text
            if key not in keys:
                keys[key] = len(keys)
            token_map.append(keys[key])
        if len(keys) != M.engram_compressed_vocab_size:
            raise ValueError("tokenizer compressed vocabulary does not match the Engram checkpoint")
        self.token_map = torch.tensor(token_map, dtype=torch.int64)
        self.pad_id = token_map[M.engram_pad_token_id]
        self.depth = M.engram_max_ngram_size - 1
        bound = max(1, (np.iinfo(np.int64).max // len(keys)) // 2)
        self.multipliers = []
        self.primes = []
        self.offsets = []
        seen = set()
        for layer, table_rows in zip(M.engram_layer_ids, M.engram_num_embeddings, strict=True):
            rng = np.random.default_rng(10007 * layer)
            multipliers = rng.integers(0, bound, size=self.depth + 1, dtype=np.int64) * 2 + 1
            self.multipliers.append(torch.from_numpy(multipliers))
            primes = []
            for _ in range(self.depth):
                candidate = M.engram_vocab_size - 1
                for _ in range(M.engram_n_heads):
                    candidate += 1
                    while candidate in seen or not self._is_prime(candidate):
                        candidate += 1
                    seen.add(candidate)
                    primes.append(candidate)
            if sum(primes) > table_rows:
                raise ValueError("Engram hash buckets exceed their checkpoint table")
            self.primes.append(torch.tensor(primes, dtype=torch.int64))
            self.offsets.append(torch.tensor([0, *np.cumsum(primes[:-1]).tolist()], dtype=torch.int64))

    @staticmethod
    def _is_prime(value):
        if value < 2 or value % 2 == 0:
            return value == 2
        return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))

    def prepare_context(
        self, token_ids, position_ids, query_start_loc, lookback_token_ids,
        token_mask=None, lookback_mask=None,
    ):
        """Pack raw current and historical IDs; -1 pads contexts across excluded tokens."""
        values = (token_ids, position_ids, query_start_loc, lookback_token_ids)
        if any(value.dtype not in (torch.int32, torch.int64) or value.device.type != "cpu" for value in values):
            raise ValueError("Engram token metadata must use CPU integer tensors")
        if token_ids.ndim != 1 or position_ids.shape != token_ids.shape or query_start_loc.ndim != 1:
            raise ValueError("Engram requires token/position vectors and packed request boundaries")
        requests = query_start_loc.numel() - 1
        if requests < 0 or lookback_token_ids.shape != (requests, self.depth):
            raise ValueError("Engram requires one lookback row per packed request")
        if query_start_loc.numel() == 0 or query_start_loc[0] != 0 or query_start_loc[-1] != token_ids.numel():
            raise ValueError("Engram request boundaries must cover all current tokens")
        if bool((query_start_loc[1:] < query_start_loc[:-1]).any()) or bool((position_ids < 0).any()):
            raise ValueError("Engram requires monotone boundaries and nonnegative positions")
        if bool(((token_ids < 0) | (token_ids >= len(self.token_map))).any()):
            raise ValueError("Engram token is outside the tokenizer vocabulary")
        if bool(((lookback_token_ids < -1) | (lookback_token_ids >= len(self.token_map))).any()):
            raise ValueError("Engram lookback token is outside the tokenizer vocabulary")
        if token_mask is None:
            token_mask = torch.ones_like(token_ids, dtype=torch.bool)
        if lookback_mask is None:
            lookback_mask = torch.ones_like(lookback_token_ids, dtype=torch.bool)
        for mask, shape in ((token_mask, token_ids.shape), (lookback_mask, lookback_token_ids.shape)):
            if mask.device.type != "cpu" or mask.shape != shape or mask.dtype not in (torch.bool, torch.int32):
                raise ValueError("Engram masks must match their token tensors and use CPU bool/int32 values")
            if bool(((mask != 0) & (mask != 1)).any()):
                raise ValueError("Engram masks must contain zero/one values")
        context = torch.full((token_ids.numel(), self.depth + 1), -1, dtype=torch.int32)
        for request in range(requests):
            start, end = query_start_loc[request:request + 2].tolist()
            if start == end:
                continue
            positions = position_ids[start:end].long()
            chunk_start = int(positions[0])
            if not torch.equal(positions, torch.arange(chunk_start, chunk_start + end - start)):
                raise ValueError("Engram requires consecutive positions within each request chunk")
            blocked = torch.zeros(end - start, dtype=torch.bool)
            for shift in range(self.depth + 1):
                position = positions - shift
                in_chunk = position >= chunk_start
                source = torch.full((end - start,), -1, dtype=torch.int64)
                dead = torch.zeros(end - start, dtype=torch.bool)
                batch_rows = torch.arange(start, end)[in_chunk] - shift
                source[in_chunk] = token_ids[batch_rows]
                dead[in_chunk] = ~token_mask[batch_rows].bool()
                history = (position >= 0) & ~in_chunk
                columns = (chunk_start - 1 - position[history]).long()
                previous = lookback_token_ids[request, columns].long()
                if bool((previous < 0).any()):
                    raise ValueError("Engram requires actual lookback tokens at nonnegative positions")
                source[history] = previous
                dead[history] = ~lookback_mask[request, columns].bool()
                blocked |= (position < 0) | dead
                source[blocked] = -1
                context[start:end, shift] = source.to(torch.int32)
        return context

    def hash_ids(
        self, token_ids, position_ids, query_start_loc, lookback_token_ids,
        token_mask=None, lookback_mask=None,
    ):
        """Compute the host reference for the device token-context hash."""
        context = self.prepare_context(token_ids, position_ids, query_start_loc, lookback_token_ids, token_mask, lookback_mask)
        compressed = self.token_map[context.clamp_min(0).long()]
        compressed[context < 0] = self.pad_id
        hashes = torch.empty(token_ids.numel(), len(self.primes), self.depth * M.engram_n_heads, dtype=torch.int32)
        for layer, multipliers in enumerate(self.multipliers):
            rolling = torch.zeros(token_ids.numel(), dtype=torch.int64)
            for shift in range(self.depth + 1):
                rolling ^= compressed[:, shift] * multipliers[shift]
                if shift:
                    first, last = (shift - 1) * M.engram_n_heads, shift * M.engram_n_heads
                    result = rolling[:, None] % self.primes[layer][first:last] + self.offsets[layer][first:last]
                    hashes[:, layer, first:last] = result.to(torch.int32)
        return hashes

    def prepare_rows(
        self, token_ids, position_ids, query_start_loc, lookback_token_ids,
        token_mask=None, lookback_mask=None,
    ):
        """Gather only referenced FP8 table rows and dequantize group-32 scales to BF16."""
        hashes = self.hash_ids(
            token_ids, position_ids, query_start_loc, lookback_token_ids, token_mask, lookback_mask,
        )
        width = self.depth * M.engram_n_heads * M.engram_head_dim
        rows = torch.empty(len(self.primes), token_ids.numel(), width, dtype=torch.bfloat16)
        if not token_ids.numel():
            return rows
        for slot, layer in enumerate(M.engram_layer_ids):
            prefix = f"layers.{layer}.engram.embed"
            ids = hashes[:, slot].long()
            payload = self.checkpoint.tensor_rows(prefix + ".weight", ids)
            scale = self.checkpoint.tensor_rows(prefix + ".scale", ids)
            shape = (*ids.shape, M.engram_head_dim)
            scale_shape = (*ids.shape, M.engram_head_dim // 32)
            if payload.dtype != torch.float8_e4m3fn or payload.shape != shape:
                raise ValueError(f"unsupported native Engram table: {prefix}.weight")
            if scale.dtype != torch.float8_e8m0fnu or scale.shape != scale_shape:
                raise ValueError(f"unsupported native Engram scales: {prefix}.scale")
            decoded = payload.float().unflatten(-1, (-1, 32)) * scale.float().unsqueeze(-1)
            rows[slot] = decoded.flatten(-2).to(torch.bfloat16).reshape(token_ids.numel(), width)
        return rows


class EngramDeviceTables:
    """Stream native striped tables or build a closed, original-ID compact device table set."""

    def __init__(self, checkpoint, shards, ranks, compact_hashes=None):
        if not 1 <= shards <= ranks:
            raise ValueError("Engram shards must be between one and the DP rank count")
        self.checkpoint = checkpoint
        self.shards = shards
        self.ranks = ranks
        self.compact = compact_hashes is not None
        self.ids = None
        if self.compact:
            tensors = list(compact_hashes)
            if not tensors or any(value.shape[-2:] != (len(M.engram_layer_ids), 24) for value in tensors):
                raise ValueError("compact Engram collection requires [tokens, layers, heads] hashes")
            self.ids = [[torch.empty(0, dtype=torch.int64) for _ in M.engram_layer_ids] for _ in range(ranks)]
            for slot, table_rows in enumerate(M.engram_num_embeddings):
                values = torch.cat([value[..., slot, :].reshape(-1).long() for value in tensors])
                if bool(((values < 0) | (values >= table_rows)).any()):
                    raise ValueError("compact Engram collection contains invalid original row IDs")
                values = values.unique(sorted=True)
                for owner in range(shards):
                    self.ids[owner][slot] = values[values.remainder(shards) == owner]
            self.rows_per_layer = max(1, max(ids.numel() for owner in self.ids for ids in owner))
        else:
            self.rows_per_layer = max((rows + shards - 1) // shards for rows in M.engram_num_embeddings)
        self.map_rows = self.rows_per_layer if self.compact else 1

    def check_hashes(self, hashes):
        """Reject absent compact rows before starting any device lookup or backbone."""
        if hashes.shape[-2:] != (len(M.engram_layer_ids), 24):
            raise ValueError("Engram hashes must end in [layers, 24]")
        for layer, size in enumerate(M.engram_num_embeddings):
            values = hashes[..., layer, :].reshape(-1).long()
            if bool(((values < 0) | (values >= size)).any()):
                raise ValueError("Engram original row ID is outside its table")
            if self.compact:
                for owner in range(self.shards):
                    requested = values[values.remainder(self.shards) == owner]
                    known = self.ids[owner][layer]
                    positions = torch.searchsorted(known, requested)
                    valid = positions < known.numel()
                    if known.numel():
                        valid &= known[positions.clamp_max(known.numel() - 1)] == requested
                    if not bool(valid.all()):
                        absent = requested[~valid].unique().tolist()
                        raise ValueError(f"compact Engram table misses original row IDs at layer {layer}: {absent[:8]}")

    def rank_tensors(self, rank):
        """Materialize bounded compact shards; full tables use the streaming uploader."""
        if not self.compact:
            raise ValueError("full Engram shards must use streaming upload")
        shape = (len(M.engram_layer_ids), self.rows_per_layer)
        table = torch.zeros((*shape, M.engram_head_dim), dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((*shape, M.engram_head_dim // 32), dtype=torch.float32).to(torch.float8_e8m0fnu)
        mapping = torch.full(shape, torch.iinfo(torch.int32).max, dtype=torch.int32)
        for slot, layer in enumerate(M.engram_layer_ids):
            ids = self.ids[rank][slot]
            if ids.numel():
                prefix = f"layers.{layer}.engram.embed"
                table[slot, :ids.numel()].view(torch.uint8).copy_(self.checkpoint.tensor_rows(prefix + ".weight", ids).view(torch.uint8))
                scale[slot, :ids.numel()].view(torch.uint8).copy_(self.checkpoint.tensor_rows(prefix + ".scale", ids).view(torch.uint8))
                mapping[slot, :ids.numel()] = ids.to(torch.int32)
        return table, scale, mapping

    def allocate(self, runtime, rank):
        """Allocate native FP8 payload, native E8M0 scales and original-ID map on one rank."""
        shape = (len(M.engram_layer_ids), self.rows_per_layer)
        return {
            "table": runtime.alloc_tensor((*shape, M.engram_head_dim), torch.float8_e4m3fn, worker_id=rank),
            "scale": runtime.alloc_tensor((*shape, M.engram_head_dim // 32), torch.float8_e8m0fnu, worker_id=rank),
            "row_ids": runtime.alloc_tensor((len(M.engram_layer_ids), self.map_rows), torch.int32, worker_id=rank),
        }

    def upload(self, runtime, rank, handles, chunk_rows=16384):
        """Upload checkpoint bytes without retaining a host offload table."""
        if self.compact:
            for name, value in zip(("table", "scale", "row_ids"), self.rank_tensors(rank), strict=True):
                runtime.copy_to(handles[name].data_ptr, value.data_ptr(), value.numel() * value.element_size(), worker_id=rank)
            return
        for slot, layer in enumerate(M.engram_layer_ids):
            actual = max(0, (M.engram_num_embeddings[slot] - rank + self.shards - 1) // self.shards)
            if rank >= self.shards:
                actual = 0
            for start in range(0, self.rows_per_layer, chunk_rows):
                count = min(chunk_rows, self.rows_per_layer - start)
                valid = min(count, max(0, actual - start))
                for name, suffix, width, dtype in (
                    ("table", "weight", M.engram_head_dim, torch.float8_e4m3fn),
                    ("scale", "scale", M.engram_head_dim // 32, torch.float8_e8m0fnu),
                ):
                    value = torch.zeros((count, width), dtype=torch.uint8).view(dtype)
                    if valid:
                        ids = rank + torch.arange(start, start + valid, dtype=torch.int64) * self.shards
                        fetched = self.checkpoint.tensor_rows(f"layers.{layer}.engram.embed.{suffix}", ids)
                        value[:valid].view(torch.uint8).copy_(fetched.view(torch.uint8))
                    offset = (slot * self.rows_per_layer + start) * width
                    runtime.copy_to(handles[name].data_ptr, value.data_ptr(), value.numel(), dst_offset=offset, worker_id=rank)
        mapping = torch.zeros((len(M.engram_layer_ids), 1), dtype=torch.int32)
        runtime.copy_to(handles["row_ids"].data_ptr, mapping.data_ptr(), mapping.numel() * 4, worker_id=rank)


if __name__ == "__main__" and "--validate-lookup" in os.sys.argv:
    parser = argparse.ArgumentParser(description=__doc__, parents=[_parser])
    parser.add_argument("--validate-lookup", action="store_true")
    parser.add_argument("--ep", type=int, default=WORLD, choices=[2, 4, 8])
    parser.add_argument("-d", "--device", default=os.environ.get("TASK_DEVICE", ",".join(map(str, range(WORLD)))))
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(device) for device in args.device.split(",")]
    if len(devices) != WORLD or len(set(devices)) != WORLD:
        parser.error("one distinct allocated device is required per DP rank")
    validate_lookup(devices, args.compile_only)

if __name__ == "__main__" and "--validate-lookup" not in os.sys.argv:
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--num-tokens", type=int, default=13)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.tokens < 1 or not 0 <= args.num_tokens <= args.tokens:
        parser.error("require 0 <= num-tokens <= tokens and positive tokens")
    result = run(
        fn=l2_engram,
        specs=build_engram_tensor_specs(args.tokens, args.num_tokens),
        golden_fn=golden_engram_case,
        config={
            "backend_type": args.backend_type,
            "execution_mode": args.execution_mode,
            "device_id": args.device,
        },
        compare_fn={"out": compare_engram},
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
