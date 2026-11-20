# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""DSpark context KV publication and sliding-window attention with HC pre/post processing."""

from __future__ import annotations

import pypto.language as pl
import torch

from quant import packed_cache_compare, pack_quantized_cache, unpack_quantized_cache

from config import (
    DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH, FLASH, MOE_TOKENS, ORI_BLOCKS_DYN,
    WINDOW_CACHE_BYTES,
)
from hc import hc_head_norm, hc_project, hc_sinkhorn
from o_proj import o_proj_hc_post
from qkv_proj_rope import qkv_quant, q_proj_rope, kv_proj, window_publish
from qkv_proj_rope import golden_rope_interleave
from quant import quantize_mxfp8_cache
from rmsnorm import golden_rms_norm


# Dynamic shape variables.
CONTEXT_T_DYN = pl.dynamic("DSPARK_CONTEXT_T_DYN")
CONTEXT_PAYLOAD_T_DYN = pl.dynamic("DSPARK_CONTEXT_PAYLOAD_T_DYN")
CONTEXT_SCALE_T_DYN = pl.dynamic("DSPARK_CONTEXT_SCALE_T_DYN")

# model config
HC_MULT = FLASH.hc_mult
HC_DIM = FLASH.hc_dim
MIX_HC = FLASH.mix_hc
D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
LOCAL_H = FLASH.num_attention_heads
LOCAL_O_GROUPS = FLASH.o_groups
LOCAL_O_WIDTH = LOCAL_O_GROUPS * FLASH.o_lora_rank
O_GROUP_IN = FLASH.num_attention_heads * HEAD_DIM // FLASH.o_groups
O_LORA = FLASH.o_lora_rank
Q_LORA = FLASH.q_lora_rank
ROPE_DIM = FLASH.qk_rope_head_dim

SOFTMAX_SCALE = HEAD_DIM**-0.5

# tiling
QUERY_HEAD_TILE = 64


@pl.jit.incore(level=pl.Level.AIC)
def dspark_cube(
    q: pl.Tensor,
    cache: pl.Tensor,
    ids: pl.Tensor,
    sink: pl.Tensor,
    out: pl.Out[pl.Tensor],
    n: pl.Scalar[pl.INT32],
    heads: pl.Scalar[pl.INT32],
):
    pl.func_attr({"split_aiv": True})
    kv_buf = pl.system.reserve_buffer(name="kv", size=196608)
    p_buf = pl.system.reserve_buffer(name="prob", size=24576)
    s_peer = pl.system.import_peer_buffer(name="scores", peer_func="dspark_vector")
    v_peer = pl.system.import_peer_buffer(name="values", peer_func="dspark_vector")
    pl.system.aic_initialize_pipe(s_peer, dir_mask=1, slot_size=8192, slot_num=1, id=0)
    pl.system.aic_initialize_pipe(v2c_consumer_buf=kv_buf, dir_mask=2, slot_size=65536, slot_num=3, id=1)
    pl.system.aic_initialize_pipe(v2c_consumer_buf=p_buf, dir_mask=2, slot_size=24576, slot_num=1, id=2)
    pl.system.aic_initialize_pipe(v_peer, dir_mask=1, slot_size=65536, slot_num=1, id=3)
    # Each mixed worker owns a cyclic sequence of query tokens.
    worker = pl.tile.get_block_idx()
    stride = pl.max(1, pl.min(n, 32))
    for token in pl.range(worker, n, stride):
        query = pl.tile.load(q, [token * QUERY_HEAD_TILE, 0], [QUERY_HEAD_TILE, HEAD_DIM], target_memory=pl.Mem.Mat)
        # Keep all three K64 panels in L1 through their corresponding PV products.
        kv_0 = pl.system.tpop_from_aiv(shape=[64, 512], dtype=pl.BF16, split=1, id=1)
        kt_0 = pl.tile.transpose_view(kv_0)
        scores_0 = pl.tile.matmul(query, kt_0)
        pl.system.tpush_to_aiv(scores_0, split=1, id=0)
        kv_1 = pl.system.tpop_from_aiv(shape=[64, 512], dtype=pl.BF16, split=1, id=1)
        kt_1 = pl.tile.transpose_view(kv_1)
        scores_1 = pl.tile.matmul(query, kt_1)
        pl.system.tpush_to_aiv(scores_1, split=1, id=0)
        kv_2 = pl.system.tpop_from_aiv(shape=[64, 512], dtype=pl.BF16, split=1, id=1)
        kt_2 = pl.tile.transpose_view(kv_2)
        scores_2 = pl.tile.matmul(query, kt_2)
        pl.system.tpush_to_aiv(scores_2, split=1, id=0)
        all_probs = pl.system.tpop_from_aiv(shape=[64, 192], dtype=pl.BF16, split=1, id=2)
        prob_0 = pl.tile.slice(all_probs, [64, 64], [0, 0])
        pv_0 = pl.tile.matmul(prob_0, kv_0)
        # Return KV credit before sending PV so the vector lanes can publish the next token.
        pl.system.tfree_to_aiv(kv_0, id=1)
        pl.system.tpush_to_aiv(pv_0, split=1, id=3)
        prob_1 = pl.tile.slice(all_probs, [64, 64], [0, 64])
        pv_1 = pl.tile.matmul(prob_1, kv_1)
        pl.system.tfree_to_aiv(kv_1, id=1)
        pl.system.tpush_to_aiv(pv_1, split=1, id=3)
        prob_2 = pl.tile.slice(all_probs, [64, 64], [0, 128])
        pv_2 = pl.tile.matmul(prob_2, kv_2)
        pl.system.tfree_to_aiv(kv_2, id=1)
        pl.system.tpush_to_aiv(pv_2, split=1, id=3)
        pl.system.tfree_to_aiv(all_probs, id=2)
    return out


@pl.jit.inline(auto_scope=False)
def _dspark_slot_panels(ids: pl.Tensor, token: pl.Scalar[pl.INDEX], width: pl.constexpr):
    """Classify 32 cache slots at a time by maximum index and contiguous run."""
    token_ids = pl.tile.load(ids, [token, 0], [1, width], target_memory=pl.Mem.Vec)
    all_slots = pl.reshape(token_ids, [width])
    summary = pl.tile.create([1, 16], dtype=pl.INT32, target_memory=pl.Mem.Vec)
    summary_flat = pl.reshape(summary, [16])
    with pl.reg_scope():
        for panel in pl.hardware_loop(width // 32):
            panel_mask = pl.reg.full_mask(lanes=32)
            physical = pl.reg.load(all_slots, panel * 32, lanes=32)
            highest = pl.reg.reduce_max(physical, panel_mask)
            written_highest = pl.reg.store(summary_flat, panel * 2, highest)
            first_slot = pl.reg.load(all_slots, panel * 32, lanes=1)
            origin = pl.reg.broadcast(first_slot, lanes=32, dtype=pl.INT32)
            offsets = pl.reg.index_gen(0, lanes=32, dtype=pl.INT32)
            sequence = pl.reg.add(origin, offsets, panel_mask)
            unequal = pl.reg.bitwise_xor(physical, sequence, panel_mask)
            unsigned = pl.reg.interpret_cast(unequal, dtype=pl.UINT32)
            difference = pl.reg.reduce_max(unsigned, panel_mask)
            signed_diff = pl.reg.interpret_cast(difference, dtype=pl.INT32)
            written_diff = pl.reg.store(summary_flat, panel * 2 + 1, signed_diff)
    return token_ids, summary


@pl.jit.inline(auto_scope=False)
def _dspark_publish_kv(
    cache: pl.Tensor,
    token_ids: pl.Tile[[1, width], pl.INT32, pl.Mem.Vec],  # noqa: F821
    summary: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec],
    table_flat: pl.Tile[[128], pl.UINT16, pl.Mem.Vec],
    panel: pl.Scalar[pl.INDEX],
    width: pl.constexpr,
):
    """Gather 32 packed rows and expand E4M3FN magnitudes with their BF16 scales."""
    largest = pl.tile.read(summary, [0, panel * 2])
    if largest < 0:
        empty = pl.tile.full([32, 512], dtype=pl.BF16, value=0.0)
        pl.system.tpush_to_aic(empty, split=1, id=1)
    else:
        start = pl.tile.read(token_ids, [0, panel * 32])
        difference32 = pl.tile.read(summary, [0, panel * 2 + 1])
        raw = pl.tile.full([32, 544], dtype=pl.UINT8, value=0)
        if start >= 0 and start <= 2147483616 and difference32 == 0:
            raw = pl.tile.gather_row(
                raw, cache, [0, 0], [pl.cast(start, pl.INDEX), 0], [32, 544]
            )
        else:
            for row in pl.range(32):
                slot = pl.tile.read(token_ids, [0, panel * 32 + row])
                if slot >= 0:
                    raw = pl.tile.gather_row(
                        raw, cache, [row, 0], [pl.cast(slot, pl.INDEX), 0], [1, 544]
                    )
        bytes_flat = pl.reinterpret_view(raw, pl.INT8, shape=[17408])
        factors = pl.reinterpret_view(raw, pl.UINT16, shape=[8704])
        kv = pl.tile.create([32, 512], dtype=pl.BF16, target_memory=pl.Mem.Vec)
        kv_flat = pl.reshape(kv, [16384])
        with pl.reg_scope():
            scale_mask = pl.reg.full_mask(lanes=128)
            positive_codes = pl.reg.load(table_flat, 0, lanes=128)
            low_bits = pl.reg.broadcast(127, lanes=128, dtype=pl.UINT16)
            sign_bit = pl.reg.broadcast(32768, lanes=128, dtype=pl.UINT16)
            for cache_row in pl.hardware_loop(32):
                for channel_chunk in pl.hardware_loop(4):
                    payload_offset = cache_row * 544 + channel_chunk * 128
                    codes = pl.reg.load(bytes_flat, payload_offset, lanes=128)
                    signed_codes = pl.reg.convert(codes, dtype=pl.INT16)
                    table_indices = pl.reg.interpret_cast(signed_codes, dtype=pl.UINT16)
                    magnitude = pl.reg.bitwise_and(table_indices, low_bits, scale_mask)
                    sign16 = pl.reg.bitwise_and(table_indices, sign_bit, scale_mask)
                    magnitude16 = pl.reg.gather_lanes(positive_codes, magnitude)
                    result_bits = pl.reg.bitwise_or(magnitude16, sign16, scale_mask)
                    payload_values = pl.reg.interpret_cast(result_bits, dtype=pl.BF16)
                    scale_bits = pl.reg.load(
                        factors, cache_row * 272 + 256 + channel_chunk * 4, lanes=4
                    )
                    expanded_scale_bits = pl.reg.group_broadcast(scale_bits, lanes=128)
                    mx_scale = pl.reg.interpret_cast(expanded_scale_bits, dtype=pl.BF16)
                    decoded_values = pl.reg.mul(payload_values, mx_scale, scale_mask)
                    stored_kv = pl.reg.store(
                        kv_flat, cache_row * 512 + channel_chunk * 128, decoded_values
                    )
        pl.system.tpush_to_aic(kv, split=1, id=1)


@pl.jit.incore(level=pl.Level.AIV)
def dspark_vector(
    q: pl.Tensor,
    cache: pl.Tensor,
    ids: pl.Tensor,
    sink: pl.Tensor,
    out: pl.Out[pl.Tensor],
    n: pl.Scalar[pl.INT32],
    heads: pl.Scalar[pl.INT32],
):
    pl.func_attr({"split_aiv": True, "dual_aiv_dispatch": True})
    s_buf = pl.system.reserve_buffer(name="scores", size=8192)
    v_buf = pl.system.reserve_buffer(name="values", size=65536)
    kv_peer = pl.system.import_peer_buffer(name="kv", peer_func="dspark_cube")
    p_peer = pl.system.import_peer_buffer(name="prob", peer_func="dspark_cube")
    pl.system.aiv_initialize_pipe(s_buf, dir_mask=1, slot_size=8192, slot_num=1, id=0)
    pl.system.aiv_initialize_pipe(v2c_consumer_buf=kv_peer, dir_mask=2, slot_size=65536, slot_num=3, id=1)
    pl.system.aiv_initialize_pipe(v2c_consumer_buf=p_peer, dir_mask=2, slot_size=24576, slot_num=1, id=2)
    pl.system.aiv_initialize_pipe(v_buf, dir_mask=1, slot_size=65536, slot_num=1, id=3)
    worker = pl.tile.get_block_idx()
    lane = pl.tile.get_subblock_idx()
    stride = pl.max(1, pl.min(n, 32))
    # Convert the 128 nonnegative E4M3FN encodings once per worker.
    pairs = pl.tile.create([1, 64], dtype=pl.UINT16, target_memory=pl.Mem.Vec)
    code_pairs = pl.reshape(pairs, [64])
    with pl.reg_scope():
        lut_init_mask = pl.reg.full_mask(lanes=64)
        pair_number = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT16)
        even = pl.reg.shift_lefts(pair_number, 1, lut_init_mask)
        odd = pl.reg.adds(even, 1, lut_init_mask)
        high = pl.reg.shift_lefts(odd, 8, lut_init_mask)
        packed_pair = pl.reg.bitwise_or(even, high, lut_init_mask)
        written_codes = pl.reg.store(code_pairs, 0, packed_pair)
    table_bytes = pl.reinterpret_view(pairs, pl.INT8, shape=[1, 128])
    table_fp8 = pl.reinterpret_view(table_bytes, pl.FP8E4M3FN, shape=[1, 128])
    table_bf16 = pl.cast(table_fp8, pl.BF16, mode="rint")
    table_flat = pl.reinterpret_view(table_bf16, pl.UINT16, shape=[128])
    for token in pl.range(worker, n, stride):
        token_ids, summary = _dspark_slot_panels(ids, token, 192)
        all_slots = pl.reshape(token_ids, [192])
        for stage in pl.range(3):
            if token == worker or stage > 0:
                _dspark_publish_kv(cache, token_ids, summary, table_flat, stage * 2 + lane, 192)
        score_tile = pl.tile.create([96, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        for panel in pl.range(3):
            score_in = pl.system.tpop_from_aic(shape=[32, 64], dtype=pl.FP32, split=1, id=0)
            score_tile[panel * 32 : panel * 32 + 32, :] = score_in
            pl.system.tfree_to_aic(score_in, id=0)
        scores_flat = pl.reshape(score_tile, [6144])
        probabilities = pl.tile.create([32, 192], dtype=pl.BF16, target_memory=pl.Mem.Vec)
        probs_flat = pl.reshape(probabilities, [6144])
        # Rows hold the final normalization and the two online-softmax numerator corrections.
        state = pl.tile.create([3, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
        state_flat = pl.reshape(state, [96])
        if LOCAL_H < QUERY_HEAD_TILE:
            sinks = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            sink_count = pl.max(0, pl.min(32, LOCAL_H - lane * 32))
            for sink_head in pl.range(sink_count):
                head_sink = pl.tensor.read(sink, [0, lane * 32 + sink_head])
                pl.tile.write(sinks, [0, sink_head], head_sink)
        else:
            sinks = pl.tile.load(sink, [0, lane * 32], [1, 32], target_memory=pl.Mem.Vec)
        sink_flat = pl.reshape(sinks, [32])
        with pl.reg_scope():
            cols = pl.reg.full_mask(lanes=64)
            one = pl.reg.full_mask(lanes=1)
            floor = pl.reg.broadcast(-1e30, lanes=64, dtype=pl.FP32)
            indices_0 = pl.reg.load(all_slots, 0, lanes=64)
            valid_0 = pl.reg.compares(indices_0, 0, cols, predicate="ge")
            indices_1 = pl.reg.load(all_slots, 64, lanes=64)
            valid_1 = pl.reg.compares(indices_1, 0, cols, predicate="ge")
            indices_2 = pl.reg.load(all_slots, 128, lanes=64)
            valid_2 = pl.reg.compares(indices_2, 0, cols, predicate="ge")
            for head in pl.hardware_loop(32):
                dots_0 = pl.reg.load(scores_flat, head * 64, lanes=64)
                logits_0 = pl.reg.muls(dots_0, SOFTMAX_SCALE, cols)
                masked_0 = pl.reg.select(valid_0, logits_0, floor)
                peak_0 = pl.reg.reduce_max(masked_0, cols)
                dots_1 = pl.reg.load(scores_flat, 2048 + head * 64, lanes=64)
                logits_1 = pl.reg.muls(dots_1, SOFTMAX_SCALE, cols)
                masked_1 = pl.reg.select(valid_1, logits_1, floor)
                peak_1 = pl.reg.reduce_max(masked_1, cols)
                dots_2 = pl.reg.load(scores_flat, 4096 + head * 64, lanes=64)
                logits_2 = pl.reg.muls(dots_2, SOFTMAX_SCALE, cols)
                masked_2 = pl.reg.select(valid_2, logits_2, floor)
                peak_2 = pl.reg.reduce_max(masked_2, cols)
                peak_floor = pl.reg.broadcast(-1e30, lanes=1, dtype=pl.FP32)
                running_0 = pl.reg.maximum(peak_floor, peak_0, one)
                max_0 = pl.reg.broadcast(running_0, lanes=64, dtype=pl.FP32)
                weight_0 = pl.reg.expdif(logits_0, max_0, valid_0)
                sum_0 = pl.reg.reduce_sum(weight_0, cols, reassoc=True)
                prob_0 = pl.reg.convert(weight_0, dtype=pl.BF16, rounding="R", saturate=False)
                write_p_0 = pl.reg.store(probs_flat, head * 192, prob_0)
                running_1 = pl.reg.maximum(running_0, peak_1, one)
                max_1 = pl.reg.broadcast(running_1, lanes=64, dtype=pl.FP32)
                weight_1 = pl.reg.expdif(logits_1, max_1, valid_1)
                sum_1 = pl.reg.reduce_sum(weight_1, cols, reassoc=True)
                prob_1 = pl.reg.convert(weight_1, dtype=pl.BF16, rounding="R", saturate=False)
                write_p_1 = pl.reg.store(probs_flat, head * 192 + 64, prob_1)
                corr_1 = pl.reg.expdif(running_0, running_1, one)
                write_c_1 = pl.reg.store(state_flat, 32 + head, corr_1)
                scaled_1 = pl.reg.mul(sum_0, corr_1, one)
                den_1 = pl.reg.add(scaled_1, sum_1, one)
                running_2 = pl.reg.maximum(running_1, peak_2, one)
                max_2 = pl.reg.broadcast(running_2, lanes=64, dtype=pl.FP32)
                weight_2 = pl.reg.expdif(logits_2, max_2, valid_2)
                sum_2 = pl.reg.reduce_sum(weight_2, cols, reassoc=True)
                prob_2 = pl.reg.convert(weight_2, dtype=pl.BF16, rounding="R", saturate=False)
                write_p_2 = pl.reg.store(probs_flat, head * 192 + 128, prob_2)
                corr_2 = pl.reg.expdif(running_1, running_2, one)
                write_c_2 = pl.reg.store(state_flat, 64 + head, corr_2)
                scaled_2 = pl.reg.mul(den_1, corr_2, one)
                den_2 = pl.reg.add(scaled_2, sum_2, one)
                sink_value = pl.reg.load(sink_flat, head, lanes=1)
                final_peak = pl.reg.maximum(running_2, sink_value, one)
                weight_scale = pl.reg.expdif(running_2, final_peak, one)
                sink_weight = pl.reg.expdif(sink_value, final_peak, one)
                scaled_sum = pl.reg.mul(den_2, weight_scale, one)
                final_sum = pl.reg.add(scaled_sum, sink_weight, one)
                safe_sum = pl.reg.maximums(final_sum, 1.0, one)
                ratio = pl.reg.div(weight_scale, safe_sum, one)
                norm_written = pl.reg.store(state_flat, head, ratio)
        pl.system.tpush_to_aic(probabilities, split=1, id=2)
        # Stage the next K64 while the cube computes the current PV products.
        if token + stride < n:
            next_ids, next_summary = _dspark_slot_panels(ids, token + stride, 64)
            _dspark_publish_kv(cache, next_ids, next_summary, table_flat, lane, 64)
        first = pl.system.tpop_from_aic(shape=[32, 512], dtype=pl.FP32, split=1, id=3)
        numerator = pl.tile.move(first, pl.Mem.Vec)
        pl.system.tfree_to_aic(first, id=3)
        numerator_flat = pl.reshape(numerator, [16384])
        with pl.reg_scope():
            mask = pl.reg.full_mask(lanes=64)
            for head in pl.hardware_loop(32):
                correction = pl.reg.load(state_flat, 32 + head, lanes=1)
                factor = pl.reg.broadcast(correction, lanes=64, dtype=pl.FP32)
                for chunk in pl.hardware_loop(8):
                    offset = head * 512 + chunk * 64
                    initial = pl.reg.load(numerator_flat, offset, lanes=64)
                    weighted = pl.reg.mul(initial, factor, mask)
                    stored = pl.reg.store(numerator_flat, offset, weighted)
        second = pl.system.tpop_from_aic(shape=[32, 512], dtype=pl.FP32, split=1, id=3)
        pair = pl.tile.add(numerator, second)
        pl.system.tfree_to_aic(second, id=3)
        pair_flat = pl.reshape(pair, [16384])
        with pl.reg_scope():
            pair_mask = pl.reg.full_mask(lanes=64)
            for pair_head in pl.hardware_loop(32):
                pair_correction = pl.reg.load(state_flat, 64 + pair_head, lanes=1)
                pair_factor = pl.reg.broadcast(pair_correction, lanes=64, dtype=pl.FP32)
                for pair_chunk in pl.hardware_loop(8):
                    pair_offset = pair_head * 512 + pair_chunk * 64
                    pair_initial = pl.reg.load(pair_flat, pair_offset, lanes=64)
                    pair_weighted = pl.reg.mul(pair_initial, pair_factor, pair_mask)
                    pair_stored = pl.reg.store(pair_flat, pair_offset, pair_weighted)
        third = pl.system.tpop_from_aic(shape=[32, 512], dtype=pl.FP32, split=1, id=3)
        total = pl.tile.add(pair, third)
        pl.system.tfree_to_aic(third, id=3)
        total_flat = pl.reshape(total, [16384])
        res = pl.tile.create([32, 512], dtype=pl.BF16, target_memory=pl.Mem.Vec)
        output_flat = pl.reshape(res, [16384])
        with pl.reg_scope():
            final_mask = pl.reg.full_mask(lanes=64)
            for final_head in pl.hardware_loop(32):
                inv = pl.reg.load(state_flat, final_head, lanes=1)
                norm = pl.reg.broadcast(inv, lanes=64, dtype=pl.FP32)
                for final_chunk in pl.hardware_loop(8):
                    final_offset = final_head * 512 + final_chunk * 64
                    accumulated = pl.reg.load(total_flat, final_offset, lanes=64)
                    normalized = pl.reg.mul(accumulated, norm, final_mask)
                    rounded_out = pl.reg.convert(normalized, dtype=pl.BF16, rounding="R", saturate=False)
                    stored_out = pl.reg.store(output_flat, final_offset, rounded_out)
        if heads == 64:
            pl.tile.store(res, [token * LOCAL_H + lane * 32, 0], out)
        elif heads > lane * 32:
            valid_heads = pl.min(pl.cast(heads, pl.INDEX) - lane * 32, 32)
            active_result = pl.tile.set_validshape(res, valid_heads, 512)
            pl.tile.store(active_result, [token * LOCAL_H + lane * 32, 0], out)
    return out


@pl.jit.incore(level=pl.Level.CORE_GROUP)
def dspark_group(
    q: pl.Tensor,
    cache: pl.Tensor,
    ids: pl.Tensor,
    sink: pl.Tensor,
    out: pl.Out[pl.Tensor],
    n: pl.Scalar[pl.INT32],
    heads: pl.Scalar[pl.INT32],
):
    out = dspark_cube(q, cache, ids, sink, out, n, heads)
    out = dspark_vector(q, cache, ids, sink, out, n, heads)
    return out


@pl.jit.inline(auto_scope=False)
def dspark_attention(
    query: pl.Tensor[[MOE_TOKENS, LOCAL_H * HEAD_DIM], pl.BF16],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[MOE_TOKENS, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    if LOCAL_H < QUERY_HEAD_TILE:
        padded_query = pl.create_tensor([MOE_TOKENS, QUERY_HEAD_TILE * HEAD_DIM], dtype=pl.BF16)
        for row in pl.spmd(num_tokens, name_hint="dspark_pad_heads"):
            padded = pl.full([1, QUERY_HEAD_TILE * HEAD_DIM], dtype=pl.BF16, value=0.0)
            active_query = query[row : row + 1, :]
            padded[:, :LOCAL_H * HEAD_DIM] = active_query
            padded_query[row : row + 1, :] = padded
        query_heads = pl.reshape(padded_query, [MOE_TOKENS * QUERY_HEAD_TILE, HEAD_DIM])
    else:
        query_heads = pl.reshape(query, [MOE_TOKENS * QUERY_HEAD_TILE, HEAD_DIM])
    cache_row_count = pl.tensor.dim(cache, 0) * 128
    cache_rows = pl.reshape(cache, [cache_row_count, WINDOW_CACHE_BYTES])
    output_heads = pl.reshape(output, [MOE_TOKENS * LOCAL_H, HEAD_DIM])
    sink_heads = pl.reshape(sink, [1, LOCAL_H])
    workers = pl.max(1, pl.min(num_tokens, 32))
    with pl.spmd(workers, name_hint="dspark_attention"):
        dspark_group(query_heads, cache_rows, indices, sink_heads, output_heads, num_tokens, LOCAL_H)


@pl.jit.inline(auto_scope=False)
def dspark_context_kv(
    kv_input_payload: pl.Tensor,
    kv_input_scales: pl.Tensor,
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    context_slots: pl.Tensor,
    window_cache: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project shared quantized context inputs through one layer's own weights."""
    tokens = pl.tensor.dim(context_slots, 0)
    projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj(
        kv_input_payload, kv_input_scales,
        wkv, wkv_scale,
        projected, num_tokens, worker_limit=32, row_tile=80,
    )
    window_publish(
        projected, kv_norm_weight, rope_cos, rope_sin,
        context_slots, window_cache,
        num_tokens,
    )
    return window_cache


@pl.jit
def dspark_context_kv_test(
    kv_input_payload: pl.Tensor[[CONTEXT_PAYLOAD_T_DYN, D], pl.FP8E4M3FN],
    kv_input_scales: pl.Tensor[[1, CONTEXT_SCALE_T_DYN], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[CONTEXT_T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[CONTEXT_T_DYN, ROPE_DIM // 2], pl.FP32],
    context_slots: pl.Tensor[[CONTEXT_T_DYN], pl.INT64],
    window_cache: pl.InOut[pl.Tensor[[3, ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Publish through a middle-layer view while retaining both guard layers."""
    kv_input_payload.bind_dynamic(0, CONTEXT_PAYLOAD_T_DYN)
    kv_input_scales.bind_dynamic(1, CONTEXT_SCALE_T_DYN)
    rope_cos.bind_dynamic(0, CONTEXT_T_DYN)
    rope_sin.bind_dynamic(0, CONTEXT_T_DYN)
    context_slots.bind_dynamic(0, CONTEXT_T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    dspark_context_kv(
        kv_input_payload, kv_input_scales, wkv, wkv_scale,
        kv_norm_weight, rope_cos, rope_sin, context_slots, window_cache[1], num_tokens,
    )
    return window_cache


@pl.jit.inline(auto_scope=False)
def dspark_attn(
    x_hc: pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[MOE_TOKENS], pl.INT64],
    window_indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    attention_pre_mix: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32]],
    attention_hidden: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run HC pre-mix, query attention, output projection, and HC post."""
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    normalized_attention = pl.create_tensor([MOE_TOKENS, D], dtype=pl.BF16)
    hc_head_norm(x_hc, incoming_pre_mix, attn_norm_weight, normalized_attention, worker_limit=64)
    inv_rms, partials = hc_project(x_hc, hc_attn_fn, rms_worker_limit=64)

    post_mix = pl.create_tensor([MOE_TOKENS, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([MOE_TOKENS, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, attention_pre_mix, post_mix, residual_mix)
    proj_payload, proj_scales, proj_rows = qkv_quant(normalized_attention, num_tokens, row_tile=80)
    qr = pl.create_tensor([MOE_TOKENS, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([MOE_TOKENS, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, q,
        num_tokens, qa_worker_limit=32, qb_worker_limit=32,
        padded_rows=proj_rows,
        qa_row_tile=80, qb_row_tile=80,
    )
    kv_projection = pl.create_tensor([MOE_TOKENS, HEAD_DIM], dtype=pl.BF16)
    kv_proj(
        proj_payload, proj_scales,
        wkv, wkv_scale,
        kv_projection, num_tokens, worker_limit=32, row_tile=80,
    )
    window_publish(
        kv_projection, kv_norm_weight, rope_cos, rope_sin,
        window_slots, window_cache,
        num_tokens,
    )
    attention_output = pl.create_tensor([MOE_TOKENS, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    dspark_attention(q, window_cache, window_indices, attn_sink, attention_output, num_tokens)

    o_proj_hc_post(
        attention_output, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        x_hc, post_mix, residual_mix, attention_hidden, num_tokens,
        oa_row_tile=80, ob_row_tile=128,
    )

    return attention_hidden, attention_pre_mix, window_cache


@pl.jit(auto_scope=False)
def l2_dspark_attn(
    x_hc: pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[MOE_TOKENS, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[MOE_TOKENS], pl.INT64],
    window_indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8]],
    attention_pre_mix: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32]],
    attention_hidden: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Dispatch the production DSpark attention composition."""
    return dspark_attn(
        x_hc, incoming_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin, window_slots, window_indices, window_cache,
        attention_pre_mix, attention_hidden,
        num_tokens,
    )


def golden_context_kv_cache(
    payload: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    cache_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish one draft layer's context KV into its own MXFP8 SWA cache."""
    from quant import decode_e8m0, unpack_fp8_nz, unpack_mx_b_scale

    if weight.dtype == torch.float8_e4m3fn:
        weight = unpack_fp8_nz(weight)
    if slots.shape != payload.shape[:1]:
        raise ValueError("context slots must have one entry per token")
    if cache.shape[:-1] != cache_scale.shape[:-1] or cache.shape[-1] != cache_scale.shape[-1] * 32:
        raise ValueError("context cache payload and scale shapes disagree")
    active = slots >= 0
    rows = slots[active].long()
    cache_rows = cache.numel() // cache.shape[-1]
    if rows.numel() and (int(rows.max()) >= cache_rows or rows.unique().numel() != rows.numel()):
        raise ValueError("context slots must be in range and unique")
    updated_cache = cache.clone()
    updated_scale = cache_scale.clone()
    if rows.numel():
        activation = payload.float().unflatten(-1, (-1, 32))
        weight_groups = weight.float().unflatten(0, (-1, 32))
        partials = torch.einsum("...gk,gkn->...gn", activation, weight_groups)
        scales_a = decode_e8m0(activation_scale)
        scales_b = decode_e8m0(unpack_mx_b_scale(weight_scale))
        partials = partials * scales_a.unsqueeze(-1) * scales_b
        projected = partials.sum(dim=-2).to(torch.bfloat16)
        projected = golden_rms_norm(projected, norm_weight)
        rope_dim = cos.shape[-1] * 2
        rotated = golden_rope_interleave(projected[..., -rope_dim:], cos, sin)
        kv = torch.cat((projected[..., :-rope_dim], rotated), dim=-1)
        payload, scale = quantize_mxfp8_cache(kv[active])
        updated_cache.reshape(-1, cache.shape[-1])[rows] = payload
        updated_scale.reshape(-1, cache_scale.shape[-1])[rows] = scale
    return updated_cache, updated_scale


def golden_dspark_attn(tensors):
    """Compose QKV, shared-history attention, output projection and HC references."""
    import torch
    from hc import golden_hc_head_norm, golden_hc_mixes, golden_hc_post
    from o_proj import golden_o_proj
    from qkv_proj_rope import golden_qkv_proj_rope
    from quant import quantize_mxfp8_cache
    from sparse_attention import golden_gather_sparse_attention
    from quant import _publish_quantized_cache

    active = int(tensors["num_tokens"])
    pre, post, residual = golden_hc_mixes(
        tensors["x_hc"], tensors["hc_attn_fn"], tensors["hc_attn_scale"], tensors["hc_attn_base"],
    )
    tensors["attention_pre_mix"].copy_(pre)
    normalized = golden_hc_head_norm(tensors["x_hc"], tensors["incoming_pre_mix"], tensors["attn_norm_weight"])
    payload, factors = unpack_quantized_cache(tensors["window_cache"], False)
    inputs = dict(tensors, x=normalized, window_cache=payload, window_cache_scale=factors)
    for name in ("x", "rope_cos", "rope_sin", "window_slots", "window_indices"):
        inputs[name] = inputs[name][:active]
    projection = torch.zeros((MOE_TOKENS, D), dtype=torch.bfloat16)
    packed_cache = tensors["window_cache"]
    if active:
        query, window_kv, _ = golden_qkv_proj_rope(
            inputs["x"], inputs["wq_a"], inputs["wq_a_scale"], inputs["q_norm_weight"],
            inputs["wq_b"], inputs["wq_b_scale"], inputs["wkv"], inputs["wkv_scale"], inputs["kv_norm_weight"],
            inputs["rope_cos"], inputs["rope_sin"], sequential_query=True,
        )
        cache, scale = _publish_quantized_cache(
            inputs["window_cache"], inputs["window_cache_scale"],
            window_kv, inputs["window_slots"], quantize_mxfp8_cache,
        )
        packed_cache = pack_quantized_cache(cache, scale)
        indices = inputs["window_indices"]
        attention_output = golden_gather_sparse_attention(
            query, indices, packed_cache, indices, packed_cache, inputs["attn_sink"],
            compressed_keys=0, key_tile=64,
        )
        projected = golden_o_proj(
            attention_output, inputs["rope_cos"], inputs["rope_sin"], inputs["wo_a"], inputs["wo_b"], inputs["wo_b_scale"],
            wo_a_scale=inputs["wo_a_scale"], sequential_projection=True,
        )
        projection[:active].copy_(projected)
    tensors["attention_hidden"].copy_(golden_hc_post(projection, tensors["x_hc"], post, residual))
    tensors["window_cache"].copy_(packed_cache)


def build_attention_specs(seed: int = 43):
    """Build five shared-history queries, HC inputs and runtime specifications."""
    import torch
    from golden import ScalarSpec, TensorSpec
    import inspect
    from hc import make_hc_inputs
    import math
    from quant import pack_fp8_nz, pack_mx_b_scale
    from quant import quantize_mxfp8_upward_reference, quantize_grouped_mxfp8_weight
    from quant import quantize_mxfp8_cache

    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen)

    def weight(k, n):
        payload, scales = quantize_mxfp8_upward_reference(rand(n, k) / math.sqrt(k))
        packed_scales = pack_mx_b_scale(scales.T.contiguous()).view(torch.float8_e8m0fnu)
        return payload.T.contiguous(), packed_scales

    qa, qas = weight(D, Q_LORA)
    qb, qbs = weight(Q_LORA, LOCAL_H * HEAD_DIM)
    kv, kvs = weight(D, HEAD_DIM)
    ob, obs = weight(LOCAL_O_WIDTH, D)
    qa = pack_fp8_nz(qa)
    qb = pack_fp8_nz(qb)
    kv = pack_fp8_nz(kv)
    ob = pack_fp8_nz(ob)
    cache, cache_scale = quantize_mxfp8_cache(rand(DSPARK_QUERY_WIDTH + 1, 128, 1, HEAD_DIM).to(torch.bfloat16))
    hc_inputs = make_hc_inputs(MOE_TOKENS, seed, "random")
    hc_inputs["x_hc"][DSPARK_QUERY_WIDTH:] = 0
    values = dict(
        x_hc=hc_inputs["x_hc"], incoming_pre_mix=hc_inputs["pre_mix"],
        hc_attn_fn=hc_inputs["hc_attn_fn"], hc_attn_scale=hc_inputs["hc_attn_scale"],
        hc_attn_base=hc_inputs["hc_attn_base"], attn_norm_weight=hc_inputs["attn_norm_weight"],
        wq_a=qa, wq_a_scale=qas, q_norm_weight=(rand(Q_LORA) * 0.1 + 1).to(torch.bfloat16),
        wq_b=qb, wq_b_scale=qbs,
        wkv=kv, wkv_scale=kvs, kv_norm_weight=(rand(HEAD_DIM) * 0.1 + 1).to(torch.bfloat16),
        attn_sink=rand(LOCAL_H) * 2,
    )
    wo_a = (rand(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(torch.bfloat16)
    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(wo_a)
    values.update(wo_b=ob, wo_b_scale=obs)
    positions = torch.arange(2, 2 + DSPARK_QUERY_WIDTH)
    frequencies = 10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
    angles = positions.float()[:, None] * frequencies
    values["rope_cos"] = angles.cos()
    values["rope_sin"] = angles.sin()
    values["window_slots"] = 128 + positions
    indices = torch.full((DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH), -1, dtype=torch.int32)
    # Contiguous panels in both lanes, followed by a partial panel.
    visible_keys = FLASH.sliding_window + DSPARK_QUERY_WIDTH
    indices[0, :visible_keys] = torch.arange(128, 128 + visible_keys, dtype=torch.int32)
    # Irregular physical slots across decode, gather, K64 and lane boundaries.
    boundary_columns = torch.tensor([0, 15, 16, 31, 32, 63, 64, 95, 96, 127, 128, 191])
    boundary_slots = torch.tensor([127, 128, 255, 256, 383, 384, 511, 512, 639, 640, 767, 0], dtype=torch.int32)
    indices[1, boundary_columns] = boundary_slots
    indices[2, boundary_columns] = boundary_slots.flip(0)
    indices[2, 95] = indices[2, 96]
    # Only lane one contributes; the last query remains fully masked.
    indices[3, 96:128] = torch.arange(512, 544, dtype=torch.int32)
    indices[3, 191] = 767
    values["window_indices"] = indices
    for name in ("rope_cos", "rope_sin", "window_slots", "window_indices"):
        current = values[name]
        fill = -1 if name in ("window_slots", "window_indices") else 0
        padded = torch.full((MOE_TOKENS, *current.shape[1:]), fill, dtype=current.dtype)
        padded[:DSPARK_QUERY_WIDTH] = current
        values[name] = padded
    values["window_cache"] = pack_quantized_cache(cache, cache_scale)
    specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
    specs.append(TensorSpec("attention_pre_mix", [MOE_TOKENS, HC_MULT], torch.float32))
    specs.append(TensorSpec("attention_hidden", [MOE_TOKENS, HC_MULT, D], torch.float32))
    specs.append(ScalarSpec("num_tokens", torch.int32, DSPARK_QUERY_WIDTH, compile_runtime=True))
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in inspect.signature(l2_dspark_attn._func).parameters]


def _compare_context_cache(
    _actual,
    _expected,
    *,
    actual_outputs,
    expected_outputs,
    inputs,
    **_kwargs,
):
    from quant import dequantize_mxfp8_cache

    slots = inputs["context_slots"][:int(inputs["num_tokens"])]
    layer_rows = actual_outputs["window_cache"].shape[1] * 128
    active = slots[slots >= 0].long().unique() + layer_rows
    actual_payload, actual_scale = unpack_quantized_cache(actual_outputs["window_cache"], False)
    expected_payload, expected_scale = unpack_quantized_cache(expected_outputs["window_cache"], False)
    actual = dequantize_mxfp8_cache(actual_payload, actual_scale).reshape(-1, HEAD_DIM)
    expected = dequantize_mxfp8_cache(expected_payload, expected_scale).reshape(-1, HEAD_DIM)
    error_norm = (actual[active].float() - expected[active].float()).norm()
    expected_norm = expected[active].float().norm().clamp_min(1e-12)
    relative_l2 = error_norm / expected_norm
    inactive = torch.ones(actual.shape[0], dtype=torch.bool)
    inactive[active] = False
    untouched = all(
        torch.equal(
            actual_field.contiguous().view(torch.uint8).reshape(actual.shape[0], -1)[inactive],
            expected_field.contiguous().view(torch.uint8).reshape(actual.shape[0], -1)[inactive],
        )
        for actual_field, expected_field in ((actual_payload, expected_payload), (actual_scale, expected_scale))
    )
    passed = bool(torch.isfinite(relative_l2) and relative_l2 <= 0.05 and untouched)
    detail = f"active context KV relative L2={relative_l2.item():.6g}; inactive rows unchanged={untouched}"
    return passed, detail


def run_context_kv_case(*, device_id: int = 0, compile_only: bool = False, execution_mode: bool = False):
    """Validate context KV publication and untouched guard layers."""
    from golden import ScalarSpec, TensorSpec, run
    from quant import pack_fp8_nz, pack_mx_b_scale, pack_mx_a_scale, unpack_mx_a_scale
    from quant import _quantize_mxfp8_activation, encode_e8m0

    torch.manual_seed(42)
    tokens = 8
    main_x = torch.randn(tokens, D, dtype=torch.bfloat16) * 0.1
    padded_rows = (tokens + 79) // 80 * 80
    activation, activation_scale = _quantize_mxfp8_activation(main_x, scale_alg=0)
    input_payload = torch.zeros(padded_rows, D, dtype=torch.float8_e4m3fn)
    input_payload[:tokens] = activation.flatten(-2)
    scale_codes = torch.zeros(padded_rows, D // 32, dtype=torch.uint8)
    scale_codes[:tokens] = encode_e8m0(activation_scale)
    input_scales = pack_mx_a_scale(scale_codes).reshape(1, -1).view(torch.float8_e8m0fnu)
    weight = (torch.randn(D, HEAD_DIM) * 0.01).to(torch.float8_e4m3fn)
    weight = pack_fp8_nz(weight)
    weight_scale = pack_mx_b_scale(torch.full((D // 32, HEAD_DIM), 127, dtype=torch.uint8))
    weight_scale = weight_scale.view(torch.float8_e8m0fnu)
    cos = torch.ones(tokens, ROPE_DIM // 2)
    sin = torch.zeros_like(cos)
    sin[0].fill_(1.0)
    cos[0].zero_()
    cache = torch.empty(3, 2, 128, 1, HEAD_DIM)
    cache_scale = torch.empty((3, 2, 128, 1, HEAD_DIM // 32), dtype=torch.bfloat16)
    for layer in range(3):
        cache[layer].fill_((layer + 1) * 0.25)
        cache_scale[layer].fill_((layer + 1) * 4.0)
    cache = cache.to(torch.float8_e4m3fn)
    slots = torch.tensor([0, 1, 127, 128, 255, -1, -1, -1], dtype=torch.int64)

    def golden(values):
        cache_payload, cache_factors = unpack_quantized_cache(values["window_cache"][1], False)
        active = int(values["num_tokens"])
        codes = values["kv_input_scales"].view(torch.uint8).reshape(-1, D // 32)
        activation_codes = unpack_mx_a_scale(codes)[:active]
        payload, scale = golden_context_kv_cache(
            values["kv_input_payload"][:active], activation_codes,
            values["wkv"], values["wkv_scale"], values["kv_norm_weight"],
            values["rope_cos"][:active], values["rope_sin"][:active],
            values["context_slots"][:active],
            cache_payload, cache_factors,
        )
        values["window_cache"][1].copy_(pack_quantized_cache(payload, scale))

    packed_cache = pack_quantized_cache(cache, cache_scale)
    specs = [
        TensorSpec("kv_input_payload", list(input_payload.shape), torch.float8_e4m3fn, init_value=input_payload),
        TensorSpec("kv_input_scales", list(input_scales.shape), torch.float8_e8m0fnu, init_value=input_scales),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=weight),
        TensorSpec("wkv_scale", [D // 32, HEAD_DIM], torch.float8_e8m0fnu, init_value=weight_scale),
        TensorSpec("kv_norm_weight", [HEAD_DIM], torch.bfloat16, init_value=1.0),
        TensorSpec("rope_cos", [tokens, ROPE_DIM // 2], torch.float32, init_value=cos),
        TensorSpec("rope_sin", [tokens, ROPE_DIM // 2], torch.float32, init_value=sin),
        TensorSpec("context_slots", [tokens], torch.int64, init_value=slots),
        TensorSpec("window_cache", [3, 2, 128, 1, WINDOW_CACHE_BYTES], torch.uint8, init_value=packed_cache),
        ScalarSpec("num_tokens", torch.int32, tokens, compile_runtime=True),
    ]
    compare = {"window_cache": _compare_context_cache}
    result = run(
        fn=dspark_context_kv_test,
        specs=specs,
        golden_fn=golden,
        compare_fn=compare,
        config={"backend_type": "a5", "device_id": device_id,
                "execution_mode": execution_mode},
        compile_only=compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark non-causal SWA validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--context-kv", action="store_true", help="validate context KV publication only")
    args = parser.parse_args()
    if args.context_kv:
        run_context_kv_case(
            device_id=args.device, compile_only=args.compile_only, execution_mode=args.execution_mode,
        )
        raise SystemExit(0)
    if args.execution_mode:
        parser.error("DSpark native attention requires A5 device execution; simulator support is unavailable")
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run

    result = run(
        fn=l2_dspark_attn,
        specs=build_attention_specs(),
        golden_fn=golden_dspark_attn,
        compare_fn={
            "attention_hidden": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
            "window_cache": packed_cache_compare(
                mapped_pool_ratio_allclose(
                    "window_slots", pool_name="window cache", mapping_shape=(MOE_TOKENS,),
                    block_size=128, leading_rank_axis=False,
                    atol=2 ** -9, rtol=2 ** -3, max_error_ratio=0.01,
                ),
                mapped_pool_ratio_allclose(
                    "window_slots", pool_name="window cache scale", mapping_shape=(MOE_TOKENS,),
                    block_size=128, leading_rank_axis=False,
                    atol=0.0, rtol=0.0, max_error_ratio=0.0,
                ),
                fp4=False,
            ),
        },
        config={"backend_type": args.backend_type, "device_id": args.device,
                "execution_mode": args.execution_mode},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
