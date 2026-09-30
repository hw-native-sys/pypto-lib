# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Resident four-rank CED prefill: one complete L2 invocation on each A5.

The host entry only creates communication windows and launches ranks. Encoder,
global publication, decoder replay, routing and the vocabulary head execute
inside the rank entry. Original FP8/FP4 weights stay packed in device memory;
their exact decoded values feed the selected arithmetic policy.
"""

import pypto.language as pl
import pypto.language.distributed as pld

from models.deepseek_v4_1_flash.config import (
    D,
    FLASH,
    HC_MULT,
    HEAD_DIM,
    INDEX_DIM,
    INDEX_TOPK,
    ROPE_DIM,
    STATE_CAPACITY,
)
from models.deepseek_v4_1_flash.engram import prefill_engram_gate_inline as engram_gate_inline
from models.deepseek_v4_1_flash.prefill_c1a_common import (
    prefill_attention_inline as attention_inline,
    prefill_publish_decoder_inline as publish_decoder_inline,
)
from models.deepseek_v4_1_flash.attention_ops import (
    prefill_norm_inline as norm_inline,
    prefill_packed_linear_inline as packed_linear_inline,
)
from models.deepseek_v4_1_flash.quantization import prefill_round_inline as round_inline
from models.deepseek_v4_1_flash.hc_post import (
    prefill_add_inline as add_inline,
    prefill_hc_post_inline as hc_post_inline,
)
from models.deepseek_v4_1_flash.hc_mixes import (
    prefill_hc_coefficients_inline as hc_coefficients_inline,
    prefill_hc_normalize_logits_inline as hc_normalize_logits_inline,
)
from models.deepseek_v4_1_flash.hc_pre import prefill_hc_pre_inline as hc_pre_inline
from models.deepseek_v4_1_flash.expert_routed import prefill_routed_moe_inline as routed_moe_inline
from models.deepseek_v4_1_flash.expert_shared import prefill_shared_expert_inline as shared_expert_inline
from models.deepseek_v4_1_flash.attention_ops import prefill_linear_inline as linear_inline
from models.deepseek_v4_1_flash.ep_transport import prefill_combine_routed_experts as combine_routed_experts

from concurrent.futures import ThreadPoolExecutor
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.golden import hc_pre, rms_norm
from models.deepseek_v4_1_flash.golden import (
    prefill_linear as reference_linear,
    prefill_round_activation as round_activation,
)
from models.deepseek_v4_1_flash.metadata import PrefillContext
from models.deepseek_v4_1_flash.metadata import PrefillState
from pathlib import Path
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig, StackedDeviceTensor
from types import SimpleNamespace
import argparse
import importlib.util
import json
import math
import os
import struct
import sys
import time
import torch

RANKS = 4
LOCAL_EXPERTS = FLASH.n_routed_experts // RANKS
VOCAB = FLASH.vocab_size
TOPK = FLASH.num_experts_per_tok
EXPERTS = FLASH.n_routed_experts
T = pl.dynamic("V41_PREFILL_ROWS")
TE = pl.dynamic("V41_ENCODER_ROWS")
TD = pl.dynamic("V41_DECODER_ROWS")
REQUESTS = pl.dynamic("V41_PREFILL_REQUESTS")
B8 = pl.dynamic("V41_PREFILL_FP8_BLOCKS")
B4 = pl.dynamic("V41_PREFILL_FP4_BLOCKS")
F32 = pl.dynamic("V41_PREFILL_DENSE_VALUES")
BW = pl.dynamic("V41_PREFILL_WINDOW_BLOCKS")
BG = pl.dynamic("V41_PREFILL_GLOBAL_BLOCKS")
BS = pl.dynamic("V41_PREFILL_STATE_BLOCKS")
PE = pl.dynamic("V41_PREFILL_ENCODER_PAGES")
PD = pl.dynamic("V41_PREFILL_DECODER_PAGES")
PAGES = pl.dynamic("V41_PREFILL_STAGE_PAGES")
WIDTH = pl.dynamic("V41_PREFILL_COPY_WIDTH")
SOURCE_ROWS = pl.dynamic("V41_PREFILL_SOURCE_ROWS")
COMM_ROWS = pl.dynamic("V41_PREFILL_COMM_ROWS")


@pl.jit.inline(auto_scope=False)
def copy_rows_inline(
    source: pl.Tensor[[T, WIDTH], pl.FP32],
    output: pl.Tensor[[T, WIDTH], pl.FP32],
):
    rows = pl.tensor.dim(source, 0)
    width = pl.tensor.dim(source, 1)
    for worker in pl.spmd(32, name_hint="prefill_copy_rows"):
        for group in pl.range(worker, (rows + 15) // 16, 32):
            for row in pl.range(group * 16, pl.min(rows, group * 16 + 16)):
                for col in pl.range(0, width, 256):
                    active = pl.min(256, width - col)
                    part = pl.load(source, [row, col], [1, 256], valid_shape=[1, active])
                    output = pl.store(part, [row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def gather_rows_inline(
    source: pl.Tensor[[SOURCE_ROWS, WIDTH], pl.FP32],
    row_ids: pl.Tensor[[T], pl.INT32],
    output: pl.Tensor[[T, WIDTH], pl.FP32],
):
    rows = pl.tensor.dim(output, 0)
    width = pl.tensor.dim(output, 1)
    for worker in pl.spmd(32, name_hint="prefill_gather_rows"):
        for group in pl.range(worker, (rows + 15) // 16, 32):
            for row in pl.range(group * 16, pl.min(rows, group * 16 + 16)):
                src = pl.cast(pl.read(row_ids, [row]), pl.INDEX)
                for col in pl.range(0, width, 256):
                    active = pl.min(256, width - col)
                    part = pl.load(source, [src, col], [1, 256], valid_shape=[1, active])
                    output = pl.store(part, [row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def initialize_hc_inline(
    embedding: pl.Tensor[[T, D], pl.FP32],
    hidden: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    pre: pl.Tensor[[T, HC_MULT], pl.FP32],
):
    rows = pl.tensor.dim(embedding, 0)
    flat = pl.reshape(hidden, [rows, HC_MULT * D])
    for worker in pl.spmd(32, name_hint="prefill_initialize_hc"):
        for group in pl.range(worker, (rows + 15) // 16, 32):
            for row in pl.range(group * 16, pl.min(rows, group * 16 + 16)):
                mix = pl.tile.full([1, 8], dtype=pl.FP32, value=0.0)
                pre = pl.store(pl.set_validshape(mix, 1, HC_MULT), [row, 0], pre)
                first = pl.tile.full([1, 8], dtype=pl.FP32, value=1.0)
                pre = pl.store(pl.set_validshape(first, 1, 1), [row, 0], pre)
                for col in pl.range(0, D, 256):
                    value = pl.load(embedding, [row, col], [1, 256])
                    for stream in pl.unroll(HC_MULT):
                        flat = pl.store(value, [row, stream * D + col], flat)
    return hidden, pre


@pl.jit.inline(auto_scope=False)
def hc_mixes_inline(
    x: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    bank: pl.Tensor[[F32], pl.FP32],
    offsets: pl.Tensor[[40, 20], pl.INT32],
    layer: pl.Scalar[pl.INDEX],
    first_column: pl.Scalar[pl.INDEX],
    pre: pl.Tensor[[T, HC_MULT], pl.FP32],
    post: pl.Tensor[[T, HC_MULT], pl.FP32],
    residual: pl.Tensor[[T, HC_MULT, HC_MULT], pl.FP32],
):
    rows = pl.tensor.dim(x, 0)
    weight_offset = pl.cast(pl.read(offsets, [layer, first_column]), pl.INDEX)
    weight_values = bank[weight_offset : weight_offset + HC_MULT * D * 24]
    weight = pl.reshape(weight_values, [HC_MULT * D, 24])
    scale_offset = pl.cast(pl.read(offsets, [layer, first_column + 1]), pl.INDEX)
    scale_values = bank[scale_offset : scale_offset + 3]
    scale = pl.reshape(scale_values, [3])
    base_offset = pl.cast(pl.read(offsets, [layer, first_column + 2]), pl.INDEX)
    base_values = bank[base_offset : base_offset + 24]
    base = pl.reshape(base_values, [24])
    with pl.scope():
        raw = pl.create_tensor([rows, 24], dtype=pl.FP32)
        normalized = pl.create_tensor([rows, 24], dtype=pl.FP32)
        call_view_3 = pl.reshape(x, [rows, HC_MULT * D])
        linear_inline(call_view_3, weight, raw)
        call_view_4 = pl.reshape(x, [rows, HC_MULT, D])
        hc_normalize_logits_inline(call_view_4, raw, normalized)
        call_view_5 = pl.reshape(pre, [rows, HC_MULT])
        call_view_6 = pl.reshape(post, [rows, HC_MULT])
        call_view_7 = pl.reshape(residual, [rows, HC_MULT, HC_MULT])
        hc_coefficients_inline(
            normalized,
            scale,
            base,
            call_view_5,
            call_view_6,
            call_view_7,
        )
    return pre, post, residual


@pl.jit.inline(auto_scope=False)
def collapse_norm_inline(
    x: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    pre: pl.Tensor[[T, HC_MULT], pl.FP32],
    norm_weight: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[T, D], pl.FP32],
    official: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(x, 0)
    with pl.scope():
        collapsed = pl.create_tensor([rows, D], dtype=pl.FP32)
        rounded = pl.create_tensor([rows, D], dtype=pl.FP32)
        call_view_8 = pl.reshape(x, [rows, HC_MULT, D])
        call_view_9 = pl.reshape(pre, [rows, HC_MULT])
        hc_pre_inline(call_view_8, call_view_9, collapsed)
        round_inline(collapsed, rounded, official)
        call_view_10 = pl.reshape(output, [rows, D])
        norm_inline(rounded, norm_weight, call_view_10, official)
    return output


@pl.jit.inline(auto_scope=False)
def post_hc_inline(
    update: pl.Tensor[[T, D], pl.FP32],
    skip: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T, HC_MULT], pl.FP32],
    residual: pl.Tensor[[T, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    official: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(update, 0)
    with pl.scope():
        mixed = pl.create_tensor([rows, HC_MULT, D], dtype=pl.FP32)
        call_view_11 = pl.reshape(update, [rows, D])
        call_view_12 = pl.reshape(skip, [rows, HC_MULT, D])
        call_view_13 = pl.reshape(post, [rows, HC_MULT])
        call_view_14 = pl.reshape(residual, [rows, HC_MULT, HC_MULT])
        hc_post_inline(
            call_view_11,
            call_view_12,
            call_view_13,
            call_view_14,
            mixed,
        )
        call_view_15 = pl.reshape(mixed, [rows, HC_MULT * D])
        call_view_16 = pl.reshape(output, [rows, HC_MULT * D])
        round_inline(
            call_view_15,
            call_view_16,
            official,
        )
    return output


@pl.jit.inline(auto_scope=False)
def apply_engram_inline(
    x: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    lookup: pl.Tensor[[T, 6144], pl.FP32],
    bank8: pl.Tensor[[B8, 1024], pl.INT8],
    scales8: pl.Tensor[[B8], pl.UINT8],
    offsets8: pl.Tensor[[40, 10], pl.INT32],
    dense: pl.Tensor[[F32], pl.FP32],
    offsets: pl.Tensor[[40, 20], pl.INT32],
    layer: pl.Scalar[pl.INDEX],
    output: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    official: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(x, 0)
    with pl.scope():
        lookup_rounded = pl.create_tensor([rows, 6144], dtype=pl.FP32)
        projected = pl.create_tensor([rows, (HC_MULT + 1) * D], dtype=pl.FP32)
        gated = pl.create_tensor([rows, HC_MULT, D], dtype=pl.FP32)
        call_view_17 = pl.reshape(lookup, [rows, 6144])
        round_inline(call_view_17, lookup_rounded, official)
        block = pl.cast(pl.read(offsets8, [layer, 9]), pl.INDEX)
        packed_linear_inline(lookup_rounded, bank8, scales8, block, projected, official, 2)
        weight_offset = pl.cast(pl.read(offsets, [layer, 19]), pl.INDEX)
        weight_values = dense[weight_offset : weight_offset + HC_MULT * D]
        weight = pl.reshape(weight_values, [HC_MULT, D])
        call_view_18 = pl.reshape(x, [rows, HC_MULT, D])
        engram_gate_inline(call_view_18, projected, weight, gated)
        call_view_19 = pl.reshape(gated, [rows, HC_MULT * D])
        call_view_20 = pl.reshape(output, [rows, HC_MULT * D])
        round_inline(
            call_view_19,
            call_view_20,
            official,
        )
    return output


@pl.jit.inline(auto_scope=False)
def prefill_stage_inline(
    initial: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    initial_pre: pl.Tensor[[T, HC_MULT], pl.FP32],
    engram_lookup: pl.Tensor[[2, TE, 6144], pl.FP32],
    bank8: pl.Tensor[[B8, 1024], pl.INT8],
    scales8: pl.Tensor[[B8], pl.UINT8],
    offsets8: pl.Tensor[[40, 10], pl.INT32],
    bank4: pl.Tensor[[B4, 512], pl.INT8],
    scales4: pl.Tensor[[B4, 32], pl.UINT8],
    expert_offsets: pl.Tensor[[40, LOCAL_EXPERTS, 3], pl.INT32],
    dense: pl.Tensor[[F32], pl.FP32],
    offsets: pl.Tensor[[40, 20], pl.INT32],
    cosine: pl.Tensor[[2, T, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[2, T, ROPE_DIM // 2], pl.FP32],
    compressed_cos: pl.Tensor[[T, ROPE_DIM // 2], pl.FP32],
    compressed_sin: pl.Tensor[[T, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T], pl.INT32],
    window_indices: pl.Tensor[[T, 128], pl.INT32],
    compressed_slots: pl.Tensor[[T], pl.INT32],
    previous_rows: pl.Tensor[[T], pl.INT32],
    blocks: pl.Tensor[[T, PAGES], pl.INT32],
    lengths: pl.Tensor[[T], pl.INT32],
    extents: pl.Tensor[[2, T, 2], pl.INT32],
    state_slots: pl.Tensor[[T], pl.INT32],
    window_cache: pl.Tensor[[40, BW, 128, HEAD_DIM], pl.FP32],
    global_cache: pl.Tensor[[4, BG, 128, HEAD_DIM], pl.FP32],
    index_cache: pl.Tensor[[4, BG, 128, INDEX_DIM], pl.FP32],
    state_cache: pl.Tensor[[3, BS, STATE_CAPACITY, 2 * HEAD_DIM], pl.FP32],
    gathered: pld.DistributedTensor[[COMM_ROWS, D], pl.FP32],
    ready: pld.DistributedTensor[[RANKS, 128], pl.INT32],
    consumed: pld.DistributedTensor[[RANKS, 128], pl.INT32],
    output: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    output_pre: pl.Tensor[[T, HC_MULT], pl.FP32],
    routes: pl.Tensor[[20, T, TOPK], pl.INT32],
    first_layer: pl.Scalar[pl.INDEX],
    rank: pl.Scalar[pl.INT32],
    capacity: pl.Scalar[pl.INT32],
    official: pl.Scalar[pl.INT32],
):
    """Run one twenty-layer phase with device-owned state and expert dispatch."""
    rows = pl.tensor.dim(initial, 0)
    window_blocks = pl.tensor.dim(window_cache, 1)
    global_blocks = pl.tensor.dim(global_cache, 1)
    state_blocks = pl.tensor.dim(state_cache, 1)
    pages = pl.tensor.dim(blocks, 1)
    compressed_cos_view = pl.reshape(compressed_cos, [rows, ROPE_DIM // 2])
    compressed_sin_view = pl.reshape(compressed_sin, [rows, ROPE_DIM // 2])
    window_slots_view = pl.reshape(window_slots, [rows])
    window_indices_view = pl.reshape(window_indices, [rows, 128])
    compressed_slots_view = pl.reshape(compressed_slots, [rows])
    previous_rows_view = pl.reshape(previous_rows, [rows])
    blocks_view = pl.reshape(blocks, [rows, pages])
    lengths_view = pl.reshape(lengths, [rows])
    state_slots_view = pl.reshape(state_slots, [rows])
    hidden_pair = pl.create_tensor([2 * rows, HC_MULT, D], dtype=pl.FP32)
    pre_pair = pl.create_tensor([2 * rows, HC_MULT], dtype=pl.FP32)
    topk = pl.create_tensor([rows, INDEX_TOPK], dtype=pl.INT32)
    initial_flat = pl.reshape(initial, [rows, HC_MULT * D])
    shape_view_101 = pl.slice(hidden_pair, [rows, HC_MULT, D], [0, 0, 0])
    pair_first = pl.reshape(shape_view_101, [rows, HC_MULT * D])
    copy_rows_inline(initial_flat, pair_first)
    pre_first = pl.slice(pre_pair, [rows, HC_MULT], [0, 0])
    initial_pre_view = pl.reshape(initial_pre, [rows, HC_MULT])
    copy_rows_inline(initial_pre_view, pre_first)
    for worker in pl.spmd(32, name_hint="prefill_empty_global_indices"):
        for row in pl.range(worker, rows, 32):
            topk = pl.store(pl.tile.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1), [row, 0], topk)
    for layer in pl.range(first_layer, first_layer + 20):
        with pl.scope():
            source = pl.min(pl.max((layer - 2) // 6, 0), 3)
            rope_kind = pl.min(layer // 2, 1)
            ratio = 1
            if layer < 20:
                ratio = 2
            mode = 3
            if layer < 2:
                mode = 0
            if layer == 2:
                mode = 1
            if layer == 8:
                mode = 1
            if layer == 14:
                mode = 1
            if layer >= 20:
                if layer % 4 == 0:
                    mode = 2
            current = pl.slice(hidden_pair, [rows, HC_MULT, D], [(layer % 2) * rows, 0, 0])
            delayed_pre = pl.slice(pre_pair, [rows, HC_MULT], [(layer % 2) * rows, 0])
            next_hidden = pl.slice(hidden_pair, [rows, HC_MULT, D], [((layer + 1) % 2) * rows, 0, 0])
            next_pre = pl.slice(pre_pair, [rows, HC_MULT], [((layer + 1) % 2) * rows, 0])
            attended_input = pl.create_tensor([rows, HC_MULT, D], dtype=pl.FP32)
            has_engram = 0
            if layer == 1:
                has_engram = 1
            if layer == 14:
                has_engram = 1
            if has_engram == 1:
                engram_index = layer // 14
                shape_view_119 = pl.slice(engram_lookup, [1, rows, 6144], [engram_index, 0, 0])
                lookup = pl.reshape(shape_view_119, [rows, 6144])
                apply_engram_inline(
                    current,
                    lookup,
                    bank8,
                    scales8,
                    offsets8,
                    dense,
                    offsets,
                    layer,
                    attended_input,
                    official,
                )
            else:
                call_view_23 = pl.reshape(current, [rows, HC_MULT * D])
                call_view_24 = pl.reshape(attended_input, [rows, HC_MULT * D])
                copy_rows_inline(
                    call_view_23,
                    call_view_24,
                )
            attn_pre = pl.create_tensor([rows, HC_MULT], dtype=pl.FP32)
            attn_post = pl.create_tensor([rows, HC_MULT], dtype=pl.FP32)
            attn_residual = pl.create_tensor([rows, HC_MULT, HC_MULT], dtype=pl.FP32)
            hc_mixes_inline(
                attended_input,
                dense,
                offsets,
                layer,
                0,
                attn_pre,
                attn_post,
                attn_residual,
            )
            norm_offset = pl.cast(pl.read(offsets, [layer, 3]), pl.INDEX)
            norm_weight_values = dense[norm_offset : norm_offset + D]
            norm_weight = pl.reshape(norm_weight_values, [D])
            hidden = pl.create_tensor([rows, D], dtype=pl.FP32)
            collapse_norm_inline(attended_input, delayed_pre, norm_weight, hidden, official)
            shape_view_109 = pl.slice(cosine, [1, rows, ROPE_DIM // 2], [rope_kind, 0, 0])
            cos = pl.reshape(shape_view_109, [rows, ROPE_DIM // 2])
            shape_view_110 = pl.slice(sine, [1, rows, ROPE_DIM // 2], [rope_kind, 0, 0])
            sin = pl.reshape(shape_view_110, [rows, ROPE_DIM // 2])
            shape_view_111 = pl.slice(extents, [1, rows, 2], [rope_kind, 0, 0])
            layer_extents = pl.reshape(shape_view_111, [rows, 2])
            shape_view_112 = pl.slice(window_cache, [1, window_blocks, 128, HEAD_DIM], [layer, 0, 0, 0])
            window = pl.reshape(shape_view_112, [window_blocks, 128, HEAD_DIM])
            shape_view_113 = pl.slice(global_cache, [1, global_blocks, 128, HEAD_DIM], [source, 0, 0, 0])
            global_kv = pl.reshape(shape_view_113, [global_blocks, 128, HEAD_DIM])
            shape_view_114 = pl.slice(index_cache, [1, global_blocks, 128, INDEX_DIM], [source, 0, 0, 0])
            index_kv = pl.reshape(shape_view_114, [global_blocks, 128, INDEX_DIM])
            shape_view_115 = pl.slice(
                state_cache, [1, state_blocks, STATE_CAPACITY, HEAD_DIM * 2], [pl.min(source, 2), 0, 0, 0]
            )
            state = pl.reshape(shape_view_115, [state_blocks, STATE_CAPACITY, HEAD_DIM * 2])
            attention = pl.create_tensor([rows, D], dtype=pl.FP32)
            attention_inline(
                hidden,
                bank8,
                scales8,
                offsets8,
                dense,
                offsets,
                layer,
                cos,
                sin,
                compressed_cos_view,
                compressed_sin_view,
                window_slots_view,
                window_indices_view,
                compressed_slots_view,
                previous_rows_view,
                blocks_view,
                lengths_view,
                layer_extents,
                state_slots_view,
                window,
                global_kv,
                index_kv,
                state,
                topk,
                attention,
                mode,
                ratio,
                official,
            )
            middle = pl.create_tensor([rows, HC_MULT, D], dtype=pl.FP32)
            post_hc_inline(attention, attended_input, attn_post, attn_residual, middle, official)
            ffn_post = pl.create_tensor([rows, HC_MULT], dtype=pl.FP32)
            ffn_residual = pl.create_tensor([rows, HC_MULT, HC_MULT], dtype=pl.FP32)
            hc_mixes_inline(
                middle,
                dense,
                offsets,
                layer,
                7,
                next_pre,
                ffn_post,
                ffn_residual,
            )
            ffn_norm_offset = pl.cast(pl.read(offsets, [layer, 10]), pl.INDEX)
            ffn_weight_values = dense[ffn_norm_offset : ffn_norm_offset + D]
            ffn_weight = pl.reshape(ffn_weight_values, [D])
            ffn_input = pl.create_tensor([rows, D], dtype=pl.FP32)
            collapse_norm_inline(middle, attn_pre, ffn_weight, ffn_input, official)
            gate_offset = pl.cast(pl.read(offsets, [layer, 11]), pl.INDEX)
            gate_values = dense[gate_offset : gate_offset + D * EXPERTS]
            gate_weight = pl.reshape(gate_values, [D, EXPERTS])
            bias_offset = pl.cast(pl.read(offsets, [layer, 12]), pl.INDEX)
            gate_bias_values = dense[bias_offset : bias_offset + EXPERTS]
            gate_bias = pl.reshape(gate_bias_values, [EXPERTS])
            shape_view_116 = pl.slice(expert_offsets, [1, LOCAL_EXPERTS, 3], [layer, 0, 0])
            local_offsets = pl.reshape(shape_view_116, [LOCAL_EXPERTS, 3])
            shape_view_117 = pl.slice(routes, [1, rows, TOPK], [layer - first_layer, 0, 0])
            layer_routes = pl.reshape(shape_view_117, [rows, TOPK])
            local = pl.create_tensor([rows * TOPK, D], dtype=pl.FP32)
            routed_moe_inline(
                ffn_input,
                bank4,
                scales4,
                local_offsets,
                gate_weight,
                gate_bias,
                local,
                layer_routes,
                rank,
                official,
            )
            combined = pl.create_tensor([rows, D], dtype=pl.FP32)
            combine_routed_experts(
                local, layer_routes, gathered, ready, consumed, combined, rank, layer + 1, capacity
            )
            shared = pl.create_tensor([rows, D], dtype=pl.FP32)
            flat_offsets8 = pl.reshape(offsets8, [400])
            shared_offsets = pl.slice(flat_offsets8, [3], [layer * 10 + 6])
            shared_expert_inline(ffn_input, bank8, scales8, shared_offsets, shared, official)
            added = pl.create_tensor([rows, D], dtype=pl.FP32)
            add_inline(combined, shared, added)
            ffn_output = pl.create_tensor([rows, D], dtype=pl.FP32)
            round_inline(added, ffn_output, official)
            post_hc_inline(ffn_output, middle, ffn_post, ffn_residual, next_hidden, official)
    shape_view_102 = pl.slice(hidden_pair, [rows, HC_MULT, D], [0, 0, 0])
    final_hidden = pl.reshape(shape_view_102, [rows, HC_MULT * D])
    call_view_1 = pl.reshape(output, [rows, HC_MULT * D])
    copy_rows_inline(final_hidden, call_view_1)
    call_view_2 = pl.slice(pre_pair, [rows, HC_MULT], [0, 0])
    final_pre_view = pl.reshape(output_pre, [rows, HC_MULT])
    copy_rows_inline(call_view_2, final_pre_view)
    return output, output_pre, window_cache, global_cache, index_cache, state_cache, routes


@pl.jit(auto_scope=False)
def prefill_fwd(
    embedding: pl.Tensor[[TE, D], pl.FP32],
    engram_lookup: pl.Tensor[[2, TE, 6144], pl.FP32],
    bank8: pl.Tensor[[B8, 1024], pl.INT8],
    scales8: pl.Tensor[[B8], pl.UINT8],
    offsets8: pl.Tensor[[40, 10], pl.INT32],
    bank4: pl.Tensor[[B4, 512], pl.INT8],
    scales4: pl.Tensor[[B4, 32], pl.UINT8],
    expert_offsets: pl.Tensor[[40, LOCAL_EXPERTS, 3], pl.INT32],
    dense: pl.Tensor[[F32], pl.FP32],
    offsets: pl.Tensor[[40, 20], pl.INT32],
    encoder_cos: pl.Tensor[[2, TE, ROPE_DIM // 2], pl.FP32],
    encoder_sin: pl.Tensor[[2, TE, ROPE_DIM // 2], pl.FP32],
    compressed_cos: pl.Tensor[[TE, ROPE_DIM // 2], pl.FP32],
    compressed_sin: pl.Tensor[[TE, ROPE_DIM // 2], pl.FP32],
    decoder_cos: pl.Tensor[[2, TD, ROPE_DIM // 2], pl.FP32],
    decoder_sin: pl.Tensor[[2, TD, ROPE_DIM // 2], pl.FP32],
    encoder_slots: pl.Tensor[[TE], pl.INT32],
    encoder_indices: pl.Tensor[[TE, 128], pl.INT32],
    compressed_slots: pl.Tensor[[TE], pl.INT32],
    previous_rows: pl.Tensor[[TE], pl.INT32],
    encoder_blocks: pl.Tensor[[TE, PE], pl.INT32],
    encoder_lengths: pl.Tensor[[TE], pl.INT32],
    encoder_extents: pl.Tensor[[2, TE, 2], pl.INT32],
    state_slots: pl.Tensor[[TE], pl.INT32],
    publisher_slots: pl.Tensor[[TE], pl.INT32],
    decoder_slots: pl.Tensor[[TD], pl.INT32],
    decoder_indices: pl.Tensor[[TD, 128], pl.INT32],
    decoder_blocks: pl.Tensor[[TD, PD], pl.INT32],
    decoder_lengths: pl.Tensor[[TD], pl.INT32],
    decoder_extents: pl.Tensor[[2, TD, 2], pl.INT32],
    replay_rows: pl.Tensor[[TD], pl.INT32],
    last_rows: pl.Tensor[[REQUESTS], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[40, BW, 128, HEAD_DIM], pl.FP32]],
    global_cache: pl.InOut[pl.Tensor[[4, BG, 128, HEAD_DIM], pl.FP32]],
    index_cache: pl.InOut[pl.Tensor[[4, BG, 128, INDEX_DIM], pl.FP32]],
    state_cache: pl.InOut[pl.Tensor[[3, BS, STATE_CAPACITY, 2 * HEAD_DIM], pl.FP32]],
    final_norm: pl.Tensor[[D], pl.FP32],
    head: pl.Tensor[[D, VOCAB], pl.FP32],
    encoder_out: pl.Out[pl.Tensor[[TE, HC_MULT, D], pl.FP32]],
    encoder_pre: pl.Out[pl.Tensor[[TE, HC_MULT], pl.FP32]],
    decoder_out: pl.Out[pl.Tensor[[TD, HC_MULT, D], pl.FP32]],
    decoder_pre: pl.Out[pl.Tensor[[TD, HC_MULT], pl.FP32]],
    encoder_routes: pl.Out[pl.Tensor[[20, TE, TOPK], pl.INT32]],
    decoder_routes: pl.Out[pl.Tensor[[20, TD, TOPK], pl.INT32]],
    normalized: pl.Out[pl.Tensor[[TD, D], pl.FP32]],
    logits: pl.Out[pl.Tensor[[REQUESTS, VOCAB], pl.FP32]],
    gathered: pld.DistributedTensor[[COMM_ROWS, D], pl.FP32],
    ready: pld.DistributedTensor[[RANKS, 128], pl.INT32],
    consumed: pld.DistributedTensor[[RANKS, 128], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    official: pl.Scalar[pl.INT32],
):
    """All forty layers, the CED boundary and the head in one rank invocation."""
    embedding.bind_dynamic(0, TE)
    replay_rows.bind_dynamic(0, TD)
    last_rows.bind_dynamic(0, REQUESTS)
    bank8.bind_dynamic(0, B8)
    bank4.bind_dynamic(0, B4)
    dense.bind_dynamic(0, F32)
    window_cache.bind_dynamic(1, BW)
    global_cache.bind_dynamic(1, BG)
    state_cache.bind_dynamic(1, BS)
    encoder_blocks.bind_dynamic(1, PE)
    decoder_blocks.bind_dynamic(1, PD)
    gathered.bind_dynamic(0, COMM_ROWS)
    enc_rows = pl.tensor.dim(embedding, 0)
    dec_rows = pl.tensor.dim(replay_rows, 0)
    requests = pl.tensor.dim(last_rows, 0)
    global_blocks = pl.tensor.dim(global_cache, 1)
    with pl.scope():
        initial = pl.create_tensor([enc_rows, HC_MULT, D], dtype=pl.FP32)
        initial_pre = pl.create_tensor([enc_rows, HC_MULT], dtype=pl.FP32)
        initialize_hc_inline(embedding, initial, initial_pre)
        prefill_stage_inline(
            initial,
            initial_pre,
            engram_lookup,
            bank8,
            scales8,
            offsets8,
            bank4,
            scales4,
            expert_offsets,
            dense,
            offsets,
            encoder_cos,
            encoder_sin,
            compressed_cos,
            compressed_sin,
            encoder_slots,
            encoder_indices,
            compressed_slots,
            previous_rows,
            encoder_blocks,
            encoder_lengths,
            encoder_extents,
            state_slots,
            window_cache,
            global_cache,
            index_cache,
            state_cache,
            gathered,
            ready,
            consumed,
            encoder_out,
            encoder_pre,
            encoder_routes,
            0,
            rank,
            enc_rows,
            official,
        )
    # Publication must consume all final encoder rows before the replay gather.
    with pl.scope():
        publish_hidden = pl.create_tensor([enc_rows, D], dtype=pl.FP32)
        publisher_norm_offset = pl.cast(pl.read(offsets, [20, 3]), pl.INDEX)
        publisher_norm_values = dense[publisher_norm_offset : publisher_norm_offset + D]
        publisher_norm = pl.reshape(publisher_norm_values, [D])
        collapse_norm_inline(encoder_out, encoder_pre, publisher_norm, publish_hidden, official)
        shape_view_103 = pl.slice(encoder_cos, [1, enc_rows, ROPE_DIM // 2], [1, 0, 0])
        publisher_cos = pl.reshape(shape_view_103, [enc_rows, ROPE_DIM // 2])
        shape_view_104 = pl.slice(encoder_sin, [1, enc_rows, ROPE_DIM // 2], [1, 0, 0])
        publisher_sin = pl.reshape(shape_view_104, [enc_rows, ROPE_DIM // 2])
        shape_view_105 = pl.slice(global_cache, [1, global_blocks, 128, HEAD_DIM], [3, 0, 0, 0])
        publisher_global = pl.reshape(shape_view_105, [global_blocks, 128, HEAD_DIM])
        shape_view_106 = pl.slice(index_cache, [1, global_blocks, 128, INDEX_DIM], [3, 0, 0, 0])
        publisher_index = pl.reshape(shape_view_106, [global_blocks, 128, INDEX_DIM])
        publish_decoder_inline(
            publish_hidden,
            dense,
            offsets,
            publisher_cos,
            publisher_sin,
            publisher_slots,
            publisher_global,
            publisher_index,
            official,
        )
    with pl.scope():
        replay = pl.create_tensor([dec_rows, HC_MULT, D], dtype=pl.FP32)
        replay_pre = pl.create_tensor([dec_rows, HC_MULT], dtype=pl.FP32)
        call_view_21 = pl.reshape(encoder_out, [enc_rows, HC_MULT * D])
        call_view_22 = pl.reshape(replay, [dec_rows, HC_MULT * D])
        gather_rows_inline(
            call_view_21,
            replay_rows,
            call_view_22,
        )
        gather_rows_inline(encoder_pre, replay_rows, replay_pre)
        inactive_slots = pl.create_tensor([dec_rows], dtype=pl.INT32)
        target = pl.reshape(inactive_slots, [1, dec_rows])
        for worker in pl.spmd(32, name_hint="prefill_no_decoder_publication"):
            for start in pl.range(worker * 16, dec_rows, 32 * 16):
                active = pl.min(16, dec_rows - start)
                value = pl.tile.full([1, 16], dtype=pl.INT32, value=-1)
                target = pl.store(pl.set_validshape(value, 1, active), [0, start], target)
        shape_view_107 = pl.slice(decoder_cos, [1, dec_rows, ROPE_DIM // 2], [1, 0, 0])
        decoder_plain_cos = pl.reshape(shape_view_107, [dec_rows, ROPE_DIM // 2])
        shape_view_108 = pl.slice(decoder_sin, [1, dec_rows, ROPE_DIM // 2], [1, 0, 0])
        decoder_plain_sin = pl.reshape(shape_view_108, [dec_rows, ROPE_DIM // 2])
        prefill_stage_inline(
            replay,
            replay_pre,
            engram_lookup,
            bank8,
            scales8,
            offsets8,
            bank4,
            scales4,
            expert_offsets,
            dense,
            offsets,
            decoder_cos,
            decoder_sin,
            decoder_plain_cos,
            decoder_plain_sin,
            decoder_slots,
            decoder_indices,
            inactive_slots,
            inactive_slots,
            decoder_blocks,
            decoder_lengths,
            decoder_extents,
            inactive_slots,
            window_cache,
            global_cache,
            index_cache,
            state_cache,
            gathered,
            ready,
            consumed,
            decoder_out,
            decoder_pre,
            decoder_routes,
            20,
            rank,
            enc_rows,
            official,
        )
    collapse_norm_inline(decoder_out, decoder_pre, final_norm, normalized, official)
    with pl.scope():
        selected = pl.create_tensor([requests, D], dtype=pl.FP32)
        gather_rows_inline(normalized, last_rows, selected)
        linear_inline(selected, head, logits)
    return encoder_out, encoder_pre, decoder_out, decoder_pre, normalized, logits


@pl.jit.host
def l3_prefill_fwd(
    embedding: pl.Tensor[[RANKS, TE, D], pl.FP32],
    engram_lookup: pl.Tensor[[RANKS, 2, TE, 6144], pl.FP32],
    bank8: pl.Tensor[[RANKS, B8, 1024], pl.INT8],
    scales8: pl.Tensor[[RANKS, B8], pl.UINT8],
    offsets8: pl.Tensor[[RANKS, 40, 10], pl.INT32],
    bank4: pl.Tensor[[RANKS, B4, 512], pl.INT8],
    scales4: pl.Tensor[[RANKS, B4, 32], pl.UINT8],
    expert_offsets: pl.Tensor[[RANKS, 40, LOCAL_EXPERTS, 3], pl.INT32],
    dense: pl.Tensor[[RANKS, F32], pl.FP32],
    offsets: pl.Tensor[[RANKS, 40, 20], pl.INT32],
    encoder_cos: pl.Tensor[[RANKS, 2, TE, ROPE_DIM // 2], pl.FP32],
    encoder_sin: pl.Tensor[[RANKS, 2, TE, ROPE_DIM // 2], pl.FP32],
    compressed_cos: pl.Tensor[[RANKS, TE, ROPE_DIM // 2], pl.FP32],
    compressed_sin: pl.Tensor[[RANKS, TE, ROPE_DIM // 2], pl.FP32],
    decoder_cos: pl.Tensor[[RANKS, 2, TD, ROPE_DIM // 2], pl.FP32],
    decoder_sin: pl.Tensor[[RANKS, 2, TD, ROPE_DIM // 2], pl.FP32],
    encoder_slots: pl.Tensor[[RANKS, TE], pl.INT32],
    encoder_indices: pl.Tensor[[RANKS, TE, 128], pl.INT32],
    compressed_slots: pl.Tensor[[RANKS, TE], pl.INT32],
    previous_rows: pl.Tensor[[RANKS, TE], pl.INT32],
    encoder_blocks: pl.Tensor[[RANKS, TE, PE], pl.INT32],
    encoder_lengths: pl.Tensor[[RANKS, TE], pl.INT32],
    encoder_extents: pl.Tensor[[RANKS, 2, TE, 2], pl.INT32],
    state_slots: pl.Tensor[[RANKS, TE], pl.INT32],
    publisher_slots: pl.Tensor[[RANKS, TE], pl.INT32],
    decoder_slots: pl.Tensor[[RANKS, TD], pl.INT32],
    decoder_indices: pl.Tensor[[RANKS, TD, 128], pl.INT32],
    decoder_blocks: pl.Tensor[[RANKS, TD, PD], pl.INT32],
    decoder_lengths: pl.Tensor[[RANKS, TD], pl.INT32],
    decoder_extents: pl.Tensor[[RANKS, 2, TD, 2], pl.INT32],
    replay_rows: pl.Tensor[[RANKS, TD], pl.INT32],
    last_rows: pl.Tensor[[RANKS, REQUESTS], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS, 40, BW, 128, HEAD_DIM], pl.FP32]],
    global_cache: pl.InOut[pl.Tensor[[RANKS, 4, BG, 128, HEAD_DIM], pl.FP32]],
    index_cache: pl.InOut[pl.Tensor[[RANKS, 4, BG, 128, INDEX_DIM], pl.FP32]],
    state_cache: pl.InOut[pl.Tensor[[RANKS, 3, BS, STATE_CAPACITY, 2 * HEAD_DIM], pl.FP32]],
    final_norm: pl.Tensor[[RANKS, D], pl.FP32],
    head: pl.Tensor[[RANKS, D, VOCAB], pl.FP32],
    encoder_out: pl.Out[pl.Tensor[[RANKS, TE, HC_MULT, D], pl.FP32]],
    encoder_pre: pl.Out[pl.Tensor[[RANKS, TE, HC_MULT], pl.FP32]],
    decoder_out: pl.Out[pl.Tensor[[RANKS, TD, HC_MULT, D], pl.FP32]],
    decoder_pre: pl.Out[pl.Tensor[[RANKS, TD, HC_MULT], pl.FP32]],
    encoder_routes: pl.Out[pl.Tensor[[RANKS, 20, TE, TOPK], pl.INT32]],
    decoder_routes: pl.Out[pl.Tensor[[RANKS, 20, TD, TOPK], pl.INT32]],
    normalized: pl.Out[pl.Tensor[[RANKS, TD, D], pl.FP32]],
    logits: pl.Out[pl.Tensor[[RANKS, REQUESTS, VOCAB], pl.FP32]],
    official: pl.Scalar[pl.INT32],
    capacity: pl.Scalar[pl.INT32],
):
    """Launch exactly one complete CED orchestration per expert rank."""
    embedding.bind_dynamic(1, TE)
    gathered_buffer = pld.alloc_window_buffer([RANKS * capacity * TOPK, D], dtype=pl.FP32)
    ready_buffer = pld.alloc_window_buffer([RANKS, 128], dtype=pl.INT32)
    consumed_buffer = pld.alloc_window_buffer([RANKS, 128], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        gathered = pld.window(gathered_buffer, [RANKS * capacity * TOPK, D], dtype=pl.FP32)
        ready = pld.window(ready_buffer, [RANKS, 128], dtype=pl.INT32)
        consumed = pld.window(consumed_buffer, [RANKS, 128], dtype=pl.INT32)
        prefill_fwd(
            embedding[rank],
            engram_lookup[rank],
            bank8[rank],
            scales8[rank],
            offsets8[rank],
            bank4[rank],
            scales4[rank],
            expert_offsets[rank],
            dense[rank],
            offsets[rank],
            encoder_cos[rank],
            encoder_sin[rank],
            compressed_cos[rank],
            compressed_sin[rank],
            decoder_cos[rank],
            decoder_sin[rank],
            encoder_slots[rank],
            encoder_indices[rank],
            compressed_slots[rank],
            previous_rows[rank],
            encoder_blocks[rank],
            encoder_lengths[rank],
            encoder_extents[rank],
            state_slots[rank],
            publisher_slots[rank],
            decoder_slots[rank],
            decoder_indices[rank],
            decoder_blocks[rank],
            decoder_lengths[rank],
            decoder_extents[rank],
            replay_rows[rank],
            last_rows[rank],
            window_cache[rank],
            global_cache[rank],
            index_cache[rank],
            state_cache[rank],
            final_norm[rank],
            head[rank],
            encoder_out[rank],
            encoder_pre[rank],
            decoder_out[rank],
            decoder_pre[rank],
            encoder_routes[rank],
            decoder_routes[rank],
            normalized[rank],
            logits[rank],
            gathered,
            ready,
            consumed,
            rank,
            official,
            device=rank,
        )


PREFILL_TENSORS = (
    "embedding",
    "engram_lookup",
    "bank8",
    "scales8",
    "offsets8",
    "bank4",
    "scales4",
    "expert_offsets",
    "dense",
    "offsets",
    "encoder_cos",
    "encoder_sin",
    "compressed_cos",
    "compressed_sin",
    "decoder_cos",
    "decoder_sin",
    "encoder_slots",
    "encoder_indices",
    "compressed_slots",
    "previous_rows",
    "encoder_blocks",
    "encoder_lengths",
    "encoder_extents",
    "state_slots",
    "publisher_slots",
    "decoder_slots",
    "decoder_indices",
    "decoder_blocks",
    "decoder_lengths",
    "decoder_extents",
    "replay_rows",
    "last_rows",
    "window_cache",
    "global_cache",
    "index_cache",
    "state_cache",
    "final_norm",
    "head",
    "encoder_out",
    "encoder_pre",
    "decoder_out",
    "decoder_pre",
    "encoder_routes",
    "decoder_routes",
    "normalized",
    "logits",
)


_DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
}


def decode_checkpoint_linear(weight: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Decode stored weights without activation quantization or requantizing weights.

    The returned matrix is output-major [N,K], matching the checkpoint and
    torch.nn.functional.linear. MXFP8 uses 32x32 scale blocks; MXFP4 packs the
    even input column in the low nibble and uses one scale per 32 input columns.
    """
    if weight.dtype in (torch.float32, torch.bfloat16):
        if scale is not None:
            raise ValueError("unquantized checkpoint weights must not have a scale")
        return weight.float().contiguous()
    if scale is None or scale.dtype != torch.float8_e8m0fnu:
        raise ValueError("quantized checkpoint weights require E8M0 scales")
    codes = scale.contiguous().view(torch.uint8).int()
    if bool((codes == 255).any()):
        raise ValueError("checkpoint E8M0 scales must be finite")
    factors = torch.exp2(codes.float() - 127)
    if weight.dtype == torch.float8_e4m3fn:
        if weight.ndim != 2 or tuple(weight.shape) != (scale.shape[0] * 32, scale.shape[1] * 32):
            raise ValueError("MXFP8 checkpoint weight/scale shapes disagree")
        result = weight.float() * factors.repeat_interleave(32, 0).repeat_interleave(32, 1)
    elif weight.dtype == torch.int8:
        if (
            weight.ndim != 2
            or weight.shape[0] != scale.shape[0]
            or weight.shape[1] * 2 != scale.shape[1] * 32
        ):
            raise ValueError("MXFP4 checkpoint weight/scale shapes disagree")
        packed = weight.contiguous().view(torch.uint8)
        nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
        values = magnitude[(nibbles & 7).long()]
        values = torch.where((nibbles & 8) != 0, -values, values)
        result = values * factors.repeat_interleave(32, -1)
    else:
        raise ValueError(f"unsupported checkpoint linear dtype {weight.dtype}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("decoded checkpoint linear contains non-finite values")
    return result.contiguous()


class PrefillCheckpoint:
    """Read requested original tensors and decode linear matrices as output-major FP32."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        config = json.loads((self.root / "config.json").read_text())
        text = config["text_config"]
        quant = config.get("quantization_config", {})
        if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [32, 32]:
            raise ValueError("checkpoint must use the official block-32 FP8/FP4 format")
        required = {
            "hidden_size": C.D,
            "num_hidden_layers": C.FLASH.num_hidden_layers,
            "num_attention_heads": C.H,
            "head_dim": C.HEAD_DIM,
            "sliding_window": C.FLASH.sliding_window,
            "kv_source_layer_ids": list(C.FLASH.kv_source_layer_ids),
            "index_source_layer_ids": list(C.FLASH.index_source_layer_ids),
        }
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"checkpoint {name}={text.get(name)!r}, expected {expected!r}")
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers = {}

    def tensor(self, name: str) -> torch.Tensor:
        """Return an owned CPU tensor without loading the rest of its shard."""
        shard = self.weight_map[name]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard leaves its root: {shard}")
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"safetensors header is too large: {shard}")
            if shard not in self._headers:
                self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype = _DTYPES[entry["dtype"]]
            shape = entry["shape"]
            start, end = entry["data_offsets"]
            expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if start < 0 or end - start != expected_bytes or 8 + header_size + end > path.stat().st_size:
                raise ValueError(f"invalid safetensors range for {name}")
            file.seek(8 + header_size + start)
            payload = bytearray(file.read(expected_bytes))
        if len(payload) != expected_bytes:
            raise ValueError(f"truncated safetensors payload for {name}")
        return torch.frombuffer(payload, dtype=dtype).reshape(shape)

    def tensor_rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Read selected table rows, including from the very large Engram tables."""
        if rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("table row ids must have an integer dtype")
        ids = rows.to(dtype=torch.int64, device="cpu").flatten()
        shard = self.weight_map[name]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard leaves its root: {shard}")
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"safetensors header is too large: {shard}")
            if shard not in self._headers:
                self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype, shape = _DTYPES[entry["dtype"]], entry["shape"]
            if len(shape) != 2 or bool(((ids < 0) | (ids >= shape[0])).any()):
                raise ValueError(f"invalid table rows for {name}")
            row_bytes = shape[1] * torch.empty((), dtype=dtype).element_size()
            start, end = entry["data_offsets"]
            if (
                start < 0
                or end - start != shape[0] * row_bytes
                or 8 + header_size + end > path.stat().st_size
            ):
                raise ValueError(f"invalid safetensors range for {name}")
            unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            payload = bytearray(unique.numel() * row_bytes)
            for index, row in enumerate(unique.tolist()):
                file.seek(8 + header_size + start + row * row_bytes)
                data = file.read(row_bytes)
                if len(data) != row_bytes:
                    raise ValueError(f"truncated safetensors row for {name}")
                payload[index * row_bytes : (index + 1) * row_bytes] = data
        if not ids.numel():
            return torch.empty(*rows.shape, shape[1], dtype=dtype)
        table = torch.frombuffer(payload, dtype=dtype).reshape(-1, shape[1])
        # CPU indexing is unavailable for FP8; index its byte representation.
        return (
            table.view(torch.uint8)
            .reshape(unique.numel(), row_bytes)[inverse]
            .contiguous()
            .view(dtype)
            .reshape(*rows.shape, shape[1])
        )

    def linear_format(self, stem: str) -> str:
        """Return the released projection format without loading its weight payload.

        Checkpoint storage and inference arithmetic differ for WO-A, the gate,
        the LM head, and ratio-two compressors. Values returned by ``linear``
        stay unchanged; the execution precision policy decides when to round.
        """
        name = stem + ".weight"
        shard = self.weight_map[name]
        if shard not in self._headers:
            path = (self.root / shard).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError(f"checkpoint shard leaves its root: {shard}")
            with path.open("rb") as file:
                header_size = struct.unpack("<Q", file.read(8))[0]
                if header_size > 64 * 1024 * 1024:
                    raise ValueError(f"safetensors header is too large: {shard}")
                self._headers[shard] = json.loads(file.read(header_size))
        storage = self._headers[shard][name]["dtype"]
        formats = {"F32": "fp32", "BF16": "bf16", "F8_E4M3": "mxfp8", "I8": "mxfp4"}
        if storage not in formats:
            raise ValueError(f"unsupported checkpoint projection dtype {storage}")
        if stem == "head" or stem.endswith(".ffn.gate"):
            return "fp32"
        if stem.endswith(".attn.wo_a"):
            return "bf16"
        parts = stem.split(".")
        if len(parts) == 5 and parts[0] == "layers" and parts[2:4] == ["attn", "compressor"]:
            if C.FLASH.compress_ratios[int(parts[1])] > 1:
                return "fp32"
        return formats[storage]

    def linear(self, stem: str) -> torch.Tensor:
        weight = self.tensor(stem + ".weight")
        scale = self.tensor(stem + ".scale") if weight.dtype in (torch.int8, torch.float8_e4m3fn) else None
        return decode_checkpoint_linear(weight, scale)

    def expert(self, layer_id: int, expert_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return w1,w3,w2 for one routed expert; retain no all-expert expansion."""
        stem = f"layers.{layer_id}.ffn.experts.{expert_id}"
        return tuple(self.linear(stem + "." + projection) for projection in ("w1", "w3", "w2"))

    def shared_expert(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = f"layers.{layer_id}.ffn.shared_experts"
        return tuple(self.linear(stem + "." + projection) for projection in ("w1", "w3", "w2"))

    def attention(self, layer_id: int) -> dict[str, torch.Tensor | dict[str, str]]:
        """Return an unsharded attention bundle with no activation/cache scales."""
        if not 0 <= layer_id < C.FLASH.num_hidden_layers:
            raise ValueError("attention layer must belong to the text backbone")
        stem = f"layers.{layer_id}.attn."
        result = {name: self.linear(stem + name) for name in ("wq_a", "wq_b", "wkv", "wo_b")}
        result.update(
            q_norm_weight=self.tensor(stem + "q_norm.weight").float(),
            kv_norm_weight=self.tensor(stem + "kv_norm.weight").float(),
            attn_sink=self.tensor(stem + "attn_sink").float(),
            wo_a=self.linear(stem + "wo_a").reshape(C.O_GROUPS, C.O_LORA, C.O_GROUP_IN),
        )
        formats = {name: self.linear_format(stem + name) for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")}
        if layer_id in C.FLASH.kv_source_layer_ids:
            publisher = self.publisher(layer_id)
            formats.update(publisher.pop("formats"))
            result.update(publisher)
            if C.FLASH.compress_ratios[layer_id] == 2:
                result["compressor_wgate"] = self.linear(stem + "compressor.wgate")
                formats["compressor_wgate"] = self.linear_format(stem + "compressor.wgate")
        if layer_id in C.FLASH.index_source_layer_ids:
            result.update(
                index_wq_b=self.linear(stem + "indexer.wq_b"),
                index_weights_proj=self.linear(stem + "indexer.weights_proj"),
            )
            formats.update(
                index_wq_b=self.linear_format(stem + "indexer.wq_b"),
                index_weights_proj=self.linear_format(stem + "indexer.weights_proj"),
            )
        result["formats"] = formats
        return result

    def publisher(self, layer_id: int = 20) -> dict[str, torch.Tensor | dict[str, str]]:
        """Return compressor/index-key matrices of an existing KV source layer."""
        stem = f"layers.{layer_id}.attn."
        return dict(
            compressor_wkv=self.linear(stem + "compressor.wkv"),
            compressor_norm_weight=self.tensor(stem + "compressor.norm.weight").float(),
            index_wk=self.linear(stem + "indexer.wk"),
            index_norm_weight=self.tensor(stem + "indexer.k_norm.weight").float(),
            formats={
                "compressor_wkv": self.linear_format(stem + "compressor.wkv"),
                "index_wk": self.linear_format(stem + "indexer.wk"),
            },
        )

    def engram(self, layer_id: int, hash_ids: torch.Tensor) -> dict[str, torch.Tensor | dict[str, str]]:
        """Decode selected table rows and the projection without BF16 narrowing."""
        if layer_id not in C.FLASH.engram_layer_ids:
            raise ValueError("Engram weights belong to encoder layers 1 and 14")
        stem = f"layers.{layer_id}.engram."
        payload = self.tensor_rows(stem + "embed.weight", hash_ids)
        scale = self.tensor_rows(stem + "embed.scale", hash_ids)
        factors = scale.float().repeat_interleave(32, dim=-1)
        lookup = (payload.float() * factors).flatten(1)
        return {
            "lookup": lookup,
            "wkv": self.linear(stem + "wkv"),
            "weight": (self.tensor(stem + "q_weight").float() * self.tensor(stem + "k_weight").float()),
            "formats": {"wkv": self.linear_format(stem + "wkv")},
        }


class ResidentPrefillWeights:
    """Plan four resident expert shards without expanding quantized weights.

    FP8 matrices use 32-by-32 blocks; FP4 matrices use 32-by-16 byte blocks
    and 32 E8M0 scale bytes. Blocks are ordered by output block, then input
    block. Device decoders recover the original FP32 values before computing.
    Each upload reads only one checkpoint tensor at a time.
    """

    ranks = 4
    local_experts = C.FLASH.n_routed_experts // ranks
    # These columns are the flat tensor ABI shared with prefill_fwd.
    fp8_names = (
        "attn.wq_a",
        "attn.wq_b",
        "attn.wkv",
        "attn.wo_a",
        "attn.wo_b",
        "attn.indexer.wq_b",
        "ffn.shared_experts.w1",
        "ffn.shared_experts.w3",
        "ffn.shared_experts.w2",
        "engram.wkv",
    )
    dense_names = (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm.weight",
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "attn.attn_sink",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "ffn_norm.weight",
        "ffn.gate.weight",
        "ffn.gate.bias",
        "attn.compressor.wkv.weight",
        "attn.compressor.wgate.weight",
        "attn.compressor.norm.weight",
        "attn.indexer.wk.weight",
        "attn.indexer.k_norm.weight",
        "attn.indexer.weights_proj.weight",
        "engram.q_weight",
    )
    matrix_columns = frozenset((0, 7, 11, 13, 14, 16, 18))

    def __init__(self, checkpoint: PrefillCheckpoint):
        self.checkpoint = checkpoint
        self.fp8_offsets = torch.zeros((40, len(self.fp8_names)), dtype=torch.int32)
        self.dense_offsets = torch.zeros((40, len(self.dense_names)), dtype=torch.int32)
        self.expert_offsets = torch.empty((40, self.local_experts, 3), dtype=torch.int32)
        self.fp8_entries, self.dense_entries = [], []
        self.fp8_blocks = self.dense_values = 0
        for layer in range(40):
            for column, suffix in enumerate(self.fp8_names):
                stem = f"layers.{layer}.{suffix}"
                if stem + ".weight" not in checkpoint.weight_map:
                    continue
                header = self._header(stem + ".weight")
                n, k = header["shape"]
                if header["dtype"] != "F8_E4M3" or n % 32 or k % 32:
                    raise ValueError(f"resident FP8 projection has an unsupported layout: {stem}")
                blocks = n * k // 1024
                self.fp8_offsets[layer, column] = self.fp8_blocks
                self.fp8_entries.append((stem, self.fp8_blocks, blocks))
                self.fp8_blocks += blocks
            for column, suffix in enumerate(self.dense_names):
                name = f"layers.{layer}.{suffix}"
                if name not in checkpoint.weight_map:
                    continue
                header = self._header(name)
                if header["dtype"] not in ("F32", "BF16"):
                    raise ValueError(f"resident dense projection is quantized: {name}")
                values = math.prod(header["shape"])
                self.dense_offsets[layer, column] = self.dense_values
                self.dense_entries.append((name, column, self.dense_values, values))
                self.dense_values += values
        expert_blocks = C.D * C.MOE_INTER // 1024
        self.fp4_blocks = 40 * self.local_experts * 3 * expert_blocks
        self.expert_offsets.copy_(
            torch.arange(40 * self.local_experts * 3, dtype=torch.int32).reshape(40, self.local_experts, 3)
            * expert_blocks
        )
        if max(self.fp4_blocks, self.fp8_blocks, self.dense_values) >= 2**31:
            raise ValueError("resident weight offsets exceed the signed INT32 metadata ABI")

    def _header(self, name):
        checkpoint = self.checkpoint
        shard = checkpoint.weight_map[name]
        if shard not in checkpoint._headers:
            path = (checkpoint.root / shard).resolve()
            if not path.is_relative_to(checkpoint.root):
                raise ValueError(f"checkpoint shard leaves its root: {shard}")
            with path.open("rb") as file:
                size = struct.unpack("<Q", file.read(8))[0]
                if size > 64 * 1024 * 1024:
                    raise ValueError(f"safetensors header is too large: {shard}")
                checkpoint._headers[shard] = json.loads(file.read(size))
        return checkpoint._headers[shard][name]

    @property
    def tensor_layouts(self):
        return {
            "fp8_bank": ((self.fp8_blocks, 1024), torch.int8),
            "fp8_scales": ((self.fp8_blocks,), torch.uint8),
            "fp4_bank": ((self.fp4_blocks, 512), torch.int8),
            "fp4_scales": ((self.fp4_blocks, 32), torch.uint8),
            "dense_bank": ((self.dense_values,), torch.float32),
        }

    @property
    def bytes_per_rank(self):
        return sum(
            math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in self.tensor_layouts.values()
        )

    def upload(self, runtime, rank, handles):
        """Fill allocated rank-local banks through bounded host staging tensors."""
        if rank not in range(self.ranks):
            raise ValueError("resident prefill requires four contiguous expert shards")
        checkpoint = self.checkpoint

        def copy(name, data, element_offset):
            data = data.contiguous()
            runtime.copy_to(
                handles[name].data_ptr,
                data.data_ptr(),
                data.numel() * data.element_size(),
                dst_offset=element_offset * data.element_size(),
                worker_id=rank,
            )

        for stem, offset, blocks in self.fp8_entries:
            raw = checkpoint.tensor(stem + ".weight").view(torch.int8)
            n, k = raw.shape
            packed = raw.reshape(n // 32, 32, k // 32, 32).permute(0, 2, 1, 3).contiguous()
            scale = checkpoint.tensor(stem + ".scale").view(torch.uint8)
            if scale.numel() != blocks:
                raise ValueError(f"FP8 scale count disagrees with resident blocks: {stem}")
            copy("fp8_bank", packed, offset * 1024)
            copy("fp8_scales", scale, offset)
        for name, column, offset, _values in self.dense_entries:
            value = checkpoint.tensor(name).float()
            if column == 19:
                value = value * checkpoint.tensor(name.replace("q_weight", "k_weight")).float()
            elif column in self.matrix_columns:
                value = value.t()
            copy("dense_bank", value, offset)
        for layer in range(40):
            for local in range(self.local_experts):
                expert = rank * self.local_experts + local
                stem = f"layers.{layer}.ffn.experts.{expert}"
                for column, projection in enumerate(("w1", "w3", "w2")):
                    raw = checkpoint.tensor(f"{stem}.{projection}.weight")
                    scale = checkpoint.tensor(f"{stem}.{projection}.scale").view(torch.uint8)
                    if raw.dtype != torch.int8 or raw.shape[0] % 32 or raw.shape[1] % 16:
                        raise ValueError(f"resident expert is not block-32 FP4: {stem}.{projection}")
                    n, packed_k = raw.shape
                    offset = int(self.expert_offsets[layer, local, column])
                    packed = raw.reshape(n // 32, 32, packed_k // 16, 16).permute(0, 2, 1, 3).contiguous()
                    scales = scale.reshape(n // 32, 32, packed_k // 16).permute(0, 2, 1).contiguous()
                    copy("fp4_bank", packed, offset * 512)
                    copy("fp4_scales", scales, offset * 32)


GLOBAL_L2_LIMIT = 0.02


WORST_ROW_L2_LIMIT = 0.05


def metrics(actual, reference):
    a, r = actual.double().flatten(1), reference.double().flatten(1)
    delta = a - r
    finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    return {
        "relative_l2": float(delta.norm() / r.norm().clamp_min(1e-30)),
        "worst_row_l2": float((delta.norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)).max()),
        "finite": finite,
    }


def checkpoint_inputs(checkpoint, lengths, prompts=None):
    """Build token IDs and Engram hashes using the original tokenizer and hash code."""
    from tokenizers import Tokenizer

    root = checkpoint.root
    spec = importlib.util.spec_from_file_location("ced_official_engram", root / "inference/engram.py")
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    backend = Tokenizer.from_file(str(root / "tokenizer.json"))

    class TokenizerView:
        backend_tokenizer = backend

        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    args = SimpleNamespace(**json.loads((root / "inference/config.json").read_text()))
    args.max_batch_size, args.max_seq_len = len(lengths), max(lengths)
    hash_state = official.NgramHashState(args, official.EngramLayout.from_args(args), TokenizerView())
    prompts = prompts or [
        "The encoder computes prompt representations and the decoder uses a bounded causal replay window. ",
        "Independent requests keep separate causal caches and token positions. ",
    ]
    sequences = []
    for request, length in enumerate(lengths):
        tokens = backend.encode(prompts[request % len(prompts)]).ids
        if not tokens:
            raise ValueError("prompt must encode at least one token")
        sequences.append(torch.tensor((tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]))
    padded = torch.full((len(lengths), max(lengths)), FLASH.engram_pad_token_id, dtype=torch.long)
    for request, sequence in enumerate(sequences):
        padded[request, : sequence.numel()] = sequence
    batched = hash_state(padded, 0)
    hashes = torch.cat([batched[request, :length] for request, length in enumerate(lengths)])
    return sequences, {layer: hashes[:, index] for index, layer in enumerate(FLASH.engram_layer_ids)}


def build_reference(checkpoint, context, sequences, hashes):
    """Evolve a separate CPU state; never supply its activations or routes to L2."""
    state = context.initial_state(sequences)
    result = {
        "routes": [],
        "precision": context.precision,
        "lengths": context.lengths,
        "sequences": sequences,
    }
    for layer in range(40):
        if layer == 20:
            result["encoder"] = state
            context.publish_decoder(state.x_hc, state.pre_mix)
            state = PrefillState(*context.take_decoder_rows(state.x_hc, state.pre_mix))
        state = context.block_reference(layer, state, hash_ids=hashes.get(layer))
        result["routes"].append(context.last_routes.clone())
        print("CED_REFERENCE_LAYER", layer, flush=True)
    result["decoder"] = state
    collapsed = round_activation(hc_pre(state.x_hc, state.pre_mix), context.precision)
    normalized = round_activation(
        rms_norm(collapsed, checkpoint.tensor("norm.weight").float()), context.precision
    )
    result["normalized"] = normalized
    rows = context.replay.query_start_loc[1:].long() - 1
    head = checkpoint.tensor("head.weight")
    result["logits"] = torch.cat(
        [
            reference_linear(normalized[rows], head[column : column + 4096].float())
            for column in range(0, head.shape[0], 4096)
        ],
        dim=-1,
    )
    for name in ("window", "global_cache", "index_cache", "state_cache"):
        result[name] = getattr(context, name)
    return result


def resident_inputs(checkpoint, context, sequences, hashes, weights):
    """Prepare immutable metadata and selected embedding/Engram rows once."""
    enc0 = context.layer_inputs(0, load_weights=False).metadata
    enc = context.layer_inputs(2, load_weights=False)
    dec = context.layer_inputs(20, load_weights=False)
    pub = context.publisher_inputs(load_weights=False).metadata
    enc_meta, dec_meta = enc.metadata, dec.metadata
    state_slots = torch.full((context.num_tokens,), -1, dtype=torch.int32)
    for request, length in enumerate(context.lengths):
        start = int(enc_meta.query_start_loc[request])
        end = start + length
        for row in range(max(start, end - C.STATE_CAPACITY), end):
            state_slots[row] = request * C.STATE_CAPACITY + int(enc_meta.position_ids[row]) % C.STATE_CAPACITY
    lookups = []
    for layer in FLASH.engram_layer_ids:
        stem = f"layers.{layer}.engram.embed."
        payload = checkpoint.tensor_rows(stem + "weight", hashes[layer])
        scale = checkpoint.tensor_rows(stem + "scale", hashes[layer])
        lookups.append((payload.float() * scale.float().repeat_interleave(32, -1)).flatten(1))
    global_blocks = max(value.shape[0] for value in context.global_cache.values())
    sources = list(FLASH.kv_source_layer_ids)
    values = {
        "embedding": checkpoint.tensor_rows("embed.weight", torch.cat(sequences)).float(),
        "engram_lookup": torch.stack(lookups),
        "offsets8": weights.fp8_offsets,
        "expert_offsets": weights.expert_offsets,
        "offsets": weights.dense_offsets,
        "encoder_cos": torch.stack((enc0.rope_cos, enc_meta.rope_cos)),
        "encoder_sin": torch.stack((enc0.rope_sin, enc_meta.rope_sin)),
        "compressed_cos": enc_meta.compressed_rope_cos,
        "compressed_sin": enc_meta.compressed_rope_sin,
        "decoder_cos": torch.stack((dec_meta.rope_cos, dec_meta.rope_cos)),
        "decoder_sin": torch.stack((dec_meta.rope_sin, dec_meta.rope_sin)),
        "encoder_slots": enc_meta.window_slots.int(),
        "encoder_indices": enc_meta.window_indices.int(),
        "compressed_slots": enc_meta.compressed_slots.int(),
        "previous_rows": enc.previous_rows,
        "encoder_blocks": enc_meta.index_block_table[enc_meta.request_ids.long()],
        "encoder_lengths": enc_meta.compressed_lens.int(),
        "encoder_extents": torch.stack((enc0.attention_extents, enc_meta.attention_extents)),
        "state_slots": state_slots,
        "publisher_slots": pub.compressed_slots.int(),
        "decoder_slots": dec_meta.window_slots.int(),
        "decoder_indices": dec_meta.window_indices.int(),
        "decoder_blocks": dec_meta.index_block_table[dec_meta.request_ids.long()],
        "decoder_lengths": dec_meta.compressed_lens.int(),
        "decoder_extents": torch.stack((dec_meta.attention_extents, dec_meta.attention_extents)),
        "replay_rows": context.decoder_rows.int(),
        "last_rows": (context.replay.query_start_loc[1:] - 1).int(),
        "window_cache": torch.stack(list(context.window.values())),
        "global_cache": torch.zeros(len(sources), global_blocks, 128, C.HEAD_DIM),
        "index_cache": torch.zeros(len(sources), global_blocks, 128, C.INDEX_DIM),
        "state_cache": torch.stack(list(context.state_cache.values())),
        "final_norm": checkpoint.tensor("norm.weight").float(),
        "head": checkpoint.tensor("head.weight").float().t().contiguous(),
    }
    return {name: value.contiguous() for name, value in values.items()}


def run_resident(checkpoint, context, sequences, hashes, device_ids):
    """Upload original weights once, then execute exactly one distributed call."""

    weights = ResidentPrefillWeights(checkpoint)
    config = RunConfig(
        platform="a5",
        ring_heap=(2 * 1024**3,) * 4,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
    )
    compiled = l3_prefill_fwd.compile(config=config)
    inputs = resident_inputs(checkpoint, context, sequences, hashes, weights)
    enc, dec, requests = context.num_tokens, context.decoder_num_tokens, len(context.lengths)
    outputs = {
        "encoder_out": ((enc, C.HC_MULT, C.D), torch.float32),
        "encoder_pre": ((enc, C.HC_MULT), torch.float32),
        "decoder_out": ((dec, C.HC_MULT, C.D), torch.float32),
        "decoder_pre": ((dec, C.HC_MULT), torch.float32),
        "encoder_routes": ((20, enc, FLASH.num_experts_per_tok), torch.int32),
        "decoder_routes": ((20, dec, FLASH.num_experts_per_tok), torch.int32),
        "normalized": ((dec, C.D), torch.float32),
        "logits": ((requests, FLASH.vocab_size), torch.float32),
    }
    banks = {
        "bank8": "fp8_bank",
        "scales8": "fp8_scales",
        "bank4": "fp4_bank",
        "scales4": "fp4_scales",
        "dense": "dense_bank",
    }
    actual = {}
    print("CED_RESIDENT_BYTES_PER_RANK", weights.bytes_per_rank, flush=True)
    with compiled.prepare(config) as runtime:
        per_rank = [{} for _ in device_ids]
        for rank, tensors in enumerate(per_rank):
            for name in PREFILL_TENSORS:
                if name in banks:
                    shape, dtype = weights.tensor_layouts[banks[name]]
                    tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                elif name in outputs:
                    shape, dtype = outputs[name]
                    tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                else:
                    value = inputs[name]
                    tensors[name] = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)

        def upload(rank):
            weights.upload(runtime, rank, {target: per_rank[rank][name] for name, target in banks.items()})
            print("CED_RANK_WEIGHTS_READY", rank, flush=True)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(upload, range(4)))
        stacked = {
            name: StackedDeviceTensor(
                [rank[name] for rank in per_rank], (4, *per_rank[0][name].shape), tuple(range(4))
            )
            for name in PREFILL_TENSORS
        }
        print("CED_DEVICE_START", flush=True)
        runtime(
            *[stacked[name] for name in PREFILL_TENSORS],
            int(context.precision == "official"),
            enc,
            config=config,
        )
        print("CED_DEVICE_COMPLETE", flush=True)
        for name in (*outputs, "window_cache", "global_cache", "index_cache", "state_cache"):
            first = per_rank[0][name]
            value = torch.empty((4, *first.shape), dtype=first.dtype)
            runtime.copy_stacked_from(stacked[name], value)
            actual[name] = value
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lengths", default="129,7")
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--devices", default=os.environ.get("TASK_DEVICE"))
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--precision", choices=("fp32", "official"), default="fp32")
    options = parser.parse_args(argv)
    device_ids = [int(value) for value in (options.devices or "").split(",") if value]
    if len(device_ids) != 4 or len(set(device_ids)) != 4:
        raise ValueError("provide exactly four distinct task-allocated A5 devices")
    torch.set_num_threads(8)
    checkpoint = PrefillCheckpoint(options.checkpoint)
    lengths = [int(n) for n in options.lengths.split(",")]
    context = PrefillContext(checkpoint, lengths, options.capacity, precision=options.precision)
    reference_context = PrefillContext(checkpoint, lengths, options.capacity, precision=options.precision)
    sequences, hashes = checkpoint_inputs(checkpoint, lengths, options.prompt)
    start = time.monotonic()
    reference = build_reference(checkpoint, reference_context, sequences, hashes)
    actual = run_resident(checkpoint, context, sequences, hashes, device_ids)
    failures = []
    report = {
        "precision": options.precision,
        "reference_accumulation": "fp64",
        "platform": "a5",
        "device_count": 4,
        "host_invocations": 1,
        "l2_invocations_per_rank": 1,
        "full_40_layers": True,
        "lengths": lengths,
        "global_l2_limit": GLOBAL_L2_LIMIT,
        "worst_row_l2_limit": WORST_ROW_L2_LIMIT,
    }

    def record(name, device, expected):
        result = metrics(device, expected)
        if (
            not result["finite"]
            or result["relative_l2"] > GLOBAL_L2_LIMIT
            or result["worst_row_l2"] > WORST_ROW_L2_LIMIT
        ):
            failures.append(name)
        print("CED_METRIC", name, json.dumps(result), flush=True)
        return result

    for name, value in actual.items():
        if any(not torch.equal(value[0], value[rank]) for rank in range(1, 4)):
            failures.append("rank_consistency_" + name)
    report["ranks_identical"] = not failures
    for phase in ("encoder", "decoder"):
        report[phase] = record(phase, actual[phase + "_out"][0], reference[phase].x_hc)
        report[phase]["pre_mix"] = record(
            phase + "_pre_mix", actual[phase + "_pre"][0], reference[phase].pre_mix
        )
    report["final_norm"] = record("final_norm", actual["normalized"][0], reference["normalized"])
    report["last_token_logits"] = record("last_token_logits", actual["logits"][0], reference["logits"])
    report["top1_matches"] = (actual["logits"][0].argmax(-1) == reference["logits"].argmax(-1)).tolist()
    routes = [*actual["encoder_routes"][0], *actual["decoder_routes"][0]]
    report["changed_expert_rows"] = [
        int((a.sort(-1).values != r.sort(-1).values).any(-1).sum())
        for a, r in zip(routes, reference["routes"])
    ]
    report["caches"] = {}
    for name, expected_name in (
        ("window_cache", "window"),
        ("global_cache", "global_cache"),
        ("index_cache", "index_cache"),
        ("state_cache", "state_cache"),
    ):
        device_rows, reference_rows = [], []
        for slot, expected in enumerate(reference[expected_name].values()):
            device_rows.append(actual[name][0, slot, : expected.shape[0]].reshape(-1, expected.shape[-1]))
            reference_rows.append(expected.reshape(-1, expected.shape[-1]))
        report["caches"][name] = record(name, torch.cat(device_rows), torch.cat(reference_rows))
    for layer in range(40):
        if bool(actual["window_cache"][:, layer, -1].any()):
            failures.append(f"window_sentinel_{layer}")
    for name in ("global_cache", "index_cache", "state_cache"):
        for slot, source in enumerate(reference[name]):
            used = reference[name][source].shape[0] - 1
            if bool(actual[name][:, slot, used:].any()):
                failures.append(f"{name}_sentinel_{source}")
    report["seconds"] = time.monotonic() - start
    report["failures"] = failures
    report["passed"] = not failures
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report, indent=2) + "\n")
    print("CED_FULL_40_COMPLETE", json.dumps(report), flush=True)
    if failures:
        raise AssertionError("CED resident prefill validation failed: " + ", ".join(failures))
    return report


_SCRIPT_ENTRY_POINT = "__" + "main__"


if __name__ == _SCRIPT_ENTRY_POINT:
    main()
