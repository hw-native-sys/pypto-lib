# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode reuse attention: projection over a source layer's paged rows and selection.

This file owns the shared reuse chain (`decode_reuse_attention`, `decode_reuse`),
its Torch golden and fixture, and the decode fixture helpers `_rope_tables` and
`_weights`/`_history`/`_sample_topk`.  `decode_c1a_reuse.py` is a thin ratio-1
entry over this chain; the other decode mode files import the fixture helpers
without duplicating them.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from attention import attention_core
from config import (
    BLOCK_SIZE, CMP_BLOCKS_DYN, DECODE_MAX_TOKENS, FLASH as M, LOCAL_HEADS, ORI_BLOCKS_DYN,
    SELECTED_KEYS, STATE_CAPACITY, T_DYN, TP_SIZE, WINDOW_KEYS,
)
from hc import hc_post, hc_pre
from indexer import IDX_DIM, IDX_HEADS
from mla_proj import mla_proj
from o_proj import o_proj
from paged_cache import gather_selected_keys, publish_cache_row, snapshot_rows
from rmsnorm import rms_norm
from tp_reduce import tp_output_reduce


# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
Q_LORA = M.q_lora_rank
LOCAL_Q = LOCAL_HEADS * HEAD_DIM
O_GROUPS = M.o_groups // (M.num_attention_heads // LOCAL_HEADS)
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
O_LORA = M.o_lora_rank
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
MIX_PAD = 32  # hc_mixes reads the padded [MIX_PAD, HC_DIM] coefficient table
HC_DIM = M.hc_dim
INDEX_TOPK = M.index_topk
LOCAL_IDX = IDX_HEADS * IDX_DIM

# tiling
QUERY_CHUNK = 16


@pl.jit.inline(auto_scope=False)
def decode_reuse_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """One TP rank's decode attention over the source layer's selected rows."""
    tokens = pl.tensor.dim(x, 0)
    q = pl.create_tensor([tokens, LOCAL_HEADS, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([tokens, 1], dtype=pl.FP32)
    mla_proj(
        x, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        rope_cos, rope_sin, q, kv, qr, qr_scale,
    )
    q_rows = pl.reshape(q, [tokens * LOCAL_HEADS, HEAD_DIM])
    q_flat, q_tid = snapshot_rows(q_rows, width=HEAD_DIM)
    kv_snapshot, kv_tid = snapshot_rows(kv, width=HEAD_DIM)
    window_ready = pl.system.task_dummy(deps=[kv_tid])
    window_cache_out, window_tid = publish_cache_row(
        kv_snapshot, window_slots, window_cache, num_tokens, width=HEAD_DIM, ready=window_ready,
    )

    attended = pl.create_tensor([tokens, LOCAL_Q], dtype=pl.BF16)
    gather_ready = pl.system.task_dummy(deps=[window_tid, q_tid])
    for begin in pl.range(0, num_tokens, QUERY_CHUNK):
        active = pl.min(QUERY_CHUNK, num_tokens - begin)
        window_chunk = pl.slice(window_indices, [active, WINDOW_KEYS], [begin, 0])
        topk_chunk = pl.slice(topk_indices, [active, INDEX_TOPK], [begin, 0])
        selected = pl.create_tensor([active, SELECTED_KEYS, HEAD_DIM], dtype=pl.BF16)
        combined = pl.create_tensor([active, SELECTED_KEYS], dtype=pl.INT32)
        selected_gathered, combined_gathered, gathered = gather_selected_keys(
            window_cache, window_chunk, compressed_cache, topk_chunk,
            selected, combined, active, gather_ready,
        )
        q_chunk = pl.slice(q_flat, [active * LOCAL_HEADS, HEAD_DIM], [begin * LOCAL_HEADS, 0])
        attended_chunk = pl.create_tensor([active, LOCAL_Q], dtype=pl.BF16)
        attended_gathered = attention_core(
            q_chunk, selected_gathered, combined_gathered, sink, attended_chunk,
            keys=SELECTED_KEYS, ready=gathered,
        )
        attended[begin : begin + active, 0:LOCAL_Q] = attended_gathered
    partial = o_proj(
        attended, wo_a, wo_b, rope_cos, rope_sin, partial, groups=O_GROUPS, group_in=O_GROUP_IN, o_lora=O_LORA,
    )
    return partial


@pl.jit.inline(auto_scope=False)
def decode_reuse(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Layer-level reuse block: HC pre, norm, attention, TP reduce, HC post."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hidden = hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        hidden, next_pre_mix, post_mix, residual_mix,
    )
    attn_input = rms_norm(hidden, attn_norm_weight, attn_input, width=D, output_fp32=False, column_tile=256)
    partial_in = pl.create_tensor([tokens, D], dtype=pl.BF16)
    partial_out = decode_reuse_attention(
        attn_input, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, topk_indices, partial_in, num_tokens,
    )
    attn_out = tp_output_reduce(partial_out, output_window, output_arrived, attn_out, group_base, tp_rank, tokens,
                     attention_epoch)
    output = hc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output, attn_input


@pl.jit
def decode_reuse_attention_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    partial: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Bounded reuse entry (one card, pre-TP-reduce output)."""
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    partial.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    decode_reuse_attention(
        x, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, topk_indices, partial, pl.tensor.dim(x, 0),
    )
    return partial


# ---------------------------------------------------------------------------
# Layer-level host-compile entry (compile-only; the fwd driver owns device runs)
# ---------------------------------------------------------------------------

LAYER_INPUT_NAMES = (
    "wq_a",
    "wq_a_scale",
    "q_norm_w",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "wkv_scale",
    "kv_norm_w",
    "sink",
    "wo_a",
    "wo_b",
    "rope_cos",
    "rope_sin",
    "window_slots",
    "window_indices",
    "window_cache",
    "compressed_cache",
    "topk_indices",
)


@pl.jit
def decode_reuse_layer_rank(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[DECODE_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run one TP rank of the decode_reuse layer chain (host-compile harness)."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    hidden.bind_dynamic(0, T_DYN)
    attn_input.bind_dynamic(0, T_DYN)
    attn_out.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    next_pre_mix.bind_dynamic(0, T_DYN)
    output_out, attn_input_out = decode_reuse(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w, sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, compressed_cache, topk_indices,
        output_window, output_arrived, hidden, attn_input, attn_out, output, next_pre_mix,
        0, 0, num_tokens, 1,
    )
    return output_out, attn_input_out


@pl.jit.host
def decode_reuse_layer_entry(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_Q], pl.INT8],
    wq_b_scale: pl.Tensor[[LOCAL_Q], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    sink: pl.Tensor[[LOCAL_HEADS], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Allocate one TP window and host-compile the decode_reuse layer chain."""
    data_buf = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.BF16)
    signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    output_window = pld.window(data_buf, [DECODE_MAX_TOKENS, D], dtype=pl.BF16)
    output_arrived = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
    decode_reuse_layer_rank(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w, sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, compressed_cache, topk_indices,
        output_window, output_arrived, hidden, attn_input, attn_out, output, next_pre_mix,
        num_tokens, device=0,
    )


def build_layer_specs(seed: int = 0):
    from golden import ScalarSpec, TensorSpec

    base = {spec.name: spec for spec in build_reuse_specs(seed=seed)}
    tokens = base["x"].shape[0]
    generator = torch.Generator().manual_seed(seed + 100)
    specs = [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32,
                   init_value=lambda: torch.randn(tokens, HC_MULT, D, generator=generator)),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32,
                   init_value=lambda: torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))),
        TensorSpec("hc_attn_fn", [MIX_PAD, HC_DIM], torch.float32,
                   init_value=lambda: torch.randn(MIX_PAD, HC_DIM, generator=generator) / HC_DIM**0.5),
        TensorSpec("hc_attn_scale", [3], torch.float32,
                   init_value=lambda: torch.randn(3, generator=generator)),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32,
                   init_value=lambda: torch.randn(MIX_HC, generator=generator)),
        TensorSpec("attn_norm_weight", [D], torch.bfloat16,
                   init_value=lambda: torch.randn(D, generator=generator) * 0.1 + 1.0),
    ]
    specs.extend(base[name] for name in LAYER_INPUT_NAMES)
    specs.extend([
        TensorSpec("hidden", [tokens, D], torch.bfloat16),
        TensorSpec("attn_input", [tokens, D], torch.bfloat16),
        TensorSpec("attn_out", [tokens, D], torch.bfloat16),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
        TensorSpec("next_pre_mix", [tokens, HC_MULT], torch.float32),
        ScalarSpec("num_tokens", torch.int32, tokens),
    ])
    return specs

# ---------------------------------------------------------------------------
# Bounded validation entry (one card, pre-TP-reduce output)
# ---------------------------------------------------------------------------


def _rope_tables(count: int, start: int = 0):
    half = 32
    inv_freq = 1.0 / torch.pow(
        torch.tensor(M.rope_theta, dtype=torch.float32),
        torch.arange(0, half, dtype=torch.float32) * 2.0 / (2 * half),
    )
    angle = torch.arange(start, start + count, dtype=torch.float32).reshape(-1, 1) * inv_freq.reshape(1, -1)
    return torch.cos(angle), torch.sin(angle)


def _weights():
    """Layer weights shared by every decode fixture."""
    from utils import pack_nz, quant_w_per_channel

    def quantized(rows, cols):
        return pack_nz(quant_w_per_channel(torch.randn(rows, cols) * 0.05)[0].transpose(0, 1).contiguous())

    def scale_of(rows, cols):
        return quant_w_per_channel(torch.randn(rows, cols) * 0.05)[1]

    to_bf16 = lambda value: value.to(torch.bfloat16)  # noqa: E731 - fixture-local cast
    return {
        "wq_a": quantized(Q_LORA, D),
        "wq_a_scale": scale_of(Q_LORA, D),
        "q_norm_w": to_bf16(torch.randn(Q_LORA) * 0.1 + 1.0),
        "wq_b": quantized(LOCAL_Q, Q_LORA),
        "wq_b_scale": scale_of(LOCAL_Q, Q_LORA),
        "wkv": quantized(HEAD_DIM, D),
        "wkv_scale": scale_of(HEAD_DIM, D),
        "kv_norm_w": to_bf16(torch.randn(HEAD_DIM) * 0.1 + 1.0),
        "comp_wkv": to_bf16(torch.randn(HEAD_DIM, D) * 0.02),
        "comp_wgate": to_bf16(torch.randn(HEAD_DIM, D) * 0.02),
        "comp_norm": to_bf16(torch.randn(HEAD_DIM) * 0.05 + 1.0),
        "idx_wq_b": quantized(LOCAL_IDX, Q_LORA),
        "idx_wq_b_scale": scale_of(LOCAL_IDX, Q_LORA),
        "idx_weights_proj": to_bf16(torch.randn(D, IDX_HEADS) * 0.02),
        "idx_wk": to_bf16(torch.randn(IDX_DIM, HEAD_DIM) * 0.02),
        "idx_k_norm": to_bf16(torch.randn(IDX_DIM) * 0.05 + 1.0),
        "wo_a": to_bf16(torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) * 0.02),
        "wo_b": to_bf16(torch.randn(D, O_GROUPS * O_LORA) * 0.02),
        "sink": torch.randn(LOCAL_HEADS) * 0.5,
    }


def _history(weights, plan, x, positions, request_id, ratio, cos, sin, caches, state_cache, state_block_table):
    """Publish child-golden prefill rows for one request's preceding chunk.

    Computes the MLA KV row, the source-layer compressed latent and the index
    key of every supplied position, then scatters them into the supplied caches
    at the physical rows of ``plan``.  ``ratio`` selects the C1A (1) or C2A (2)
    source compressor; for C2A the carried state evolves in ``state_cache``.
    """
    from compressor import golden_c1a_compressor, golden_c2a_compressor
    from indexer import golden_index_keys
    from mla_proj import _golden_mla_proj
    from utils import compressed_metadata, scatter_cache_rows, window_metadata

    count = int(positions.numel())
    mla = {
        "x": x, "wq_a": weights["wq_a"], "wq_a_scale": weights["wq_a_scale"], "q_norm_w": weights["q_norm_w"],
        "wq_b": weights["wq_b"], "wq_b_scale": weights["wq_b_scale"], "wkv": weights["wkv"],
        "wkv_scale": weights["wkv_scale"], "kv_norm_w": weights["kv_norm_w"],
        "rope_cos": cos[positions], "rope_sin": sin[positions],
        "q": torch.zeros(count, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        "kv": torch.zeros(count, HEAD_DIM, dtype=torch.bfloat16),
        "qr": torch.zeros(count, Q_LORA, dtype=torch.int8),
        "qr_scale": torch.zeros(count, 1, dtype=torch.float32),
    }
    _golden_mla_proj(mla)
    compressed = {
        "x": x, "norm_w": weights["comp_norm"], "wkv": weights["comp_wkv"],
        "output": torch.zeros(count, HEAD_DIM, dtype=torch.bfloat16),
    }
    if ratio == 1:
        golden_c1a_compressor(compressed)
        compressed_positions = positions
    else:
        compressed.update({
            "wgate": weights["comp_wgate"], "position_ids": positions,
            "request_ids": torch.full((count,), request_id, dtype=torch.int32),
            "state_block_table": state_block_table, "state_cache": state_cache,
        })
        golden_c2a_compressor(compressed)
        compressed_positions = torch.div(positions, ratio, rounding_mode="floor")
    requests = torch.full((count,), request_id, dtype=torch.int32)
    table = plan["index_block_table"]
    window_slots, _, _ = window_metadata(positions, requests, table, M.sliding_window)
    if caches.get("window") is not None:
        scatter_cache_rows(caches["window"], window_slots, mla["kv"])
    if caches.get("compressed") is not None or caches.get("index") is not None:
        compressed_slots, _ = compressed_metadata(positions, requests, table, ratio)
        if caches.get("compressed") is not None:
            scatter_cache_rows(caches["compressed"], compressed_slots, compressed["output"])
        if caches.get("index") is not None:
            keys = golden_index_keys({
                "latent": compressed["output"], "wk": weights["idx_wk"], "k_norm": weights["idx_k_norm"],
                "compressed_cos": cos[compressed_positions], "compressed_sin": sin[compressed_positions],
            })
            scatter_cache_rows(caches["index"], compressed_slots, keys)


def _sample_topk(plan, positions, request_ids, ratio, seed=0):
    """Sample up to 512 visible compressed rows per token and map them to physical rows."""
    table = plan["index_block_table"]
    generator = torch.Generator().manual_seed(seed)
    tokens = int(positions.numel())
    topk = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    for token in range(tokens):
        visible = min(INDEX_TOPK, (int(positions[token]) + 1) // ratio)
        if visible <= 0:
            continue
        logical = torch.randperm((int(positions[token]) + 1) // ratio, generator=generator)[:visible]
        page = torch.div(logical, BLOCK_SIZE, rounding_mode="floor")
        offset = logical % BLOCK_SIZE
        physical = table[int(request_ids[token]), page].to(torch.long)
        topk[token, :visible] = (physical * BLOCK_SIZE + offset).to(torch.int32)
    return topk


def build_reuse_specs(ratio: int = 2, handoff: bool = False, requests=None, seed: int = 0):
    """Paged two-request reuse fixture with source rows published by the source layer."""
    from golden import TensorSpec
    from utils import build_request_plan

    requests = requests or [(1250, 1), (1253, 1)]
    plan = build_request_plan(requests, ratio=ratio)
    pages = plan["pages"]
    positions = plan["position_ids"]
    request_ids = plan["request_ids"]
    tokens = int(positions.numel())
    max_position = int(positions.max())
    generator = torch.Generator().manual_seed(seed)
    weights = _weights()
    cos, sin = _rope_tables(max_position + 2)
    window_cache = torch.randn(pages, BLOCK_SIZE, 1, HEAD_DIM, generator=generator).to(torch.bfloat16) * 0.1
    compressed_cache = torch.randn(pages, BLOCK_SIZE, 1, HEAD_DIM, generator=generator).to(torch.bfloat16) * 0.1
    state_cache = torch.randn(plan["state_block_table"].shape[0], STATE_CAPACITY, 2 * HEAD_DIM,
                              generator=generator)
    x = torch.randn(tokens, D, generator=generator) * 0.1
    cache_rows = {"window": window_cache, "compressed": compressed_cache}

    # The source layer publishes its compressed rows for the active tokens before
    # this layer runs.  With ``handoff`` a preceding chunk also seeds the caches
    # and the source compressor state, so this call continues that chunk.
    row = 0
    for request, (start, length) in enumerate(requests):
        if handoff:
            history_positions = torch.arange(start - 6, start, dtype=torch.int32)
            history_x = torch.randn(6, D, generator=generator) * 0.1
            _history(weights, plan, history_x, history_positions, request, ratio, cos, sin,
                     cache_rows, state_cache, plan["state_block_table"])
        _history(weights, plan, x[row : row + length], torch.arange(start, start + length, dtype=torch.int32),
                 request, ratio, cos, sin, {"compressed": compressed_cache}, state_cache,
                 plan["state_block_table"])
        row += length

    values = dict(weights)
    values.update({"x": x, "rope_cos": cos[positions], "rope_sin": sin[positions]})
    topk = _sample_topk(plan, positions, request_ids, ratio, seed)
    specs = [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=values["x"]),
        TensorSpec("wq_a", [D, Q_LORA], torch.int8, init_value=values["wq_a"]),
        TensorSpec("wq_a_scale", [Q_LORA], torch.float32, init_value=values["wq_a_scale"]),
        TensorSpec("q_norm_w", [Q_LORA], torch.bfloat16, init_value=values["q_norm_w"]),
        TensorSpec("wq_b", [Q_LORA, LOCAL_Q], torch.int8, init_value=values["wq_b"]),
        TensorSpec("wq_b_scale", [LOCAL_Q], torch.float32, init_value=values["wq_b_scale"]),
        TensorSpec("wkv", [D, HEAD_DIM], torch.int8, init_value=values["wkv"]),
        TensorSpec("wkv_scale", [HEAD_DIM], torch.float32, init_value=values["wkv_scale"]),
        TensorSpec("kv_norm_w", [HEAD_DIM], torch.bfloat16, init_value=values["kv_norm_w"]),
        TensorSpec("sink", [LOCAL_HEADS], torch.float32, init_value=values["sink"]),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=values["wo_a"]),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.bfloat16, init_value=values["wo_b"]),
        TensorSpec("rope_cos", [tokens, 32], torch.float32, init_value=values["rope_cos"]),
        TensorSpec("rope_sin", [tokens, 32], torch.float32, init_value=values["rope_sin"]),
        TensorSpec("window_slots", [tokens], torch.int64, init_value=plan["window_slots"]),
        TensorSpec("window_indices", [tokens, WINDOW_KEYS], torch.int32, init_value=plan["window_indices"]),
        TensorSpec("window_cache", [pages, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=window_cache),
        TensorSpec("compressed_cache", [pages, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=compressed_cache),
        TensorSpec("topk_indices", [tokens, INDEX_TOPK], torch.int32, init_value=topk),
        TensorSpec("partial", [tokens, D], torch.bfloat16),
    ]
    return specs


def golden_decode_reuse(tensors):
    """Compose the per-operator references over the paged fixtures."""
    from attention import golden_attention_core
    from mla_proj import _golden_mla_proj
    from o_proj import golden_o_proj
    from paged_cache import golden_gather_selected_keys
    from utils import scatter_cache_rows

    tokens = tensors["x"].shape[0]
    mla = {name: tensors[name] for name in (
        "x", "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
        "kv_norm_w", "rope_cos", "rope_sin")}
    mla["q"] = torch.zeros(tokens, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    mla["kv"] = torch.zeros(tokens, HEAD_DIM, dtype=torch.bfloat16)
    mla["qr"] = torch.zeros(tokens, Q_LORA, dtype=torch.int8)
    mla["qr_scale"] = torch.zeros(tokens, 1, dtype=torch.float32)
    _golden_mla_proj(mla)
    scatter_cache_rows(tensors["window_cache"], tensors["window_slots"], mla["kv"].reshape(-1, HEAD_DIM))

    gather = {
        "window_cache": tensors["window_cache"], "window_indices": tensors["window_indices"],
        "compressed_cache": tensors["compressed_cache"], "topk_indices": tensors["topk_indices"],
        "selected": torch.zeros(tokens, SELECTED_KEYS, HEAD_DIM, dtype=torch.bfloat16),
        "combined": torch.zeros(tokens, SELECTED_KEYS, dtype=torch.int32),
    }
    golden_gather_selected_keys(gather)
    attention = {
        "q": mla["q"], "selected": gather["selected"], "combined": gather["combined"],
        "sink": tensors["sink"],
        "output": torch.zeros(tokens, LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16),
    }
    golden_attention_core(attention)
    out = {
        "attended": attention["output"].reshape(tokens, LOCAL_Q).float(),
        "wo_a": tensors["wo_a"], "wo_b": tensors["wo_b"],
        "output": torch.zeros(tokens, D, dtype=torch.bfloat16),
    }
    out["cos"] = tensors["rope_cos"]
    out["sin"] = tensors["rope_sin"]
    golden_o_proj(out)
    tensors["partial"][:] = out["output"]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.compile_only:
        config = dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                      device_id=args.device)
        compile_cases = (
            ("operator chain", decode_reuse_attention_test, build_reuse_specs(2, handoff=False, seed=args.seed)),
            ("layer chain", decode_reuse_layer_entry, build_layer_specs(seed=args.seed)),
        )
        for label, fn, specs in compile_cases:
            print(f"[DECODE] host compile: {label}", flush=True)
            result = run(fn=fn, specs=specs, golden_fn=None, config=config, compare_fn={},
                         compile_only=True)
            print(result)
            if not result.passed:
                raise SystemExit(result.error or 1)
        raise SystemExit(0)
    compare = {
        "partial": ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=1e-2),
        "window_cache": ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2),
    }
    cases = [
        ("reuse", build_reuse_specs(2, handoff=False, seed=args.seed)),
        ("reuse_handoff", build_reuse_specs(2, handoff=True, requests=[(1240, 9), (1252, 9)], seed=args.seed)),
    ]
    failures = []
    for name, specs in cases:
        print(f"[DECODE] {name}", flush=True)
        result = run(
            fn=decode_reuse_attention_test,
            specs=specs,
            golden_fn=golden_decode_reuse,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-2,
            atol=1e-2,
            compare_fn=compare,
            compile_only=args.compile_only,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
