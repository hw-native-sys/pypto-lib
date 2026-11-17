# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prefill C1A reuse attention on the paged dynamic-batch interface.

The C1A source layer owns the ratio-1 compressed and index caches; this layer
reads the published Top-K physical rows and the compressed pool without
re-projecting keys or re-scoring.  It still publishes its own window KV rows,
gathers the packed window + Top-K key set per token and runs the MLA core, then
the output projection.  The chain is layer-level and the bounded entry validates
the pre-TP-reduce output on one card.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from attention import attention_core
from config import (
    BLOCK_SIZE, CMP_BLOCKS_DYN, FLASH as M, LOCAL_HEADS, ORI_BLOCKS_DYN, PREFILL_MAX_TOKENS, SELECTED_KEYS,
    TP_SIZE, T_DYN, WINDOW_KEYS,
)
from hc import hc_post, hc_pre
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

# tiling
QUERY_CHUNK = 16


@pl.jit.inline(auto_scope=False)
def prefill_c1a_reuse_attention(
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
    """One TP rank's packed C1A reuse attention; only the window cache is written."""
    tokens = pl.tensor.dim(x, 0)
    q = pl.create_tensor([tokens, LOCAL_HEADS, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([tokens, 1], dtype=pl.FP32)
    mla_proj(
        x, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        rope_cos, rope_sin, q, kv, qr, qr_scale,
    )
    # Snapshot every early intermediate the later stages consume: the fused
    # planner may otherwise hand a dead InOut buffer to a later temporary.
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
def prefill_c1a_reuse(
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
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
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
    """Layer-level C1A reuse block: HC pre, norm, attention, TP reduce, HC post."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hidden = hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        hidden, next_pre_mix, post_mix, residual_mix,
    )
    _attn_input_ret = rms_norm(hidden, attn_norm_weight, attn_input, width=D, output_fp32=False, column_tile=256)
    partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
    prefill_c1a_reuse_attention(
        attn_input, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, topk_indices, partial, num_tokens,
    )
    attn_out = tp_output_reduce(partial, output_window, output_arrived, attn_out, group_base, tp_rank, tokens, attention_epoch)
    _output_ret = hc_post(attn_out, x_hc, post_mix, residual_mix, output)
    return output, attn_input


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


def _fixture(tokens: int = 8, seed: int = 0):
    from utils import build_request_plan

    plan = build_request_plan([(tokens + 2, tokens // 2), (tokens + 5, tokens - tokens // 2)], ratio=1)
    cos, sin = _rope_tables(256)
    positions = plan["position_ids"].long()
    return plan, cos, sin, positions


def _selection(plan, tokens: int):
    """Physical compressed rows of every visible row, as the source published them."""
    indices = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    table = plan["index_block_table"]
    for token in range(tokens):
        visible = int(plan["compressed_lens"][token])
        request = int(plan["request_ids"][token])
        for logical in range(min(visible, INDEX_TOPK)):
            physical = int(table[request, logical // BLOCK_SIZE])
            indices[token, logical] = physical * BLOCK_SIZE + logical % BLOCK_SIZE
    return indices


def build_tensor_specs(tokens: int = 8, seed: int = 0):
    from golden import TensorSpec
    from utils import pack_nz, quant_w_per_channel

    def quantized(output_width, input_width):
        return pack_nz(
            quant_w_per_channel(torch.randn(output_width, input_width) * 0.05)[0].transpose(0, 1).contiguous()
        )

    def scale_of(output_width, input_width):
        return quant_w_per_channel(torch.randn(output_width, input_width) * 0.05)[1]

    plan, cos, sin, positions = _fixture(tokens, seed)
    pages = plan["pages"]
    generator = torch.Generator().manual_seed(seed)
    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=lambda: torch.randn(tokens, D) * 0.1),
        TensorSpec("wq_a", [D, Q_LORA], torch.int8, init_value=lambda: quantized(Q_LORA, D)),
        TensorSpec("wq_a_scale", [Q_LORA], torch.float32, init_value=lambda: scale_of(Q_LORA, D)),
        TensorSpec("q_norm_w", [Q_LORA], torch.bfloat16, init_value=lambda: torch.randn(Q_LORA) * 0.1 + 1.0),
        TensorSpec("wq_b", [Q_LORA, LOCAL_Q], torch.int8, init_value=lambda: quantized(LOCAL_Q, Q_LORA)),
        TensorSpec("wq_b_scale", [LOCAL_Q], torch.float32, init_value=lambda: scale_of(LOCAL_Q, Q_LORA)),
        TensorSpec("wkv", [D, HEAD_DIM], torch.int8, init_value=lambda: quantized(HEAD_DIM, D)),
        TensorSpec("wkv_scale", [HEAD_DIM], torch.float32, init_value=lambda: scale_of(HEAD_DIM, D)),
        TensorSpec("kv_norm_w", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM) * 0.1 + 1.0),
        TensorSpec("sink", [LOCAL_HEADS], torch.float32, init_value=lambda: torch.randn(LOCAL_HEADS) * 0.5),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16,
                   init_value=lambda: torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) * 0.02),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.bfloat16,
                   init_value=lambda: torch.randn(D, O_GROUPS * O_LORA) * 0.02),
        TensorSpec("rope_cos", [tokens, 32], torch.float32, init_value=cos[positions]),
        TensorSpec("rope_sin", [tokens, 32], torch.float32, init_value=sin[positions]),
        TensorSpec("window_slots", [tokens], torch.int64, init_value=plan["window_slots"]),
        TensorSpec("window_indices", [tokens, WINDOW_KEYS], torch.int32, init_value=plan["window_indices"]),
        TensorSpec("window_cache", [pages, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=lambda: torch.randn(pages, BLOCK_SIZE, 1, HEAD_DIM, generator=generator) * 0.1),
        TensorSpec("compressed_cache", [pages, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=lambda: torch.randn(pages, BLOCK_SIZE, 1, HEAD_DIM, generator=generator) * 0.1),
        TensorSpec("topk_indices", [tokens, INDEX_TOPK], torch.int32,
                   init_value=_selection(plan, tokens)),
        TensorSpec("partial", [tokens, D], torch.bfloat16),
    ]


def golden_c1a_reuse_attention(tensors):
    """Compose the per-operator references on the paged reuse fixture."""
    from attention import golden_attention_core
    from mla_proj import _golden_mla_proj
    from o_proj import golden_o_proj
    from paged_cache import golden_gather_selected_keys
    from utils import scatter_cache_rows

    mla = {name: tensors[name] for name in (
        "x", "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
        "kv_norm_w", "rope_cos", "rope_sin")}
    mla["q"] = torch.zeros(tensors["x"].shape[0], LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    mla["kv"] = torch.zeros(tensors["x"].shape[0], HEAD_DIM, dtype=torch.bfloat16)
    mla["qr"] = torch.zeros(tensors["x"].shape[0], Q_LORA, dtype=torch.int8)
    mla["qr_scale"] = torch.zeros(tensors["x"].shape[0], 1, dtype=torch.float32)
    _golden_mla_proj(mla)
    scatter_cache_rows(tensors["window_cache"], tensors["window_slots"], mla["kv"])

    gather = {
        "window_cache": tensors["window_cache"], "window_indices": tensors["window_indices"],
        "compressed_cache": tensors["compressed_cache"], "topk_indices": tensors["topk_indices"],
        "selected": torch.zeros(tensors["x"].shape[0], SELECTED_KEYS, HEAD_DIM, dtype=torch.bfloat16),
        "combined": torch.zeros(tensors["x"].shape[0], SELECTED_KEYS, dtype=torch.int32),
    }
    golden_gather_selected_keys(gather)
    attention = {
        "q": mla["q"], "selected": gather["selected"], "combined": gather["combined"],
        "sink": tensors["sink"],
        "output": torch.zeros(tensors["x"].shape[0], LOCAL_HEADS, HEAD_DIM, dtype=torch.bfloat16),
    }
    golden_attention_core(attention)
    out = {
        "attended": attention["output"].reshape(tensors["x"].shape[0], LOCAL_Q).float(),
        "wo_a": tensors["wo_a"], "wo_b": tensors["wo_b"],
        "output": torch.zeros(tensors["x"].shape[0], D, dtype=torch.bfloat16),
    }
    out["cos"] = tensors["rope_cos"]
    out["sin"] = tensors["rope_sin"]
    golden_o_proj(out)
    tensors["partial"][:] = out["output"]


@pl.jit
def prefill_c1a_reuse_test(
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
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    partial.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    prefill_c1a_reuse_attention(
        x, wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, topk_indices, partial, pl.tensor.dim(x, 0),
    )
    return window_cache, partial


# ---------------------------------------------------------------------------
# Layer-level host-compile entry (compile-only; the fwd driver owns device runs)
# ---------------------------------------------------------------------------

LAYER_INPUT_NAMES = (
    "x_hc", "pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
    "wq_a", "wq_a_scale", "q_norm_w", "wq_b", "wq_b_scale", "wkv", "wkv_scale", "kv_norm_w",
    "sink", "wo_a", "wo_b", "rope_cos", "rope_sin", "window_slots", "window_indices", "window_cache",
    "compressed_cache", "topk_indices",
)

ATTENTION_INPUT_NAMES = LAYER_INPUT_NAMES[6:]


def _layer_specs(tokens: int = 8, seed: int = 0):
    from golden import ScalarSpec, TensorSpec

    base = {spec.name: spec for spec in build_tensor_specs(tokens, seed)}
    generator = torch.Generator().manual_seed(seed + 100)
    specs = [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32,
                   init_value=lambda: torch.randn(tokens, HC_MULT, D, generator=generator)),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32,
                   init_value=lambda: torch.sigmoid(torch.randn(tokens, HC_MULT, generator=generator))),
        TensorSpec("hc_attn_fn", [MIX_PAD, HC_DIM], torch.float32,
                   init_value=lambda: torch.randn(MIX_PAD, HC_DIM, generator=generator) / HC_DIM**0.5),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=lambda: torch.randn(3, generator=generator)),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32,
                   init_value=lambda: torch.randn(MIX_HC, generator=generator)),
        TensorSpec("attn_norm_weight", [D], torch.bfloat16,
                   init_value=lambda: torch.randn(D, generator=generator) * 0.1 + 1.0),
    ]
    specs.extend(base[name] for name in ATTENTION_INPUT_NAMES)
    specs.extend([
        TensorSpec("hidden", [tokens, D], torch.bfloat16),
        TensorSpec("attn_input", [tokens, D], torch.bfloat16),
        TensorSpec("attn_out", [tokens, D], torch.bfloat16),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
        TensorSpec("next_pre_mix", [tokens, HC_MULT], torch.float32),
        ScalarSpec("num_tokens", torch.int32, tokens),
    ])
    return specs


@pl.jit
def prefill_c1a_reuse_layer_rank(
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
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run one TP rank of the C1A reuse layer chain (host-compile harness)."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    hidden.bind_dynamic(0, T_DYN)
    attn_input.bind_dynamic(0, T_DYN)
    attn_out.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    next_pre_mix.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    prefill_c1a_reuse(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, topk_indices, output_window, output_arrived,
        hidden, attn_input, attn_out, output, next_pre_mix,
        0, 0, num_tokens, 1,
    )
    return output, attn_input


@pl.jit.host
def prefill_c1a_reuse_layer_entry(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_PAD, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    # W8A8 weights are NZ-packed on device; the host ABI carries the packed
    # buffer under its logical shape, as the forward driver does.
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
    """Allocate one TP window and host-compile the C1A reuse layer chain."""
    data_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    output_window = pld.window(data_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    output_arrived = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
    prefill_c1a_reuse_layer_rank(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_w, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_w,
        sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        compressed_cache, topk_indices, output_window, output_arrived,
        hidden, attn_input, attn_out, output, next_pre_mix,
        num_tokens, device=0,
    )


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device)
    if args.compile_only:
        print("[C1A-REUSE] host compile: operator chain", flush=True)
        result = run(fn=prefill_c1a_reuse_test, specs=build_tensor_specs(args.tokens, seed=args.seed),
                     golden_fn=None, config=config, compare_fn={}, compile_only=True)
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)
        print("[C1A-REUSE] host compile: layer chain", flush=True)
        result = run(fn=prefill_c1a_reuse_layer_entry, specs=_layer_specs(args.tokens, seed=args.seed),
                     golden_fn=None, config=config, compare_fn={}, compile_only=True)
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)
        raise SystemExit(0)

    compare = {
        "partial": ratio_allclose(atol=2e-2, rtol=1e-2, max_error_ratio=1e-2),
        "window_cache": ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2),
    }
    result = run(
        fn=prefill_c1a_reuse_test,
        specs=build_tensor_specs(args.tokens, seed=args.seed),
        golden_fn=golden_c1a_reuse_attention,
        config=config,
        rtol=2e-2,
        atol=2e-2,
        compare_fn=compare,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)
