# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-sharded packed-prefill SWA with local mHC and head-TP attention."""


import math
import pypto.language as pl
import pypto.language.distributed as pld
import torch
from utils import token_shards
from golden import mapped_pool_ratio_allclose
from sparse_attention import (
    SWA_RUN_DESCRIPTOR_WIDTH as RUN_DESCRIPTOR_WIDTH, SWA_RUN_PARTS as RUN_PARTS,
    analyze_index_runs, attend_shared_cache, decode_shared_cache_after_publish,
    prefill_publish_window_batched, swa_official_reference as golden_reference,
    make_swa_inputs as make_inputs, make_packed_swa_inputs as make_packed_inputs,
)
from config import (
    MAX_BATCH_PER_DP, WINDOW_CACHE_GROUP, FLASH as M, ORI_BLOCKS_DYN, T_DYN, TP_SIZE, PREFILL_MAX_TOKENS,
    SP_T_DYN,
)
from hc import golden_mhc_mixes
from hc import golden_hc_post, prefill_sp_post
from hc import golden_hc_collapse, hc_pre, zero_bf16_padding
from rmsnorm import golden_rms_norm, rms_norm_tiled
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj_rope
from o_proj import retire_prefill_signals, o_proj, prefill_sp_output_reduce_scatter
from prefill_cp_allgather import make_prefill_sp_input_allgather


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4

D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc


prefill_sp_input_allgather = make_prefill_sp_input_allgather(PREFILL_MAX_TOKENS)


# tiling
QUERY_TILE = 128
WORKER_TILE = 64


@pl.jit.inline(auto_scope=False)
def prefill_swa_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
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
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    partial: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Write BF16 head-TP partials for packed causal SWA; write slots must be unique."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_prefill_previous_epoch", allow_early_resolve=False) as cache_ready:
        for peer in pl.range(TP_SIZE):
            previous_epoch = (attention_epoch - 1) * 2
            pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
    tokens = pl.tensor.dim(x, 0)
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    # Gate every projection tile before scratch storage is reused by a new epoch.
    with pl.spmd(WORKER_TILE, name_hint="prefill_swa_begin", deps=[cache_ready]):
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, num_tokens, WORKER_TILE):
            kv_projection[row : row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_input_payload, kv_input_scales = quantize_projection_input(x, num_tokens)
    kv_proj_rope(
        kv_input_payload, kv_input_scales,
        wkv, wkv_scale, kv_norm_weight,
        rope_cos, rope_sin,
        kv_projection, kv,
        num_tokens,
    )
    chunk_done = prefill_publish_window_batched(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)
    cache_blocks = pl.tensor.dim(window_cache, 0)
    cache_rows = cache_blocks * 128
    decoded_cache = pl.create_tensor([cache_blocks, 128, 1, HEAD_DIM], dtype=pl.BF16)
    zero_row = pl.create_tensor([32, HEAD_DIM], dtype=pl.BF16)
    # Decode once for the entire packed prompt. The explicit publication
    # dependency protects the hoisted reader independently of query staging.
    decoded_ready = decode_shared_cache_after_publish(window_cache, window_cache_scale, decoded_cache, zero_row, chunk_done)

    chunk_x = pl.create_tensor([QUERY_TILE, D], dtype=pl.BF16)
    chunk_cos = pl.create_tensor([QUERY_TILE, ROPE_DIM // 2], dtype=pl.FP32)
    chunk_sin = pl.create_tensor([QUERY_TILE, ROPE_DIM // 2], dtype=pl.FP32)
    chunk_indices = pl.create_tensor([QUERY_TILE, 128], dtype=pl.INT32)
    qr = pl.create_tensor([QUERY_TILE, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    descriptors = pl.create_tensor([QUERY_TILE, RUN_PARTS, RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
    attended = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    chunk_partial = pl.create_tensor([QUERY_TILE, D], dtype=pl.BF16)

    for start in pl.range(0, num_tokens, QUERY_TILE):
        active = pl.min(QUERY_TILE, num_tokens - start)
        # The prior chunk must finish reading every reusable scratch buffer.
        with pl.spmd(WORKER_TILE, name_hint="prefill_swa_stage", deps=[chunk_done]) as stage_tid:
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, active, WORKER_TILE):
                source_row = start + row
                for col in pl.range(0, D, 512):
                    chunk_x[row:row + 1, col:col + 512] = x[source_row:source_row + 1, col:col + 512]
                chunk_cos[row:row + 1, :] = rope_cos[source_row:source_row + 1, :]
                chunk_sin[row:row + 1, :] = rope_sin[source_row:source_row + 1, :]
                chunk_indices[row:row + 1, :] = window_indices[source_row:source_row + 1, :]
        q_input_payload, q_input_scales = quantize_projection_input(chunk_x, active)
        q_proj_rope(
            q_input_payload, q_input_scales,
            wq_a, wq_a_scale, q_norm_weight,
            wq_b, wq_b_scale,
            chunk_cos, chunk_sin,
            qr, q,
            active,
        )
        index_ready = analyze_index_runs(chunk_indices, descriptors, active, cache_rows)
        attend_shared_cache(
            q, decoded_cache, zero_row,
            descriptors, chunk_indices, attn_sink,
            attended, active, decoded_ready, index_ready,
        )
        o_proj(
            attended, wo_a, wo_a_scale, wo_b,
            wo_b_scale, chunk_cos, chunk_sin,
            chunk_partial, active,
        )
        with pl.spmd(WORKER_TILE, name_hint="prefill_swa_collect") as collect_tid:
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, active, WORKER_TILE):
                for col in pl.range(0, D, 512):
                    partial[start + row:start + row + 1, col:col + 512] = chunk_partial[row:row + 1, col:col + 512]
        chunk_done = collect_tid
    return partial


@pl.jit.inline(auto_scope=False)
def prefill_swa(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
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
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_out: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    output: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Gather local mHC inputs, reduce-scatter SWA partials, and update local streams."""
    tokens = pl.tensor.dim(x_hc, 0)
    collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    local_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, incoming_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        collapsed, next_pre_mix, post_mix, residual_mix,
        hf32=False,
    )
    norm_tokens = pl.tensor.dim(collapsed, 0)
    for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(collapsed, attn_norm_weight, local_hidden, norm_block, norm_tokens, output_fp32=False)
    zero_bf16_padding(local_hidden, pl.min(tokens, pl.tensor.dim(local_hidden, 0)))
    prefill_sp_input_allgather(
        local_hidden, input_window, input_arrived, hidden,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    group_tokens = pl.tensor.dim(hidden, 0)
    partial = pl.create_tensor([group_tokens, D], dtype=pl.BF16)
    if num_tokens > 0:
        prefill_swa_attention(
            hidden, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale,
            output_arrived, partial, num_tokens, attention_epoch,
        )
    prefill_sp_output_reduce_scatter(
        partial, output_window, output_arrived, attn_out,
        group_base, tp_rank, num_tokens, attention_epoch,
    )
    prefill_sp_post(
        attn_out, x_hc, post_mix, residual_mix, output, next_pre_mix, tp_rank, num_tokens,
    )
    return output, next_pre_mix


HC_INPUT_NAMES = (
    "x_hc", "incoming_pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
    "wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale",
    "kv_norm_weight", "attn_sink", "wo_a", "wo_a_scale", "wo_b", "wo_b_scale", "rope_cos", "rope_sin",
    "window_slots", "window_indices", "window_cache", "window_cache_scale",
)


# Layer-0 mHC fixture statistics from DeepSeek-V4.1-Flash at
# dba1be0a40aa45a94ad051997016db3960a90277, model-00003-of-00048.safetensors.
HC_FIXTURE_FN_STD = 0.0225357748568058
HC_FIXTURE_SCALE = (0.06304993480443954, 0.011977050453424454, 0.06153993308544159)
HC_FIXTURE_BASE = (
    2.845095157623291, -3.343229293823242, -3.233515501022339, -2.217123031616211,
    -4.17180871963501, -3.9524407386779785, -19.57306671142578, -1.4172816276550293,
    0.20820771157741547, -25.451234817504883, -5.102978229522705, 0.09652144461870193,
    0.5044570565223694, -0.4407903552055359, -25.251018524169922, -21.587751388549805,
    -25.49090003967285, 0.1881018877029419, -0.11225703358650208, -22.04298973083496,
    -19.504518508911133, -21.67133140563965, -22.463973999023438, 13.438657760620117,
)


def make_hc_inputs(base: dict, seed: int, fixture: str = "checkpoint") -> dict:
    """Replace the collapsed hidden with four HC streams and mix weights."""
    gen = torch.Generator().manual_seed(seed + 2000)
    tokens = base["x"].shape[0]
    values = dict(base)
    values.pop("x")
    if fixture == "checkpoint":
        values["x_hc"] = torch.rand(tokens, HC_MULT, D, generator=gen) * 2 - 1
        values["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM, generator=gen) * HC_FIXTURE_FN_STD
        values["hc_attn_scale"] = torch.tensor(HC_FIXTURE_SCALE)
        values["hc_attn_base"] = torch.tensor(HC_FIXTURE_BASE)
        values["incoming_pre_mix"] = torch.zeros(tokens, HC_MULT)
        values["incoming_pre_mix"][:, 0] = 1
    elif fixture == "random":
        values["x_hc"] = torch.randn(tokens, HC_MULT, D, generator=gen)
        values["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM, generator=gen) / math.sqrt(HC_DIM)
        values["hc_attn_scale"] = torch.randn(3, generator=gen)
        values["hc_attn_base"] = torch.randn(MIX_HC, generator=gen)
        values["incoming_pre_mix"] = torch.sigmoid(torch.randn(tokens, HC_MULT, generator=gen))
    else:
        raise ValueError(f"unsupported fixture: {fixture}")
    values["attn_norm_weight"] = torch.ones(D, dtype=torch.bfloat16)
    if bool((base["x"] == 0).all()):
        values["x_hc"].zero_()
    return values


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_prefill_swa(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
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
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    output: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32]],
    hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_out: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run one rank of HC-orchestrated prefill SWA once."""
    x_hc.bind_dynamic(0, SP_T_DYN)
    hidden.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    active_tokens = pl.read(num_tokens, [0])
    prefill_swa(
        x_hc, incoming_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale,
        input_window, input_arrived, output_window, output_arrived, hidden, attn_out, output, next_pre_mix,
        rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active_tokens, attention_epoch,
    )
    retire_prefill_signals(output, input_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    retire_prefill_signals(output, output_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    return output, next_pre_mix, hidden, attn_out, window_cache, window_cache_scale


@pl.jit.host
def l3_prefill_swa(
    x_hc: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    output: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32]],
    hidden: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    attn_out: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, D], pl.BF16]],
    num_tokens: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Dispatch stacked ranks onto TP windows for the HC prefill program."""
    x_hc.bind_dynamic(0, RANKS_DYN)
    x_hc.bind_dynamic(1, SP_T_DYN)
    hidden.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    input_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    input_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    data_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        input_data = pld.window(input_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        input_signal = pld.window(input_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        data = pld.window(data_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        signal = pld.window(signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        # The rank takes these scales as MX_B_NN; a bare slice is ND, so annotate it.
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        l2_prefill_swa(
            x_hc[rank], incoming_pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank],
            wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
            wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank],
            wo_a[rank], wo_a_scale[rank], wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank],
            window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank],
            output[rank], next_pre_mix[rank], hidden[rank], attn_out[rank], input_data, input_signal, data, signal,
            num_tokens[rank], rank, attention_epoch,
            device=rank,
        )


def build_hc_specs(args):
    """Stacked TP/DP specs with HC streams in and HC streams out."""
    from golden import ScalarSpec, TensorSpec

    world_size = TP_SIZE * args.dp
    packed = True
    if packed:
        lengths = [args.tokens // args.requests + (r < args.tokens % args.requests) for r in range(args.requests)]
        if args.requests > 1 and lengths[-1] > 16:
            lengths[0] += 7
            lengths[-1] -= 7
        prefixes = ([127, 128, 511, 1048000] * 8)[:args.requests] if args.case == "prefix" else [0] * args.requests
        pages = 1 + sum((p + n + 128) // 128 - max(0, p - 127) // 128 + 1 for p, n in zip(prefixes, lengths))
    else:
        pages = args.tokens + 1
    local_tokens = (args.tokens + TP_SIZE - 1) // TP_SIZE
    shapes = (
        [local_tokens, HC_MULT, D], [local_tokens, HC_MULT], [MIX_HC, HC_DIM], [3], [MIX_HC], [D],
        [D, Q_LORA], [D // 32, Q_LORA], [Q_LORA],
        [Q_LORA, LOCAL_H * HEAD_DIM], [Q_LORA // 32, LOCAL_H * HEAD_DIM],
        [D, HEAD_DIM], [D // 32, HEAD_DIM], [HEAD_DIM], [LOCAL_H],
        [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], [LOCAL_O_WIDTH, D], [LOCAL_O_WIDTH // 32, D],
        [args.tokens, ROPE_DIM // 2], [args.tokens, ROPE_DIM // 2], [args.tokens], [args.tokens, 128],
        [pages, 128, 1, HEAD_DIM], [pages, 128, 1, HEAD_DIM // 32],
    )
    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    dtypes = (torch.float32, torch.float32, torch.float32, torch.float32, torch.float32, bf,
              fp, mx, bf, fp, mx, fp, mx, bf, torch.float32, fp, mx, fp, mx,
              torch.float32, torch.float32, torch.int64, torch.int32, fp, bf)
    values = {}

    def initialize(name):
        """Materialize one stacked HC input, sharing DP-group replicas."""
        if not values:
            ranks = []
            for rank in range(world_size):
                seed = args.seed + rank
                if packed:
                    value = make_packed_inputs(args.tokens, args.requests, seed, args.case)
                else:
                    value = make_inputs(args.tokens, seed)
                    if args.case == "masked":
                        value["window_slots"].fill_(-1)
                        value["window_indices"].fill_(-1)
                    elif args.case == "zero":
                        value["x"].zero_()
                        value["window_cache"].view(torch.uint8).zero_()
                    elif args.case == "sink":
                        value["attn_sink"].fill_(1000)
                ranks.append(make_hc_inputs(value, seed, args.fixture))
            replicated = (
                "x_hc", "incoming_pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
                "wq_a", "wq_a_scale", "q_norm_weight", "wkv", "wkv_scale",
                "kv_norm_weight", "rope_cos", "rope_sin", "window_slots", "window_indices",
                "window_cache", "window_cache_scale",
            )
            for rank, value in enumerate(ranks):
                for key in replicated:
                    value[key] = ranks[rank // TP_SIZE * TP_SIZE][key]
            counts = getattr(args, "dp_tokens", None) or [args.tokens] * args.dp
            for rank, value in enumerate(ranks):
                active = counts[rank // TP_SIZE]
                start = rank % TP_SIZE * local_tokens
                shard = torch.full((local_tokens, HC_MULT, D), 17.0)
                valid = max(0, min(local_tokens, active - start))
                shard[:valid].copy_(value["x_hc"][start:start + valid])
                value["x_hc"] = shard
                pre_shard = torch.zeros(local_tokens, HC_MULT)
                pre_shard[:valid].copy_(value["incoming_pre_mix"][start:start + valid])
                value["incoming_pre_mix"] = pre_shard
                value["window_slots"] = value["window_slots"].clone()
                value["window_slots"][active:] = -1
                value["window_indices"] = value["window_indices"].clone()
                value["window_indices"][active:] = -1
            for key in HC_INPUT_NAMES:
                dtype = ranks[0][key].dtype
                shards = [r[key].view(torch.uint8) if dtype in (fp, mx) else r[key] for r in ranks]
                values[key] = torch.stack(shards).view(dtype)
        return values[name]

    counts = getattr(args, "dp_tokens", None) or [args.tokens] * args.dp
    rank_counts = torch.tensor(counts, dtype=torch.int32).repeat_interleave(TP_SIZE).reshape(world_size, 1)
    specs = [TensorSpec(name, [world_size, *shape], dtype, init_value=lambda n=name: initialize(n), resident="stacked")
             for name, shape, dtype in zip(HC_INPUT_NAMES, shapes, dtypes)]
    specs += [TensorSpec("output", [world_size, local_tokens, HC_MULT, D], torch.float32, resident="stacked"),
              TensorSpec("next_pre_mix", [world_size, local_tokens, HC_MULT], torch.float32, resident="stacked"),
              TensorSpec("hidden", [world_size, args.tokens, D], bf, resident="stacked"),
              TensorSpec("attn_out", [world_size, local_tokens, D], bf, resident="stacked"),
              TensorSpec("num_tokens", [world_size, 1], torch.int32, init_value=lambda: rank_counts, resident="stacked"),
              ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True,
                         benchmark_step=None)]
    return specs


def reference_attention(tensors, hidden, base):
    """One TP group's SWA on ``hidden``: the FP32 TP sum in BF16 and each rank's (cache, scale)."""

    partials, caches = [], []
    active = int(tensors["num_tokens"][base, 0]) if "num_tokens" in tensors else hidden.shape[0]
    for rank in range(base, base + TP_SIZE):
        inputs = {name: tensors[name][rank] for name in HC_INPUT_NAMES if name not in (
            "x_hc", "incoming_pre_mix", "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_weight",
        )}
        inputs["x"] = hidden[:active]
        for name in ("rope_cos", "rope_sin", "window_slots", "window_indices"):
            inputs[name] = inputs[name][:active]
        partial = torch.zeros_like(hidden, dtype=torch.float32)
        cache, scale = inputs["window_cache"].clone(), inputs["window_cache_scale"].clone()
        if active:
            active_partial, cache, scale = golden_reference(inputs, shared_cache=True)
            partial[:active].copy_(active_partial)
        partials.append(partial)
        caches.append((cache, scale))
    return sum(partials).bfloat16(), caches


def golden_prefill_swa_case(tensors):
    """Reconstruct full residuals, run replicated attention, and split the reference."""
    world_size, local_tokens = tensors["x_hc"].shape[:2]
    tokens = tensors["hidden"].shape[1]
    for base in range(0, world_size, TP_SIZE):
        group = slice(base, base + TP_SIZE)
        active = int(tensors["num_tokens"][base, 0]) if "num_tokens" in tensors else tokens
        x_hc = tensors["x_hc"][group].flatten(0, 1)[:tokens].clone()
        x_hc[active:] = 0
        pre_mix, post_mix, residual_mix = golden_mhc_mixes(
            x_hc, tensors["hc_attn_fn"][base],
            tensors["hc_attn_scale"][base], tensors["hc_attn_base"][base],
        )
        incoming = tensors["incoming_pre_mix"][group].flatten(0, 1)[:tokens]
        hidden = golden_rms_norm(golden_hc_collapse(x_hc, incoming).to(torch.bfloat16), tensors["attn_norm_weight"][base])
        pre_mix[active:] = 0
        tensors["next_pre_mix"][group].copy_(token_shards(pre_mix, local_tokens))
        reduced, caches = reference_attention(tensors, hidden, base)
        for rank, (cache, scale) in enumerate(caches, base):
            tensors["window_cache"][rank].copy_(cache)
            tensors["window_cache_scale"][rank].copy_(scale)
        output = golden_hc_post(reduced, x_hc, post_mix, residual_mix)
        output[active:] = 0
        tensors["hidden"][group].copy_(hidden.unsqueeze(0).expand(TP_SIZE, -1, -1))
        tensors["attn_out"][group].copy_(token_shards(reduced, local_tokens))
        tensors["output"][group].copy_(token_shards(output, local_tokens))


if __name__ == "__main__":
    import argparse
    import os

    from golden import ratio_allclose, run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill SWA + mHC: A5 precision and timing")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", default=None, help="comma-separated device IDs; default: 0 through TP*DP-1")
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--ep", type=int, default=8, choices=[2, 4, 8])
    parser.add_argument("--dp", type=int, default=1, choices=[1, 2])
    parser.add_argument("--tokens", "--batch", type=int, default=128)
    parser.add_argument("--dp-tokens", help="comma-separated active counts per DP group, each in [0, tokens]")
    parser.add_argument("--requests", type=int, help="packed requests; default min(tokens, 4)")
    parser.add_argument("--case", default="mixed", choices=["mixed", "prefix", "shuffle", "masked", "zero", "sink"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fixture", choices=["checkpoint", "random"], default="checkpoint",
                        help="checkpoint mHC scale/base with synthetic weights, or random stress parameters")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", help="replay a compatible data directory containing in/ and out/")
    parser.add_argument("--enable-chip-swimlane", type=int, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true")
    args = parser.parse_args()
    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    args.requests = args.requests if args.requests is not None else min(args.tokens, 4)
    try:
        devices = list(range(TP_SIZE * args.dp)) if args.device is None else [int(d) for d in args.device.split(",")]
    except ValueError:
        parser.error("device IDs must be comma-separated integers")
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices) or min(devices) < 0:
        parser.error("device IDs must be distinct nonnegative integers, with count TP * DP")
    if not 1 <= args.tokens <= PREFILL_MAX_TOKENS or not 1 <= args.requests <= min(MAX_BATCH_PER_DP, args.tokens):
        parser.error(f"tokens must be in [1, {PREFILL_MAX_TOKENS}]; requests in [1, min({MAX_BATCH_PER_DP}, tokens)]")
    if args.dp_tokens is not None:
        try:
            args.dp_tokens = tuple(int(n) for n in args.dp_tokens.split(","))
        except ValueError:
            parser.error("dp-tokens must contain comma-separated integer counts")
        if len(args.dp_tokens) != args.dp or any(n < 0 or n > args.tokens for n in args.dp_tokens):
            parser.error("dp-tokens must contain one count per DP group, each in [0, tokens]")
    torch.set_num_threads(8)
    print(f"[SWA+HC] tokens={args.tokens} requests={args.requests} TP={TP_SIZE} DP={args.dp} "
          f"case={args.case} fixture={args.fixture} seed={args.seed} devices={devices}")
    result = run(
        fn=l3_prefill_swa,
        specs=build_hc_specs(args),
        golden_fn=golden_prefill_swa_case,
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            ring_heap=PREFILL_ATTN_RING_HEAP,
        ),
        compare_fn={
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "hidden": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            # Zero, masked and sink outputs are exactly zero, so any residue is a failure.
            "attn_out": (
                ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
                if args.case in ("masked", "zero", "sink") else ratio_allclose(atol=1e-2, rtol=1.0 / 128)
            ),
            "output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "window_cache": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache",
                mapping_shape=(len(devices), args.tokens),
                block_size=128, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "window_cache_scale": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache scale",
                mapping_shape=(len(devices), args.tokens),
                block_size=128, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
        },
    )
    if args.compile_only and result.passed:
        print("[SWA+HC] Compilation passed; device accuracy was NOT validated.")
    if not result.passed:
        raise SystemExit(result.error or 1)
