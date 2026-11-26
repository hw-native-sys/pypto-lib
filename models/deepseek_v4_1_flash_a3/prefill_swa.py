# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Token-sharded packed-prefill SWA with local mHC and head-TP attention."""


import pypto.language as pl
from .activation_quant import quantize_projection_activation
import pypto.language.distributed as pld
from .sparse_attention import attend_shared_cache_direct, decode_shared_cache_after_publish, prefill_publish_window
from .config import WINDOW_CACHE_GROUP, FLASH as M, TP_SIZE, PREFILL_MAX_TOKENS
from .dsl_types import ORI_BLOCKS_DYN, T_DYN, SP_T_DYN
from .hc import prefill_sp_post
from .hc import hc_pre, zero_bf16_padding
from .rmsnorm import rms_norm_tiled
from .qkv_proj_rope import q_proj_rope, kv_proj_rope
from .o_proj import prefill_o_proj_nz as o_proj, prefill_sp_output_reduce_scatter, retire_prefill_signals
from .prefill_cp_allgather import make_prefill_sp_input_allgather


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


@pl.jit.inline(auto_scope=False)
def prefill_swa_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
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
    begin_workers = pl.max(1, pl.min(num_tokens, pl.system.available_aiv_count()))
    with pl.spmd(begin_workers, name_hint="prefill_swa_begin", deps=[cache_ready]):
        worker = pl.tile.get_block_idx()
        for row in pl.range(worker, num_tokens, begin_workers):
            kv_projection[row : row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_input_payload, kv_input_scales = quantize_projection_activation(x, num_tokens, D, True, True)
    kv_proj_rope(
        kv_input_payload, kv_input_scales,
        wkv, wkv_scale, kv_norm_weight,
        rope_cos, rope_sin,
        kv_projection, kv,
        num_tokens,
    )
    published = prefill_publish_window(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)
    cache_rows = pl.tensor.dim(window_cache, 0) * 128
    key_rows = cache_rows + tokens * 128
    keys = pl.create_tensor([key_rows, HEAD_DIM], dtype=pl.BF16)
    zero_row = pl.create_tensor([32, HEAD_DIM], dtype=pl.BF16)
    decoded_ready = decode_shared_cache_after_publish(window_cache, window_cache_scale, keys, zero_row, published)

    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    q_proj_rope(
        kv_input_payload, kv_input_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, q, num_tokens,
    )

    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attend_shared_cache_direct(
        q, keys, cache_rows, zero_row,
        window_indices, attn_sink,
        attended, num_tokens, decoded_ready,
    )
    o_proj(attended, wo_a, wo_b, rope_cos, rope_sin, partial, num_tokens)
    return partial


@pl.jit.inline(auto_scope=False)
def prefill_swa(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8],
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
        prefill=True,
    )
    norm_tokens = pl.tensor.dim(collapsed, 0)
    norm_tiles = (norm_tokens + 7) // 8
    norm_workers = pl.max(1, pl.min(norm_tiles, pl.system.available_aiv_count()))
    for norm_worker in pl.spmd(norm_workers, name_hint="rms_norm"):
        for norm_block in pl.range(norm_worker, norm_tiles, norm_workers):
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
            hidden, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, output_arrived, partial, num_tokens, attention_epoch,
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
    "kv_norm_weight", "attn_sink", "wo_a",  "wo_b",  "rope_cos", "rope_sin",
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


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_prefill_swa(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8]],
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
        x_hc, incoming_pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_weight, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight, attn_sink, wo_a, wo_b, rope_cos, rope_sin, window_slots, window_indices, window_cache, window_cache_scale, input_window, input_arrived, output_window, output_arrived, hidden, attn_out, output, next_pre_mix, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active_tokens, attention_epoch,
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
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[RANKS_DYN, 1, Q_LORA], pl.FP32],
    q_norm_weight: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[RANKS_DYN, 1, LOCAL_H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[RANKS_DYN, 1, HEAD_DIM], pl.FP32],
    kv_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.UINT8]],
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
        wq_a_scale_r: pl.Tensor[[1, Q_LORA], pl.FP32, pl.ND] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[
            [1, LOCAL_H * HEAD_DIM], pl.FP32, pl.ND
        ] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[1, HEAD_DIM], pl.FP32, pl.ND] = wkv_scale[rank]
        l2_prefill_swa(
            x_hc[rank], incoming_pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r, wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_b[rank], rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank], output[rank], next_pre_mix[rank], hidden[rank], attn_out[rank], input_data, input_signal, data, signal, num_tokens[rank], rank, attention_epoch, device=rank,
        )
