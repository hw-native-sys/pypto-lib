# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill sliding-window attention for encoder layers 0 and 1."""

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult, golden_swa_attention
from models.deepseek_v4_1_flash.attention_tp import prefill_tp_output_all_reduce
from models.deepseek_v4_1_flash.config import (
    D,
    HEAD_DIM,
    LOCAL_H,
    LOCAL_O_GROUPS,
    LOCAL_O_WIDTH,
    NOPE_DIM,
    O_GROUP_IN,
    O_LORA,
    ORI_BLOCKS_DYN,
    PREFILL_MAX_TOKENS,
    Q_LORA,
    ROPE_DIM,
    T_DYN,
    TP_SIZE,
    WINDOW_CACHE_GROUP,
)
from models.deepseek_v4_1_flash.decode_swa import (
    EPS,
    M_TILE,
    MX_M_TILE,
    N_TILE,
    SOFTMAX_SCALE,
    grouped_output,
    project_kv,
    project_ob,
    project_qb,
)


# tiling
QUERY_TILE = 128
WORKER_TILE = 64
PROJECTION_K_TILE = 32


@pl.jit.inline
def prefill_project_qa(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Accumulate scale-corrected group-32 QLoRA products in FP32."""
    scale_storage = pl.tensor.view(scale, [Q_LORA // 16, D // 2], layout=pl.ND)
    for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
        t0 = mt * MX_M_TILE
        for block in pl.spmd(Q_LORA // N_TILE, name_hint="prefill_swa_q_lora"):
            n0 = block * N_TILE
            rows = pl.min(MX_M_TILE, num_tokens - t0)
            acc = pl.tile.full([MX_M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(D // (2 * PROJECTION_K_TILE)):
                # MX_B_NN stores adjacent K-group scales in interleaved pairs.
                raw = pl.load(scale_storage, [n0 // 16, kb * 32], [N_TILE // 16, 32])
                raw_u8 = pl.reinterpret_view(raw, pl.UINT8)
                codes = pl.cast(pl.reinterpret_view(raw_u8, pl.INT8), pl.INT32)
                codes = pl.ands(codes, 255)
                scale_bits = pl.maximum(pl.shls(codes, 23), 4194304)
                scale_pair = pl.reinterpret_view(scale_bits, pl.FP32)
                for half in pl.unroll(2):
                    k0 = (kb * 2 + half) * PROJECTION_K_TILE
                    if half == 0:
                        gathered_scale = pl.tile.gather_mask(scale_pair, mask_pattern=pl.tile.MaskPattern.P0101)
                    else:
                        gathered_scale = pl.tile.gather_mask(scale_pair, mask_pattern=pl.tile.MaskPattern.P1010)
                    sb = pl.reshape(gathered_scale, [1, N_TILE])
                    source = pl.load(x, [t0, k0], [MX_M_TILE, PROJECTION_K_TILE], valid_shape=[rows, PROJECTION_K_TILE])
                    source = pl.fillpad(source, pad_value=pl.PadValue.zero)
                    source = pl.set_validshape(source, MX_M_TILE, PROJECTION_K_TILE)
                    value = pl.cast(source, pl.FP32)
                    reduce_tmp = pl.create_tile([MX_M_TILE, PROJECTION_K_TILE], dtype=pl.FP32)
                    maximum = pl.row_max(pl.abs(value), tmp_tile=reduce_tmp)
                    maximum = pl.maximum(maximum, 1e-4)
                    bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
                    exponent = pl.shrs(pl.add(bits, 8388607), 23)
                    sa = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                    quantized = pl.row_expand_div(value, sa)
                    payload = pl.cast(quantized, pl.FP8E4M3FN, mode="rint")
                    a = pl.cast(payload, pl.BF16)
                    weight_payload = pl.load(weight, [k0, n0], [PROJECTION_K_TILE, N_TILE])
                    b = pl.cast(weight_payload, pl.BF16)
                    dot = pl.matmul(a, b)
                    part = pl.row_expand_mul(dot, sa)
                    part = pl.col_expand_mul(part, sb)
                    acc = pl.add(acc, part)
            result = pl.cast(acc, pl.BF16, mode="rint")
            result = pl.set_validshape(result, rows, N_TILE)
            output = pl.store(result, [t0, n0], output)
    return output


def make_prefill_norm(width):
    @pl.jit.inline
    def normalize(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for worker in pl.spmd(WORKER_TILE, name_hint="prefill_swa_rmsnorm"):
            for block in pl.range(worker, (num_tokens + 7) // 8, WORKER_TILE):
                t = block * 8
                rows = pl.min(8, num_tokens - t)
                source = pl.slice(x, [8, width], [t, 0], valid_shape=[rows, width])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, width)
                value = pl.cast(source, pl.FP32)
                square_sum = pl.row_sum(pl.mul(value, value))
                variance = pl.add(pl.mul(square_sum, 1.0 / width), EPS)
                inv = pl.rsqrt(variance, high_precision=True)
                gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
                normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
                output[t:t + 8, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, width)
        return output

    return normalize


def make_prefill_rope(heads, inverse=False):
    sign = -1.0 if inverse else 1.0

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for worker in pl.spmd(WORKER_TILE, name_hint="prefill_swa_rope"):
            for block in pl.range(worker, num_tokens * heads, WORKER_TILE):
                t = block // heads
                h = block % heads
                base = h * HEAD_DIM
                output[t:t + 1, base:base + NOPE_DIM] = x[t:t + 1, base:base + NOPE_DIM]
                tail = pl.cast(x[t:t + 1, base + NOPE_DIM:base + HEAD_DIM], pl.FP32)
                even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                c = cos[t:t + 1, :]
                s = pl.mul(sin[t:t + 1, :], sign)
                re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
                im = pl.add(pl.mul(even, s), pl.mul(odd, c))
                rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
                rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
                rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
                output[t:t + 1, base + NOPE_DIM:base + HEAD_DIM] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rotate


@pl.jit.inline
def prefill_publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(WORKER_TILE, name_hint="prefill_swa_cache_publish", deps=[cache_ready]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, WORKER_TILE):
            slot_i64 = pl.read(slots, [t])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                source = pl.slice(kv, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, HEAD_DIM * 2)
                value = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM // 16, 32])
                amax = pl.maximum(pl.row_max(pl.abs(value)), 1e-4)
                raw = pl.mul(amax, 1.0 / 448.0)
                bits = pl.reinterpret_view(raw, pl.INT32)
                exponent = pl.shrs(pl.add(bits, 8388607), 23)
                scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                payload = pl.cast(pl.row_expand_div(value, scale), pl.FP8E4M3FN, mode="rint")
                flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(payload, [1, HEAD_DIM * 2]), 1, HEAD_DIM)
                signed_exponent = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
                codes = pl.cast(signed_exponent, pl.INT8)
                encoded = pl.reinterpret_view(pl.reinterpret_view(codes, pl.UINT8), pl.FP8E8M0)
                encoded_row = pl.reshape(encoded, [1, HEAD_DIM // 16])
                scale_flat[slot:slot + 1, :] = pl.set_validshape(encoded_row, 1, HEAD_DIM // 32)
    return publish_tid


@pl.jit.inline
def prefill_gather_window(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    selected: pl.Tensor[[T_DYN, 128, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    tokens = pl.tensor.dim(selected, 0)
    selected_rows = tokens * 128
    gathered = pl.reshape(selected, [selected_rows, HEAD_DIM])
    with pl.spmd(WORKER_TILE, name_hint="prefill_swa_cache_gather") as gather_tid:
        worker = pl.tile.get_block_idx()
        for block in pl.range(worker, num_tokens * 8, WORKER_TILE):
            t = block // 8
            for i in pl.range(block % 8 * 16, block % 8 * 16 + 16):
                row_i32 = pl.read(indices, [t, i])
                dst = t * 128 + i
                if row_i32 >= 0:
                    row = pl.cast(row_i32, pl.INDEX)
                    value = pl.reshape(pl.cast(flat[row:row + 1, :], pl.FP32), [HEAD_DIM // 32, 32])
                    scale_row = pl.slice(scale_flat, [1, 32], [row, 0], valid_shape=[1, HEAD_DIM // 32])
                    raw_codes = pl.reinterpret_view(scale_row, pl.UINT8)
                    signed_codes = pl.cast(pl.reinterpret_view(raw_codes, pl.INT8), pl.INT32)
                    codes = pl.ands(signed_codes, 255)
                    scale_values = pl.reinterpret_view(pl.maximum(pl.shls(codes, 23), 4194304), pl.FP32)
                    scale = pl.reshape(scale_values[:, :HEAD_DIM // 32], [HEAD_DIM // 32, 1])
                    decoded = pl.cast(pl.row_expand_mul(value, scale), pl.BF16, mode="rint")
                    gathered[dst:dst + 1, :] = pl.reshape(decoded, [1, HEAD_DIM])
                else:
                    gathered[dst:dst + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    return gather_tid


@pl.jit.inline
def prefill_attend_window(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[T_DYN, 128, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(query, 0)
    head_rows = tokens * LOCAL_H
    cache_rows = tokens * 128
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    for worker in pl.spmd(32, name_hint="prefill_swa_online_attention"):
        for block in pl.range(worker, num_tokens * (LOCAL_H // M_TILE), 32):
            t = block // (LOCAL_H // M_TILE)
            h = block % (LOCAL_H // M_TILE) * M_TILE
            q0 = t * LOCAL_H + h
            maximum = pl.full([1, M_TILE], dtype=pl.FP32, value=-1e30)
            denominator = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
            numerator = pl.full([M_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
            for part in pl.range(2):
                k0 = t * 128 + part * 64
                q = qflat[q0:q0 + M_TILE, :]
                kv = kflat[k0:k0 + 64, :]
                scores = pl.matmul(q, kv, b_trans=True)
                scores = pl.mul(scores, SOFTMAX_SCALE)
                idx = pl.cast(indices[t:t + 1, part * 64:part * 64 + 64], pl.FP32)
                valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
                bias = pl.mul(pl.sub(valid, 1.0), 1e30)
                scores = pl.col_expand_add(scores, bias)
                next_max = pl.maximum(maximum, pl.reshape(pl.row_max(scores), [1, M_TILE]))
                correction = pl.exp(pl.sub(maximum, next_max))
                shifted_scores = pl.row_expand_sub(scores, pl.reshape(next_max, [M_TILE, 1]))
                probabilities = pl.col_expand_mul(pl.exp(shifted_scores), valid)
                corrected_denominator = pl.mul(denominator, correction)
                probability_sum = pl.reshape(pl.row_sum(probabilities), [1, M_TILE])
                denominator = pl.add(corrected_denominator, probability_sum)
                weights = pl.cast(probabilities, pl.BF16, mode="rint")
                weighted = pl.matmul(weights, kv)
                numerator = pl.add(pl.row_expand_mul(numerator, pl.reshape(correction, [M_TILE, 1])), weighted)
                maximum = next_max
            sinks = pl.reshape(sink[h:h + M_TILE], [1, M_TILE])
            final_max = pl.maximum(maximum, sinks)
            correction = pl.exp(pl.sub(maximum, final_max))
            denominator = pl.add(pl.mul(denominator, correction), pl.exp(pl.sub(sinks, final_max)))
            normalized_correction = pl.div(correction, denominator, high_precision=True)
            result = pl.row_expand_mul(numerator, pl.reshape(normalized_correction, [M_TILE, 1]))
            oflat[q0:q0 + M_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
    return output


prefill_normalize_q = make_prefill_norm(Q_LORA)
prefill_normalize_kv = make_prefill_norm(HEAD_DIM)
prefill_rotate_q = make_prefill_rope(LOCAL_H)
prefill_rotate_kv = make_prefill_rope(1)
prefill_rotate_output = make_prefill_rope(LOCAL_H, inverse=True)


def golden_prefill_swa(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
) -> AttentionGoldenResult:
    return golden_swa_attention(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
    )


@pl.jit.inline(auto_scope=False)
def prefill_swa(
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
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.FP8E8M0],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.FP32],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Write packed causal SWA output; active physical write slots must be unique."""
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
            kv_projection[row:row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    project_kv(x, wkv, wkv_scale, kv_projection, num_tokens)
    kv_normalized = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    prefill_normalize_kv(kv_projection, kv_norm_weight, kv_normalized, num_tokens)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    prefill_rotate_kv(kv_normalized, rope_cos, rope_sin, kv, num_tokens)
    chunk_done = prefill_publish_window(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)

    chunk_x = pl.create_tensor([QUERY_TILE, D], dtype=pl.BF16)
    chunk_cos = pl.create_tensor([QUERY_TILE, ROPE_DIM // 2], dtype=pl.FP32)
    chunk_sin = pl.create_tensor([QUERY_TILE, ROPE_DIM // 2], dtype=pl.FP32)
    chunk_indices = pl.create_tensor([QUERY_TILE, 128], dtype=pl.INT32)
    qa = pl.create_tensor([QUERY_TILE, Q_LORA], dtype=pl.BF16)
    qr = pl.create_tensor([QUERY_TILE, Q_LORA], dtype=pl.BF16)
    qb = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    q = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    selected = pl.create_tensor([QUERY_TILE, 128, HEAD_DIM], dtype=pl.BF16)
    attended = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    unrotated = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    latent = pl.create_tensor([QUERY_TILE, LOCAL_O_WIDTH], dtype=pl.BF16)
    chunk_partial = pl.create_tensor([QUERY_TILE, D], dtype=pl.FP32)
    partial = pl.create_tensor([tokens, D], dtype=pl.FP32)

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
        prefill_project_qa(chunk_x, wq_a, wq_a_scale, qa, active)
        prefill_normalize_q(qa, q_norm_weight, qr, active)
        project_qb(qr, wq_b, wq_b_scale, qb, active)
        prefill_rotate_q(qb, chunk_cos, chunk_sin, q, active)
        prefill_gather_window(window_cache, window_cache_scale, chunk_indices, selected, active)
        prefill_attend_window(q, selected, chunk_indices, attn_sink, attended, active)
        prefill_rotate_output(attended, chunk_cos, chunk_sin, unrotated, active)
        grouped_output(unrotated, wo_a, latent, active)
        project_ob(latent, wo_b, wo_b_scale, chunk_partial, active)
        with pl.spmd(WORKER_TILE, name_hint="prefill_swa_collect") as collect_tid:
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, active, WORKER_TILE):
                for col in pl.range(0, D, 512):
                    partial[start + row:start + row + 1, col:col + 512] = chunk_partial[row:row + 1, col:col + 512]
        chunk_done = collect_tid
    prefill_tp_output_all_reduce(
        partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch,
    )
    return output


__all__ = ["golden_prefill_swa", "prefill_swa"]


if __name__ == "__main__":
    from models.deepseek_v4_1_flash._golden_smoke import run_attention_golden

    run_attention_golden(golden_prefill_swa, ratio=0, mode="swa")
