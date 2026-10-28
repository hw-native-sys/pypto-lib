# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DSpark sliding-window attention over history and all five query rows."""

import pypto.language as pl

from config import DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH, FLASH, ORI_BLOCKS_DYN, TP_SIZE, T_DYN
from o_proj import o_proj
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj_rope
from sparse_attention import publish_window_row


D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
LOCAL_H = FLASH.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = FLASH.o_groups // TP_SIZE
LOCAL_O_WIDTH = LOCAL_O_GROUPS * FLASH.o_lora_rank
O_GROUP_IN = FLASH.num_attention_heads * HEAD_DIM // FLASH.o_groups
O_LORA = FLASH.o_lora_rank
Q_LORA = FLASH.q_lora_rank
ROPE_DIM = FLASH.qk_rope_head_dim


M_TILE = 16
SOFTMAX_SCALE = HEAD_DIM**-0.5


@pl.jit.inline
def gather_dspark_window(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    indices: pl.Tensor[[T_DYN, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    selected: pl.Tensor[[T_DYN, DSPARK_SWA_INDEX_WIDTH * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    cache_rows = pl.tensor.dim(cache, 0) * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    for block in pl.spmd(num_tokens * (DSPARK_SWA_INDEX_WIDTH // 16), name_hint="dspark_cache_gather"):
        token = block // (DSPARK_SWA_INDEX_WIDTH // 16)
        for i in pl.range(block % (DSPARK_SWA_INDEX_WIDTH // 16) * 16, block % (DSPARK_SWA_INDEX_WIDTH // 16) * 16 + 16):
            row_i32 = pl.read(indices, [token, i])
            column = i * HEAD_DIM
            if row_i32 >= 0:
                row = pl.cast(row_i32, pl.INDEX)
                value = pl.reshape(pl.cast(flat[row:row + 1, :], pl.FP32), [HEAD_DIM // 32, 32])
                scale_row = pl.slice(scale_flat, [1, 32], [row, 0], valid_shape=[1, HEAD_DIM // 32])
                scale = pl.reshape(pl.cast(scale_row[:, :HEAD_DIM // 32], pl.FP32), [HEAD_DIM // 32, 1])
                decoded = pl.cast(pl.row_expand_mul(value, scale), pl.BF16, mode="rint")
                selected[token:token + 1, column:column + HEAD_DIM] = pl.reshape(decoded, [1, HEAD_DIM])
            else:
                selected[token:token + 1, column:column + HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
    return selected


@pl.jit.inline
def attend_dspark_window(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[T_DYN, DSPARK_SWA_INDEX_WIDTH * HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[T_DYN, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(query, 0)
    head_rows = tokens * LOCAL_H
    cache_rows = tokens * DSPARK_SWA_INDEX_WIDTH
    qflat = pl.reshape(query, [head_rows, HEAD_DIM])
    kflat = pl.reshape(selected, [cache_rows, HEAD_DIM])
    oflat = pl.reshape(output, [head_rows, HEAD_DIM])
    for block in pl.spmd(num_tokens * (LOCAL_H // M_TILE), name_hint="dspark_online_attention"):
        token = block // (LOCAL_H // M_TILE)
        head = block % (LOCAL_H // M_TILE) * M_TILE
        q0 = token * LOCAL_H + head
        maximum = pl.full([1, M_TILE], dtype=pl.FP32, value=-1e30)
        denominator = pl.full([1, M_TILE], dtype=pl.FP32, value=0.0)
        numerator = pl.full([M_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for part in pl.range(DSPARK_SWA_INDEX_WIDTH // 64):
            k0 = token * DSPARK_SWA_INDEX_WIDTH + part * 64
            q = qflat[q0:q0 + M_TILE, :]
            kv = kflat[k0:k0 + 64, :]
            scores = pl.mul(pl.matmul(q, kv, b_trans=True), SOFTMAX_SCALE)
            idx = pl.cast(indices[token:token + 1, part * 64:part * 64 + 64], pl.FP32)
            valid = pl.minimum(pl.maximum(pl.add(idx, 1.0), 0.0), 1.0)
            bias = pl.mul(pl.sub(valid, 1.0), 1e30)
            scores = pl.col_expand_add(scores, bias)
            next_max = pl.maximum(maximum, pl.reshape(pl.row_max(scores), [1, M_TILE]))
            correction = pl.exp(pl.sub(maximum, next_max))
            probabilities = pl.col_expand_mul(
                pl.exp(pl.row_expand_sub(scores, pl.reshape(next_max, [M_TILE, 1]))), valid
            )
            denominator = pl.add(
                pl.mul(denominator, correction), pl.reshape(pl.row_sum(probabilities), [1, M_TILE])
            )
            weights = pl.cast(probabilities, pl.BF16, mode="rint")
            weighted = pl.matmul(weights, kv)
            numerator = pl.add(
                pl.row_expand_mul(numerator, pl.reshape(correction, [M_TILE, 1])), weighted
            )
            maximum = next_max
        sinks = pl.reshape(sink[head:head + M_TILE], [1, M_TILE])
        final_max = pl.maximum(maximum, sinks)
        correction = pl.exp(pl.sub(maximum, final_max))
        denominator = pl.add(pl.mul(denominator, correction), pl.exp(pl.sub(sinks, final_max)))
        result = pl.row_expand_mul(
            numerator, pl.reshape(pl.div(correction, denominator), [M_TILE, 1])
        )
        oflat[q0:q0 + M_TILE, :] = pl.cast(result, pl.BF16, mode="rint")
    return output


@pl.jit.inline(auto_scope=False)
def dspark_attention(
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
    window_indices: pl.Tensor[[T_DYN, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x, 0)
    qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    proj_payload, proj_scales = quantize_projection_input(x, num_tokens)
    q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, q,
        num_tokens, 256, 2147483647,
    )
    kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(
        proj_payload, proj_scales,
        wkv, wkv_scale, kv_norm_weight,
        rope_cos, rope_sin,
        kv_projection, kv,
        num_tokens,
    )
    publish_window_row(kv, window_slots, window_cache, window_cache_scale, num_tokens, max_workers=2147483647)
    selected = pl.create_tensor([tokens, DSPARK_SWA_INDEX_WIDTH * HEAD_DIM], dtype=pl.BF16)
    gather_dspark_window(window_cache, window_cache_scale, window_indices, selected, num_tokens)
    attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attend_dspark_window(q, selected, window_indices, attn_sink, attended, num_tokens)
    o_proj(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale,
        rope_cos, rope_sin, output, num_tokens,
    )
    return output, window_cache, window_cache_scale


@pl.jit
def dspark_attention_test(
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
    window_indices: pl.Tensor[[T_DYN, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16]],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(0, ORI_BLOCKS_DYN)
    output.bind_dynamic(0, T_DYN)
    return dspark_attention(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale, output, num_tokens,
    )


def make_attention_inputs(seed: int = 43):
    """Prepare five queries sharing history and all current KV rows."""
    import torch

    from sparse_attention import make_swa_inputs

    values = make_swa_inputs(batch=DSPARK_QUERY_WIDTH, seed=seed)
    positions = torch.arange(2, 2 + DSPARK_QUERY_WIDTH)
    angles = positions.float()[:, None] * (
        10000.0 ** (-torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM)
    )
    values["rope_cos"] = angles.cos()
    values["rope_sin"] = angles.sin()
    values["window_slots"] = 128 + positions
    indices = torch.full((DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH), -1, dtype=torch.int32)
    indices[:, : DSPARK_QUERY_WIDTH + 2] = torch.arange(
        128, 130 + DSPARK_QUERY_WIDTH, dtype=torch.int32
    )
    values["window_indices"] = indices
    return values


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark non-causal SWA validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    import torch

    from golden import ScalarSpec, TensorSpec, ratio_allclose, run
    from sparse_attention import SWA_INPUT_NAMES, swa_official_reference

    values = make_attention_inputs()

    def golden(tensors):
        output, cache, cache_scale = swa_official_reference(tensors, key_tile=64)
        tensors["output"].copy_(output)
        tensors["window_cache"].copy_(cache)
        tensors["window_cache_scale"].copy_(cache_scale)

    def compare_payload(actual, expected, **_kwargs):
        return bool(torch.allclose(actual.float(), expected.float(), rtol=0.05, atol=0.05)), "FP8 payload"

    specs = [TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name])
             for name in SWA_INPUT_NAMES]
    specs += [
        TensorSpec("output", [DSPARK_QUERY_WIDTH, D], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, DSPARK_QUERY_WIDTH, compile_runtime=True),
    ]
    result = run(
        fn=dspark_attention_test,
        specs=specs,
        golden_fn=golden,
        compare_fn={
            "window_cache": compare_payload,
            "output": ratio_allclose(atol=0.03, rtol=0.03, max_error_ratio=0.0),
        },
        config={
            "backend_type": args.backend_type, "execution_mode": args.execution_mode,
            "device_id": args.device,
        },
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
