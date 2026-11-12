# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""DSpark sliding-window attention over history and all five query rows."""

from pathlib import Path
from pypto.runtime import pto_isa_include_dir
from quant import packed_cache_compare

import pypto.language as pl

from quant import pack_quantized_cache, unpack_quantized_cache

from config import (
    DSPARK_QUERY_WIDTH, DSPARK_SWA_INDEX_WIDTH, FLASH, MOE_TOKENS, ORI_BLOCKS_DYN,
    WINDOW_CACHE_BYTES,
)
from o_proj import o_proj_hc_post
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj, window_publish


HC_MULT = FLASH.hc_mult
D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
LOCAL_H = FLASH.num_attention_heads
LOCAL_O_GROUPS = FLASH.o_groups
LOCAL_O_WIDTH = LOCAL_O_GROUPS * FLASH.o_lora_rank
O_GROUP_IN = FLASH.num_attention_heads * HEAD_DIM // FLASH.o_groups
O_LORA = FLASH.o_lora_rank
Q_LORA = FLASH.q_lora_rank
ROPE_DIM = FLASH.qk_rope_head_dim


QUERY_HEAD_TILE = max(32, LOCAL_H)
SOFTMAX_SCALE = HEAD_DIM**-0.5


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "dspark_gather.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _gather_dspark_window_native(
    cache: pl.Tensor,
    indices: pl.Tensor,
    selected: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Gather and dequantize DSpark windows with guarded query reuse."""
    ...


@pl.jit.inline(auto_scope=False)
def gather_dspark_window(
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    selected: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    groups = (num_tokens + DSPARK_QUERY_WIDTH - 1) // DSPARK_QUERY_WIDTH
    workers = pl.max(1, pl.min(groups * (DSPARK_SWA_INDEX_WIDTH // 32), 64))
    with pl.spmd(workers, name_hint="dspark_cache_gather"):
        selected = _gather_dspark_window_native(cache, indices, selected, num_tokens)
    return selected


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "dspark_attention_core.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "dspark_attention_core.cpp",
               include_dirs=(pto_isa_include_dir(),), dual_aiv_dispatch=True)
def _attend_dspark_native(
    query: pl.Tensor,
    selected: pl.Tensor,
    indices: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Attend selected K64 tiles with paired ownership and BF16 probabilities."""
    ...


@pl.jit.inline(auto_scope=False)
def attend_dspark_window(
    query: pl.Tensor[[MOE_TOKENS, LOCAL_H * HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH * HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.Tensor[[MOE_TOKENS, LOCAL_H * HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    workers = pl.max(1, pl.min(num_tokens, 32))
    if LOCAL_H < 32:
        padded_query = pl.create_tensor([MOE_TOKENS, QUERY_HEAD_TILE * HEAD_DIM], dtype=pl.BF16)
        for row in pl.spmd(num_tokens, name_hint="dspark_pad_heads"):
            padded = pl.full([1, QUERY_HEAD_TILE * HEAD_DIM], dtype=pl.BF16, value=0.0)
            active_query = query[row : row + 1, :]
            padded = pl.assemble(padded, active_query, [0, 0])
            padded_query[row : row + 1, :] = padded
        with pl.spmd(workers, name_hint="dspark_online_attention"):
            output = _attend_dspark_native(padded_query, selected, indices, sink, output, num_tokens, LOCAL_H)
    else:
        with pl.spmd(workers, name_hint="dspark_online_attention"):
            output = _attend_dspark_native(query, selected, indices, sink, output, num_tokens, LOCAL_H)
    return output


@pl.jit.inline(auto_scope=False)
def dspark_attention_partial(
    x: pl.Tensor,
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_slots: pl.Tensor,
    window_indices: pl.Tensor,
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    num_tokens: pl.Scalar[pl.INT32],
):
    qr = pl.create_tensor([MOE_TOKENS, Q_LORA], dtype=pl.BF16)
    q = pl.create_tensor([MOE_TOKENS, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    proj_payload, proj_scales, proj_payload_padded_rows = quantize_projection_input(x, num_tokens, row_tile=80)
    q_proj_result = q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, q,
        num_tokens, qa_worker_limit=32, qb_worker_limit=32,
        padded_rows=proj_payload_padded_rows,
        qa_row_tile=80, qb_row_tile=80,
    )
    kv_projection = pl.create_tensor([MOE_TOKENS, HEAD_DIM], dtype=pl.BF16)
    window_input_ready = pl.system.task_dummy(deps=[])
    kv_projection = kv_proj(
        proj_payload, proj_scales,
        wkv, wkv_scale,
        kv_projection, num_tokens, worker_limit=32, row_tile=80,
    )
    window_publish(
        kv_projection, kv_norm_weight, rope_cos, rope_sin,
        window_slots, window_cache,
        num_tokens, window_input_ready,
    )
    selected = pl.create_tensor([MOE_TOKENS, DSPARK_SWA_INDEX_WIDTH * HEAD_DIM], dtype=pl.BF16)
    selected = gather_dspark_window(window_cache, window_indices, selected, num_tokens)
    attended = pl.create_tensor([MOE_TOKENS, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    attended = attend_dspark_window(q, selected, window_indices, attn_sink, attended, num_tokens)
    return attended, window_cache


@pl.jit
def dspark_attention_test(
    x: pl.Tensor[[MOE_TOKENS, D], pl.BF16],
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
    residual: pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[MOE_TOKENS, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[MOE_TOKENS, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    attended, window_cache = dspark_attention_partial(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, num_tokens,
    )
    o_proj_hc_post(
        attended, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        residual, post_mix, residual_mix, output, num_tokens, oa_row_tile=80, ob_row_tile=128,
    )
    return output, window_cache


def golden_dspark_attention(tensors):
    """Compose QKV, shared-history attention, output projection and HC references."""
    import torch
    from hc import golden_hc_post
    from o_proj import golden_o_proj
    from qkv_proj_rope import golden_qkv_proj_rope
    from quant import quantize_mxfp8_cache
    from sparse_attention import golden_gather_sparse_attention
    from quant import _publish_quantized_cache

    active = int(tensors["num_tokens"])
    payload, factors = unpack_quantized_cache(tensors["window_cache"], False)
    inputs = dict(tensors, window_cache=payload, window_cache_scale=factors)
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
            inputs["window_cache"], inputs["window_cache_scale"], window_kv, inputs["window_slots"], quantize_mxfp8_cache,
        )
        packed_cache = pack_quantized_cache(cache, scale)
        indices = inputs["window_indices"]
        attended = golden_gather_sparse_attention(
            query, indices, packed_cache, indices, packed_cache, inputs["attn_sink"],
            compressed_keys=0, key_tile=64,
        )
        projected = golden_o_proj(
            attended, inputs["rope_cos"], inputs["rope_sin"], inputs["wo_a"], inputs["wo_b"], inputs["wo_b_scale"],
            wo_a_scale=inputs["wo_a_scale"], sequential_projection=True,
        )
        projection[:active].copy_(projected)
    tensors["output"].copy_(golden_hc_post(projection, tensors["residual"], tensors["post_mix"], tensors["residual_mix"]))
    tensors["window_cache"].copy_(packed_cache)


def build_attention_specs(seed: int = 43):
    """Build five shared-history queries, HC inputs and runtime specifications."""
    import torch
    from golden import ScalarSpec, TensorSpec
    from o_proj import build_o_proj_hc_post_specs
    import math
    from quant import pack_fp8_nz, pack_mx_b_scale
    from quant import quantize_mxfp8_upward_reference, quantize_grouped_mxfp8_weight
    from quant import quantize_mxfp8_cache

    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen)

    def weight(k, n):
        payload, scales = quantize_mxfp8_upward_reference(rand(n, k) / math.sqrt(k))
        return (payload.T.contiguous(), pack_mx_b_scale(scales.T.contiguous()).view(torch.float8_e8m0fnu))

    qa, qas = weight(D, Q_LORA)
    qb, qbs = weight(Q_LORA, LOCAL_H * HEAD_DIM)
    kv, kvs = weight(D, HEAD_DIM)
    ob, obs = weight(LOCAL_O_WIDTH, D)
    qa = pack_fp8_nz(qa)
    qb = pack_fp8_nz(qb)
    kv = pack_fp8_nz(kv)
    ob = pack_fp8_nz(ob)
    cache, cache_scale = quantize_mxfp8_cache(rand(DSPARK_QUERY_WIDTH + 1, 128, 1, HEAD_DIM).to(torch.bfloat16))
    values = dict(
        x=rand(DSPARK_QUERY_WIDTH, D).to(torch.bfloat16),
        wq_a=qa, wq_a_scale=qas, q_norm_weight=(rand(Q_LORA) * 0.1 + 1).to(torch.bfloat16),
        wq_b=qb, wq_b_scale=qbs,
        wkv=kv, wkv_scale=kvs, kv_norm_weight=(rand(HEAD_DIM) * 0.1 + 1).to(torch.bfloat16),
        attn_sink=rand(LOCAL_H) * 2,
    )
    wo_a = (rand(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN) / math.sqrt(O_GROUP_IN)).to(torch.bfloat16)
    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(wo_a)
    values.update(wo_b=ob, wo_b_scale=obs)
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
    for name in ("x", "rope_cos", "rope_sin", "window_slots", "window_indices"):
        current = values[name]
        fill = -1 if name in ("window_slots", "window_indices") else 0
        padded = torch.full((MOE_TOKENS, *current.shape[1:]), fill, dtype=current.dtype)
        padded[:DSPARK_QUERY_WIDTH] = current
        values[name] = padded
    values["window_cache"] = pack_quantized_cache(cache, cache_scale)
    specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
    hc_specs = build_o_proj_hc_post_specs(MOE_TOKENS, pad_to_tile=False)
    specs += [spec for spec in hc_specs if spec.name in ("residual", "post_mix", "residual_mix", "output")]
    specs.append(ScalarSpec("num_tokens", torch.int32, DSPARK_QUERY_WIDTH, compile_runtime=True))
    return specs


if __name__ == "__main__":
    import argparse
    import functools

    parser = argparse.ArgumentParser(description="V4.1 DSpark non-causal SWA validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.execution_mode:
        parser.error("DSpark native attention requires A5 device execution; simulator support is unavailable")
    from golden import ratio_allclose, run
    from o_proj import compare_o_proj_hc_post

    result = run(
        fn=dspark_attention_test,
        specs=build_attention_specs(),
        golden_fn=golden_dspark_attention,
        compare_fn={
            "window_cache": packed_cache_compare(
                ratio_allclose(rtol=0.05, atol=0.05, max_error_ratio=0.0),
                ratio_allclose(max_error_ratio=0.0),
                fp4=False,
            ),
            "output": functools.partial(
                compare_o_proj_hc_post, compare=ratio_allclose(atol=0.03, rtol=0.03, max_error_ratio=0.001),
            ),
        },
        config={"backend_type": args.backend_type, "device_id": args.device,
                "execution_mode": args.execution_mode},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
