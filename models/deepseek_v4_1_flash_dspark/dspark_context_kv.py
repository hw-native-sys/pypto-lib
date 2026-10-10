# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Publish one V4.1 DSpark draft layer's context KV into its SWA cache."""

import pypto.language as pl
import torch

from models.deepseek_v4_1_flash.attention_ops import make_mx_projection, make_norm, make_rope
from models.deepseek_v4_1_flash.config import (
    D,
    HEAD_DIM,
    ORI_BLOCKS_DYN,
    ROPE_DIM,
    T_DYN,
)
from models.deepseek_v4_1_flash.decode_attn_swa import publish_window
from models.deepseek_v4_1_flash.golden import rms_norm, rope_interleave
from models.deepseek_v4_1_flash.quantization import mxfp8_linear, quantize_mxfp8_cache
from models.deepseek_v4_1_flash_dspark.config import DSPARK_QUERY_WIDTH


def golden_context_kv_cache(
    main_x: torch.Tensor,
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
    if slots.shape != main_x.shape[:1]:
        raise ValueError("context slots must have one entry per token")
    if cache.shape[:-1] != cache_scale.shape[:-1] or cache.shape[-1] != cache_scale.shape[-1] * 32:
        raise ValueError("context cache payload and scale shapes disagree")
    active = slots >= 0
    rows = slots[active].long()
    if rows.numel() and (
        int(rows.max()) >= cache.numel() // cache.shape[-1] or rows.unique().numel() != rows.numel()
    ):
        raise ValueError("context slots must be in range and unique")
    updated_cache = cache.clone()
    updated_scale = cache_scale.clone()
    if rows.numel():
        projected = rms_norm(mxfp8_linear(main_x, weight, weight_scale), norm_weight)
        rope_dim = cos.shape[-1] * 2
        kv = torch.cat(
            (projected[..., :-rope_dim], rope_interleave(projected[..., -rope_dim:], cos, sin)),
            dim=-1,
        )
        payload, scale = quantize_mxfp8_cache(kv[active])
        updated_cache.reshape(-1, cache.shape[-1])[rows] = payload
        updated_scale.view(torch.uint8).reshape(-1, cache_scale.shape[-1])[rows] = scale.view(torch.uint8)
    return updated_cache, updated_scale


_project = make_mx_projection(D, HEAD_DIM, name_hint="dspark_context_kv_proj")
_normalize = make_norm(HEAD_DIM, name_hint="dspark_context_kv_norm")
_rotate = make_rope(1, name_hint="dspark_context_kv_rope")


@pl.jit.inline(auto_scope=False)
def dspark_context_kv(
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    context_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Use the draft layer's own WKV, norm, and paged cache mapping."""
    tokens = pl.tensor.dim(main_x, 0)
    projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    normalized = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    _project(main_x, wkv, wkv_scale, projected, num_tokens)
    _normalize(projected, kv_norm_weight, normalized, num_tokens)
    _rotate(normalized, rope_cos, rope_sin, rotated, num_tokens)
    ready = pl.system.task_dummy(deps=[])
    publish_window(rotated, context_slots, window_cache, window_cache_scale, num_tokens, ready)
    return window_cache, window_cache_scale


@pl.jit
def dspark_context_kv_test(
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    context_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0]],
    num_tokens: pl.Scalar[pl.INT32],
):
    main_x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    context_slots.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(0, ORI_BLOCKS_DYN)
    return dspark_context_kv(
        main_x,
        wkv,
        wkv_scale,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        context_slots,
        window_cache,
        window_cache_scale,
        num_tokens,
    )


def _compare_context_cache(
    _actual,
    _expected,
    *,
    actual_outputs,
    expected_outputs,
    inputs,
    **_kwargs,
):
    from models.deepseek_v4_1_flash.quantization import dequantize_mxfp8_cache

    slots = inputs["context_slots"]
    active = slots[slots >= 0].long().unique()
    actual = dequantize_mxfp8_cache(
        actual_outputs["window_cache"], actual_outputs["window_cache_scale"]
    ).reshape(-1, HEAD_DIM)
    expected = dequantize_mxfp8_cache(
        expected_outputs["window_cache"], expected_outputs["window_cache_scale"]
    ).reshape(-1, HEAD_DIM)
    relative_l2 = (actual[active].float() - expected[active].float()).norm() / expected[
        active
    ].float().norm().clamp_min(1e-12)
    inactive = torch.ones(actual.shape[0], dtype=torch.bool)
    inactive[active] = False
    untouched = all(
        torch.equal(
            actual_outputs[name].contiguous().view(torch.uint8).reshape(actual.shape[0], -1)[inactive],
            expected_outputs[name].contiguous().view(torch.uint8).reshape(actual.shape[0], -1)[inactive],
        )
        for name in ("window_cache", "window_cache_scale")
    )
    return bool(torch.isfinite(relative_l2) and relative_l2 <= 0.05 and untouched), (
        f"active context KV relative L2={relative_l2.item():.6g}; inactive rows unchanged={untouched}"
    )


def run_context_case(*, platform: str = "a5sim", device_id: int = 0, compile_only: bool = False):
    """Compile or numerically validate one draft layer's context publication."""
    from golden import ScalarSpec, TensorSpec, run
    from models.deepseek_v4_1_flash.quantization import pack_mx_b_scale

    torch.manual_seed(42)
    tokens = DSPARK_QUERY_WIDTH
    main_x = torch.randn(tokens, D, dtype=torch.bfloat16) * 0.1
    weight = (torch.randn(D, HEAD_DIM) * 0.01).to(torch.float8_e4m3fn)
    weight_scale = pack_mx_b_scale(torch.full((D // 32, HEAD_DIM), 127, dtype=torch.uint8))
    weight_scale = weight_scale.view(torch.float8_e8m0fnu)
    cos = torch.ones(tokens, ROPE_DIM // 2)
    sin = torch.zeros_like(cos)
    sin[0].fill_(1.0)
    cos[0].zero_()
    cache = torch.zeros(1, 128, 1, HEAD_DIM, dtype=torch.float8_e4m3fn)
    cache_scale = torch.full((1, 128, 1, HEAD_DIM // 32), 127, dtype=torch.uint8)
    cache_scale = cache_scale.view(torch.float8_e8m0fnu)

    def golden(values):
        payload, scale = golden_context_kv_cache(
            values["main_x"],
            values["wkv"],
            values["wkv_scale"],
            values["kv_norm_weight"],
            values["rope_cos"],
            values["rope_sin"],
            values["context_slots"],
            values["window_cache"],
            values["window_cache_scale"],
        )
        values["window_cache"].copy_(payload)
        values["window_cache_scale"].copy_(scale)

    specs = [
        TensorSpec("main_x", [tokens, D], torch.bfloat16, init_value=main_x),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=weight),
        TensorSpec("wkv_scale", [D // 32, HEAD_DIM], torch.float8_e8m0fnu, init_value=weight_scale),
        TensorSpec("kv_norm_weight", [HEAD_DIM], torch.bfloat16, init_value=1.0),
        TensorSpec("rope_cos", [tokens, ROPE_DIM // 2], torch.float32, init_value=cos),
        TensorSpec("rope_sin", [tokens, ROPE_DIM // 2], torch.float32, init_value=sin),
        TensorSpec("context_slots", [tokens], torch.int64, init_value=torch.arange(tokens) * 2 + 3),
        TensorSpec("window_cache", [1, 128, 1, HEAD_DIM], torch.float8_e4m3fn, init_value=cache),
        TensorSpec(
            "window_cache_scale", [1, 128, 1, HEAD_DIM // 32], torch.float8_e8m0fnu, init_value=cache_scale
        ),
        ScalarSpec("num_tokens", torch.int32, tokens, compile_runtime=True),
    ]
    compare = {name: _compare_context_cache for name in ("window_cache", "window_cache_scale")}
    return run(
        fn=dspark_context_kv_test,
        specs=specs,
        golden_fn=golden,
        compare_fn=compare,
        config={"platform": platform, "device_id": device_id},
        compile_only=compile_only,
    )


if __name__ == "__" + "main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark context KV validation")
    parser.add_argument("-p", "--platform", choices=("a5", "a5sim"), default="a5sim")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    result = run_context_case(platform=args.platform, device_id=args.device, compile_only=args.compile_only)
    print(result)
    if not result.passed:
        raise SystemExit(1)
