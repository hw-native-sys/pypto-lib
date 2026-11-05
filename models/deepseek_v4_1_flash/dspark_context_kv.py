# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Publish one V4.1 DSpark draft layer's context KV into its SWA cache."""

import pypto.language as pl
import torch

from config import FLASH, ORI_BLOCKS_DYN
from qkv_proj_rope import golden_rope_interleave, quantize_projection_input, kv_proj, window_publish
from gemm import golden_mxfp8_linear_grouped_sum
from quant import quantize_mxfp8_cache
from rmsnorm import golden_rms_norm


T_DYN = pl.dynamic("V41_DSPARK_CONTEXT_T_DYN")
D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
ROPE_DIM = FLASH.qk_rope_head_dim


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
    from quant import unpack_fp8_nz

    if weight.dtype == torch.float8_e4m3fn:
        weight = unpack_fp8_nz(weight)
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
        projected = golden_rms_norm(golden_mxfp8_linear_grouped_sum(main_x, weight, weight_scale), norm_weight)
        rope_dim = cos.shape[-1] * 2
        kv = torch.cat(
            (projected[..., :-rope_dim], golden_rope_interleave(projected[..., -rope_dim:], cos, sin)),
            dim=-1,
        )
        payload, scale = quantize_mxfp8_cache(kv[active])
        updated_cache.reshape(-1, cache.shape[-1])[rows] = payload
        updated_scale.reshape(-1, cache_scale.shape[-1])[rows] = scale
    return updated_cache, updated_scale


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
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Use the draft layer's own WKV, norm, and paged cache mapping."""
    tokens = pl.tensor.dim(main_x, 0)
    projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    kv_input_payload, kv_input_scales = quantize_projection_input(main_x, num_tokens)
    window_input_ready = pl.system.task_dummy(deps=[])
    projected = kv_proj(kv_input_payload, kv_input_scales, wkv, wkv_scale, projected, num_tokens)
    window_publish(
        projected, kv_norm_weight, rope_cos, rope_sin,
        context_slots, window_cache, window_cache_scale,
        num_tokens, window_input_ready,
    )
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
    window_cache: pl.InOut[pl.Tensor[[3, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[3, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Publish through a middle-layer view while retaining both guard layers."""
    main_x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    context_slots.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(1, ORI_BLOCKS_DYN)
    dspark_context_kv(
        main_x,
        wkv,
        wkv_scale,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        context_slots,
        window_cache[1],
        window_cache_scale[1],
        num_tokens,
    )
    return window_cache, window_cache_scale


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

    slots = inputs["context_slots"]
    layer_rows = actual_outputs["window_cache"].shape[1] * 128
    active = slots[slots >= 0].long().unique() + layer_rows
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark context KV validation")
    parser.add_argument("-p", "--platform", choices=("a5", "a5sim"), default="a5sim")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    from golden import ScalarSpec, TensorSpec, run
    from quant import pack_fp8_nz, pack_mx_b_scale

    torch.manual_seed(42)
    tokens = 8
    main_x = torch.randn(tokens, D, dtype=torch.bfloat16) * 0.1
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
        payload, scale = golden_context_kv_cache(
            values["main_x"],
            values["wkv"],
            values["wkv_scale"],
            values["kv_norm_weight"],
            values["rope_cos"],
            values["rope_sin"],
            values["context_slots"],
            values["window_cache"][1],
            values["window_cache_scale"][1],
        )
        values["window_cache"][1].copy_(payload)
        values["window_cache_scale"][1].copy_(scale)

    specs = [
        TensorSpec("main_x", [tokens, D], torch.bfloat16, init_value=main_x),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=weight),
        TensorSpec("wkv_scale", [D // 32, HEAD_DIM], torch.float8_e8m0fnu, init_value=weight_scale),
        TensorSpec("kv_norm_weight", [HEAD_DIM], torch.bfloat16, init_value=1.0),
        TensorSpec("rope_cos", [tokens, ROPE_DIM // 2], torch.float32, init_value=cos),
        TensorSpec("rope_sin", [tokens, ROPE_DIM // 2], torch.float32, init_value=sin),
        TensorSpec("context_slots", [tokens], torch.int64, init_value=slots),
        TensorSpec("window_cache", [3, 2, 128, 1, HEAD_DIM], torch.float8_e4m3fn, init_value=cache),
        TensorSpec(
            "window_cache_scale", [3, 2, 128, 1, HEAD_DIM // 32], torch.bfloat16, init_value=cache_scale
        ),
        ScalarSpec("num_tokens", torch.int32, tokens, compile_runtime=True),
    ]
    compare = {name: _compare_context_cache for name in ("window_cache", "window_cache_scale")}
    result = run(
        fn=dspark_context_kv_test,
        specs=specs,
        golden_fn=golden,
        compare_fn=compare,
        config={"backend_type": "a5", "device_id": args.device,
                "execution_mode": args.execution_mode or args.platform == "a5sim"},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)
